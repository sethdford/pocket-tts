/**
 * amx_flow_fused.c — Fused LSD decode on CPU via Accelerate BLAS (AMX-backed).
 *
 * Runs the entire SimpleMLPAdaLN flow network + LSD Euler integration loop
 * in a single C function call. Zero Python overhead, zero temporary numpy
 * allocations. All intermediates live on the stack (~30KB, fits in L1 cache).
 *
 * BLAS calls go through cblas_sgemv (Apple Accelerate), which dispatches to
 * the AMX coprocessor on M1-M3 or ARM SME on M4+.
 *
 * Build: cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *        -o libamx_flow_fused.dylib amx_flow_fused.c
 */

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/* Maximum model dimension. Stack buffers are sized to this. */
#define MAX_DIM 1024
#define MAX_3DIM 3072  /* 3 * MAX_DIM for adaLN modulation */

/* -----------------------------------------------------------------------
 * Primitive ops (all inline, auto-vectorizable by clang with -O3)
 * ----------------------------------------------------------------------- */

static inline void vec_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static inline void vec_rms_norm(float *out, const float *x, const float *alpha,
                                float eps, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * alpha[i] * inv_rms;
    }
}

static inline void vec_layer_norm(float *out, const float *x,
                                  const float *weight, const float *bias,
                                  float eps, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);

    if (weight && bias) {
        for (int i = 0; i < n; i++) {
            out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            out[i] = (x[i] - mean) * inv_std;
        }
    }
}

static inline void vec_modulate(float *out, const float *x,
                                const float *shift, const float *scale, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * (1.0f + scale[i]) + shift[i];
    }
}

static inline void vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

static inline void vec_add(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

static inline void vec_addscaled(float *dst, const float *a,
                                 const float *b, float s, int n) {
    /* dst = a + b * s */
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i] * s;
}

static inline void vec_gate_residual(float *x, const float *gate,
                                     const float *h, int n) {
    /* x += gate * h */
    for (int i = 0; i < n; i++) x[i] += gate[i] * h[i];
}

static inline void vec_avg(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) out[i] = (a[i] + b[i]) * 0.5f;
}

/* -----------------------------------------------------------------------
 * Linear projection: out = weight @ x + bias
 * Uses cblas_sgemv (AMX-backed on Apple Silicon).
 *
 * weight: (out_dim, in_dim) row-major
 * x:      (in_dim,)
 * bias:   (out_dim,) or NULL
 * out:    (out_dim,)
 * ----------------------------------------------------------------------- */
static inline void linear(float *out, const float *weight, const float *x,
                          const float *bias, int out_dim, int in_dim) {
    if (bias) {
        vec_copy(out, bias, out_dim);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    out_dim, in_dim, 1.0f, weight, in_dim, x, 1, 1.0f, out, 1);
    } else {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    out_dim, in_dim, 1.0f, weight, in_dim, x, 1, 0.0f, out, 1);
    }
}

/* -----------------------------------------------------------------------
 * Weight buffer layout
 *
 * All weights are packed contiguously as float32 in this order:
 *
 * [input_proj_w (mc*ic)] [input_proj_b (mc)]
 * [cond_embed_w (mc*cc)] [cond_embed_b (mc)]
 * For each timestep embedder (x2):
 *   [freqs (fes/2)]
 *   [mlp0_w (mc*fes)] [mlp0_b (mc)]
 *   [mlp2_w (mc*mc)]  [mlp2_b (mc)]
 *   [rms_alpha (mc)]
 * For each res_block (x num_res_blocks):
 *   [ln_w (mc)] [ln_b (mc)]
 *   [ada_w (3mc*mc)] [ada_b (3mc)]
 *   [mlp0_w (mc*mc)] [mlp0_b (mc)]
 *   [mlp2_w (mc*mc)] [mlp2_b (mc)]
 * Final layer:
 *   [ada_w (2mc*mc)]   [ada_b (2mc)]
 *   [norm_w (mc)]      [norm_b (mc)]  (zeros if no affine)
 *   [linear_w (ic*mc)] [linear_b (ic)]
 *
 * mc = model_channels, ic = in_channels, cc = cond_channels, fes = freq_embed_size
 * ----------------------------------------------------------------------- */

/* Helper: advance pointer by n floats */
#define ADV(ptr, n) ((ptr) += (n))

/**
 * lsd_decode_fused — Run the full LSD decode on CPU/AMX in one call.
 *
 * @param weights       Pre-packed weight buffer (see layout above)
 * @param conditioning  Transformer output vector, length cond_channels (1024)
 * @param noise         Starting noise vector, length in_channels (32)
 * @param output        Output latent vector, length in_channels (32)
 * @param num_steps     Number of Euler integration steps (typically 4)
 * @param mc            Model channels (512)
 * @param ic            Input/output channels (32)
 * @param cc            Conditioning channels (1024)
 * @param fes           Frequency embedding size (256)
 * @param num_blocks    Number of residual blocks (6)
 * @param rms_eps       Epsilon for RMSNorm (1e-5)
 * @param ln_eps        Epsilon for LayerNorm (1e-6)
 * @param has_final_affine  Whether final LayerNorm has affine params (0 or 1)
 */
void lsd_decode_fused(
    const float *weights,
    const float *conditioning,
    const float *noise,
    float *output,
    int num_steps,
    int mc,
    int ic,
    int cc,
    int fes,
    int num_blocks,
    float rms_eps,
    float ln_eps,
    int has_final_affine
) {
    /* --- Parse weight pointers from packed buffer --- */
    const float *p = weights;

    const float *input_proj_w = p; ADV(p, mc * ic);
    const float *input_proj_b = p; ADV(p, mc);

    const float *cond_embed_w = p; ADV(p, mc * cc);
    const float *cond_embed_b = p; ADV(p, mc);

    /* Timestep embedders (x2) */
    int half_fes = fes / 2;
    const float *te_freqs[2];
    const float *te_mlp0_w[2], *te_mlp0_b[2];
    const float *te_mlp2_w[2], *te_mlp2_b[2];
    const float *te_rms_alpha[2];

    for (int t = 0; t < 2; t++) {
        te_freqs[t]     = p; ADV(p, half_fes);
        te_mlp0_w[t]    = p; ADV(p, mc * fes);
        te_mlp0_b[t]    = p; ADV(p, mc);
        te_mlp2_w[t]    = p; ADV(p, mc * mc);
        te_mlp2_b[t]    = p; ADV(p, mc);
        te_rms_alpha[t] = p; ADV(p, mc);
    }

    /* ResBlock pointers */
    const float *rb_ln_w[16], *rb_ln_b[16];
    const float *rb_ada_w[16], *rb_ada_b[16];
    const float *rb_mlp0_w[16], *rb_mlp0_b[16];
    const float *rb_mlp2_w[16], *rb_mlp2_b[16];

    for (int b = 0; b < num_blocks && b < 16; b++) {
        rb_ln_w[b]   = p; ADV(p, mc);
        rb_ln_b[b]   = p; ADV(p, mc);
        rb_ada_w[b]  = p; ADV(p, 3 * mc * mc);
        rb_ada_b[b]  = p; ADV(p, 3 * mc);
        rb_mlp0_w[b] = p; ADV(p, mc * mc);
        rb_mlp0_b[b] = p; ADV(p, mc);
        rb_mlp2_w[b] = p; ADV(p, mc * mc);
        rb_mlp2_b[b] = p; ADV(p, mc);
    }

    /* Final layer */
    const float *final_ada_w    = p; ADV(p, 2 * mc * mc);
    const float *final_ada_b    = p; ADV(p, 2 * mc);
    const float *final_norm_w   = p; ADV(p, mc);
    const float *final_norm_b   = p; ADV(p, mc);
    const float *final_linear_w = p; ADV(p, ic * mc);
    const float *final_linear_b = p; ADV(p, ic);

    /* --- Stack-allocated intermediate buffers --- */
    float current[MAX_DIM];       /* Current latent (ic) */
    float x[MAX_DIM];             /* Working state (mc) */
    float y[MAX_DIM];             /* Conditioning (mc) */
    float te_s[MAX_DIM];          /* Timestep embed s (mc) */
    float te_t[MAX_DIM];          /* Timestep embed t (mc) */
    float cond_emb[MAX_DIM];      /* Conditioning embed (mc) */
    float h[MAX_DIM];             /* Temp for MLP (mc) */
    float te_buf[MAX_DIM];        /* Temp for timestep embed (mc or fes) */
    float mod[MAX_3DIM];          /* adaLN modulation output (3mc or 2mc) */
    float ada_in[MAX_DIM];        /* SiLU(y) for adaLN (mc) */

    /* --- Pre-compute conditioning embedding (constant across all steps) --- */
    linear(cond_emb, cond_embed_w, conditioning, cond_embed_b, mc, cc);

    /* --- Pre-compute scalar_template = noise[0] (matching MLX behavior) --- */
    float scalar_template = noise[0];

    /* --- Copy noise to current state --- */
    vec_copy(current, noise, ic);

    float inv_steps = 1.0f / (float)num_steps;

    /* --- LSD Euler integration loop --- */
    for (int step = 0; step < num_steps; step++) {
        float s_val = (float)step * inv_steps * scalar_template;
        float t_val = (float)(step + 1) * inv_steps * scalar_template;

        /* ---- Timestep embeddings ---- */
        for (int te_idx = 0; te_idx < 2; te_idx++) {
            float tv = (te_idx == 0) ? s_val : t_val;
            const float *freqs = te_freqs[te_idx];

            /* Sinusoidal embedding: [cos(tv*freqs), sin(tv*freqs)] */
            float embedding[MAX_DIM];
            for (int i = 0; i < half_fes; i++) {
                float arg = tv * freqs[i];
                embedding[i] = cosf(arg);
                embedding[half_fes + i] = sinf(arg);
            }

            /* MLP: Linear -> SiLU -> Linear -> RMSNorm */
            linear(te_buf, te_mlp0_w[te_idx], embedding, te_mlp0_b[te_idx], mc, fes);
            vec_silu(te_buf, mc);
            float te_out[MAX_DIM];
            linear(te_out, te_mlp2_w[te_idx], te_buf, te_mlp2_b[te_idx], mc, mc);
            if (te_idx == 0) {
                vec_rms_norm(te_s, te_out, te_rms_alpha[te_idx], rms_eps, mc);
            } else {
                vec_rms_norm(te_t, te_out, te_rms_alpha[te_idx], rms_eps, mc);
            }
        }

        /* ---- Combined conditioning: y = (te_s + te_t) * 0.5 + cond_emb ---- */
        vec_avg(y, te_s, te_t, mc);
        vec_add(y, y, cond_emb, mc);

        /* ---- Input projection: current (ic) -> x (mc) ---- */
        linear(x, input_proj_w, current, input_proj_b, mc, ic);

        /* ---- ResBlocks ---- */
        for (int b = 0; b < num_blocks; b++) {
            /* adaLN modulation: SiLU(y) -> Linear -> split(shift, scale, gate) */
            vec_copy(ada_in, y, mc);
            vec_silu(ada_in, mc);
            linear(mod, rb_ada_w[b], ada_in, rb_ada_b[b], 3 * mc, mc);
            const float *shift = mod;
            const float *scale = mod + mc;
            const float *gate  = mod + 2 * mc;

            /* LayerNorm + modulate */
            vec_layer_norm(h, x, rb_ln_w[b], rb_ln_b[b], ln_eps, mc);
            vec_modulate(h, h, shift, scale, mc);

            /* MLP: Linear -> SiLU -> Linear */
            float mlp_out[MAX_DIM];
            linear(mlp_out, rb_mlp0_w[b], h, rb_mlp0_b[b], mc, mc);
            vec_silu(mlp_out, mc);
            linear(h, rb_mlp2_w[b], mlp_out, rb_mlp2_b[b], mc, mc);

            /* Residual: x += gate * h */
            vec_gate_residual(x, gate, h, mc);
        }

        /* ---- Final layer: adaLN + LayerNorm + Linear -> flow_dir (ic) ---- */
        vec_copy(ada_in, y, mc);
        vec_silu(ada_in, mc);
        linear(mod, final_ada_w, ada_in, final_ada_b, 2 * mc, mc);
        const float *fshift = mod;
        const float *fscale = mod + mc;

        vec_layer_norm(h, x, has_final_affine ? final_norm_w : NULL,
                       has_final_affine ? final_norm_b : NULL, ln_eps, mc);
        vec_modulate(h, h, fshift, fscale, mc);

        float flow_dir[MAX_DIM];
        linear(flow_dir, final_linear_w, h, final_linear_b, ic, mc);

        /* ---- Euler step: current += flow_dir * inv_steps ---- */
        vec_addscaled(current, current, flow_dir, inv_steps, ic);
    }

    /* --- Write result --- */
    vec_copy(output, current, ic);
}
