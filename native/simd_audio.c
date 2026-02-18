/**
 * simd_audio.c — ARM NEON SIMD-accelerated audio processing for pocket-tts.
 *
 * Replaces numpy-based PCM conversion with zero-allocation NEON intrinsics.
 * Processes 8 float32 samples per cycle (clip + scale + narrow to int16)
 * in a single pass, eliminating 3 intermediate numpy array allocations.
 *
 * Also provides a fast float32→bytes conversion for WAV streaming that
 * writes directly to a caller-provided output buffer.
 *
 * Build: cc -O3 -shared -fPIC -arch arm64 -o libsimd_audio.dylib simd_audio.c
 */

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/**
 * Convert float32 audio samples to int16 PCM in a single NEON pass.
 *
 * Fuses: clip(x, -1, 1) → scale(x * 32767) → cast(int16)
 * into vectorized NEON operations processing 8 samples per iteration.
 *
 * @param input   Source float32 buffer (audio samples in [-1, 1] range)
 * @param output  Destination int16 buffer (must be pre-allocated, n elements)
 * @param n       Number of samples to convert
 */
void float32_to_pcm16(const float *input, int16_t *output, size_t n) {
    const float32x4_t scale = vdupq_n_f32(32767.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t pos_one = vdupq_n_f32(1.0f);

    size_t i = 0;

    /* Process 8 samples per iteration (2 × 128-bit NEON registers) */
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(input + i);
        float32x4_t v1 = vld1q_f32(input + i + 4);

        /* Clip to [-1.0, 1.0] */
        v0 = vmaxq_f32(vminq_f32(v0, pos_one), neg_one);
        v1 = vmaxq_f32(vminq_f32(v1, pos_one), neg_one);

        /* Scale to int16 range */
        v0 = vmulq_f32(v0, scale);
        v1 = vmulq_f32(v1, scale);

        /* Convert float32 → int32 → int16 (narrow) */
        int32x4_t i0 = vcvtq_s32_f32(v0);
        int32x4_t i1 = vcvtq_s32_f32(v1);
        int16x4_t s0 = vmovn_s32(i0);
        int16x4_t s1 = vmovn_s32(i1);

        /* Combine and store 8 int16 values */
        int16x8_t combined = vcombine_s16(s0, s1);
        vst1q_s16(output + i, combined);
    }

    /* Process remaining 4 samples */
    if (i + 4 <= n) {
        float32x4_t v = vld1q_f32(input + i);
        v = vmaxq_f32(vminq_f32(v, pos_one), neg_one);
        v = vmulq_f32(v, scale);
        int32x4_t iv = vcvtq_s32_f32(v);
        int16x4_t sv = vmovn_s32(iv);
        vst1_s16(output + i, sv);
        i += 4;
    }

    /* Scalar remainder */
    for (; i < n; i++) {
        float v = input[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        output[i] = (int16_t)(v * 32767.0f);
    }
}

/**
 * Convert float32 audio to PCM16 bytes in a single pass (no intermediate alloc).
 *
 * This is the complete replacement for:
 *   chunk_int16 = np.clip(audio_chunk, -1.0, 1.0)
 *   chunk_int16 = (chunk_int16 * 32767).astype(np.int16)
 *   chunk_bytes = chunk_int16.tobytes()
 *
 * @param input       Source float32 buffer
 * @param output_bytes  Destination byte buffer (must be n * 2 bytes)
 * @param n           Number of samples
 */
void float32_to_pcm16_bytes(const float *input, uint8_t *output_bytes, size_t n) {
    float32_to_pcm16(input, (int16_t *)output_bytes, n);
}

