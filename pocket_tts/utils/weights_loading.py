import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

# Conv1d weight suffixes: PyTorch shape (out, in/groups, K) -> MLX shape (out, K, in/groups)
_CONV1D_WEIGHT_SUFFIXES = (".conv.weight", ".output_proj.weight")

# ConvTranspose1d weight suffixes: PyTorch shape (in, out/groups, K) -> MLX shape (out/groups, K, in)
_CONVTR_WEIGHT_SUFFIXES = (".convtr.weight",)


def _transpose_conv_weight(key: str, arr: mx.array) -> mx.array:
    """Transpose conv weights from PyTorch layout to MLX layout if needed."""
    if arr.ndim == 3:
        if any(key.endswith(s) for s in _CONV1D_WEIGHT_SUFFIXES):
            # PyTorch Conv1d: (out_ch, in_ch/g, K) -> MLX: (out_ch, K, in_ch/g)
            arr = mx.swapaxes(arr, 1, 2)
        elif any(key.endswith(s) for s in _CONVTR_WEIGHT_SUFFIXES):
            # PyTorch ConvTranspose1d: (in_ch, out_ch/g, K)
            # MLX ConvTranspose1d: (out_ch, K, in_ch) for groups=1
            # MLX grouped conv_transpose1d: (out_ch, K, in_ch/g)
            if arr.shape[1] == 1:
                # Depthwise grouped case: (dim, 1, K) -> (dim, K, 1)
                arr = mx.swapaxes(arr, 1, 2)
            else:
                # Non-grouped: (in_ch, out_ch, K) -> (out_ch, K, in_ch)
                arr = mx.transpose(arr, axes=(1, 2, 0))
    return arr


def get_flow_lm_state_dict(path: Path) -> dict[str, mx.array]:
    state_dict = {}
    tensors = mx.load(str(path))
    for key, arr in tensors.items():
        if (
            key.startswith("flow.w_s_t.")
            or key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
            or key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
        ):
            continue
        new_name = key
        if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
            new_name = "conditioner.embed.weight"
        if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
            new_name = "speaker_proj_weight"
        state_dict[new_name] = _transpose_conv_weight(new_name, arr)
    return state_dict


def get_mimi_state_dict(path: Path) -> dict[str, mx.array]:
    state_dict = {}
    tensors = mx.load(str(path))
    for key, arr in tensors.items():
        if key.startswith("model.quantizer.vq.") or key == "model.quantizer.logvar_proj.weight":
            continue
        new_name = key.removeprefix("model.")
        state_dict[new_name] = _transpose_conv_weight(new_name, arr)
    return state_dict


def load_safetensors_as_mlx(path: Path) -> dict[str, mx.array]:
    """Load a safetensors file directly into MLX arrays, converting conv weights."""
    tensors = mx.load(str(path))
    return {key: _transpose_conv_weight(key, arr) for key, arr in tensors.items()}
