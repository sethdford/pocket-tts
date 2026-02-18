import math
import warnings

import mlx.core as mx
import mlx.nn as nn

from pocket_tts.modules.stateful_module import StatefulModule


def get_extra_padding_for_conv1d(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`.
    Input x is in NLC format: (batch, length, channels).
    """
    length = x.shape[1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: mx.array, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Input x is in NLC format: (batch, length, channels).
    Extra padding is added at the end along the length dimension.
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    if extra_padding > 0:
        x = mx.pad(x, [(0, 0), (0, int(extra_padding)), (0, 0)])
    return x


class StreamingConv1d(StatefulModule):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization. Uses MLX NLC (batch, length, channels) format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ["constant", "replicate"], pad_mode
        self.pad_mode = pad_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._kernel_size_val = kernel_size
        self._stride_val = stride
        self._dilation_val = dilation
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @property
    def _stride(self) -> int:
        return self._stride_val

    @property
    def _kernel_size(self) -> int:
        return self._kernel_size_val

    @property
    def _effective_kernel_size(self) -> int:
        return (self._kernel_size - 1) * self._dilation_val + 1

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, mx.array]:
        stride = self._stride
        kernel = self._effective_kernel_size
        # NLC format: (batch, length, channels)
        previous = mx.zeros((batch_size, kernel - stride, self.in_channels))
        first = mx.ones((batch_size,), dtype=mx.bool_)
        return dict(previous=previous, first=first)

    def __call__(self, x, model_state: dict | None):
        # x is NLC: (B, T, C)
        B, T, C = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self.get_state(model_state)
        TP = state["previous"].shape[1]  # length dim in NLC
        if TP and self.pad_mode == "replicate":
            assert T >= TP, "Not enough content to pad streaming."
            init = x[:, :1, :]  # (B, 1, C)
            state["previous"] = mx.where(
                state["first"].reshape(-1, 1, 1), init, state["previous"]
            )
        if TP:
            x = mx.concatenate([state["previous"], x], axis=1)
        y = self.conv(x)
        if TP:
            state["previous"] = x[:, -TP:, :]
            if self.pad_mode == "replicate":
                state["first"] = mx.zeros_like(state["first"])
        return y


class _GroupedConvTranspose1d(nn.Module):
    """ConvTranspose1d with groups support via mx.conv_transpose1d.
    Uses same attribute names (weight, bias) as nn.ConvTranspose1d for weight loading.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._groups = groups
        self._stride = stride
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            -scale, scale, (out_channels, kernel_size, in_channels // groups)
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(x, self.weight, stride=self._stride, groups=self._groups)
        if self.bias is not None:
            y = y + self.bias
        return y


class StreamingConvTranspose1d(StatefulModule):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization. Uses MLX NLC (batch, length, channels) format.
    Supports groups via _GroupedConvTranspose1d when groups > 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self._kernel_size_val = kernel_size
        self._stride_val = stride
        self._has_bias = bias
        if groups == 1:
            self.convtr = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, bias=bias
            )
        else:
            self.convtr = _GroupedConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
            )

    @property
    def _stride(self) -> int:
        return self._stride_val

    @property
    def _kernel_size(self) -> int:
        return self._kernel_size_val

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, mx.array]:
        K = self._kernel_size
        S = self._stride
        # NLC format: (batch, length, channels)
        return dict(partial=mx.zeros((batch_size, K - S, self.out_channels)))

    def __call__(self, x, mimi_state: dict):
        state = self.get_state(mimi_state)
        layer_state = state["partial"]
        y = self.convtr(x)
        # y is NLC: (B, T_out, C_out)
        PT = layer_state.shape[1]  # length dim in NLC
        if PT > 0:
            # Add overlap from previous frame
            y_updated = mx.concatenate([
                (y[:, :PT, :] + layer_state),
                y[:, PT:, :]
            ], axis=1)
            bias = self.convtr.bias if self._has_bias else None
            for_partial = y_updated[:, -PT:, :]
            if bias is not None:
                for_partial = for_partial - bias.reshape(1, 1, -1)
            state["partial"] = for_partial
            y = y_updated[:, :-PT, :]
        return y
