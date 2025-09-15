"""Deep learning architectures used by BotCopier models."""

from __future__ import annotations

from typing import Sequence

try:  # pragma: no cover - optional dependency handling
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


if _HAS_TORCH:

    class PositionalEncoding(nn.Module):
        """Learnable positional encoding for temporal sequences."""

        def __init__(self, window: int, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, window, dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            if self.weight.size(1) < x.size(1):
                raise ValueError("sequence length exceeds positional encoding size")
            return x + self.weight[:, : x.size(1)]


    class TabTransformer(nn.Module):
        """Transformer encoder tailored for tabular temporal windows."""

        def __init__(
            self,
            num_features: int,
            window: int,
            *,
            dim: int = 64,
            depth: int = 2,
            heads: int = 4,
            ff_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if window < 1:
                raise ValueError("window must be positive")
            self.num_features = num_features
            self.window = window
            self.input_proj = nn.Linear(num_features, dim)
            encoder_layer = nn.TransformerEncoderLayer(
                dim,
                heads,
                ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, depth)
            self.positional = PositionalEncoding(window, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3:
                raise ValueError("expected input of shape (batch, window, features)")
            if x.size(-1) != self.num_features:
                raise ValueError("feature dimension mismatch")
            if x.size(1) > self.window:
                raise ValueError("sequence length exceeds configured window")
            h = self.input_proj(x)
            h = self.positional(h)
            h = self.encoder(h)
            h = self.norm(h)
            h = h.mean(dim=1)
            h = self.dropout(h)
            return self.head(h).squeeze(-1)


    class TemporalBlock(nn.Module):
        """Residual block used by :class:`TemporalConvNet`."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(dropout)
            self.conv2 = nn.utils.weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(dropout)
            self.downsample = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else None
            )
            self.out_relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.drop1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.drop2(out)
            res = x if self.downsample is None else self.downsample(x)
            out = out[:, :, : res.size(2)]  # match residual length
            return self.out_relu(out + res)


    class TemporalConvNet(nn.Module):
        """Temporal convolutional network with exponentially dilated layers."""

        def __init__(
            self,
            num_inputs: int,
            channels: Sequence[int],
            *,
            kernel_size: int = 3,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if kernel_size < 1:
                raise ValueError("kernel_size must be positive")
            layers: list[nn.Module] = []
            in_channels = num_inputs
            for i, out_channels in enumerate(channels):
                dilation = 2**i
                layers.append(
                    TemporalBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3:
                raise ValueError("expected input of shape (batch, channels, time)")
            return self.network(x)


    class TCNClassifier(nn.Module):
        """Binary classifier built on top of :class:`TemporalConvNet`."""

        def __init__(
            self,
            num_features: int,
            window: int,
            *,
            channels: Sequence[int] | None = None,
            kernel_size: int = 3,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.num_features = num_features
            self.window = window
            chs = tuple(channels or (64, 64))
            self.tcn = TemporalConvNet(
                num_features,
                chs,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.head = nn.Linear(chs[-1], 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3:
                raise ValueError("expected input of shape (batch, window, features)")
            if x.size(-1) == self.window and x.size(1) == self.num_features:
                # allow (batch, features, window)
                x = x
            else:
                if x.size(-1) != self.num_features:
                    raise ValueError("feature dimension mismatch")
                x = x.transpose(1, 2)
            feat = self.tcn(x)
            out = feat[:, :, -1]
            return self.head(out).squeeze(-1)


else:  # pragma: no cover - fallback definitions when torch is unavailable

    class TabTransformer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TabTransformer")

    class TemporalConvNet:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TemporalConvNet")

    class TCNClassifier:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TemporalConvNet")


__all__ = ["TabTransformer", "TemporalConvNet", "TCNClassifier"]

