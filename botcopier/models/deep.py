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
            window: int | None = None,
            *,
            dim: int = 64,
            depth: int = 2,
            heads: int = 4,
            ff_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if heads < 1:
                raise ValueError("heads must be positive")
            if dim % heads != 0:
                raise ValueError("dim must be divisible by heads")
            effective_window = int(window or 1)
            if effective_window < 1:
                raise ValueError("window must be positive")
            self.num_features = int(num_features)
            self.window = effective_window
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
            self.positional = PositionalEncoding(effective_window, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() != 3:
                raise ValueError(
                    "expected input of shape (batch, window, features) or (batch, features)"
                )
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


    class CrossModalTransformer(nn.Module):
        """Fuse price and news sequences using dual encoders with cross-attention."""

        def __init__(
            self,
            price_features: int,
            news_features: int,
            price_window: int,
            news_window: int,
            *,
            dim: int = 64,
            depth: int = 2,
            heads: int = 4,
            ff_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if heads < 1:
                raise ValueError("heads must be positive")
            if dim % heads != 0:
                raise ValueError("dim must be divisible by heads")
            if price_window < 1 or news_window < 1:
                raise ValueError("windows must be positive")
            if price_features < 1 or news_features < 1:
                raise ValueError("feature dimensions must be positive")

            self.price_features = int(price_features)
            self.news_features = int(news_features)
            self.price_window = int(price_window)
            self.news_window = int(news_window)

            encoder_layer = nn.TransformerEncoderLayer(
                dim,
                heads,
                ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            news_layer = nn.TransformerEncoderLayer(
                dim,
                heads,
                ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )

            self.price_proj = nn.Linear(price_features, dim)
            self.news_proj = nn.Linear(news_features, dim)
            self.price_pos = PositionalEncoding(self.price_window, dim)
            self.news_pos = PositionalEncoding(self.news_window, dim)
            self.price_encoder = nn.TransformerEncoder(encoder_layer, depth)
            self.news_encoder = nn.TransformerEncoder(news_layer, depth)
            self.cross_attn = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            fused_dim = dim * 3
            self.fuse_norm = nn.LayerNorm(fused_dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(fused_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, 1),
            )

        def _validate_inputs(
            self, price: torch.Tensor, news: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if price.dim() == 2:
                price = price.unsqueeze(1)
            if news.dim() == 2:
                news = news.unsqueeze(1)
            if price.dim() != 3 or news.dim() != 3:
                raise ValueError(
                    "expected price and news inputs of shape (batch, window, features)"
                )
            if price.size(-1) != self.price_features:
                raise ValueError("price feature dimension mismatch")
            if news.size(-1) != self.news_features:
                raise ValueError("news feature dimension mismatch")
            if price.size(1) > self.price_window:
                raise ValueError("price sequence exceeds configured window")
            if news.size(1) > self.news_window:
                raise ValueError("news sequence exceeds configured window")
            return price, news

        def forward(self, price: torch.Tensor, news: torch.Tensor) -> torch.Tensor:
            price, news = self._validate_inputs(price, news)
            price_h = self.price_proj(price)
            price_h = self.price_pos(price_h)
            price_h = self.price_encoder(price_h)

            news_h = self.news_proj(news)
            news_h = self.news_pos(news_h)
            news_h = self.news_encoder(news_h)

            attn_out, _ = self.cross_attn(price_h, news_h, news_h)
            price_pool = price_h.mean(dim=1)
            attn_pool = attn_out.mean(dim=1)
            news_pool = news_h.mean(dim=1)
            fused = torch.cat([price_pool, attn_pool, news_pool], dim=-1)
            fused = self.dropout(self.fuse_norm(fused))
            return self.head(fused).squeeze(-1)


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
            self.num_inputs = int(num_inputs)
            self.channels = tuple(int(c) for c in channels)
            layers: list[nn.Module] = []
            in_channels = num_inputs
            for i, out_channels in enumerate(self.channels):
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
            if x.size(1) != self.num_inputs:
                raise ValueError("channel dimension mismatch")
            return self.network(x)


    class MixtureOfExperts(nn.Module):
        """Simple mixture of experts with a softmax gating network."""

        def __init__(
            self,
            n_features: int,
            n_regime_features: int,
            n_experts: int,
            *,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if n_experts < 1:
                raise ValueError("n_experts must be positive")
            if n_features < 1 or n_regime_features < 1:
                raise ValueError("feature dimensions must be positive")
            self.n_features = int(n_features)
            self.n_regime_features = int(n_regime_features)
            self.n_experts = int(n_experts)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.experts = nn.ModuleList(
                [nn.Linear(n_features, 1) for _ in range(self.n_experts)]
            )
            self.gating = nn.Linear(n_regime_features, self.n_experts)

        def forward(
            self, x: torch.Tensor, regime_features: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if x.dim() != 2:
                raise ValueError("expected expert features of shape (batch, features)")
            if regime_features.dim() != 2:
                raise ValueError("expected regime features of shape (batch, regime_features)")
            if x.size(0) != regime_features.size(0):
                raise ValueError("batch dimension mismatch between experts and regimes")
            if x.size(1) != self.n_features:
                raise ValueError("expert feature dimension mismatch")
            if regime_features.size(1) != self.n_regime_features:
                raise ValueError("regime feature dimension mismatch")
            x = self.dropout(x)
            gate_logits = self.gating(regime_features)
            gate = torch.softmax(gate_logits, dim=1)
            expert_logits = torch.cat([expert(x) for expert in self.experts], dim=1)
            expert_prob = torch.sigmoid(expert_logits)
            out = (gate * expert_prob).sum(dim=1, keepdim=True)
            return out, gate


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

    class CrossModalTransformer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for CrossModalTransformer")

    class TemporalConvNet:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TemporalConvNet")

    class TCNClassifier:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TemporalConvNet")

    class MixtureOfExperts:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for MixtureOfExperts")


__all__ = [
    "TabTransformer",
    "CrossModalTransformer",
    "TemporalConvNet",
    "TCNClassifier",
    "MixtureOfExperts",
]

