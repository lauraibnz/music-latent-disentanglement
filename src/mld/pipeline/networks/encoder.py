import torch
import torch.nn as nn
import gin

from .blocks import ConvBlock1D


@gin.configurable
class Encoder1D(nn.Module):
    """Simple 1D encoder for latent sequence processing."""

    def __init__(
        self,
        in_channels=1,
        channels=(64, 128, 256, 512),
        kernel_size=5,
        strides=(2, 2, 2, 2),
        average_out=False,
        use_tanh=True,
    ):
        super().__init__()

        if len(channels) != len(strides):
            raise ValueError(
                f"channels and strides must have the same length, got {len(channels)} and {len(strides)}."
            )

        self.average_out = bool(average_out)
        self.use_tanh = bool(use_tanh)

        blocks = []
        prev_channels = int(in_channels)
        for out_channels, stride in zip(channels, strides):
            blocks.append(
                ConvBlock1D(
                    prev_channels,
                    int(out_channels),
                    int(kernel_size),
                    int(stride),
                )
            )
            prev_channels = int(out_channels)

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = int(channels[-1])

        self.total_ratio = 1
        for stride in strides:
            self.total_ratio *= int(stride)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, T]
        Returns:
            Encoded tensor [B, C', T'] or [B, C'] if average_out=True
        """
        x = self.blocks(x)

        if self.average_out:
            x = torch.mean(x, dim=-1)

        if self.use_tanh:
            x = torch.tanh(x)

        return x

    def get_output_length(self, input_length):
        """Calculate output sequence length given input length."""
        return int(input_length) // self.total_ratio


@gin.configurable
class TemporalPitchHead1D(nn.Module):
    """
    Time-aware pitch head operating on structure embeddings.

    Input:
        x: [B, C, T] (or [B, C], interpreted as T=1)
    Output:
        logits: [B, num_pitch_classes, T]
    """

    def __init__(
        self,
        in_channels=32,
        num_pitch_classes=128,
        hidden_channels=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        in_channels = int(in_channels)
        num_pitch_classes = int(num_pitch_classes)
        hidden_channels = int(hidden_channels)
        dropout = float(dropout)

        layers = []
        c_in = in_channels
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Conv1d(c_in, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            c_in = hidden_channels
        layers.append(nn.Conv1d(c_in, num_pitch_classes, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError(f"TemporalPitchHead1D expects [B, C, T] or [B, C], got {tuple(x.shape)}")
        return self.net(x)
