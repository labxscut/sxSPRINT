"""SPRINT neural network model.

The implementation is adapted from the project notebook and keeps the core
architecture used in the manuscript: a residual Conv1D SNP encoder, a PRS
projection branch, late SNP-PRS fusion, and an adaptive classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Residual 1D convolution block for SNP feature maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class SPRINT(nn.Module):
    """SNP-PRS Residual Integration model."""

    def __init__(
        self,
        snp_input_dim: int,
        prs_input_dim: int,
        snp_channels: tuple[int, int] = (32, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.snp_encoder = nn.Sequential(
            nn.Conv1d(1, snp_channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock1D(snp_channels[0], snp_channels[0], stride=1, dropout=dropout),
            ResidualBlock1D(snp_channels[0], snp_channels[1], stride=2, dropout=dropout),
            nn.Flatten(),
        )

        with torch.no_grad():
            test_input = torch.randn(1, 1, snp_input_dim)
            self.snp_embedding_dim = self.snp_encoder(test_input).shape[1]

        self.prs_projection = nn.Linear(prs_input_dim, prs_input_dim)
        fused_dim = self.snp_embedding_dim + prs_input_dim
        hidden_dim = self._adaptive_hidden_dim(fused_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _adaptive_hidden_dim(input_dim: int) -> int:
        if input_dim <= 128:
            return 64
        if input_dim <= 256:
            return 128
        if input_dim <= 512:
            return 256
        return 512

    def forward(self, snp_data: torch.Tensor, prs_data: torch.Tensor) -> torch.Tensor:
        snp_features = self.snp_encoder(snp_data.unsqueeze(1))
        prs_features = self.prs_projection(prs_data)
        fused = torch.cat((snp_features, prs_features), dim=1)
        return self.classifier(fused).squeeze(-1)
