# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer encoder for [B, T, D] inputs with causal mask."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 128,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(self.max_len, model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [B, T, D], got {x.shape}")
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        x = self.input_proj(x)
        x = x + self.pos_emb[:seq_len].unsqueeze(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.encoder(x, mask=causal_mask)
        x = x[:, -1]
        return self.output_proj(x)
