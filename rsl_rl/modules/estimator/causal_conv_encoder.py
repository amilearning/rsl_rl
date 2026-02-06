# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvEncoder(nn.Module):
    """Causal temporal conv encoder for [B, T, D] inputs."""

    def __init__(
        self,
        input_dim: int,
        channels: Iterable[int] = (64, 64, 128),
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.0,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.dilations = []
        layers = []
        in_ch = input_dim
        dilation = 1
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, self.kernel_size, dilation=dilation))
            self.dilations.append(dilation)
            in_ch = ch
            dilation *= int(dilation_base)

        self.convs = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.proj = nn.Linear(in_ch, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [B, T, D], got {x.shape}")
        x = x.transpose(1, 2)  # [B, D, T]
        for conv, dilation in zip(self.convs, self.dilations):
            left_pad = (self.kernel_size - 1) * dilation
            x = F.pad(x, (left_pad, 0))
            x = conv(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = x[..., -1]  # last timestep
        return self.proj(x)
