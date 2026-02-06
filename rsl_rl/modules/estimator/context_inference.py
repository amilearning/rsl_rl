# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from .mlp import MLP


class ContextInference(nn.Module):
    """Infer deterministic context vector from merged inputs."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dims: Iterable[int] = (256, 128),
        dropout: float = 0.0,
        min_sigma: float = 1e-4,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.min_sigma = float(min_sigma)
        self.mlp = MLP(input_dim, hidden_dims, self.context_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
