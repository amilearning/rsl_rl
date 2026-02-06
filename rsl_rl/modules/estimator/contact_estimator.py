# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from .mlp import MLP


class ContactEstimator(nn.Module):
    """Contact estimation head from encoded features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        contact_dim: int = 1,
        dropout: float = 0.0,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.mlp = MLP(input_dim, hidden_dims, contact_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        return torch.sigmoid(logits) if self.apply_sigmoid else logits
