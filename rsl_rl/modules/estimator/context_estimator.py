# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .contact_estimator import ContactEstimator
from .context_inference import ContextInference
from .feature_normalizer import FeatureNormalizer


class ContextEstimator(nn.Module):
    """End-to-end estimator with encoder, contact head, and context inference."""

    def __init__(
        self,
        encoder: nn.Module,
        contact_head: ContactEstimator,
        context_head: ContextInference,
        input_normalizer: FeatureNormalizer | None = None,
        output_normalizer: FeatureNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.contact_head = contact_head
        self.context_head = context_head
        self.input_normalizer = input_normalizer or FeatureNormalizer()
        self.output_normalizer = output_normalizer or FeatureNormalizer()

    def forward(
        self,
        x: torch.Tensor,
        prior_mu: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.input_normalizer(x)
        encoded = self.encoder(x)
        contact = self.contact_head(encoded)

        merged = torch.cat([prior_mu, encoded, contact], dim=-1)
        mu = self.context_head(merged)

     
        return {
            "encoded": encoded,
            "contact": contact,
            "mu": mu,
        }

    def sample_prior(
        self,
        cfg_dict: dict[str, Any],
        batch_size: int,
        device: str,
    ) -> torch.Tensor:
        prior_cfg = cfg_dict["model"].get("prior", {})
        prior_type = prior_cfg.get("type", "sphere")
        mu_low, mu_high = prior_cfg.get("mu_range", [-1.0, 1.0])

        if prior_type == "sphere":
            z = torch.randn(batch_size, self.context_head.context_dim, device=device)
            z = z * (self.context_head.context_dim**0.5) / (
                z.norm(dim=-1, keepdim=True) + 1e-8
            )
            mu = z
        else:
            mu = torch.empty(batch_size, self.context_head.context_dim, device=device).uniform_(
                mu_low, mu_high
            )
        return mu
