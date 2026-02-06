# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FeatureNormalizer(nn.Module):
    """Feature-wise mean/std normalization with saved buffers."""

    def __init__(
        self,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        epsilon: float = 1e-8,
        allow_unfitted: bool = True,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.allow_unfitted = allow_unfitted
        if mean is not None and std is not None:
            self.register_buffer("mean", mean)
            self.register_buffer("std", torch.clamp(std, min=self.epsilon))
        else:
            self.register_buffer("mean", None)
            self.register_buffer("std", None)

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, data: torch.Tensor) -> None:
        if data.ndim == 3:
            mean = data.mean(dim=(0, 1))
            std = data.std(dim=(0, 1))
        elif data.ndim == 2:
            mean = data.mean(dim=0)
            std = data.std(dim=0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {data.ndim}D")

        std = torch.clamp(std, min=self.epsilon)
        self.register_buffer("mean", mean.to(data.device))
        self.register_buffer("std", std.to(data.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            if self.allow_unfitted:
                return x
            raise RuntimeError("Normalizer not fitted. Call fit() first or provide mean/std.")

        if x.ndim == 3:
            mean = self.mean.unsqueeze(0).unsqueeze(0)
            std = self.std.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")

        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            if self.allow_unfitted:
                return x
            raise RuntimeError("Normalizer not fitted. Call fit() first or provide mean/std.")

        if x.ndim == 3:
            mean = self.mean.unsqueeze(0).unsqueeze(0)
            std = self.std.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")

        return x * std + mean

    def denormalize_mu_sigma(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            if self.allow_unfitted:
                return mu, sigma
            raise RuntimeError("Normalizer not fitted. Call fit() first or provide mean/std.")

        mean = self.mean.unsqueeze(0)
        std = self.std.unsqueeze(0)
        return mu * std + mean, sigma * std

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )  # type: ignore[arg-type]
        epsilon_key = prefix + "epsilon"
        epsilon_val = torch.tensor(self.epsilon)
        if not keep_vars:
            epsilon_val = epsilon_val.detach()
        if destination is not None:
            destination[epsilon_key] = epsilon_val
            return destination
        state[epsilon_key] = epsilon_val
        return state

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        epsilon_val = state_dict.get("epsilon", None)
        state_dict_no_eps = {k: v for k, v in state_dict.items() if k != "epsilon"}
        result = super().load_state_dict(state_dict_no_eps, strict=strict, assign=assign)
        if epsilon_val is not None:
            self.epsilon = float(
                epsilon_val.item() if isinstance(epsilon_val, torch.Tensor) else epsilon_val
            )
        return result
