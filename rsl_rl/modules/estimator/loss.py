# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_context_estimator_losses(
    prior_context: torch.Tensor,
    post_context: torch.Tensor,
    physical_params: torch.Tensor,
    target_contact: torch.Tensor,
    predicted_contact: torch.Tensor,
    cfg_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    # 1) Contact prediction loss
    contact_loss = F.binary_cross_entropy(predicted_contact, target_contact)

    # 2) If target_contact == 0, encourage post_context ~ prior_context (L2)
    if target_contact.ndim == 2 and target_contact.shape[-1] == 1:
        contact_mask = target_contact.squeeze(-1)
    else:
        contact_mask = target_contact
    no_contact_mask = (contact_mask == 0).float()
    l2_dist = (post_context - prior_context).norm(dim=-1)
    if no_contact_mask.sum() > 0:
        no_contact_loss = (l2_dist * no_contact_mask).sum() / (no_contact_mask.sum() + 1e-8)
    else:
        no_contact_loss = torch.zeros((), device=post_context.device)

    # 3) If target_contact == 1, ignore (no penalty)

    # 4) Post-context should lie on sphere (match prior distribution norm)
    target_norm = float(post_context.shape[-1]) ** 0.5
    sphere_loss = (post_context.norm(dim=-1) - target_norm).pow(2).mean()

    # 5) Contrastive loss: context similarity follows physical_params similarity
    # Use L2 distance for both; all pairs in batch.
    if physical_params.ndim == 3:
        phys = physical_params[:, -1]
    else:
        phys = physical_params
    phys_dist = torch.cdist(phys, phys, p=2)
    ctx_dist = torch.cdist(post_context, post_context, p=2)

    # Similarity weights from physical distance
    temp = float(cfg_dict.get("train", {}).get("contrastive_temp", 1.0))
    margin = float(cfg_dict.get("train", {}).get("contrastive_margin", 1.0))
    sim = torch.exp(-phys_dist / max(temp, 1e-6))

    # Exclude diagonal
    bsz = post_context.shape[0]
    diag = torch.eye(bsz, device=post_context.device)
    off_diag = 1.0 - diag

    pos_term = sim * ctx_dist
    neg_term = (1.0 - sim) * F.relu(margin - ctx_dist)
    contrastive_loss = ((pos_term + neg_term) * off_diag).sum() / (off_diag.sum() + 1e-8)

    total = (
        float(cfg_dict["train"].get("contact_weight", 1.0)) * contact_loss
        + float(cfg_dict["train"].get("no_contact_weight", 1.0)) * no_contact_loss
        + float(cfg_dict["train"].get("sphere_weight", 1.0)) * sphere_loss
        + float(cfg_dict["train"].get("contrastive_weight", 1.0)) * contrastive_loss
    )

    return {
        "total": total,
        "contact": contact_loss,
        "no_contact": no_contact_loss,
        "sphere": sphere_loss,
        "contrastive": contrastive_loss,
    }
