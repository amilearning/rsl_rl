# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from rsl_rl.utils import split_and_pad_trajectories


class ContextEstDataset(Dataset):
    """Chunked dataset storing observations as [time, envs, ...] plus dones.

    When the buffer reaches capacity, trajectories are split and padded using
    `split_and_pad_trajectories`, and the padded chunk is stored.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        obs: TensorDict,
        device: str = "cpu",
        log_dir: str | None = None,
        keys: Iterable[str] = (
            "context_est",
            "obj_contact",
            "physical_params",
            "momentum_change",
        ),
    ) -> None:
        self.num_envs = num_envs
        self.capacity = int(capacity)
        self.device = device
        self.log_dir = log_dir        
        self.keys = tuple(keys)

        for k in self.keys:
            if k not in obs.keys():
                raise KeyError(f"Missing observation key '{k}' in obs.")

        self._obs_buffer = TensorDict(
            {k: torch.zeros(self.capacity, *obs[k].shape, device=self.device) for k in self.keys},
            batch_size=[self.capacity, self.num_envs],
            device=self.device,
        )
        self._dones_buffer = torch.zeros(self.capacity, self.num_envs, 1, device=self.device)
        self._step = 0

        # Stored padded chunks: list of (padded_obs, masks)
        self.chunks: list[tuple[TensorDict, torch.Tensor]] = []

    def add(self, obs: TensorDict, dones: torch.Tensor) -> None:
        obs_on_device = obs.to(self.device)
        dones_on_device = dones.view(-1, 1).to(self.device)

        self._obs_buffer[self._step].copy_(obs_on_device.select(*self.keys))
        self._dones_buffer[self._step].copy_(dones_on_device)
        self._step += 1

        if self._step >= self.capacity:
            self._finalize_chunk()
            self._reset_buffers()



    def _finalize_chunk(self) -> None:
        padded_obs, masks = split_and_pad_trajectories(self._obs_buffer, self._dones_buffer)
        self.chunks.append((padded_obs, masks))

    def _reset_buffers(self) -> None:
        self._step = 0
        self._obs_buffer.zero_()
        self._dones_buffer.zero_()

    def clear(self) -> None:        
        self.chunks.clear()
        self._reset_buffers()

    def get(self) -> list[tuple[TensorDict, torch.Tensor]]:
        return self.chunks

    def save(self, path: str) -> None:
        torch.save(self.chunks, path)

    @staticmethod
    def load(path: str) -> list[tuple[TensorDict, torch.Tensor]]:
        return torch.load(path)

    def save_to_logdir(self, filename: str = "context_est_dataset.pt") -> str | None:      
        if self.log_dir is None:
            return None
        data_dir = os.path.join(self.log_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, filename)
        self.save(path)
        self.clear()
        return path

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[TensorDict, torch.Tensor]:
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError("Index out of range.")
        return self.chunks[idx]
