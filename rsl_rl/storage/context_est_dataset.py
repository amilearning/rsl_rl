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


class ContextEstDataset(Dataset):
    """Dataset storing raw observation/done buffers as [time, envs, ...]."""

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

        # Stored raw buffers: list of (obs_buffer, dones_buffer)
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
        self.chunks.append((self._obs_buffer.clone(), self._dones_buffer.clone()))

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
    def load(path: str, history_window: int = 10) -> TensorDict:
        """Load raw buffers and return windowed trajectories.

        Args:
            path: Path to saved dataset.
            history_window: Window length (number of timesteps).

        Returns:
            TensorDict with batch dimension [num_windows] and each entry shaped
            [history_window, ...].
        """
        tmp_chunks = torch.load(path, weights_only=False)
        if history_window <= 0:
            raise ValueError("history_window must be > 0")

        windowed: dict[str, list[TensorDict | torch.Tensor]] = {}

        def _ensure_time_dim(t: torch.Tensor) -> torch.Tensor:
            # Expect time dimension to be at dim=1 with length history_window.
            if t.ndim < 3:
                return t
            if t.shape[1] == history_window:
                return t
            if t.shape[-1] == history_window:
                perm = [0, t.ndim - 1] + list(range(1, t.ndim - 1))
                return t.permute(*perm)
            return t

        for obs_buffer, dones_buffer in tmp_chunks:
            dones = dones_buffer.squeeze(-1)
            if dones.ndim != 2:
                raise ValueError(f"Expected dones [T, N], got {tuple(dones.shape)}")
            time_steps, num_envs = dones.shape
            if time_steps < history_window:
                continue

            t_idx = torch.arange(time_steps, device=dones.device)

            for env_id in range(num_envs):
                dones_env = dones[:, env_id].bool()

                # Episode start indices: t=0 and timestep after a done
                start_mask = torch.zeros_like(dones_env)
                start_mask[0] = True
                if time_steps > 1:
                    start_mask[1:] = dones_env[:-1]

                reset_points = torch.where(start_mask, t_idx, torch.zeros_like(t_idx))
                last_reset = torch.cummax(reset_points, dim=0).values
                since_reset = t_idx - last_reset + 1

                valid_end = since_reset >= history_window
                valid_window_mask = valid_end[history_window - 1 :]
                if not torch.any(valid_window_mask):
                    continue

                for k, v in obs_buffer.items():
                    v_env = v[:, env_id]
                    if isinstance(v_env, TensorDict):
                        v_windows = v_env.apply(
                            lambda t: t.unfold(0, history_window, 1)
                        )
                        v_selected = v_windows.apply(lambda t: _ensure_time_dim(t[valid_window_mask]))
                    else:
                        v_windows = v_env.unfold(0, history_window, 1)
                        v_selected = _ensure_time_dim(v_windows[valid_window_mask])

                    windowed.setdefault(k, []).append(v_selected)

        if not windowed:
            raise ValueError("No valid windows found in dataset.")

        out: dict[str, TensorDict | torch.Tensor] = {}
        for k, items in windowed.items():
            if isinstance(items[0], TensorDict):
                out[k] = TensorDict.cat(items, dim=0)
            else:
                out[k] = torch.cat(items, dim=0)

        first_key = next(iter(out.keys()))
        total_windows = out[first_key].batch_size[0] if isinstance(out[first_key], TensorDict) else out[first_key].shape[0]
        return TensorDict(out, batch_size=[total_windows])
    

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
