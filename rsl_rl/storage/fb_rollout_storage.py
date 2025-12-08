# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories
import os
from datetime import datetime
import glob

class FBRolloutStorage:
    class Transition:
        def __init__(self) -> None:
            # (s, a, r, done, time_out) 
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None            
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None     
            self.time_outs: torch.Tensor | None = None        

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        log_dir,
        training_type: str,
        num_envs: int,
        max_buffer_size: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],        
    ) -> None:
        self.log_dir = log_dir
        self.training_type = training_type
        self.device = "cpu"
        self.max_buffer_size = max_buffer_size
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(max_buffer_size, *value.shape, device=self.device) for key, value in obs.items()},
            batch_size=[max_buffer_size, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(max_buffer_size, num_envs, 1, device=self.device)
        self.actions = torch.zeros(max_buffer_size, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(max_buffer_size, num_envs, 1, device=self.device).byte()
        self.time_outs = torch.zeros(max_buffer_size, num_envs, 1, device=self.device).byte()

        # Counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.max_buffer_size:
            self.save_to_disk()
            self.clear()
        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.time_outs[self.step].copy_(transition.time_outs.view(-1, 1))

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        self.step = 0


    def save_to_disk(self):
        save_dir = os.path.join(self.log_dir, "offdata")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(save_dir, f"buffer_{self.step:06d}_{timestamp}.pt")

        obs_dict = {k: v[:self.step].cpu() for k, v in self.observations.items()}

        data = {
            "observations": obs_dict,  # plain dict[str, Tensor]
            "actions": self.actions[:self.step].cpu(),
            "rewards": self.rewards[:self.step].cpu(),
            "dones": self.dones[:self.step].cpu(),
            "time_outs": self.time_outs[:self.step].cpu(),
            "size": self.step,
        }


        # data = {
        #     "observations": self.observations[:self.step].cpu(),
        #     "actions": self.actions[:self.step].cpu(),
        #     "rewards": self.rewards[:self.step].cpu(),
        #     "dones": self.dones[:self.step].cpu(),
        #     "time_outs": self.time_outs[:self.step].cpu(),
        #     "size": self.step,
        # }

        torch.save(data, file_path)
        print(f"[FBRolloutStorage] Saved {self.step} transitions → {file_path}")
        
        
            
    def load_latest(self, load_path):
        """Load the most recent saved buffer from disk into memory."""
        save_dir = os.path.join(load_path, "offdata")

        if not os.path.exists(save_dir):
            print(f"[FBRolloutStorage] No offdata folder found at {save_dir}")
            return False

        # Find all buffer files
        files = glob.glob(os.path.join(save_dir, "buffer_*.pt"))
        if len(files) == 0:
            print(f"[FBRolloutStorage] No buffer files found in {save_dir}")
            return False

        # Pick the most recent one by timestamp
        latest_file = max(files, key=os.path.getmtime)

        print(f"[FBRolloutStorage] Loading latest buffer: {latest_file}")
        data = torch.load(latest_file, weights_only=False, map_location="cpu")

        size = data["size"]

        # Resize the in-memory buffer if needed
        if size > self.max_buffer_size:
            raise ValueError(
                f"Saved buffer size {size} exceeds current max_buffer_size {self.max_buffer_size}"
            )

        obs_td = TensorDict(data["observations"], batch_size=[size, self.num_envs], device=self.device)
        self.observations[:size].copy_(obs_td)
        self.actions[:size].copy_(data["actions"])
        self.rewards[:size].copy_(data["rewards"])
        self.dones[:size].copy_(data["dones"])
        self.time_outs[:size].copy_(data["time_outs"])
     
        # Update step counter
        self.step = size

        print(f"[FBRolloutStorage] Successfully loaded {size} transitions.")

        return True



    def get_random_idxs(self, batch_size: int):   
        n_env = len(self.dones[1])     
        idxs = torch.randint(low=0,high=self.step,size=(batch_size,n_env),dtype=torch.long)
        return idxs
     
    def sample_goals(
        self,        
        idxs: torch.Tensor,   # [B] flattened indices in [0, self.step * self.num_envs)
        batch_size: int = 2,
        p_curgoal: float = 0.0,
        p_trajgoal: float = 1.0,
        p_randomgoal: float = 0.0,  # kept for API symmetry
        geom_sample: bool = True,
        discount: float = 0.98,
    ):        
        terminal = (self.dones | self.time_outs).squeeze()        
        T = terminal.shape[0]
        E = terminal.shape[1]
        idx = torch.arange(T).view(T, 1).expand(T, E)  # [T, E]
        sentinel = T
        idx_or_T = torch.where(terminal, idx, torch.full_like(idx, sentinel))  # [T, E]
        rev = torch.flip(idx_or_T, dims=[0])        # [T, E]
        # 2) prefix-min in reversed order
        rev_min, _ = torch.cummin(rev, dim=0)       # [T, E]
        # 3) flip back → suffix-min in original order
        terminal_locs = torch.flip(rev_min, dims=[0])  # [T, E]
        terminal_locs = torch.where(terminal_locs == sentinel,
                                    torch.full_like(terminal_locs, T - 1),
                                    terminal_locs)

        env_ids = torch.arange(E, device=idxs.device).unsqueeze(0).expand(idxs.shape[0], E)  # [B, E]
        
        init_ts = idxs #[B,E]
        final_ts = terminal_locs[idxs, env_ids]   # [B, E]
        final_ts = torch.maximum(final_ts, init_ts)                         # safety clamp

        # 3) Trajectory goals between init_ts and final_ts
        if geom_sample:
            # offsets ~ Geometric(1 - discount), shape [B, E]
            geom = torch.distributions.Geometric(
                probs=torch.tensor(1.0 - discount)
            )
            offsets = geom.sample(init_ts.shape).long()          # [B, E], >= 1
            middle_ts = torch.minimum(init_ts + offsets, final_ts)
        else:
            # uniform between (init_ts+1) and final_ts
            distances = torch.rand_like(init_ts, dtype=torch.float)    # [B, E] in [0,1)
            lower = torch.minimum(init_ts + 1, final_ts)
            upper = final_ts
            middle_ts = torch.round(
                lower * distances + upper * (1.0 - distances)
            ).long()

        # 4) Choose between traj-goal and random-goal
        denom = 1.0 - p_curgoal 
        prob_traj_given_not_cur = p_trajgoal / (denom+1e-6)
        mask_traj = torch.rand_like(init_ts, dtype=torch.float) < prob_traj_given_not_cur  # [B, E]

        random_goal_ts = torch.randint(
            low=0,
            high=T,
            size=(batch_size, E),
            dtype=torch.long,
        )        
        
        goal_ts = torch.where(mask_traj, middle_ts, random_goal_ts)  # [B, E]

        return goal_ts
    
    def sample(self, batch_size: int, idxs: torch.Tensor | None = None, evaluation: bool = False):
        """
        Sample a batch of transitions with value/actor goals from the offline buffer.
        """

        device = self.observations["policy"].device
        max_valid_idx = min(self.step, self.max_buffer_size) - 2  # -1 for next_state, another -1 for safety
        
        if max_valid_idx < 0:
            raise ValueError("Not enough data in buffer to sample (need at least 2 steps).")
        num_env = self.dones.shape[1]
        # -------------------------------------------------------
        # 1) Choose base indices
        # -------------------------------------------------------
        if idxs is None:
            # random indices in [0, max_valid_idx]
            idxs = torch.randint(
                low=0,
                high=max_valid_idx + 1,
                size=(batch_size,num_env),
                device=device,
                dtype=torch.long,
            )
        else:
            idxs = idxs.to(device).long()
            # clamp to be safe
            idxs = torch.clamp(idxs, 0, max_valid_idx)

        # sort them (optional but you asked for it)
        idxs, sort_indices = torch.sort(idxs, dim=0)

        cur_idxs = idxs.clone()            # [B]
        next_idxs = idxs.clone() + 1       # [B], guaranteed < step

        # -------------------------------------------------------
        # 2) Sample goals
        # -------------------------------------------------------
        # value goals: future in same trajectory (traj + maybe some random if you like)
        value_goal_idxs = self.sample_goals(
            idxs=idxs,
            batch_size = batch_size,
            p_curgoal=0.0,
            p_trajgoal=0.5,
            p_randomgoal=0.5,  # or 0.0 if you want only trajectory goals
            geom_sample=True,
            discount=0.99,
        )

        # actor goals: purely random goals (as in the paper)
        actor_goal_idxs = self.sample_goals(
            idxs=idxs,
            batch_size = batch_size,
            p_curgoal=0.0,
            p_trajgoal=0.0,
            p_randomgoal=1.0,
            geom_sample=False,
            discount=0.99,
        )

        # sanity: all indices must be in range
        value_goal_idxs = torch.clamp(value_goal_idxs, 0, self.step - 1)
        actor_goal_idxs = torch.clamp(actor_goal_idxs, 0, self.step - 1)

        # -------------------------------------------------------
        # 3) Gather data from buffers
        # -------------------------------------------------------
        env_ids = torch.arange(num_env).unsqueeze(0).expand(idxs.shape[0], num_env)  # [B, E]
        obs_buf = self.observations["policy"]  # [N, obs_dim]
        cur_obs = obs_buf[cur_idxs,env_ids]           # [B, obs_dim]
        next_obs = obs_buf[next_idxs,env_ids]         # [B, obs_dim]
        value_goal_batch = obs_buf[value_goal_idxs,env_ids]  # [B, obs_dim]
        actor_goal_batch = obs_buf[actor_goal_idxs,env_ids]  # [B, obs_dim]

        cur_action = self.actions[cur_idxs, env_ids]
        cur_reward = self.rewards[cur_idxs, env_ids]

        batch = {
            "obs": cur_obs,
            "actions": cur_action,
            "rewards": cur_reward,
            "next_obs": next_obs,
            "value_goals": value_goal_batch,
            "actor_goals": actor_goal_batch,
            "indices": cur_idxs,
        }

        return batch

