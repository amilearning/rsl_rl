# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import BCStudentTeacher
from rsl_rl.storage import BCRolloutStorage
from rsl_rl.utils import resolve_optimizer


class Cloning:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: BCStudentTeacher 
    """The student teacher model."""

    def __init__(
        self,
        policy: BCStudentTeacher ,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # Initialized later

        # Initialize the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # Initialize the transition
        self.transition = BCRolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
            "nll": nn.functional.mse_loss 
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        elif loss_type == "nll":
            raise ValueError(f"TODO NLL implementation.")     
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int],
    ) -> None:
        # Create rollout storage
        self.storage = BCRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )


    
    def rollout(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions
        self.transition.actions = self.policy.act_teacher(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach() 
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions


    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions
        self.transition.actions = self.policy.act(obs)
        self.transition.privileged_actions = self.policy.evaluate(obs)
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions

    def vec_relabeling(self, relabeling_buffer_size, env_cfg, t1_noise=None, t01_noise=None):
        # --- setup ---
        policy_dt = env_cfg.sim.dt * env_cfg.decimation
        K = env_cfg.commands.contact_cmd.num_ee

        cur_buffer_start_idx = self.storage.step - relabeling_buffer_size
        cur_buffer_end_idx   = self.storage.step - 1
        sl = slice(cur_buffer_start_idx, cur_buffer_end_idx)

        # ===== extract =====
        contact_history   = self.storage.observations['previliege'][sl].clone()              # [T,E,K] bool
        teacher_block     = self.storage.observations['teacher'][sl, :, -5 * K:].clone()     # [T,E,5K]
        cur_contact_pos_b = self.storage.observations['contact_body'][sl].clone()            # [T,E,3K]
        T, E = teacher_block.shape[:2]

        contact_des_pos_b = teacher_block[:, :, :3*K].reshape(T, E, K, 3)                    # [T,E,K,3]
        contact_des_time  = teacher_block[:, :, 3*K:].reshape(T, E, K, 2)                    # [T,E,K,2]
        cur_contact_pos_b = cur_contact_pos_b.reshape(T, E, K, 3)                            # [T,E,K,3]

        t1  = contact_des_time[..., 0].clone()                                               # [T,E,K]
        t01 = contact_des_time[..., 1].clone()                                               # [T,E,K]
        # ===== end extract =====

        device, dtype = t1.device, t1.dtype
        eps = torch.tensor(1e-6, device=device, dtype=dtype)

        H      = contact_history.to(torch.bool)                                              # [T,E,K]
        pos_cur = cur_contact_pos_b
        tgrid  = torch.arange(T, device=device).view(T,1,1).expand(T,E,K)                    # [T,E,K]

        # Noise (allow injection for testing equivalence)
        if t1_noise is None:
            t1_noise  = torch.zeros((T,E,K), device=device, dtype=dtype)
        if t01_noise is None:
            t01_noise = torch.zeros((T,E,K), device=device, dtype=dtype)

        # ---- next True/False absolute indices via suffix-min (flip + cummin) ----
        bigT = torch.full((T,E,K), T, device=device, dtype=torch.long)
        idx  = tgrid.to(torch.long)

        # next contact (True) abs index >= t
        mask_next_true = torch.where(H, idx, bigT)
        rev_true       = torch.flip(mask_next_true, dims=[0])
        rev_true_min, _= torch.cummin(rev_true, dim=0)
        next_true_abs  = torch.flip(rev_true_min, dims=[0])                                  # [T,E,K]

        # next detach (False) abs index >= t
        mask_next_false = torch.where(~H, idx, bigT)
        rev_false       = torch.flip(mask_next_false, dims=[0])
        rev_false_min, _= torch.cummin(rev_false, dim=0)
        next_false_abs  = torch.flip(rev_false_min, dims=[0])                                 # [T,E,K]

        # relative steps and masks
        next_true_rel  = (next_true_abs  - tgrid)                                            # [T,E,K]
        next_false_rel = (next_false_abs - tgrid)
        has_next_true  = next_true_abs  < T
        has_next_false = (next_false_abs < T) & (next_false_rel > 0)  # remove zero-step detaches

        rel_to_end            = (2*T - tgrid).to(dtype) * policy_dt
        next_true_rel_steps   = next_true_rel.to(dtype)  * policy_dt
        next_false_rel_steps  = next_false_rel.to(dtype) * policy_dt

        # ---- new t1 ----
        # in contact: t1 ≤ 0
        t1_in  = -(t1 + t1_noise).abs()
        # not in contact: time until next contact (≥ eps), else horizon
        t1_out = torch.where(has_next_true, next_true_rel_steps + t1_noise,
                                        rel_to_end            + t1_noise)
        t1_out = torch.clamp(t1_out, min=eps.item())
        new_t1 = torch.where(H, t1_in, t1_out)

        # ---- new t01 ----
        # in contact: -t1 + time to next detach (or end) + noise
        t01_in_raw = torch.where(has_next_false,
                                -new_t1 + next_false_rel_steps + t01_noise,
                                -new_t1 + rel_to_end           + t01_noise)
        # loop semantics: if (t1 + t01) ≤ eps  =>  t01 = -t1 + eps
        bad_in = (new_t1 + t01_in_raw) <= eps
        t01_in = torch.where(bad_in, -new_t1 + eps, t01_in_raw)

        # not in contact: duration from *future contact* to its next detach (or end)
        abs_contact         = next_true_abs
        abs_contact_clamped = torch.clamp(abs_contact, max=T-1)

        # detach abs index evaluated at the contact time
        nf_at_contact_abs = torch.gather(next_false_abs, 0, abs_contact_clamped)
        has_detach_after_contact = (nf_at_contact_abs < T) & (nf_at_contact_abs > abs_contact)

        dur_from_contact = torch.where(
            has_detach_after_contact,
            (nf_at_contact_abs - abs_contact).to(dtype) * policy_dt,
            ((2*T - abs_contact).to(dtype) * policy_dt)
        )

        t01_out_raw = dur_from_contact + t01_noise
        # loop semantics: if (t1_out + t01_out_raw) ≤ eps => t01_out = max(eps - t1_out, eps)
        bad_out       = (t1_out + t01_out_raw) <= eps
        t01_out_fixed = torch.maximum(eps - t1_out, eps)
        t01_out       = torch.where(bad_out, t01_out_fixed, t01_out_raw)

        new_t01 = torch.where(H, t01_in, t01_out)

        # ---- new desired pos ----
        # in contact → current pose; else → pose at next contact if exists; else keep desired
        idx_time = abs_contact_clamped.unsqueeze(-1).expand(T, E, K, 3)                      # [T,E,K,3]
        pos_at_contact = torch.gather(pos_cur, dim=0, index=idx_time)                        # [T,E,K,3]

        new_pos_b = torch.where(
            H.unsqueeze(-1),
            pos_cur,
            torch.where(
                (abs_contact < T).unsqueeze(-1),
                pos_at_contact,
                contact_des_pos_b
            )
        )                                                                                    # [T,E,K,3]



        contact_history   = self.storage.observations['previliege'][sl].clone()              # [T,E,K] bool
        teacher_block     = self.storage.observations['teacher'][sl, :, -5 * K:].clone()     # [T,E,5K]
        cur_contact_pos_b = self.storage.observations['contact_body'][sl].clone()            # [T,E,3K]
        T, E = teacher_block.shape[:2]

        contact_des_pos_b = teacher_block[:, :, :3*K].reshape(T, E, K, 3)                    # [T,E,K,3]
        contact_des_time  = teacher_block[:, :, 3*K:].reshape(T, E, K, 2)                    # [T,E,K,2]
        cur_contact_pos_b = cur_contact_pos_b.reshape(T, E, K, 3)                            # [T,E,K,3]

        t1  = contact_des_time[..., 0].clone()                                               # [T,E,K]
        t01 = contact_des_time[..., 1].clone()     

        new_pos_b = new_pos_b.reshape(T,E,-1)
        new_time = torch.cat([new_t1.unsqueeze(-1),new_t01.unsqueeze(-1)], dim=-1)
        new_time = new_time.reshape(T,E,-1)
        new_contact_cmd = torch.cat([new_pos_b, new_time], dim=-1)

        self.storage.observations['teacher'][sl, :, -5 * K:] = new_contact_cmd.clone()
        self.storage.observations['policy'][sl, :, -5 * K:] = new_contact_cmd.clone()
       



    def relabeling(self,relabeling_buffer_size, env_cfg):
        policy_dt = env_cfg.sim.dt *env_cfg.decimation
        
        num_ee = env_cfg.commands.contact_cmd.num_ee
        cur_buffer_start_idx = self.storage.step - relabeling_buffer_size
        cur_buffer_end_idx = self.storage.step-1        
        
        ################# extract observation data ########################
        sl = slice(cur_buffer_start_idx, cur_buffer_end_idx)
        contact_history   = self.storage.observations['previliege'][sl].clone()                 # [T,E,num_ee]
        teacher_block     = self.storage.observations['teacher'][sl, :, -5 * num_ee:].clone()   # [T,E,5*num_ee]
        cur_contact_pos_b = self.storage.observations['contact_body'][sl].clone()               # [T,E,num_ee*3]
        T, E = teacher_block.shape[:2]
        contact_des_pos_b  = teacher_block[:, :, : 3 * num_ee].reshape(T, E, num_ee, 3)   # [T,E,num_ee,3]
        contact_des_time = teacher_block[:, :, 3 * num_ee:].reshape(T, E, num_ee, 2)    # [T,E,num_ee,2]
        cur_contact_pos_b = cur_contact_pos_b.reshape(T, E, num_ee, 3)                  # [T,E,num_ee,3]
        contact_des_time_t1  = contact_des_time[..., 0]  # [T,E,num_ee]
        contact_des_time_t01 = contact_des_time[..., 1]  # [T,E,num_ee]
        ################# extract observation data END ########################
                  
        eps = 1e-6
        # # Loop over envs and end-effectors for clarity
        for t in range(T):
            for e in range(E):
                for k in range(num_ee):
                 
                    t_contact = contact_history[t, e, k].bool()
                    t1_noise = torch.zeros(1,device=contact_des_time_t1.device) # (torch.rand(1, device=contact_des_time_t1.device) - 0.5) * 0.5 * policy_dt
                    t01_noise = torch.zeros(1,device=contact_des_time_t1.device) # (torch.rand(1, device=contact_des_time_t01.device) - 0.5) * 0.5 * policy_dt
                    if t_contact:
                        # === CASE 1: currently in contact ===
                        # 1. Set desired pose to current contact pose
                        contact_des_pos_b[t, e, k] = cur_contact_pos_b[t, e, k]

                        # 2. Ensure t1 is non-positive (time since contact)
                        if contact_des_time_t1[t, e, k] > 0:
                            contact_des_time_t1[t, e, k] = -abs(contact_des_time_t1[t, e, k]) + t1_noise

                        # 3 find next detach
                        # Find next detach (first False after t)
                        h_tail = contact_history[t:, e, k].int()
                        next_detach_rel = torch.argmax(~h_tail).item() if (~h_tail).any() else None

                        if next_detach_rel is not None and next_detach_rel>0:
                            # valid detach found
                            contact_des_time_t01[t, e, k] =  -contact_des_time_t1[t, e, k] + next_detach_rel * policy_dt + t01_noise
                        else:
                            # no detach in window
                            contact_des_time_t01[t, e, k] =   -contact_des_time_t1[t, e, k] +(T - t) * policy_dt + t01_noise
                        if (contact_des_time_t1[t, e, k] + contact_des_time_t01[t, e, k] <=eps):
                            contact_des_time_t01[t, e, k] = -contact_des_time_t1[t, e, k] + eps 

                    else:   
                        # === CASE 2: currently not in contact ===
                        h_tail = contact_history[t:, e, k].int()
                        next_contact_rel = torch.argmax(h_tail).item() if h_tail.any() else None                        
                        if next_contact_rel is not None:
                            abs_contact = t + next_contact_rel
                            # 1. Set desired pose to next contact position
                            contact_des_pos_b[t, e, k] = cur_contact_pos_b[abs_contact, e, k]
                            # 2. Set time until contact
                            t1 = next_contact_rel * policy_dt + t1_noise
                            t1=t1.clamp_min(eps)
                            contact_des_time_t1[t, e, k] =t1
                            # 3. Find detach moment after the contact
                            h_after_contact = contact_history[abs_contact:, e, k].int()
                            next_detach_rel = torch.argmax(~h_after_contact).item() if (~h_after_contact).any() else None
                            if next_detach_rel is not None and next_detach_rel > 0:                                
                                contact_des_time_t01[t, e, k] = next_detach_rel * policy_dt + t01_noise
                            else:
                                contact_des_time_t01[t, e, k] = (T - abs_contact) * policy_dt+t01_noise
                                
                            if (contact_des_time_t1[t, e, k] + contact_des_time_t01[t, e, k] ) <= eps:
                                contact_des_time_t01[t, e, k] = max(float(eps - contact_des_time_t1[t, e, k]), float(eps))

                        else:
                            # no contact found in the future
                                t1  = (T - t) * policy_dt + t1_noise
                                t01 = (T - t) * policy_dt + t01_noise
                                t1  = t1.clamp_min(eps)
                                if (t1 + t01) <= eps:
                                    t01 = max(float(eps - t1), float(eps))
                                contact_des_time_t1[t, e, k]  = t1
                                contact_des_time_t01[t, e, k] = t01


        
        new_t1  = contact_des_time_t1.clone().to(device=contact_des_time_t1.device) 
        new_t01  = contact_des_time_t01.clone().to(device=contact_des_time_t01.device) 
        new_pos_b = contact_des_pos_b.clone().to(device=contact_des_pos_b.device)  # [T,E,num_ee,3]
        return new_pos_b, new_t1, new_t01
             


    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)

        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0
        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()            
            for obs, tecaher_actions, privileged_actions, dones in self.storage.generator():                
                # Inference of the student for gradient computation                
                # noise = torch.randn_like(tecaher_actions) * 1e-5
                # tecaher_actions +=noise

                actions = self.policy.act_inference(obs)
                # Behavior cloning loss
                behavior_loss = self.loss_fn(actions, tecaher_actions)

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

            # Reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
