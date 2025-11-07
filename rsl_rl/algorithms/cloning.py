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

    def relabeling_batch(self, env_cfg):

        K = env_cfg.commands.contact_cmd.num_ee
        contact_history = self.storage.observations['previliege'].clone()   # [T,E,K]
        # teacher_block = self.storage.observations['teacher'][:, :, -5*K:].clone()
        cur_contact_pos_b = self.storage.observations['contact_body'].clone().reshape(-1, contact_history.shape[1], K, 3)
        cur_time = self.storage.observations['time'].clone()                # [T,E,1]
        cur_time = cur_time.repeat(1,1,K)
        T, E, _ = contact_history.shape

        in_contact = contact_history.bool()
        not_contact = ~in_contact
        '''
        for in cotact moment, we can easily relable the target goal
        for contact time,
        t1 - (rising time) find the last contact moment. and recompute the t1. 
        t01 - (hold time) find the next detach moment. and recompute the t01
        '''
        # contact_start = in_contact.int() & (~torch.cat([torch.zeros_like(in_contact[:1]), in_contact[:-1].int()], dim=0))
        contact_start = in_contact & (~torch.cat([torch.zeros_like(in_contact[:1]), in_contact[:-1]], dim=0))
        # contact_end = not_contact & (~torch.cat([torch.zeros_like(not_contact[:1]), not_contact[:-1]], dim=0))
        contact_end_1side = in_contact & (~torch.cat([in_contact[1:], torch.zeros_like(in_contact[-1:])], dim=0))
        t_idx = torch.arange(T, device=contact_start.device).view(T,1,1).expand(T,E,K).long()

        '''
        Find the previous contact index
        '''
        left_contact_cand = torch.where(contact_start, t_idx, torch.full_like(t_idx, 0))
        left_contact_idx  = torch.cummax(left_contact_cand, dim=0)[0]          # [T,E,K], -1 means none to the left

        '''
        Find the next contact index
        '''
        BIG      = torch.full((T,E,K), T-1, device=contact_start.device, dtype=torch.long)
        cand     = torch.where(contact_start, t_idx, BIG)
        cand_r   = torch.flip(cand, dims=[0])
        min_r, _ = torch.cummin(cand_r, dim=0)
        next_ge  = torch.flip(min_r, dims=[0])                 # [T,E,K] in [0..T]
        right_contact_idx  = torch.where(next_ge == T, torch.full_like(next_ge, -1), next_ge)
        
        '''
        Find the next detach index
        '''       
        BIG = torch.full((T,E,K), T-1, device=in_contact.device, dtype=torch.long)  # sentinel
        cand_det = torch.where(contact_end_1side, t_idx, BIG)                              # [T,E,K]
        cand_det_r = torch.flip(cand_det, dims=[0])
        min_det_r, _ = torch.cummin(cand_det_r, dim=0)
        right_detach_idx = torch.flip(min_det_r, dims=[0])    
        # right_detach_idx = torch.where(right_detach_idx == T, torch.full_like(right_detach_idx, -1), right_detach_idx)
        # invalid_right_detach_mask = right_detach_idx == -1
        '''
        To determine the desired contact time:
        If currently in contact, find the previous contact time.
        If currently not in contact, find the next contact time. 
        '''
        t1_idx = torch.where(in_contact,left_contact_idx, right_contact_idx)
        t1 = torch.gather(cur_time, dim=0, index=t1_idx)-cur_time

        '''
        To determine the desired contact holding time: 
        If currently in contact, find the next contact time.--> hold time = next detach time - prev contact_time 
        If currently not in contact, find the next detach time. --> hold time = next detach time - next_contact time
        '''
        t01_in_contact = torch.gather(cur_time, dim=0, index=right_detach_idx)- torch.gather(cur_time, dim=0, index=left_contact_idx)
        t01_no_contact = torch.gather(cur_time, dim=0, index=right_detach_idx)- torch.gather(cur_time, dim=0, index=right_contact_idx)
        t01 = torch.where(in_contact,t01_in_contact,t01_no_contact)

        '''     
        To find the desired contact point: 
        If currently in contact, current contact point is the contact target. 
        If currently not in contact, find the next contact target. 
        '''
        right_contact_idx_x4 = right_contact_idx.unsqueeze(-1).expand(-1, -1, -1, 3) 
        right_contact_idx_x4 = right_contact_idx_x4.clamp(min=0, max=T-1)      
        right_retarget_pos = torch.gather(cur_contact_pos_b, dim=0, index=right_contact_idx_x4) 
        in_contact_x4 = in_contact.unsqueeze(-1).expand(-1, -1, -1, 3) 
        contact_des_pos_b = torch.where(in_contact_x4, cur_contact_pos_b, right_retarget_pos)

        '''
        relabel the dataset
        '''
        contact_pos_cmd = contact_des_pos_b.reshape(T,E,-1)
        time_cmd = torch.stack([t1,t01],dim=-1).reshape(T,E,-1)
        self.storage.observations['teacher'][:, :, -5*K:] = torch.cat([contact_pos_cmd,time_cmd],dim=-1).clone()
        self.storage.observations['policy'][:, :, -5*K:] = torch.cat([contact_pos_cmd,time_cmd],dim=-1).clone()
        


    def relabeling(self, env_cfg):
        policy_dt = env_cfg.sim.dt * env_cfg.decimation
        K = env_cfg.commands.contact_cmd.num_ee
        num_ee = env_cfg.commands.contact_cmd.num_ee
        # ===== extract =====
        contact_history   = self.storage.observations['previliege'].clone()              # [T,E,K] bool
        teacher_block     = self.storage.observations['teacher'][:, :, -5 * K:].clone()     # [T,E,5K]
        cur_contact_pos_b = self.storage.observations['contact_body'].clone()            # [T,E,3K]
        cur_time = self.storage.observations['time'].clone()            # [T,E,1]
        T, E = teacher_block.shape[:2]

        contact_des_pos_b = teacher_block[:, :, :3*K].reshape(T, E, K, 3)                    # [T,E,K,3]
        contact_des_time  = teacher_block[:, :, 3*K:].reshape(T, E, K, 2)                    # [T,E,K,2]
        cur_contact_pos_b = cur_contact_pos_b.reshape(T, E, K, 3)                            # [T,E,K,3]

        t1  = contact_des_time[..., 0].clone()                                               # [T,E,K]
        t01  = contact_des_time[..., 1].clone()     
      
        eps = 1e-6
        # # Loop over envs and end-effectors for clarity
        for t in range(T):
            for e in range(E):
                for k in range(num_ee):
                    t_contact = contact_history[t, e, k].bool()                    
                    if t_contact:
                        # === CASE 1: currently in contact ===
                        # 1. Set desired pose to current contact pose
                        contact_des_pos_b[t, e, k] = cur_contact_pos_b[t, e, k]
                        # try to find the init contact time 
                        if t > 0 :                        
                            h_head = contact_history[:t, e, k].int()
                            prior_contact_idx = t - (torch.argmin(torch.flip(h_head,dims=[0])))                      
                            t1[t,e,k] = cur_time[prior_contact_idx,e,0] - cur_time[t,e,0] 
                        else:
                            prior_contact_idx = 0
                            t1[t,e,k] = -abs(t1[t,e,k])
                        
                        h_tail = contact_history[t:, e, k].int()
                        next_detach_idx = (t+torch.argmin(h_tail)).item()
                        if next_detach_idx == t:
                            next_detach_idx = -1
                        t01[t,e,k] = cur_time[next_detach_idx,e,0] - t1[t,e,k]                       
                    else: # === CASE 2: currently not in contact ===           
                        h_tail = contact_history[t:, e, k].int()
                        next_contact_idx = torch.argmax(h_tail).item() if h_tail.any() else None                                            
                        if next_contact_idx is not None:
                            contact_des_pos_b[t, e, k] = cur_contact_pos_b[next_contact_idx, e, k]
                            t1[t,e,k] = cur_time[next_contact_idx,e,0] - cur_time[t,e,0] 

                            next_h_tail = contact_history[next_contact_idx:, e, k].int()
                            next_next_contact_idx = torch.argmax(next_h_tail).item() if next_h_tail.any() else None     
                            if next_next_contact_idx is not None:
                                t01[t,e,k] = cur_time[next_contact_idx+next_next_contact_idx,e,0] - cur_time[next_contact_idx+next_contact_idx,e,0] 
                            else:
                                t01[t,e,k] = cur_time[-1,e,0] - cur_time[next_contact_idx,e,0] 

                        else: # next_contact_idx is all None. so every time detach. 
                            if t1[t,e,k] <= 0: 
                                if t < T-1:
                                    t1[t,e,k] = abs(cur_time[-1,e,0]-cur_time[t,e,0])
                                else:
                                    t1[t,e,k] = abs(cur_time[-1,e,0]-cur_time[-2,e,0])


        
        new_t1  = t1.clone().to(device=t1.device) 
        new_t01  = t01.clone().to(device=t01.device) 
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
