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

    def relabeling(self,relabeling_buffer_size, env_cfg):
        policy_dt = env_cfg.sim.dt *env_cfg.decimation
        
        num_ee = env_cfg.commands.contact_cmd.num_ee
        cur_buffer_start_idx = self.storage.step - relabeling_buffer_size
        cur_buffer_end_idx = self.storage.step-1        
        
        ################# extract observation data ########################
        sl = slice(cur_buffer_start_idx, cur_buffer_end_idx)
        contact_history   = self.storage.observations['previliege'][sl]                 # [T,E,num_ee]
        teacher_block     = self.storage.observations['teacher'][sl, :, -5 * num_ee:]   # [T,E,5*num_ee]
        cur_contact_pos_b = self.storage.observations['contact_body'][sl]               # [T,E,num_ee*3]
        T, E = teacher_block.shape[:2]
        contact_des_pos_b  = teacher_block[:, :, : 3 * num_ee].reshape(T, E, num_ee, 3)   # [T,E,num_ee,3]
        contact_des_time = teacher_block[:, :, 3 * num_ee:].reshape(T, E, num_ee, 2)    # [T,E,num_ee,2]
        cur_contact_pos_b = cur_contact_pos_b.reshape(T, E, num_ee, 3)                  # [T,E,num_ee,3]
        contact_des_time_t1  = contact_des_time[..., 0]  # [T,E,num_ee]
        contact_des_time_t01 = contact_des_time[..., 1]  # [T,E,num_ee]
        ################# extract observation data END ########################
        new_t1  = contact_des_time_t1.clone().to(device=contact_des_time_t1.device) 
        new_t01  = contact_des_time_t01.clone().to(device=contact_des_time_t01.device) 
        new_pos_b = cur_contact_pos_b.clone().to(device=cur_contact_pos_b.device)  # [T,E,num_ee,3]
                               
        eps = 1e-6
        # # Loop over envs and end-effectors for clarity
        for t in range(T):
            for e in range(E):
                for k in range(num_ee):
                    t_contact = contact_history[t, e, k].bool()
                    t1_noise = (torch.rand(1, device=contact_des_time_t1.device) - 0.5) * 0.5 * policy_dt
                    t01_noise = (torch.rand(1, device=contact_des_time_t01.device) - 0.5) * 0.5 * policy_dt
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

                        if next_detach_rel is not None:
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
                            if next_detach_rel is not None:                                
                                contact_des_time_t01[t, e, k] = next_detach_rel * policy_dt + t01_noise
                            else:
                                contact_des_time_t01[t, e, k] = (T - next_contact_rel) * policy_dt+t01_noise
                                
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

              
        # for t in range(T):
        #     for e in range(E):
        #         for k in range(num_ee):        
        #             t_contact = contact_history[t,e,k].bool()
        #             if t_contact:  # currently ee is in contact 
        #                 contact_des_pose_b[t,e,k] = cur_contact_pos_b[t,e,k]
        #                 if contact_des_time_t1[t,e,k] >0 
        #                     contact_des_time_t1[t,e,k] = -abs(contact_des_time_t1[t,e,k])
        #                 if contact_des_time_t1[t,e,k] +  contact_des_time_t01[t,e,k] <0: 
        #                     find the next detach from contact_histpory[t:,e,k] -> idx 
        #                     if index finding within the size  
        #                         contact_des_time_t01[t,e,k] = contact_des_time_t1[t,e,k] + idx* self.policy_dt
        #                     else 
        #                         contact_des_time_t01[t,e,k] = contact_des_time_t1[t,e,k] + T* self.policy_dt
        #             else: # currently ee is not in contact
        #                 if contact_des_time_t1[t,e,k] < 0:
        #                     if contact_des_time_t1[t,e,k] + contact_des_time_t01[t,e,k]> 0:
        #                         find the next contact from contact_history[t:,e,k] --> idx
        #                         if index finding wihtin  the size 
        #                             contact_des_pose_b[t,e,k] = cur_contact_pos_b[idx,e,k]
        #                             contact_des_time_t1[t,e,k] = idx*self.policy_dt
        #                             find the next detach after idx_detach
        #                             if idx_detach found 
        #                                 contact_des_time_t01[t,e,k] = (idx_detach-idx )*self.policy_dt
        #                             elsE:
        #                                 contact_des_time_t01[t,e,k] = (T-idx)*self.policy_dt
        #                         else: 
        #                              contact_des_timet1[t,e,k]= T*self.policy_dt
        #                              contact_des_time_t01[t,e,k] = T*self.policy_dt
        #                 else:
        #                         find the next contact from contact_history[t:,e,k] --> idx
        #                         if index finding wihtin  the size 
        #                             contact_des_pose_b[t,e,k] = cur_contact_pos_b[idx,e,k]
        #                             contact_des_time_t1[t,e,k] = idx*self.policy_dt
        #                             find the next detach after idx_detach
        #                             if idx_detach found 
        #                                 contact_des_time_t01[t,e,k] = (idx_detach-idx )*self.policy_dt
        #                             elsE:
        #                                 contact_des_time_t01[t,e,k] = (T-idx)*self.policy_dt
        #                         else: 
        #                              contact_des_timet1[t,e,k]= T*self.policy_dt
        #                              contact_des_time_t01[t,e,k] = T*self.policy_dt
                            
                            


                                
                          

        #         h = contact_history[t, :, k].bool()  # [T] bool
        #         seg_starts = segment_starts_from_bool(h)
        #         if len(seg_starts) == 1 and h[seg_starts[0]].item():
        #             new_pos, new_t1, new_t01 = relabel_always_contact(h=h,
        #                                                             contact_des_time_t1=contact_des_time_t1[:, e, k],
        #                                                             contact_des_time_t01=contact_des_time_t01[:, e, k],
        #                                                             contact_pos=contact_pos[:, e, k,:],   
        #                                                             seg_starts=seg_starts
        #                                                             )
                                                                    
        # #         elif len(seg_starts) ==1 and h[seg_starts].item() is False: 
        # #             always_detach
        # #         elif len(seg_starts) ==2 and h[seg_starts].item(): 
        # #                 contact -> detach 
        # #         elif len(seg_starts) ==2 and h[seg_starts].item() is False: 
        # #                 detach  -> contact
        #         if len(seg_starts) > 2 
        #             for every 3 group of seg_starts 
        #                 is contact_detach_contact? --> run  

                  
        #         # Add a virtual end to iterate segment-by-segment
        #         seg_starts.append(T)
                
        #         if h starts with True and end with False with a single change 
        #         elif h start with False and end with True with a single change
        #         elif h starts with True and no change 
        #         elif h starts with False and no change 
        #         elif h starts with more than two change. 


        # def classify_contact_pattern(h: torch.Tensor):
        #     h_np = h.to(torch.bool).cpu().numpy().astype(int)
        #     changes = (h_np[1:] != h_np[:-1]).nonzero()[0]  # indices where value flips
        #     n_changes = len(changes)

        #     start, end = bool(h_np[0]), bool(h_np[-1])

        #     if n_changes == 1:
        #         if start and not end:
        #             return "contact→detach"
        #         elif not start and end:
        #             return "detach→contact"
        #     elif n_changes == 0:
        #         if start:
        #             return "always_contact"
        #         else:
        #             return "always_detach"
        #     else:
        #         return "multiple_changes"
            
            
        #         # We’ll need contact-moment positions; when a contact segment starts at t_c,
        #         # pos_at_contact = contact_des_pos[t_c, e, k]
        #         # For detach relabeling, we use the most recent contact moment start.

        #         last_contact_start = None  # index of most recent contact start (attach)

        #         for s in range(len(seg_starts) - 1):
        #             t0 = seg_starts[s]
        #             t1_excl = seg_starts[s + 1]  # exclusive
        #             seg_is_contact = bool(h[t0].item())

        #             if seg_is_contact:
        #                 # This segment starts with contact ON at t0 (attach moment at t0)
        #                 contact_moment_idx = t0
        #                 last_contact_start = contact_moment_idx

        #                 # Find next detach moment index (start of next segment where h becomes False)
        #                 # That’s exactly t1_excl if next segment is detach, else end of window.
        #                 next_detach_idx = t1_excl  # (could be T if never detaches in window)

        #                 # Find previous detach segment start (start of previous segment)
        #                 prev_detach_start = seg_starts[s - 1] if s > 0 else 0

        #                 hold_duration = (next_detach_idx - contact_moment_idx) * policy_dt
        #                 pos_at_contact = contact_des_pos[contact_moment_idx, e, k]  # [3]

        #                 # 1) Before contact: indices in [prev_detach_start, contact_moment_idx]
        #                 for t in range(prev_detach_start, contact_moment_idx + 1):
        #                     new_t1[t, e, k]  = (contact_moment_idx - t) * policy_dt
        #                     new_t01[t, e, k] = hold_duration
        #                     new_pos[t, e, k] = pos_at_contact

        #                 # 2) During contact: indices in [contact_moment_idx, next_detach_idx)
        #                 for t in range(contact_moment_idx, next_detach_idx):
        #                     new_t1[t, e, k]  = -(t - contact_moment_idx) * policy_dt
        #                     new_t01[t, e, k] = hold_duration
        #                     new_pos[t, e, k] = pos_at_contact

        #             else:
        #                 # This segment is DETACH (contact OFF). We relabel w.r.t. the most recent contact.
        #                 # We need the last contact start (attach) and this detach moment (at t0).
        #                 detach_moment_idx = t0

        #                 if last_contact_start is None:
        #                     # No previous contact in window; define a benign fallback:
        #                     # treat as if contact started at t=0.
        #                     last_contact_start = 0

        #                 hold_duration = (detach_moment_idx - last_contact_start) * policy_dt
        #                 pos_at_prev_contact = contact_des_pos[last_contact_start, e, k]

        #                 # “Detach relabeling”: for indices in [t0, next_seg_start) we keep elapsed-since-contact negative
        #                 for t in range(t0, t1_excl):
        #                     new_t1[t, e, k]  = -(t - last_contact_start) * policy_dt
        #                     new_t01[t, e, k] = hold_duration
        #                     new_pos[t, e, k] = pos_at_prev_contact

                            
        


        # def relabeling_contact(prev_detatch_moment_idx, contact_moment_idx, next_detach_moment_idx): 
        #             hold_duration = (next_detach_moment_idx - contact_moment_idx)*policy_dt                     
                    
        #             for each idx                            
        #                 if idx is between prev_detach_moment_idx, and contact_moment_idx 
        #                             contact_dest_time_t1 = (contact_moment_idx - idx )*policy_dt                 
        #                             contact-des_time_t01 = hold_duration
        #                             contact-des_pos = contact_pos(contact_moment_idx)
        #                 if idx is between contact_moment_idx  and nex_detach_moment_idx 
        #                     contact-dest_time_t1 = -(idx - contact_moment_idx)*policy_dt 
        #                     contact-des_time_t01 = hold_duration
        #                     contact-des_pos = contact_pos(contact_moment_idx)

        # def relabeling_detach(prev_contact_moment_idx, detach_moment_idx, next_contact_moment_idx): 
        #             hold_duration = (detach_moment_idx - prev_contact_moment_idx)*policy_dt                     
        #             for each idx 
        #                 contact_dest_time_t1 = -(idx - prev_contact_moment_idx )*policy_dt                 
        #                 contact-des_time_t01 = hold_duration
        #                 contact-des_pos = contact_pos(prev_contact_moment_idx)

        # for each end effecotr compute contact change moments so we can have like
        #             C_k-1, C_k, C_k+1 
        # C can be contact moment or detach moment. 
        # using the each three groups of semegnes we can relabeling the idx.  
        # so for each idx, we run either relabeling_contact or relabegin_detach for relabeling. and make sure the relabeling results one of each shall be the same. (pls check) 

        
                    
        

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
