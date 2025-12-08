# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ForwardMap, BackwardMap, FBActor 
from rsl_rl.storage import FBRolloutStorage
from rsl_rl.utils import resolve_optimizer
import torch.nn.functional as F
import math 

import omegaconf
import typing as tp

import dataclasses
import numpy as np
MetaDict = tp.Mapping[str, np.ndarray]
from collections import OrderedDict

@dataclasses.dataclass
class FBDDPGAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.fb_ddpg.FBDDPGAgent"
    name: str = "fb_ddpg"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    use_hiplog: bool = omegaconf.II("use_hiplog")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 5120
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 526   # 512
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 50  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    nstep: int = 1
    batch_size: int = 1024  # 512
    init_fb: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    ortho_coef: float = 1.0  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    q_loss: bool = False
    q_loss_coef: float = 0.01
    additional_metric: bool = False
    add_trunk: bool = False




class FBAlgorithm:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    def __init__(
        self,      
        log_dir,
        cfg,  
        action_dim,
        obs_dim, 
        num_learning_epochs: int = 1,
        num_mini_batches: int = 20,        
        learning_rate: float = 1e-3,        
        gamma: float = 0.99,
        device: str = "cuda",        
    ) -> None:
        # Device-related parameters
        self.log_dir = log_dir
        self.cfg = cfg
        self.alg_cfg = cfg['fb_algorithm']
        
        self.policy_cfg = cfg['fb_policy']
        self.update_z_every_step = self.alg_cfg['update_z_every_step']
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.gamma = gamma      
        self.device = device
        self.z_dim = self.policy_cfg['z_dim']
        
        self.policy  = FBActor(action_dim = action_dim,
                               obs_dim= obs_dim,
                               policy_cfg = self.policy_cfg).to(self.device)        
    
        self.forward_net = ForwardMap(action_dim = self.action_dim,
                                      obs_dim = self.obs_dim,
                                      z_dim = self.z_dim).to(self.device)

        self.forward_target_net = ForwardMap(action_dim = self.action_dim,
                                      obs_dim = self.obs_dim,
                                      z_dim = self.z_dim).to(self.device)
        
        self.backward_net = BackwardMap(obs_dim = obs_dim, 
                                        z_dim = self.z_dim).to(self.device)
        self.backward_target_net = BackwardMap(obs_dim = obs_dim, 
                                        z_dim = self.z_dim).to(self.device)        
            
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        
        self.lr_coef = 1.0
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.alg_cfg['learning_rate'])
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  
                                        {'params': self.backward_net.parameters(), 'lr': self.lr_coef * self.alg_cfg['learning_rate']}],
                                       lr=self.alg_cfg['learning_rate'])
        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()
        
        self.storage: FBRolloutStorage | None = None        
        self.transition = FBRolloutStorage.Transition()
        
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate

        self.num_updates = 0
        self.global_step = 0


    def init_storage(
        self,
        training_type: str,
        num_envs: int,        
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = FBRolloutStorage(
            self.log_dir,
            training_type,
            num_envs,
            self.alg_cfg["max_off_buffer_size"],
            obs,
            actions_shape
        )
        
    def update_transition_pre(self,obs, actions):        
        self.transition.observations = obs
        self.transition.actions = actions 
    def update_transition_post(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]):        
        self.transition.rewards= rewards
        self.transition.dones = dones
        self.transition.time_outs =  extras["time_outs"]
        self.storage.add_transitions(self.transition)
        
    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.policy, self.forward_net, self.backward_net]:
            net.train(training)
            
    def sample_z(self, size, device: str = "cuda"):
        gaussian_rdv = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)        
        z = math.sqrt(self.z_dim) * gaussian_rdv        
        return z
    
    def init_meta(self) -> MetaDict:        
        z = self.sample_z(1)
        z = z.squeeze().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta
    
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
    ) -> MetaDict:
        if global_step % self.update_z_every_step == 0:
            return self.init_meta()
        return meta
    
    def init_fb_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:        
        self.storage = FBRolloutStorage(training_type,num_envs,num_transitions_per_env,obs,actions_shape)
                

    def act(self, obs: TensorDict, eval_mode = False) -> torch.Tensor:
        # Compute the actions
        self.transition.observations = obs
        
        self.transition.actions = self.policy.act(obs)
        # self.transition.privileged_actions = self.policy.evaluate(obs)
        # Record the observations
        
        
        
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
             



#     def process_env_step(
#         self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
#     ) -> None:
#         # Update the normalizers
# #         self.policy.update_normalization(obs)

#         # Record the transition
        
                

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

