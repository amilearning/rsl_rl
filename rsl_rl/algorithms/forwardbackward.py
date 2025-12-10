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
        self.metrics: tp.Dict[str, float] = {}        
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
    
  
    def act(self, obs: TensorDict, eval_mode = False) -> torch.Tensor:
        # Compute the actions
        self.transition.observations = obs
        
        self.transition.actions = self.policy.act(obs)
        # self.transition.privileged_actions = self.policy.evaluate(obs)
        # Record the observations
        
        
        return self.transition.actions


    def sample_z(self, batch_size, env_size, device: str = "cuda"):
        gaussian_rdv = torch.randn((batch_size*env_size, self.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)        
        z = math.sqrt(self.z_dim) * gaussian_rdv                
        return z.reshape(batch_size, env_size, self.z_dim)
    
    def sample_mixed_z(
        self,
        actor_goals
    ) -> torch.Tensor:
        """
        PyTorch version of:

            batch_size = batch['observations'].shape[0]
            z = self.sample_z(batch_size, latent_dim, key)
            b_goals = self.network.select('b_value')(goal=batch['actor_goals'])
            mask = jax.random.uniform(key, shape=(batch_size, 1)) < self.config['z_mix_ratio']
            z = jnp.where(mask, z, b_goals)
        """
        batch_size = actor_goals.shape[0]
        env_size = actor_goals.shape[1]
        # sample normalized z
        z = self.sample_z(batch_size, env_size)
        
        b_goals = self.backward_net(actor_goals)

        mask = torch.rand(batch_size,env_size,device=self.device) < self.alg_cfg["z_mix_ratio"]   # bool tensor
        mask = mask.unsqueeze(-1) 
        
        z = torch.where(mask, z, b_goals)        

        return z

    
    
    
    def update(self) -> dict[str, float]:
        
        mean_behavior_loss = 0
        loss = 0
        cnt = 0
        
        

        batch = self.storage.sample(self.alg_cfg["batch_size"])        

        # pdb.set_trace()
        cur_obs = batch["obs"].to(self.device)
        cur_action = batch["actions"].to(self.device)
        discount = self.alg_cfg["gamma"]
        
        next_obs = batch["next_obs"].to(self.device)
        value_goals = batch["value_goals"].to(self.device)
        actor_goals = batch["actor_goals"].to(self.device)
        z = self.sample_mixed_z(actor_goals).detach()
           
        if not z.shape[-1] == self.policy_cfg["z_dim"]:
            raise RuntimeError("There's something wrong with the logic here")
    
    
        ''' 
        Update FB models   
        '''
        
        
        self.metrics.update(self.update_fb(cur_obs, cur_action,discount,next_obs, value_goals,z))
        self.metrics.update(self.update_policy(cur_obs, z))
        
        self.soft_update_params(self.forward_net, self.forward_target_net,
                                 self.alg_cfg["fb_target_tau"])
        self.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.alg_cfg["fb_target_tau"])
        
        return self.metrics
            
    def update_fb(self,
        cur_obs: torch.Tensor,
        cur_action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        value_goals: torch.Tensor,
        z: torch.Tensor):
        metrics: tp.Dict[str, float] = {}        
        # compute target successor measure
        with torch.no_grad():
            if self.policy_cfg["boltzmann"]:
                dist = self.policy(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = float(self.policy_cfg["stddev_schedule"])
                dist = self.policy(next_obs, z, stddev)
                next_action = dist.sample(clip=self.policy_cfg["stddev_clip"])
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
            target_B = self.backward_target_net(value_goals)  # batch x z_dim

            
            target_M1 = torch.einsum('bsd, btd -> bst', target_F1, target_B)  # batch x batch
            target_M2 = torch.einsum('bsd, btd -> bst', target_F2, target_B)  # batch x batch
            target_M = torch.min(target_M1, target_M2)
            
            
        # compute FB loss
        F1, F2 = self.forward_net(cur_obs, z, cur_action)
        B = self.backward_net(value_goals)
  
        
        
        M1 = torch.einsum('bsd, btd -> bst', F1, B)  # batch x batch
        M2 = torch.einsum('bsd, btd -> bst', F2, B)  # batch x batch
        I = torch.eye(M1.shape[1],M1.shape[2], device=M1.device)
        I = I.repeat(M1.shape[0], 1,1)
        off_diag = ~I.bool()
        fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag: tp.Any = -sum(torch.diagonal(M, dim1=1, dim2=2).mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # Target M for continuous actor
        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING

        # Cov = torch.matmul(B, B.T)
        Cov = torch.matmul(B, B.transpose(-1, -2))        
        orth_loss_diag = - 2 * torch.diagonal(Cov, dim1=1, dim2=2).mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.alg_cfg["ortho_coef"] * orth_loss
        
      

        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()      
          
        metrics['target_M'] = target_M.mean().item()
        metrics['M1'] = M1.mean().item()
        metrics['F1'] = F1.mean().item()
        metrics['B'] = B.mean().item()
        metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
        metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
        metrics['fb_loss'] = fb_loss.item()
        metrics['fb_diag'] = fb_diag.item()
        metrics['fb_offdiag'] = fb_offdiag.item()
        metrics['orth_loss'] = orth_loss.item()
        metrics['orth_loss_diag'] = orth_loss_diag.item()
        metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
        # eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
        # metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
        # metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
        if isinstance(self.fb_opt, torch.optim.Adam):
            metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]
  
      
        return metrics
    
    def soft_update_params(self,net, target_net, tau) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
            
    def update_policy(self,cur_obs: torch.Tensor, z: torch.Tensor):
        metrics: tp.Dict[str, float] = {}        
        if self.policy_cfg["boltzmann"]:
            dist = self.policy(cur_obs, z)
            action = dist.rsample()
        else:
            stddev = self.policy_cfg["stddev_schedule"]
            dist = self.policy(cur_obs, z, stddev)
            action = dist.sample(clip=self.policy_cfg["stddev_clip"])

        log_prob = dist.log_prob(action).sum(-1, keepdim=False)
        F1, F2 = self.forward_net(cur_obs, z, action)
        Q1 = torch.einsum('bsd, bsd -> bs', F1, z)
        Q2 = torch.einsum('bsd, bsd -> bs', F2, z)
   
        Q = torch.min(Q1, Q2)
        actor_loss = (self.policy_cfg["temp"] * log_prob - Q).mean() if self.policy_cfg["boltzmann"] else -Q.mean()

        # optimize actor
        self.policy_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['q'] = Q.mean().item()            
        metrics['actor_logprob'] = log_prob.mean().item()
            
        return metrics
    
    
            
        # for epoch in range(self.num_learning_epochs):
        #     self.policy.reset(hidden_states=self.last_hidden_states)
        #     self.policy.detach_hidden_states()            
        #     for obs, tecaher_actions, privileged_actions, dones in self.storage.generator():                
        #         # Inference of the student for gradient computation                
        #         # noise = torch.randn_like(tecaher_actions) * 1e-5
        #         # tecaher_actions +=noise

        #         actions = self.policy.act_inference(obs)
        #         # Behavior cloning loss
        #         behavior_loss = self.loss_fn(actions, tecaher_actions)

        #         # Total loss
        #         loss = loss + behavior_loss
        #         mean_behavior_loss += behavior_loss.item()
        #         cnt += 1

        #         # Gradient step
        #         if cnt % self.gradient_length == 0:
        #             self.optimizer.zero_grad()
        #             loss.backward()
                   
        #             if self.max_grad_norm:
        #                 nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
        #             self.optimizer.step()
        #             self.policy.detach_hidden_states()
        #             loss = 0

        #     # Reset dones
        #         self.policy.reset(dones.view(-1))
        #         self.policy.detach_hidden_states(dones.view(-1))

        # mean_behavior_loss /= cnt
        # self.storage.clear()
        # self.last_hidden_states = self.policy.get_hidden_states()
        # self.policy.detach_hidden_states()

        # # Construct the loss dictionary
        # loss_dict = {"behavior": mean_behavior_loss}

        return z


    def get_state(self) -> dict:
        """Return everything needed to resume training."""
        return {
            "policy": self.policy.state_dict(),
            "forward_net": self.forward_net.state_dict(),
            "forward_target_net": self.forward_target_net.state_dict(),
            "backward_net": self.backward_net.state_dict(),
            "backward_target_net": self.backward_target_net.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "fb_opt": self.fb_opt.state_dict(),                        
            "cfg": self.cfg,  # optional but handy
            "metrics": self.metrics, 
            "log_dir": self.log_dir,
        }
        
        
    def load_state(self, state: dict, load_optim: bool = True) -> None:
        """Load everything from a state dict created by get_state()."""

        self.policy.load_state_dict(state["policy"])
        self.forward_net.load_state_dict(state["forward_net"])
        self.backward_net.load_state_dict(state["backward_net"])
        self.forward_target_net.load_state_dict(state["forward_target_net"])
        self.backward_target_net.load_state_dict(state["backward_target_net"])

        if load_optim:
            self.policy_opt.load_state_dict(state["policy_opt"])
            self.fb_opt.load_state_dict(state["fb_opt"])

        
        
        