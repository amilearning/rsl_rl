# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from tabnanny import check
import time

import torch
import warnings
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import PPO, FBAlgorithm
from rsl_rl.env import VecEnv
from rsl_rl.modules import FBActor, ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state


class HierarchicalRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of high and low level hierarchical training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        
        self.fb_alg_cfg = train_cfg['fb_algorithm']        
        self.fb_policy_cfg = train_cfg["fb_policy"]
        
        self.device = device
        self.env = env

        self.prev_data_path = None 
        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        
        self.alg, self.fb_alg = self._construct_algorithm(obs)        


    def _construct_algorithm(self, obs: TensorDict) -> tuple[PPO, FBAlgorithm]:
        """Construct the actor-critic algorithm."""
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        
        
        alg_class = eval(self.fb_alg_cfg.pop("class_name"))
        
        ''' 
        TODO: get the hl action dim from the env command term
        '''
        self.hl_action_dim = self.env.env.env.env.command_manager.get_term('base_velocity').command.shape[-1]                
        fb_alg: FBAlgorithm = alg_class(log_dir = self.log_dir, 
                                        cfg = self.cfg,
                                        action_dim = self.hl_action_dim,
                                        obs_dim = obs['hl_policy'].shape[-1],                                        
                                        device=self.device)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        fb_alg.init_storage("fb",
                            self.env.num_envs,
                            obs,
                            [self.hl_action_dim])
        
        return alg, fb_alg
    
    
    
    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model       
        saved_dict = {
            "fb_alg_state": self.fb_alg.get_state(),
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])            
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            
        if "fb_alg_state" in loaded_dict:
            self.fb_alg.load_state(loaded_dict["fb_alg_state"], load_optim=load_optimizer)        
        # self.fb_alg.storage.load_latest(os.path.dirname(path))
        self.prev_data_path = path
        
        
        return loaded_dict["infos"]
    
    
    def fb_log_metrics(self, locs: dict, prefix: str = "FB") -> None:
        """Log FB + actor metrics stored in self.metrics to a SummaryWriter-like `writer`."""
        step = locs["it"]        
        m = self.fb_alg.metrics
        self.writer.add_scalar(f"{prefix}/target_M", m.get("target_M", 0.0), step)
        self.writer.add_scalar(f"{prefix}/M1", m.get("M1", 0.0), step)
        self.writer.add_scalar(f"{prefix}/F1", m.get("F1", 0.0), step)
        self.writer.add_scalar(f"{prefix}/B", m.get("B", 0.0), step)
        self.writer.add_scalar(f"{prefix}/B_norm", m.get("B_norm", 0.0), step)
        self.writer.add_scalar(f"{prefix}/z_norm", m.get("z_norm", 0.0), step)
        self.writer.add_scalar(f"{prefix}/fb_loss", m.get("fb_loss", 0.0), step)
        self.writer.add_scalar(f"{prefix}/fb_diag", m.get("fb_diag", 0.0), step)
        self.writer.add_scalar(f"{prefix}/fb_offdiag", m.get("fb_offdiag", 0.0), step)
        self.writer.add_scalar(f"{prefix}/orth_loss", m.get("orth_loss", 0.0), step)
        self.writer.add_scalar(f"{prefix}/orth_loss_diag", m.get("orth_loss_diag", 0.0), step)
        self.writer.add_scalar(f"{prefix}/orth_loss_offdiag", m.get("orth_loss_offdiag", 0.0), step)        
        self.writer.add_scalar(f"{prefix}/actor_loss", m.get("actor_loss", 0.0), step)
        self.writer.add_scalar(f"{prefix}/q", m.get("q", 0.0), step)
        self.writer.add_scalar(f"{prefix}/actor_logprob", m.get("actor_logprob", 0.0), step)
        
   
    def train_fb(self):
        num_updates = self.fb_alg_cfg['num_agent_updates']
        for train_it in range(num_updates):
            # 1. Run one FB update
            self.fb_alg.update()
            
            
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        hl_action_command = self.env.env.env.env.command_manager.get_term('base_velocity').command.clone()
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for itt in range(self.num_steps_per_env):                    
                    '''
                    high level policy interaction
                    '''          
                    if itt % self.cfg["hl_policy_decimation_multiplier"] == 0:                    
                        '''
                        TODO : get the hl action command from the env command term
                        '''
                        hl_obs =  obs.clone()                        
                        hl_action_command = self.env.env.env.env.command_manager.get_term('base_velocity').command.clone()
                        self.fb_alg.update_transition_pre(hl_obs, hl_action_command)                    
                    # if itt % self.cfg["hl_policy_decimation_multiplier"] == 0 and itt > 1:                                                                                            
                    #     hl_actions = self.fb_alg.act(hl_obs, is_eval = False)
                    #     if it > self.fb_alg_cfg["num_prior_data_collect_epoch"]+1:                            
                    #         self.fb_alg.update_transition_pre(hl_obs, hl_actions)                            
                    '''
                    remap the hl action command to the obs for ll policy
                    '''
                    obs['policy'][:,-self.hl_action_dim:] = hl_action_command                                        

                    '''
                    low level policy interaction
                    '''                    
                    # Sample ll_actions
                    ll_actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(ll_actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    
                    if itt % self.cfg["hl_policy_decimation_multiplier"] == 0:    
                        hl_dones = dones.clone()                            
                        hl_time_outs = extras['time_outs'].clone()
                    else:
                        hl_dones = hl_dones | dones
                        hl_time_outs = hl_time_outs | extras['time_outs']
                    
                    if itt % self.cfg["hl_policy_decimation_multiplier"] == 0:    
                        '''
                        TODO: compute hl_rewards if can 
                        '''
                        hl_rewards = rewards.clone()                                                
                        self.fb_alg.update_transition_post(hl_rewards, hl_dones, hl_time_outs)
                                                 
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            fb_start = time.time()
            if self.cfg["resume"] or it > self.fb_alg_cfg["num_prior_data_collect_epoch"]: #                                              
                self.train_fb()
            fb_stop = time.time()            
            train_fb_time = fb_stop - fb_start
            
            
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())                
                self.fb_log_metrics(locals())
                
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))



    def train_fb(self):
        num_updates = self.fb_alg_cfg['num_agent_updates']
        for train_it in range(num_updates):
            # 1. Run one FB update
            self.fb_alg.update()
            
            
    def fb_log_metrics(self, locs: dict, prefix: str = "FB") -> None:
        """Log FB + actor metrics stored in self.metrics to a SummaryWriter-like `writer`."""
        step = locs["it"]        
        m = self.fb_alg.metrics
        self.writer.add_scalar(f"{prefix}/target_M", m.get("target_M", 0.0), step)
        self.writer.add_scalar(f"{prefix}/M1", m.get("M1", 0.0), step)
        self.writer.add_scalar(f"{prefix}/F1", m.get("F1", 0.0), step)
        self.writer.add_scalar(f"{prefix}/B", m.get("B", 0.0), step)
        self.writer.add_scalar(f"{prefix}/B_norm", m.get("B_norm", 0.0), step)
        self.writer.add_scalar(f"{prefix}/z_norm", m.get("z_norm", 0.0), step)
        self.writer.add_scalar(f"{prefix}/fb_loss", m.get("fb_loss", 0.0), step)
        self.writer.add_scalar(f"{prefix}/fb_diag", m.get("fb_diag", 0.0), step)
        self.writer.add_scalar(f"{prefix}/fb_offdiag", m.get("fb_offdiag", 0.0), step)
        self.writer.add_scalar(f"{prefix}/orth_loss", m.get("orth_loss", 0.0), step)
        self.writer.add_scalar(f"{prefix}/orth_loss_diag", m.get("orth_loss_diag", 0.0), step)
        self.writer.add_scalar(f"{prefix}/orth_loss_offdiag", m.get("orth_loss_offdiag", 0.0), step)        
        self.writer.add_scalar(f"{prefix}/actor_loss", m.get("actor_loss", 0.0), step)
        self.writer.add_scalar(f"{prefix}/q", m.get("q", 0.0), step)
        self.writer.add_scalar(f"{prefix}/actor_logprob", m.get("actor_logprob", 0.0), step)
        