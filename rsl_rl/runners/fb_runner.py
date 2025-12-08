# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import FBAlgorithm
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state
import warnings 

class FBRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["policy","previliege"])

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

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

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_prior_train.pt"))
        self.global_step = 0
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        

                stop = time.time()
                collection_time = stop - start
                start = stop

                

            
            
                    
            with torch.inference_mode():
                self.env.reset()
                prev_contact = obs["previliege"].clone()
                contact_changed_any = torch.tensor(False,device=self.device)
                init_count = 0
                init_count_max = 50
                while init_count < init_count_max:                    
                    if (obs['previliege'].float().mean() >= 0.9):
                        break               
                    actions = self.alg.rollout(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))                    
                    init_count+=1

                for rollout_step in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.rollout(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                    if contact_changed_any.item() is False: 
                        contact_changed_any = (obs["previliege"] !=prev_contact).any()                    
                    prev_contact = obs["previliege"].clone()

            stop = time.time()
            collection_time = stop - start
            start = stop
            
            # self.alg.relabeling_batch( env_cfg = self.env.cfg)

           
            if contact_changed_any:                
                loss_dict = self.alg.update()
            else:
                print(f"No contact change found... consider increase currentnum_steps_per_env {self.num_steps_per_env}...")

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
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

    def _construct_algorithm(self, obs: TensorDict) -> FBAlgorithm:
        """Construct the distillation algorithm."""
        # Initialize the policy        
        # Resolve RND config
        
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
        # actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        # actor_critic: ActorCritic = actor_critic_class(
        #     obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        # ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: FBAlgorithm = alg_class(action_dim = self.env.num_actions,
                                     obs_dim = obs['policy'].shape[-1],
                                     cfg = self.cfg, 
                                     device=self.device,
                                     **self.alg_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference
    

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
        return loaded_dict["infos"]
    


    def teacher_eval(self, num_eval_iterations: int) -> None:
        self._prepare_logging_writer()
        obs = self.env.get_observations().to(self.device)
        self.eval_mode()  
        start_iter = 0
        tot_iter = num_eval_iterations                
        
        for it in range(start_iter, tot_iter): 
            pose_error_w = torch.zeros(self.num_steps_per_env, self.env.num_envs, dtype=torch.float, device=self.device)           
            
            with torch.inference_mode():
                for rollout_step in range(self.num_steps_per_env):
                    actions = self.alg.rollout(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))                    
                    if self.log_dir is not None:
                        pose_error_w[rollout_step,:] =  obs['evaluation'].squeeze().clone()
                       
            self.writer.add_scalar("Eval_teacher/pose_error_w_mean", pose_error_w.mean().cpu().numpy().tolist(), it)
            self.writer.add_scalar("Eval_teacher/pose_error_w_std", pose_error_w.std().cpu().numpy().tolist(), it)

     
     
    def student_eval(self, num_eval_iterations: int) -> None:      
        self._prepare_logging_writer()
        obs = self.env.get_observations().to(self.device)
        self.eval_mode()  # switch to train mode (for dropout for example)
        student_policy = self.get_inference_policy(device=self.env.device)
        start_iter = 0
        tot_iter = num_eval_iterations                
        
        for it in range(start_iter, tot_iter): 
            pose_error_w = torch.zeros(self.num_steps_per_env, self.env.num_envs, dtype=torch.float, device=self.device)           
            with torch.inference_mode():
                self.env.reset()
                init_count = 0
                init_count_max = 50
                while init_count < init_count_max:                    
                    if (obs['previliege'].float().mean() >= 0.9):
                        break               
                    teacher_actions = self.alg.rollout(obs)
                    obs, rewards, dones, extras = self.env.step(teacher_actions.to(self.env.device))                    
                    init_count+=1


                for rollout_step in range(self.num_steps_per_env):   
                    actions = student_policy(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))                
                    if self.log_dir is not None:
                        pose_error_w[rollout_step,] =  obs['evaluation'].squeeze().clone()
                       
            self.writer.add_scalar("Eval_student/pose_error_w_mean", pose_error_w.mean().cpu().numpy().tolist(), it)
            self.writer.add_scalar("Eval_student/pose_error_w_std", pose_error_w.std().cpu().numpy().tolist(), it)

