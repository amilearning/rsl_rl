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
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state


class HierarchicalRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of high and low level hierarchical training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.low_alg_cfg = train_cfg["low_algorithm"]        
        self.low_policy_cfg = train_cfg["low_policy"]
        self.high_actions_dim = train_cfg["high_policy_actions_dim"]
        self.high_level_decimation = train_cfg["high_level_decimation"]
        self.high_alg_cfg = train_cfg["high_algorithm"]
        self.high_policy_cfg = train_cfg["high_policy"]

        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()        
        default_sets = ["critic"]        
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)



        # Create the algorithm
        self.low_alg, self.high_alg = self._construct_algorithm(obs)         

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



    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.low_alg.policy.to(device)
            self.high_alg.policy.to(device)
        return self.low_alg.policy.act_inference


    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "low_model_state_dict": self.low_alg.policy.state_dict(),
            "low_optimizer_state_dict": self.low_alg.optimizer.state_dict(),
            "high_model_state_dict": self.high_alg.policy.state_dict(),
            "high_optimizer_state_dict": self.high_alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        low_resumed_training = self.low_alg.policy.load_state_dict(loaded_dict["low_model_state_dict"])
        high_resumed_training = self.high_alg.policy.load_state_dict(loaded_dict["high_model_state_dict"])
        # Load optimizer if used
        if load_optimizer and low_resumed_training:
            # Algorithm optimizer
            self.low_alg.optimizer.load_state_dict(loaded_dict["low_optimizer_state_dict"])            
        if load_optimizer and high_resumed_training:
            # Algorithm optimizer
            self.high_alg.optimizer.load_state_dict(loaded_dict["high_optimizer_state_dict"])            
        # Load current learning iteration
        if low_resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]


    def eval_mode(self) -> None:
        # PPO
        self.low_alg.policy.eval()
        self.high_alg.policy.eval()

    def train_mode(self) -> None:
        # PPO
        self.low_alg.policy.train()
        self.high_alg.policy.train()

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

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                
                mdp_count_for_hl = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
                sample_high_level = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
                hl_actions = self.high_alg.act(obs).clone()                                     

                for env_count in range(self.num_steps_per_env):                    
                    high_actions_ = self.high_alg.act(obs)                                                                                
                    hl_actions[sample_high_level,:] = high_actions_[sample_high_level,:].clone()                    
                    
                    obs['policy'][sample_high_level,-self.high_actions_dim:] = hl_actions[sample_high_level,:].clone()
                    self.env.env.env.env.command_manager._terms['base_velocity'].vel_command_b[sample_high_level,:] = high_actions_[sample_high_level,:].clone()
                    
                    low_actions = self.low_alg.act(obs)                    
                    mdp_count_for_hl+=1
                    # Step the environment
                    obs, low_rewards, high_rewards, dones, extras = self.env.step(low_actions.to(self.env.device))
                    # Move to device
                    obs, low_rewards, high_rewards, dones = (obs.to(self.device), low_rewards.to(self.device), high_rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.low_alg.process_env_step(obs, low_rewards, dones, extras)
 
                    ## check if we need to resample high level actions , if so, change the sample_high_level[env] as True                                   
                    sample_high_level[:] = False  
                    decimation_reached_mask = mdp_count_for_hl >= self.high_level_decimation
                    reset_mdp_count_for_hl_mask = dones | decimation_reached_mask
                    sample_high_level[reset_mdp_count_for_hl_mask] = True
                    mdp_count_for_hl[reset_mdp_count_for_hl_mask] = 0
                        
                    if len(obs[sample_high_level]) > 0:
                        self.high_alg.process_env_step(obs[sample_high_level], high_rewards[sample_high_level], dones[sample_high_level], extras[sample_high_level])
                    # Extract intrinsic rewards (only for logging)
                    
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                    
                        cur_reward_sum +=  low_rewards
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

                # Compute returns
                self.low_alg.compute_returns(obs)
                self.high_alg.compute_returns(obs)

            # Update policy
            loss_dict = self.low_alg.update()
            loss_dict = self.high_alg.update()

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

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        
        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.low_policy_cfg.get("actor_obs_normalization") is None:
                self.low_policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.low_policy_cfg.get("critic_obs_normalization") is None:
                self.low_policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.high_policy_cfg.get("actor_obs_normalization") is None:
                self.high_policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.high_policy_cfg.get("critic_obs_normalization") is None:
                self.high_policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        '''
        Iniitalize Low level policy 
        '''
        # Initialize the policy
        low_actor_critic_class = eval(self.low_policy_cfg.pop("class_name"))
        low_actor_critic: ActorCritic = low_actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.low_policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        low_alg_class = eval(self.low_alg_cfg.pop("class_name"))
        low_alg: PPO = low_alg_class(low_actor_critic, device=self.device, **self.low_alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        low_alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        '''
        Iniitalize High level policy 
        '''
        # Initialize the policy
        high_actor_critic_class = eval(self.high_policy_cfg.pop("class_name"))
        high_actor_critic: ActorCritic = high_actor_critic_class(
            obs, self.cfg["obs_groups"], self.high_actions_dim, **self.high_policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        high_alg_class = eval(self.high_alg_cfg.pop("class_name"))
        high_alg: PPO = high_alg_class(high_actor_critic, device=self.device, **self.high_alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        high_alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.high_actions_dim],
        )

        return low_alg, high_alg
    
    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        low_mean_std = self.low_alg.policy.action_std.mean()
        high_mean_std = self.high_alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/low_learning_rate", self.low_alg.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/high_learning_rate", self.high_alg.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/low_mean_noise_std", low_mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/high_mean_noise_std", high_mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        # if len(locs["rewbuffer"]) > 0:
        #     # Separate logging for intrinsic and extrinsic rewards        
        #     # Everything else
        #     self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
        #     self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        #     if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
        #         self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
        #         self.writer.add_scalar(
        #             "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
        #         )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                # f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Low Mean action noise std:":>{pad}} {low_mean_std.item():.2f}\n"""
                f"""{"High Mean action noise std:":>{pad}} {high_mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards

            # log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # # Print episode information
            # log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Low Mean action noise std:":>{pad}} {low_mean_std.item():.2f}\n"""
                f"""{"High Mean action noise std:":>{pad}} {high_mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)