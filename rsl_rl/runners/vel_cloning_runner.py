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
from rsl_rl.algorithms import Cloning
from rsl_rl.env import VecEnv
from rsl_rl.modules import BCStudentTeacher
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state
import matplotlib.pyplot as plt

class VelClonPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.gp_sampler = self.env.env.env.command_manager._terms['base_velocity'].gp_model
        self.is_active_sample = self.env.env.env.command_manager._terms['base_velocity'].cfg.active_sample
        
        

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["policy", "teacher"])

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
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

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
        
        self.save(os.path.join(self.log_dir, f"model_prior_train.pt"))

        for it in range(start_iter, tot_iter):
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
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
                    
       

            stop = time.time()
            collection_time = stop - start
            start = stop
            
            self.alg.relabeling_velolicy_batch( env_cfg = self.env.cfg)
         
                    
            loss_dict = self.alg.update()
         
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

    def _construct_algorithm(self, obs: TensorDict) -> Cloning:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: BCStudentTeacher  = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Cloning = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env*10,
            obs,
            [self.env.num_actions],
        )

        return alg


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
    
    def eval(self, num_eval_iterations: int, is_student = False, teacher_cmd_history = None) -> None:      
        # self._prepare_logging_writer()
        obs = self.env.get_observations().to(self.device)
        self.eval_mode()  # switch to train mode (for dropout for example)
        self.alg.policy.to(self.device)
        
        student_policy = self.alg.policy.act_inference # self.get_inference_policy(device=self.env.device)                                              
        teacher_policy = self.alg.policy.act_teacher_mean

        start_iter = 0
        tot_iter = num_eval_iterations                
        cmd_history = torch.zeros(tot_iter, self.num_steps_per_env, self.env.num_envs, 3,dtype=torch.float, device=self.device)
        error_history = torch.zeros(tot_iter, self.num_steps_per_env, self.env.num_envs,2, dtype=torch.float, device=self.device)
        for it in range(start_iter, tot_iter): 
            lin_vel_xy_error = torch.zeros(self.num_steps_per_env, dtype=torch.float, device=self.device)           
            ang_vel_xy_error = torch.zeros(self.num_steps_per_env, dtype=torch.float, device=self.device)           
            with torch.inference_mode():
                for rollout_step in range(self.num_steps_per_env):   
                    if teacher_cmd_history is not None:
                        obs['teacher'][:,-3:] = teacher_cmd_history[it, rollout_step, :, :].clone()
                        obs['policy'][:,-3:] = teacher_cmd_history[it, rollout_step, :, :].clone()                    
                    if is_student:
                        student_actions = student_policy(obs)
                    else:
                        teacher_actions = teacher_policy(obs)

                    cmd_history[it, rollout_step, :, :] = obs['policy'][:,-3:].clone()
                    if is_student:
                        obs, rewards, dones, extras = self.env.step(student_actions.to(self.env.device))                
                    else:
                        obs, rewards, dones, extras = self.env.step(teacher_actions.to(self.env.device))                

                    cur_lin_vel = obs['policy'][:,:2].clone()
                    cur_ang_vel = obs['policy'][:,5].clone()
                    cmd_lin_vel = obs['policy'][:,-3:-1].clone()
                    cmd_ang_vel = obs['policy'][:,-1].clone()

                    lin_vel_diff = torch.norm(cmd_lin_vel - cur_lin_vel, dim=-1)
                    ang_vel_diff = torch.norm(cmd_ang_vel.unsqueeze(dim=-1) - cur_ang_vel.unsqueeze(dim=-1), dim =-1)

                    error_history[it, rollout_step, :,0] = lin_vel_diff.clone()
                    error_history[it, rollout_step, :,1] = ang_vel_diff.clone()


                    lin_vel_xy_error[rollout_step] = lin_vel_diff.mean()
                    ang_vel_xy_error[rollout_step] =ang_vel_diff.mean()

            # self.writer.add_scalar("Eval_student/error_vel_xy", lin_vel_xy_error.mean().cpu().numpy().tolist(), it)
            # self.writer.add_scalar("Eval_student/error_vel_yaw", ang_vel_xy_error.mean().cpu().numpy().tolist(), it)

        return cmd_history, error_history
    

    def active_learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer        
        self._prepare_logging_writer()
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

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
        
        self.save(os.path.join(self.log_dir, f"model_prior_train.pt"))
        


        for it in range(start_iter, tot_iter):            
            start = time.time()
            # Rollout            
            with torch.inference_mode():
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
                    
       

            stop = time.time()
            collection_time = stop - start
            start = stop
            
            self.alg.relabeling_velolicy_batch( env_cfg = self.env.cfg)
         
                    
            loss_dict = self.alg.update()
         
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

            teacher_cmd_history, teacher_error_history = self.eval(1, is_student=False)
            cmd_history, error_history = self.eval(1, is_student=True, teacher_cmd_history= teacher_cmd_history)
          
            teacher_cmd, teacher_lin, teacher_ang = self.get_err_statistics(teacher_cmd_history[0],teacher_error_history[0])
            student_cmd, student_lin, student_ang = self.get_err_statistics(cmd_history[0],error_history[0])
       
            self.get_histogram_plot_lin_ang(it, teacher_lin.cpu().numpy(), student_lin.cpu().numpy(), teacher_ang.cpu().numpy(), student_ang.cpu().numpy())
            self.get_scatter_lin_error_over_cmd_space_plot(it,teacher_cmd.cpu().numpy(),student_cmd.cpu().numpy(), teacher_lin.cpu().numpy(),student_lin.cpu().numpy())

            if self.is_active_sample:
                gp_X = student_cmd # cmd_history.view(-1, self.gp_sampler.input_dim)
                gp_y =  torch.cat([student_lin, student_lin],dim=-1) #  error_history.view(-1, self.gp_sampler.output_dim)
                self.gp_sampler.train(gp_X, gp_y)            

            self.train_mode()

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            


    def get_err_statistics(self,cmd, err):
        T, N, _ = cmd.shape

        results = []  # list for storing results per environment

        for env in range(N):
            env_cmd = cmd[:, env, :]   # [T, 3]
            env_err = err[:, env, :]   # [T, 2]

            # Convert commands into tuples so we can group by them
            unique_cmds, inverse_idx = torch.unique(env_cmd, dim=0, return_inverse=True)

            env_result = []

            for ci, command in enumerate(unique_cmds):
                # Find all timesteps that use this command
                mask = (inverse_idx == ci)   # [T]

                # Extract linear & angular error
                lin_err_values = env_err[mask, 0]
                ang_err_values = env_err[mask, 1]

                env_result.append({
                    "command": command,                      # [3] tensor (cmd_x, cmd_y, yaw)
                    "mean_linear_error": lin_err_values.mean().item(),
                    "mean_angular_error": ang_err_values.mean().item(),
                    "num_samples": mask.sum().item()
                })

            results.append(env_result)

            
        all_commands = []
        all_lin = []
        all_ang = []
        all_num = []

        for env_result in results:
            for item in env_result:
                all_commands.append(item["command"])                    # [3]
                all_lin.append(item["mean_linear_error"])               # scalar
                all_ang.append(item["mean_angular_error"])              # scalar
                all_num.append(item["num_samples"])                     # scalar
        all_commands = torch.stack(all_commands)
        all_lin = torch.tensor(all_lin)
        all_ang = torch.tensor(all_ang)
      
        return all_commands, all_lin.unsqueeze(-1), all_ang.unsqueeze(-1)

            
    def get_scatter_lin_error_over_cmd_space_plot(self,it_num,teacher_cmd_history,student_cmd_history, teacher_lin,student_lin):
        t_cmd_x = teacher_cmd_history[:,0]
        t_cmd_y = teacher_cmd_history[:,1]

        s_cmd_x = student_cmd_history[:,0]
        s_cmd_y = student_cmd_history[:,1]

        # -----------------------
        # TEACHER PLOT
        # -----------------------
        eval_hist_log_dir =  os.path.join(self.log_dir, "eval_data_hist", self.cfg['load_checkpoint'])        
        if not os.path.exists(eval_hist_log_dir):
            os.makedirs(eval_hist_log_dir)

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(t_cmd_x, t_cmd_y, c=teacher_lin, s=5, cmap="viridis")
        plt.colorbar(sc, label="Teacher linear error")
        plt.xlabel("cmd_x")
        plt.ylabel("cmd_y")
        plt.title("Teacher: Linear Error over Command Space")
        plt.grid(True, alpha=0.05)

        teacher_path = os.path.join(eval_hist_log_dir, f"{it_num}_teacher_cmd_error_map.png")
        plt.savefig(teacher_path, dpi=200, bbox_inches="tight")
        plt.close()

        # -----------------------
        # STUDENT PLOT
        # -----------------------
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(s_cmd_x, s_cmd_y, c=student_lin, s=5, cmap="viridis")
        plt.colorbar(sc, label="Student linear error")
        plt.xlabel("cmd_x")
        plt.ylabel("cmd_y")
        plt.title("Student: Linear Error over Command Space")
        plt.grid(True, alpha=0.05)

        student_path = os.path.join(eval_hist_log_dir, f"{it_num}_student_cmd_error_map.png")
        plt.savefig(student_path, dpi=200, bbox_inches="tight")
        plt.close()


    def get_histogram_plot_lin_ang(self,it_num, teacher_lin, student_lin, teacher_ang, student_ang):
      

        # -----------------------
        # Histogram Plot
        # -----------------------
        plt.figure(figsize=(10, 4))

        # Linear
        plt.subplot(1, 2, 1)
        plt.hist(teacher_lin, bins=50, alpha=0.5, label="Teacher", density=True)
        plt.hist(student_lin, bins=50, alpha=0.5, label="Student", density=True)
        plt.xlabel("Linear velocity error")
        plt.ylabel("Density")
        plt.title("Linear Error Distribution")
        plt.legend()

        # Angular
        plt.subplot(1, 2, 2)
        plt.hist(teacher_ang, bins=50, alpha=0.5, label="Teacher", density=True)
        plt.hist(student_ang, bins=50, alpha=0.5, label="Student", density=True)
        plt.xlabel("Angular velocity error")
        plt.ylabel("Density")
        plt.title("Angular Error Distribution")
        plt.legend()


        
        eval_log_dir =  os.path.join(self.log_dir, "eval_data", self.cfg['load_checkpoint'])        
        if not os.path.exists(eval_log_dir):
            os.makedirs(eval_log_dir)
        
        hist_path = os.path.join(eval_log_dir, f"epoch_{it_num}_error_histograms.png")
        plt.savefig(hist_path, dpi=200, bbox_inches="tight")
        plt.close()