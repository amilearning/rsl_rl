

# import os

# import torch
# # import matplotlib.pyplot as plt

# import os, tempfile
# # from datetime import datetime, timedelta
# import torch

# import gpytorch
# # from torch.utils.data import DataLoader, Dataset
# from rsl_rl.env import VecEnv
# import math
# # from collections.abc import Sequence
# # from typing import TYPE_CHECKING, Tuple, Optional

# # from isaaclab.assets import Articulation
# # from isaaclab.managers import CommandTerm
# # from isaaclab.markers import VisualizationMarkers
# # from isaaclab.markers.config import CONTACT_COMMAND_MARKER_CFG, HIP_RAY_CASTER_COMMAND_MARKER_CFG
# # from isaaclab.sensors import RayCaster
# # from isaaclab.terrains import TerrainImporter
# # from isaaclab.utils.math import wrap_to_pi, yaw_quat, quat_apply, quat_apply_inverse
# # from scripts.reinforcement_learning.rsl_rl.acquisition import VarGradAcq, VarianceAcq



# class RunningNormStats:
#     def __init__(self, dim, device="cuda", eps=1e-8):
#         self.device = device
#         self.eps = eps
#         self.n = torch.tensor(0.0, device=device)
#         self.mean = torch.zeros(dim, device=device)
#         self.var = torch.ones(dim, device=device)

#     @torch.no_grad()
#     def update(self, batch: torch.Tensor):
#         """
#         Incremental update using Welford's algorithm for numerical stability.
#         batch: [N, D]
#         """
#         if batch.numel() == 0:
#             return

#         batch = batch.to(self.device)
#         n_b = batch.size(0)
#         mean_b = batch.mean(dim=0)
#         var_b = batch.var(dim=0, unbiased=False)

#         n_a = self.n
#         mean_a = self.mean
#         var_a = self.var

#         total = n_a + n_b
#         delta = mean_b - mean_a
#         new_mean = mean_a + delta * (n_b / total)
#         m_a = var_a * n_a
#         m_b = var_b * n_b
#         new_var = (m_a + m_b + delta.pow(2) * n_a * n_b / total) / total

#         self.mean.copy_(new_mean).to(device=self.device)
#         self.var.copy_(new_var).to(device=self.device)
#         self.n.copy_(total).to(device=self.device)

#     def normalize(self, x):
#         return (x - self.mean) / (torch.sqrt(self.var + self.eps))

#     def denormalize_mean(self, x):
#         return x * torch.sqrt(self.var + self.eps) + self.mean
    
#     def denormalize(self,mean, std):
#         mean_d = self.denormalize_mean(mean)
#         std_d = self.denormalize_mean(mean + std) - self.denormalize_mean(mean)
#         return mean_d, std_d
    
#     def state_dict(self):
#         return dict(mean=self.mean, var=self.var, n=self.n)

#     def load_state_dict(self, state):
#         self.mean.copy_(state["mean"]).to(device=self.device)
#         self.var.copy_(state["var"]).to(device=self.device)
#         self.n.copy_(state["n"]).to(device=self.device)

# class IndependentMultiSparseGP(gpytorch.models.ApproximateGP):
#     def __init__(self, inducing_points_num, input_dim, num_tasks):
#         # Let's use a different set of inducing points for each task
#         inducing_points = torch.rand(num_tasks, inducing_points_num, input_dim)

#         # We have to mark the CholeskyVariationalDistribution as batch
#         # so that we learn a variational distribution for each task
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
#             inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
#         )

#         variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
#             gpytorch.variational.VariationalStrategy(
#                 self, inducing_points, variational_distribution, learn_inducing_locations=True
#             ),
#             num_tasks=num_tasks,
#         )

#         super().__init__(variational_strategy)


#         # The mean and covariance modules should be marked as batch
#         # so we learn a different set of hyperparameters
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([num_tasks])), # nu = 1.5
#             batch_shape=torch.Size([num_tasks])
#         )


#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# class GPCmdSampler:
#     def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cuda") -> None:
#         ## TODO: command type generalization
#         self.cmd = env.cfg.commands.base_velocity
#         ## TODO: input and output generalization        
#         self.input_dim = 3 # linear x, y, angular z
#         self.output_dim = 2 # linear vel error, and angular vel error

#         self.cfg = train_cfg
#         self.env = env
#         self.lr = 5e-3
#         self.inducing_points = 256      
#         self.max_online_train_epochs = 5000  
#         self.log_dir = log_dir
#         self.training_data_folder = os.path.join(self.log_dir, 'training_data')
#         self.gp_model_folder = os.path.join(self.log_dir, 'gpmodels')
#         self.plot_folder = os.path.join(self.training_data_folder, 'plots')
#         self.processed_files = None
#         if not os.path.exists(self.training_data_folder):
#             os.makedirs(self.training_data_folder)
#         if not os.path.exists(self.gp_model_folder):
#             os.makedirs(self.gp_model_folder)
#         if not os.path.exists(self.plot_folder):
#             os.makedirs(self.plot_folder)
#         running_model_save_file_name = f"uptodate_model.pt"
#         self.running_model_file_path = os.path.join(self.gp_model_folder, running_model_save_file_name)

#         self.device = device

        
        
#         self.train_loader = None
#         self.val_loader = None
#         # GP model + likelihood
#         self.gp_model = IndependentMultiSparseGP(
#             inducing_points_num=self.inducing_points,
#             input_dim=self.input_dim,
#             num_tasks=self.output_dim,
#         ).to(self.device)
#         self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim).to(self.device)

#         # Optimizer: variational params + hypers + likelihood
#         self.optimizer_gp = torch.optim.Adam(
#             [
#                 {"params": self.gp_model.variational_parameters()},
#                 {"params": self.gp_model.hyperparameters()},
#                 {"params": self.likelihood.parameters()},
#             ],
#             lr=self.lr,  
#         )

#         # ELBO (num_data affects KL scaling; set approx total per-task count)
#         self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_model, num_data=1024).to(self.device)
        
#         self.norm_input = RunningNormStats(dim= self.input_dim, device = self.device)
#         self.norm_output = RunningNormStats(dim= self.output_dim,device = self.device)

#         ## 
#         # self.acq_bounds = (torch.zeros(self.input_dim, device=self.device), torch.ones(self.input_dim, device=self.device))
#         # self.acq = VarGradAcq(
#         #             model=self.gp_model,             # your gpytorch model; posterior comes from model(X)
#         #             likelihood=self.likelihood,      # optional
#         #             norm_input = self.norm_input,
#         #             norm_output = self.norm_output,
#         #             bounds=self.acq_bounds,
#         #             reduce_out="mean",
#         #             device=self.device,
#         #         )
#         print("initalize done")
    
        
#     def train(self,X,y):
#         self.gp_model.train()
#         self.likelihood.train()
        
#         # Move to device
#         X, y = X.to(self.device), y.to(self.device)
#         self.norm_input.update(X)
#         self.norm_output.update(y)

#         # ----- 1) Train/validation split (10% validation) -----
#         N = X.shape[0]
#         val_size = max(1, int(0.1 * N))          # at least 1 sample
#         perm = torch.randperm(N, device=self.device)

#         val_idx   = perm[:val_size]
#         train_idx = perm[val_size:]

#         X_train, y_train = X[train_idx], y[train_idx]
#         X_val,   y_val   = X[val_idx],   y[val_idx]


#         X_train_n = self.norm_input.normalize(X_train)
#         y_train_n = self.norm_output.normalize(y_train)

#         X_val_n   = self.norm_input.normalize(X_val)
#         y_val_n   = self.norm_output.normalize(y_val)

#         # ----- 3) Early stopping setup -----
#         best_val_loss = math.inf
#         patience = 10
#         bad_epochs = 0

#         train_loss_sum = 0.0
#         train_epoch_count = 0

#         for epoch in range(self.max_online_train_epochs):
#             # ---- Training step ----
#             self.gp_model.train()
#             self.likelihood.train()
#             self.optimizer_gp.zero_grad()

#             with gpytorch.settings.num_likelihood_samples(1):
#                 output = self.gp_model(X_train_n)
#                 loss = -self.mll(output, y_train_n)

#             loss.backward()
#             self.optimizer_gp.step()

#             train_loss_sum += loss.item()
#             train_epoch_count += 1

#             # ---- Validation step ----
#             self.gp_model.eval()
#             self.likelihood.eval()
#             with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
#                 val_output = self.gp_model(X_val_n)
#                 val_loss = -self.mll(val_output, y_val_n).item()

#             # ---- Early stopping check ----
#             if val_loss < best_val_loss - 1e-6:  # small tolerance
#                 best_val_loss = val_loss
#                 bad_epochs = 0
#             else:
#                 bad_epochs += 1

#             if bad_epochs >= patience:
#                 # Stop training if no improvement for `patience` epochs
#                 break

#         avg_train_loss = train_loss_sum / max(1, train_epoch_count)

#         # You can return both if useful
#         return avg_train_loss, best_val_loss


#     # def train(self,X,y):
#     #     self.gp_model.train()
#     #     self.likelihood.train()
#     #     epoch_loss = 0.0
#     #     epoch_count = 0        
#     #     X, y = X.to(self.device), y.to(self.device)
#     #     self.norm_input.update(X)
#     #     self.norm_output.update(y)        
#     #     X_batch = self.norm_input.normalize(X)
#     #     y_batch = self.norm_output.normalize(y)
        
#     #     for epoch in range(self.max_online_train_epochs):
#     #         epoch_count+=1
#     #         self.optimizer_gp.zero_grad()
#     #         with gpytorch.settings.num_likelihood_samples(1):
#     #             output = self.gp_model(X_batch)
#     #             loss = -self.mll(output, y_batch)
#     #         loss.backward()
#     #         self.optimizer_gp.step()
#     #         epoch_loss += loss.item()
#     #     epoch_loss = epoch_loss/epoch_count
#     #     return epoch_loss



#         # self.gp_model.train()
#         # self.likelihood.train()

#         # self.optimizer_gp.zero_grad()
#         # output = self.gp_model(X)
#         # loss = -self.mll(output, Y)
#         # loss.backward()
#         # self.optimizer_gp.step()
#         # return loss.item()
    
    
#     # @torch.no_grad()
#     # def predict(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     #     self.gp_model.eval()
#     #     self.likelihood.eval()
#     #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     #         x_batch = x_batch.to(self.device)
#     #         x_batch = self.norm_input.normalize(x_batch)
#     #         pred = self.likelihood(self.gp_model(x_batch))
#     #         mean = pred.mean
#     #         var = pred.variance
#     #         std = var.sqrt()
#     #         mean_d, std_d = self.norm_output.denormalize(mean, std)
#     #     return mean_d, std_d

