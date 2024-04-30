import torch
from torch import nn
import numpy as np
import os
import random
from base_trainer import BaseTrainer
from src.utils.general_utils import get_expon_lr_func
from src.utils.graphics_utils import build_rotation
from src.model.gaussian4d import Gaussian4D
from src.utils.camera import getNerfppNorm
from src.utils.graphics_utils import fetchPly


class Gaussian4DTrainer(BaseTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = Gaussian4D(**kwargs)
        self.gaussians = self.model.gaussians

        self.position_lr_init = kwargs['position_lr_init']
        self.position_lr_final = kwargs['position_lr_final']
        self.position_lr_delay_mult = kwargs['position_lr_delay_mult']
        self.position_lr_max_steps = kwargs['position_lr_max_steps']
        self.deformation_lr_init = kwargs['deformation_lr_init']
        self.deformation_lr_final = kwargs['deformation_lr_final']
        self.deformation_lr_delay_mult = kwargs['deformation_lr_delay_mult']
        self.grid_lr_init = kwargs['grid_lr_init']
        self.grid_lr_final = kwargs['grid_lr_final']
        self.grid_lr_delay_mult = kwargs['grid_lr_delay_mult']
        self.feature_lr = kwargs['feature_lr']
        self.opacity_lr = kwargs['opacity_lr']
        self.scaling_lr = kwargs['scaling_lr']
        self.rotation_lr = kwargs['rotation_lr']
        self.spatial_lr_scale = self.gaussians.get_spatial_lr_scale()
        self.percent_dense = kwargs['percent_dense']
        
        self.optimizer = None

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        # # members in the base class
        # device, train_dataloader, valid_dataloader, loss_fns, loss_weights, metric_fns, 
        # logger, monitor, num_epochs, epoch, np_random_seeds

        nerf_normalization = getNerfppNorm(self.train_cameras)

        ply_path = os.path.join(self.data_dir, "points3D_downsample2.ply")
        pcd = fetchPly(ply_path)
        print(f"ply data loaded, initial points:{len(pcd.points)}")


    def _run(self, inputs):
        raise NotImplementedError

    def _training_setup(self):
        self.model.training_setup()

        l = [
            {'params': [self._xyz], 'lr': self.position_lr_init * self._spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': self.deformation_lr_init * self._spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': self.grid_lr_init * self._spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init*self._spatial_lr_scale,
                                                    lr_final=self.position_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=self.position_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=self.deformation_lr_init*self._spatial_lr_scale,
                                                    lr_final=self.deformation_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=self.deformation_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=self.grid_lr_init*self._spatial_lr_scale,
                                                    lr_final=self.grid_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=self.deformation_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)    
        
    def _update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr

    def _replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        """Remove points from optimizer that are not in the mask
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_points(self, mask):
        """Remove points from the model that are not in the mask
        """
        valid_points_mask = ~mask
        # Prune the optimizer
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        # Call 3D Gaussians's pruning method
        self.gaussians.prune_points(valid_points_mask, optimizable_tensors)
        # Prune the deformation table
        self.model._deformation_accum = self.model._deformation_accum[valid_points_mask]
        self.model._deformation_table = self.model._deformation_table[valid_points_mask]
        self.model.denom = self.model.denom[valid_points_mask]
        

    def _cat_tensors_to_optimizer(self, tensors_dict):
        """ Add new points to the optimizer
        Params:
            tensors_dict: {xyz, f_dc, f_rest, opacity, scaling, rotation}
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        # Get the mask
        n_init_points = self.gaussians.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.gaussians.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        
        # Calculate new attributes
        stds = self.gaussians.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.gaussians._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.gaussians.scaling_inverse_activation(self.gaussians.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.gaussians._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.gaussians._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.gaussians._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.gaussians._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self.model._deformation_table[selected_pts_mask].repeat(N)

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }

        # Postfix
        self._densification_postfix(d, new_deformation_table)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self._prune_points(prune_filter)
        

    def _densification_postfix(self, d, new_deformation_table):
        """ Update the optimizer, 3D Gaussians, and deformation table
        Params:
            d: {xyz, f_dc, f_rest, opacity, scaling, rotation}
            new_deformation_table: the new deformation table to be concatenated     
        """
        # Add points to the optimizer
        optimizable_tensors = self._cat_tensors_to_optimizer(d)
        # Update the 3D Gaussians
        self.gaussians.update_points(optimizable_tensors)
        # Update the deformation table
        self.model._deformation_table = torch.cat([self.model._deformation_table, new_deformation_table], -1)
        # Reset the gaussians statistics
        self.gaussians._xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians._max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
        # Reset the deformation statistics
        self.model._deformation_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 3), device="cuda")
        self.model._denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")

    def _densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Get the mask
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.gaussians.get_scaling, dim=1).values <= self.percent_dense*scene_extent)      
        # Calculate new attributes
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }

        new_deformation_table = self.model._deformation_table[selected_pts_mask]

        # Postfix
        self._densification_postfix(d, new_deformation_table)


    def _get_inputs_targets(self, batch):
        return batch['image'], batch['label']

    def _compute_losses(self, output, target):
        losses = [loss(output, target) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metric_fns, metrics):
            log[metric.__class__.__name__] += _metric.item() * batch_size

    def save(self, save_dir, iteration, stage):
        if stage == "coarse":
            self.gaussians.save_ply(os.path.join(save_dir, "point_cloud", f"coarse_iteration_{iteration}", "point_cloud.ply"))
            self.model.save_deformation(os.path.join(save_dir, "point_cloud", f"coarse_iteration_{iteration}"))
        else:
            self.gaussians.save_ply(os.path.join(save_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
            self.model.save_deformation(os.path.join(save_dir, "point_cloud", f"iteration_{iteration}"))

    def load(self, pcl_path):
        self.gaussians.load_ply()