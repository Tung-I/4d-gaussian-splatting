import torch
from torch import nn
import numpy as np
import os
import logging
from src.utils.general_utils import get_expon_lr_func
from src.utils.graphics_utils import build_rotation
from tqdm import tqdm
from torch.utils.data import DataLoader
from gaussian_renderer import render
from src.model.losses import l1_loss, ssim
from src.model.metrics import psnr


class Gaussian4DTrainer:
    def __init__(self, device, num_iter, lazy_level, test_iterations, save_iterations, 
                 coarse_iteration, batch_size, shuffle, num_workers, white_background,
                 convert_SHs_python, compute_cov3D_python,
                 loss_fns, loss_weights, metric_fns, logger, monitor,
                 optimizer_kwargs, scene):
        self.device = device
        self.lazy_level = lazy_level
        self.test_iterations = test_iterations
        self.save_iterations = save_iterations
        self.coarse_iteration = coarse_iteration
        self.loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]
        self.logger = logger
        self.monitor = monitor
        self.num_iter = num_iter
        self.iter = 1

        self.position_lr_init = optimizer_kwargs['position_lr_init']
        self.position_lr_final = optimizer_kwargs['position_lr_final']
        self.position_lr_delay_mult = optimizer_kwargs['position_lr_delay_mult']
        self.position_lr_max_steps = optimizer_kwargs['position_lr_max_steps']
        self.deformation_lr_init = optimizer_kwargs['deformation_lr_init']
        self.deformation_lr_final = optimizer_kwargs['deformation_lr_final']
        self.deformation_lr_delay_mult = optimizer_kwargs['deformation_lr_delay_mult']
        self.grid_lr_init = optimizer_kwargs['grid_lr_init']
        self.grid_lr_final = optimizer_kwargs['grid_lr_final']
        self.grid_lr_delay_mult = optimizer_kwargs['grid_lr_delay_mult']
        self.feature_lr = optimizer_kwargs['feature_lr']
        self.opacity_lr = optimizer_kwargs['opacity_lr']
        self.scaling_lr = optimizer_kwargs['scaling_lr']
        self.rotation_lr = optimizer_kwargs['rotation_lr']
        
        self.optimizer = None

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        self.scene = scene
        self.gaussians = scene.gaussians
        self.deforms = scene.deforms
        self.spatial_lr_scale = self.gaussians.get_spatial_lr_scale()

        # Prepare the dataloader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        viewpoint_stack = self.scene.getTrainCameras()
        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=self.batch_size, 
                                            shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=list)
        self.random_loader = True
        self.dataloader = iter(viewpoint_stack_loader)
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        

    def train(self):
        self._training_setup() 
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        ema_loss_for_log = 0.0
        ema_psnr_for_log = 0.0

        rogress_bar = tqdm(range(0, self.num_iter), desc="Training progress")


        while self.iter <= self.num_iter:
            print()
            iter_start.record()
            logging.info(f'Iteration {self.iter}.')
            stage = "coarse" if self.iter <= self.coarse_iteration else "fine"
            train_outputs = self._run('training', self.iter, stage=stage)

            radii_list = train_outputs["radii_list"]
            visibility_filter_list = train_outputs["visibility_filter_list"]
            images = train_outputs["rendering"]
            gt_images = train_outputs["gt_images"]

            # Calculate loss and update model
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            image_tensor = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)

            self._compute_losses(image_tensor, gt_image_tensor)
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

            #######################################


    def _run(self, mode, iteration, stage=None):
        self._update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0: 
            self.gaussians.oneupSHdegree()

        # Load the next batch of viewpoints
        try:
            viewpoint_cams = next(self.dataloader)
        except StopIteration:
            print("reset dataloader into random dataloader.")
            if not self.random_loader:
                viewpoint_stack = self.scene.getTrainCameras()
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=self.batch_size, 
                                            shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=list)
            self.dataloader = iter(viewpoint_stack_loader)
        images, gt_images, radii_list, visibility_filter_list, viewspace_point_tensor_list = [], [], [], [], []

        # Batch processing of training viewpoints
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, self.scene.gaussians, self.background, stage=stage, cam_type="dynerf", 
                                convert_SHs_python=self.convert_SHs_python, compute_cov3D_python=self.compute_cov3D_python)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        # Prepare the returns
        train_output = {"rendering": images, "gt_images": gt_images, 
                        "radii_list": radii_list, "visibility_filter_list": visibility_filter_list}

        return train_output

        

    def _training_setup(self):
        self.gaussians.training_setup()
        self.deforms.training_setup() 

        l = [
            {'params': [self.gaussians._xyz], 'lr': self.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self.deforms.deformation.get_mlp_parameters()), 'lr': self.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self.deforms.deformation.get_grid_parameters()), 'lr': self.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self.gaussians._features_dc], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [self.gaussians._features_rest], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.gaussians._opacity], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [self.gaussians._scaling], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [self.gaussians._rotation], 'lr': self.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.position_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=self.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.deformation_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=self.grid_lr_init*self._spatial_lr_scale,
                                                    lr_final=self.grid_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=self.deformation_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)    
        
    def _update_learning_rate(self, iteration):
        """
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
  
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr

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
        """Remove 3DGS points that are not in the mask
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.gaussians._xyz = optimizable_tensors["xyz"]
        self.gaussians._features_dc = optimizable_tensors["f_dc"]
        self.gaussians._features_rest = optimizable_tensors["f_rest"]
        self.gaussians._opacity = optimizable_tensors["opacity"]
        self.gaussians._scaling = optimizable_tensors["scaling"]
        self.gaussians._rotation = optimizable_tensors["rotation"]
        
        self.gaussians.xyz_gradient_accum = self.gaussians.xyz_gradient_accum[valid_points_mask]
        self.gaussians.max_radii2D = self.gaussians.max_radii2D[valid_points_mask]

        self.deforms.deformation_accum = self.deforms.deformation_accum[valid_points_mask]
        self.deforms.deformation_table = self.deforms.deformation_table[valid_points_mask]
        self.deforms.denom = self.deforms.denom[valid_points_mask]
        

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
        """ Densify 3DGS based on the gradients
        """
        # Get the mask
        n_init_points = self.gaussians.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.gaussians.get_scaling, dim=1).values > self.gaussians.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        
        # Calculate the new attributes
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
        new_deformation_table = self.deforms.deformation_table[selected_pts_mask].repeat(N)

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }

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
        self.gaussians._xyz = optimizable_tensors["xyz"]
        self.gaussians._features_dc = optimizable_tensors["f_dc"]
        self.gaussians._features_rest = optimizable_tensors["f_rest"]
        self.gaussians._opacity = optimizable_tensors["opacity"]
        self.gaussians._scaling = optimizable_tensors["scaling"]
        self.gaussians._rotation = optimizable_tensors["rotation"]

        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")

        # Update the deformation fields
        self.deforms.deformation_table = torch.cat([self.deforms.deformation_table, new_deformation_table], -1)
        self.deforms.deformation_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 3), device="cuda")
        self.deforms.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")

    def _densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Get the mask
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.gaussians.get_scaling, dim=1).values <= self.gaussians.percent_dense*scene_extent)      
        # Calculate new attributes
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self.deforms.deformation_table[selected_pts_mask]

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }

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