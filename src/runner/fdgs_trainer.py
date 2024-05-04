import torch
from torch import nn
import numpy as np
import sys
import os
import logging
from src.utils.general_utils import get_expon_lr_func
from src.utils.graphics_utils import build_rotation
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.gaussian_renderer import render
from src.model.losses import l1_loss, ssim
from src.model.metrics import psnr
from src.utils.general_utils import Timer
from src.logger.fdgs_logger import training_report


class Gaussian4DTrainer:
    def __init__(self, scene, **kwargs):
        trainer_kwargs = kwargs['trainer_kwargs']  
        renderer_kwargs = kwargs['renderer_kwargs']
        optimizer_kwargs = kwargs['optimizer_kwargs']

        self.scene = scene
        self.gaussians = scene.gaussians
        self.deforms = scene.deforms
        self.spatial_lr_scale = self.gaussians.get_spatial_lr_scale

        self.device = trainer_kwargs['device']
        self.saved_dir = trainer_kwargs['saved_dir']
        self.lazy_level = trainer_kwargs['lazy_level']
        self.test_iterations = trainer_kwargs['test_iterations']
        self.save_iterations = trainer_kwargs['save_iterations']
        self.ckpt_iterations = trainer_kwargs['ckpt_iterations']
        self.coarse_iteration = trainer_kwargs['coarse_iteration']
        self.batch_size = trainer_kwargs['batch_size']
        self.shuffle = trainer_kwargs['shuffle']
        self.num_iter = trainer_kwargs['num_iter']
        self.num_workers = trainer_kwargs['num_workers']
        self.iter = 1

        self.white_background = renderer_kwargs['white_background']
        self.convert_SHs_python = renderer_kwargs['convert_SHs_python']
        self.compute_cov3D_python = renderer_kwargs['compute_cov3D_python']
        bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.position_lr_init = optimizer_kwargs['position_lr_init']
        self.position_lr_final = optimizer_kwargs['position_lr_final']
        self.position_lr_delay_mult = optimizer_kwargs['position_lr_delay_mult']
        self.position_lr_max_steps = optimizer_kwargs['position_lr_max_steps']
        self.deformation_lr_init = optimizer_kwargs['deformation_lr_init']
        self.deformation_lr_final = optimizer_kwargs['deformation_lr_final']
        self.deformation_lr_delay_mult = optimizer_kwargs['deformation_lr_delay_mult']
        self.grid_lr_init = optimizer_kwargs['grid_lr_init']
        self.grid_lr_final = optimizer_kwargs['grid_lr_final']
        self.feature_lr = optimizer_kwargs['feature_lr']
        self.opacity_lr = optimizer_kwargs['opacity_lr']
        self.scaling_lr = optimizer_kwargs['scaling_lr']
        self.rotation_lr = optimizer_kwargs['rotation_lr']
        self.time_smoothness_weight = optimizer_kwargs['time_smoothness_weight']
        self.plane_tv_weight = optimizer_kwargs['plane_tv_weight']
        self.l1_time_planes = optimizer_kwargs['l1_time_planes']
        self.lambda_dssim = optimizer_kwargs['lambda_dssim']
        self.densify_from_iter = optimizer_kwargs['densify_from_iter']
        self.densify_until_iter = optimizer_kwargs['densify_until_iter']
        self.densify_grad_threshold_coarse = optimizer_kwargs['densify_grad_threshold_coarse']
        self.densify_grad_threshold_fine_init = optimizer_kwargs['densify_grad_threshold_fine_init']
        self.densify_grad_threshold_after = optimizer_kwargs['densify_grad_threshold_after']
        self.prunin_from_iter = optimizer_kwargs['pruning_from_iter']
        self.pruning_interval = optimizer_kwargs['pruning_interval']
        self.opacity_threshold_coarse = optimizer_kwargs['opacity_threshold_coarse']
        self.opacity_threshold_fine_init = optimizer_kwargs['opacity_threshold_fine_init']
        self.opacity_threshold_fine_after = optimizer_kwargs['opacity_threshold_fine_after']
        self.spatial_lr_scale = self.gaussians.get_spatial_lr_scale
        
        self.optimizer = None
        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        # Prepare the dataloader
        viewpoint_stack = self.scene.getTrainCameras()
        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=self.batch_size, 
                                            shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=list)
        self.random_loader = True
        self.dataloader = iter(viewpoint_stack_loader)
        
        # Tensorboard writer
        self.tb_writer = SummaryWriter(self.saved_dir)

    def train(self):
        timer = Timer()
        timer.start()
        self._training_setup() 
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        ema_loss_for_log = 0.0
        ema_psnr_for_log = 0.0

        progress_bar = tqdm(range(0, self.num_iter), desc="Training progress")

        
        while self.iter <= self.num_iter:
            print()
            iter_start.record()
            logging.info(f'Iteration {self.iter}.')
            stage = "coarse" if self.iter <= self.coarse_iteration else "fine"

            train_outputs = self._run('training', stage=stage)

            radii_list = train_outputs["radii_list"]
            visibility_filter_list = train_outputs["visibility_filter_list"]
            viewspace_point_tensor_list = train_outputs["viewspace_point_tensor_list"]
            images = train_outputs["rendering"]
            gt_images = train_outputs["gt_images"]

            # Calculate loss and backpropagate
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            image_tensor = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)

            self._compute_losses(image_tensor, gt_image_tensor)
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

            loss = Ll1
            if stage == "fine" and self.time_smoothness_weight != 0:
                tv_loss = self.deformations.compute_regulation(self.time_smoothness_weight, self.l1_time_planes, self.plane_tv_weight)
                loss += tv_loss
            if self.lambda_dssim != 0:
                ssim_loss = ssim(image_tensor,gt_image_tensor)
                loss += self.lambda_dssim * (1.0-ssim_loss)

            loss.backward()
            if torch.isnan(loss).any():
                print("loss is Nan; end training.")
                os.execv(sys.executable, [sys.executable] + sys.argv)

            # Record the gradient of the viewspace points
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            iter_end.record()

            # Update progress bar and log training metrics
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
                total_point = self.gaussians._xyz.shape[0]
                if self.iter % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "psnr": f"{psnr_:.{2}f}",
                                            "point":f"{total_point}"})
                    progress_bar.update(10)
                if self.iter == self.num_iter:
                    progress_bar.close()

            # Perform logging, saving, and checkpointing 
            timer.pause()
            
            render_kwargs = {"background": self.background, "convert_SHs_python": self.convert_SHs_python, "compute_cov3D_python": self.compute_cov3D_python}
            training_report(self.tb_writer, self.iter, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            self.test_iterations, self.scene, render, render_kwargs, stage, 'dynerf')
            if (self.iter in self.save_iterations):
                print("\n[ITER {}] Saving Gaussians to {}".format(self.iter, self.saved_dir))
                self.scene.save(self.saved_dir, self.iter, stage)
            
            timer.start()

            # Densification
            if self.iter < self.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                self.deforms.add_densification_stats(visibility_filter)

                if stage == "coarse":
                    opacity_threshold = self.opacity_threshold_coarse
                    densify_threshold = self.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = self.opacity_threshold_fine_init - self.iter*(self.opacity_threshold_fine_init - self.opacity_threshold_fine_after)/(self.densify_until_iter)  
                    densify_threshold = self.densify_grad_threshold_fine_init - self.iter*(self.densify_grad_threshold_fine_init - self.densify_grad_threshold_after)/(self.densify_until_iter )  
                
                if  self.iter > self.densify_from_iter and self.iter % self.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if self.iter > self.opacity_reset_interval else None
                    self._densify(densify_threshold, self.scene.cameras_extent)

                if  self.iter > self.pruning_from_iter and self.iter % self.pruning_interval == 0 and self.gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if self.iter > self.opacity_reset_interval else None
                    self._prune(opacity_threshold, self.scene.cameras_extent, size_threshold)
                    
                if self.iter % self.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iter < self.num_iter:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)

            # Checkpointing
            if (self.iter in self.ckpt_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(self.iter))
                torch.save((self.capture(), self.iter), self.saved_dir + "/ckpt" +f"_{stage}_" + str(self.iter) + ".pth")


    def _run(self, mode, stage=None):
        self._update_learning_rate(self.iter)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iter % 1000 == 0: 
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
        train_output = {"rendering": images, "gt_images": gt_images, "radii_list": radii_list, 
                        "visibility_filter_list": visibility_filter_list, "viewspace_point_tensor_list": viewspace_point_tensor_list}

        return train_output

        

    def _training_setup(self):
        self.gaussians.training_setup()
        self.deforms.training_setup(self.gaussians.get_xyz.shape[0]) 

        l = [
            {'params': [self.gaussians._xyz], 'lr': self.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self.deforms.deformation_fields.get_mlp_parameters()), 'lr': self.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self.deforms.deformation_fields.get_grid_parameters()), 'lr': self.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
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
        self.grid_scheduler_args = get_expon_lr_func(lr_init=self.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.grid_lr_final*self.spatial_lr_scale,
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
        self.gaussians.denom = self.gaussians.denom[valid_points_mask]

        self.deforms.deformation_accum = self.deforms.deformation_accum[valid_points_mask]
        self.deforms.deformation_table = self.deforms.deformation_table[valid_points_mask]

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

        # Reset gaussians stats
        self.gaussians.xyz_gradient_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
        self.gaussians.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")

        # Update the deformation fields
        self.deforms.deformation_table = torch.cat([self.deforms.deformation_table, new_deformation_table], -1)
        self.deforms.deformation_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 3), device="cuda")

    def _prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.gaussians.max_radii2D > max_screen_size
            big_points_ws = self.gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask)
        torch.cuda.empty_cache()

    def _densify(self, max_grad, extent):
        grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom
        grads[grads.isnan()] = 0.0

        self._densify_and_clone(grads, max_grad, extent)
        self._densify_and_split(grads, max_grad, extent)


    def _init_log(self):
        return
    
    def _update_log(self, log, batch_size, loss, losses, metrics):
        return

    def save(self, save_dir, iteration, stage):
        if stage == "coarse":
            self.gaussians.save_ply(os.path.join(save_dir, "point_cloud", f"coarse_iteration_{iteration}", "point_cloud.ply"))
            self.model.save_deformation(os.path.join(save_dir, "point_cloud", f"coarse_iteration_{iteration}"))
        else:
            self.gaussians.save_ply(os.path.join(save_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
            self.model.save_deformation(os.path.join(save_dir, "point_cloud", f"iteration_{iteration}"))

    def capture(self):
        return (
            self.gaussians.active_sh_degree,
            self.gaussians._xyz,
            self.deforms._deformation.state_dict(),
            self.deforms._deformation_table,
            self.gaussians._features_dc,
            self.gaussians._features_rest,
            self.gaussians._scaling,
            self.gaussians._rotation,
            self.gaussians._opacity,
            self.gaussians.max_radii2D,
            self.gaussians.xyz_gradient_accum,
            self.gaussians.denom,
            self.optimizer.state_dict(),
            self.gaussians.spatial_lr_scale,
        )