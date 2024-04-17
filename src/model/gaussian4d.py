import torch
import os
from torch import nn
import numpy as np
from simple_knn._C import distCUDA2
from deformation import deform_network
from src.utils.general_utils import strip_symmetric, build_scaling_rotation, inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud
from utils.plane_utils import compute_plane_smoothness
from utils.sh_utils import RGB2SH
from gaussian_model import GaussianModel

class GS4DModel:
    def __init__(self, **kwargs):

        self.sh_degree  = kwargs['sh_degree']
        self.net_width = kwargs['net_width']
        self.timebase_pe = kwargs['timebase_pe']
        self.defor_depth = kwargs['defor_depth']
        self.posbase_pe = kwargs['posbase_pe']
        self.scale_rotation_pe = kwargs['scale_rotation_pe']
        self.opacity_pe = kwargs['opacity_pe']
        self.timenet_width = kwargs['timenet_width']
        self.timenet_output = kwargs['timenet_output']
        self.grid_pe = kwargs['grid_pe']
        self.percent_dense = kwargs['percent_dense']


        self.gaussians = GaussianModel(self.sh_degree, self.percent_dense).to("cuda") 
        self._deformation = deform_network(
            self.net_width, 
            self.timebase_pe, 
            self.defor_depth, 
            self.posbase_pe, 
            self.scale_rotation_pe, 
            self.opacity_pe, 
            self.timenet_width, 
            self.timenet_output, 
            self.grid_pe
        ).to("cuda") 
        self._denom = torch.zeros((self.gs.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.gs.get_xyz.shape[0],3),device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.gs.get_xyz.shape[0]),device="cuda"),0)


    def load_model(self, path):
        """ Load deformation, deformation_table, deformation_accum
        """
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict).to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))

    def prune_points(self, valid_points_mask, optimizable_tensors):
        # Call 3D Gaussians's pruning method
        self.gaussians.prune_points(valid_points_mask, optimizable_tensors)

        # Prune the deformation table
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

    def update_gaussians(self, optimizable_tensors):
        self.gaussians.update_gaussians(optimizable_tensors)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N):
        # Get the selected points based on the gradient threshold
        selected_pts_mask, d = self.gaussians.get_selected_points(grads, grad_threshold, scene_extent, N)
        # Create a new deformation table
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        # Append the new deformation table to the existing one
        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        # Reset the deformation accumulation and denominator
        self._deformation_accum = torch.zeros((self.gaussians.get_xyz.shape[0], 3), device="cuda")
        self._denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        return d, selected_pts_mask, new_deformation_table

       
        
