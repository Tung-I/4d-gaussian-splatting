import torch
import os
from torch import nn
import numpy as np
from deformation import deform_network
from utils.plane_utils import compute_plane_smoothness
from utils.sh_utils import RGB2SH
from model.gaussian3d import GaussianModel

class Gaussian4D(nn.Module):
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

        self.gaussians = GaussianModel(self.sh_degree, self.percent_dense).to("cuda") 
        self.deformation = deform_network(
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
        self.deformation_accum = None
        self.deformation_table = None
        self.denom = None

    def training_setup(self):
        self.gaussians.training_setup()
        # Note: below should be called after create_from_pcd
        self.deformation_accum = torch.zeros((self.gaussians.get_xyz.shape[0],3),device="cuda")
        self.denom = torch.zeros((self.gaussians.get_xyz.shape[0], 1), device="cuda")
        self.deformation_table = torch.gt(torch.ones((self.gaussians.get_xyz.shape[0]),device="cuda"),0)


    def load_model(self, path):
        """ Load deformation, deformation_table, deformation_accum
        """
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self.deformation.load_state_dict(weight_dict).to("cuda")
        self.deformation_table = torch.gt(torch.ones((self.gaussians.get_xyz.shape[0]),device="cuda"),0)
        self.deformation_accum = torch.zeros((self.gaussians.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self.deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self.deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")

    def save_deformation(self, path):
        torch.save(self.deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self.deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self.deformation_accum,os.path.join(path, "deformation_accum.pth"))

    def update_deformation_table(self, threshold):
        self.deformation_table = torch.gt(self.deformation_accum.max(dim=-1).values/100, threshold)

    def plane_regulation(self):
        multi_res_grids = self.deformation.deformation_net.grid.grids
        # 6 x [1, rank * F_dim, reso, reso]
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def time_regulation(self):
        multi_res_grids = self.deformation.deformation_net.grid.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def l1_regulation(self):
        multi_res_grids = self.deformation.deformation_net.grid.grids
        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight*self.plane_regulation() + \
            time_smoothness_weight*self.time_regulation() + \
            l1_time_planes_weight*self.l1_regulation()

