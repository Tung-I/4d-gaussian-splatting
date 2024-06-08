import os
import torch
from torch import nn
import numpy as np
from src.model.deformation import DeformationFields
from src.utils.plane_utils import compute_plane_smoothness
from src.utils.sh_utils import RGB2SH

class DeformationWrapper(nn.Module):
    def __init__(self, **kwargs):
        super(DeformationWrapper, self).__init__()  
        self.net_width = kwargs['net_width']
        self.timebase_pe = kwargs['timebase_pe']
        self.defor_depth = kwargs['defor_depth']
        self.posebase_pe = kwargs['posebase_pe']
        self.scale_rotation_pe = kwargs['scale_rotation_pe']
        self.opacity_pe = kwargs['opacity_pe']
        self.timenet_width = kwargs['timenet_width']
        self.timenet_output = kwargs['timenet_output']
        self.grid_pe = kwargs['grid_pe']

        self.deformation_fields = DeformationFields(kwargs).to("cuda") 
        self.deformation_accum = None
        self.deformation_table = None

    def training_setup(self, n_points):
        self.deformation_fields.to("cuda")
        self.deformation_accum = torch.zeros((n_points, 3), device="cuda")
        self.deformation_table = torch.gt(torch.ones((n_points), device="cuda"), 0)

    def load_model(self, model_dir, n_points):
        """ Load deformation, deformation_table, deformation_accum
        """
        weight_dict = torch.load(os.path.join(model_dir, "deformation.pth"), map_location="cuda")
        self.deformation_fields.load_state_dict(weight_dict)
        self.deformation_table = torch.gt(torch.ones((n_points), device="cuda"),0)
        self.deformation_accum = torch.zeros((n_points, 3), device="cuda")
        if os.path.exists(os.path.join(model_dir, "deformation_table.pth")):
            self.deformation_table = torch.load(os.path.join(model_dir, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(model_dir, "deformation_accum.pth")):
            self.deformation_accum = torch.load(os.path.join(model_dir, "deformation_accum.pth"), map_location="cuda")
        print("loading deformation fields from {}".format(model_dir))

    def save_deformation(self, path):
        torch.save(self.deformation_fields.state_dict(), os.path.join(path, "deformation.pth"))
        torch.save(self.deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self.deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def update_deformation_table(self, threshold):
        self.deformation_table = torch.gt(self.deformation_accum.max(dim=-1).values/100, threshold)

    def plane_regulation(self):
        multi_res_grids = self.deformation_fields.deformation_net.grid.grids
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
        multi_res_grids = self.deformation_fields.deformation_net.grid.grids
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
        multi_res_grids = self.deformation_fields.deformation_net.grid.grids
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
    