import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.utils.graphics_utils import batch_quaternion_multiply
from src.model.hexplane import HexPlaneField
from src.model.grid import DenseGrid


class DeformationNet(nn.Module):
    """Decoder of Gaussian attributes
    self.grid: HexPlaneField, (x, y, z) -> (value)
    """
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[],
                 no_grid=False, bounds=1.6, kplanes_config={}, multires=[], empty_voxel=False, static_mlp=False, device="cuda"):
        super(DeformationNet, self).__init__()
        self.D = D  # Depth of the hidden layers
        self.W = W
        # self.input_ch = input_ch
        # self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = no_grid
        self.grid = HexPlaneField(bounds, kplanes_config, multires).to(device)
        self.bounds = bounds
        self.kplanes_config = kplanes_config
        self.multires = multires
        self.empty_voxel = empty_voxel
        self.static_mlp = static_mlp

        self.no_dx = False
        self.no_ds = False
        self.no_dr = False
        self.no_dir = False
        self.no_do = False
        self.no_dshs = False
        self.apply_rotation = False

        if self.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.static_mlp:
            self.static_mlp_nn = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))

        self.ratio=0
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim

        # Hidden layers
        self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)] 
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        # Attribute-specific MLPs
        self.pos_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W),nn.ReLU(), nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W,self.W), nn.ReLU(), nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W,self.W), nn.ReLU(), nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W,self.W), nn.ReLU(), nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb, time_emb):
        """Query the grid feature and return the hidden representation.
        """
        grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
        
        if self.grid_pe > 1:
            grid_feature = poc_fre(grid_feature,self.grid_pe)

        hidden = torch.cat([grid_feature], -1) 
        hidden = self.feature_out(hidden)   

        return hidden
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, shs_emb=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb):
        """ Apply deformations to the input embeddings
        """
        hidden = self.query_time(rays_pts_emb, time_emb)

        # Determine the mask that filteres out the static points
        if self.static_mlp:  # mask decoded by a static mlp
            mask = self.static_mlp_nn(hidden)
        elif self.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:, :3])
        else:
            mask = torch.ones_like(opacity_emb[:, 0]).unsqueeze(-1)
 
        # Apply deformations
        if self.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx

        if self.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do

        if self.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    

class DeformationFields(nn.Module):
    def __init__(self, kwargs):
        super(DeformationFields, self).__init__()    
        self.net_width = kwargs['net_width']
        self.timebase_pe = kwargs['timebase_pe']
        self.defor_depth = kwargs['defor_depth']
        self.posebase_pe = kwargs['posebase_pe']
        self.scale_rotation_pe = kwargs['scale_rotation_pe']
        self.opacity_pe = kwargs['opacity_pe']
        # self.timenet_width = kwargs['timenet_width']
        # self.timenet_output = kwargs['timenet_output']
        self.grid_pe = kwargs['grid_pe']
        self.no_grid = kwargs['no_grid']
        self.bounds = kwargs['bounds']
        self.kplanes_config = kwargs['kplanes_config']
        self.multires = kwargs['multires']
        self.empty_voxel = kwargs['empty_voxel']
        self.static_mlp = kwargs['static_mlp']

        times_ch = 2*self.timebase_pe + 1

        # self.timenet = nn.Sequential(
        #     nn.Linear(times_ch, self.timenet_width), nn.ReLU(),
        #     nn.Linear(self.timenet_width, self.timenet_output)
        # )

        self.deformation_net = DeformationNet(W=self.net_width, D=self.defor_depth, 
                                           input_ch=(3)+(3*(self.posebase_pe))*2,
                                           grid_pe=self.grid_pe, no_grid=self.no_grid, 
                                           bounds=self.bounds, kplanes_config=self.kplanes_config, 
                                           multires=self.multires, empty_voxel=self.empty_voxel, 
                                           static_mlp=self.static_mlp)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(self.timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(self.posebase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(self.scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(self.opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, time=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, time)
    
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, time=None):
        # Encode with positional encoding (only pos, scale, rot)
        point_emb = poc_fre(point, self.pos_poc) 
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)

        # Decode the attributes
        means3D, scales, rotations, opacity, shs = self.deformation_net(
                                                point_emb, scales_emb, rotations_emb,
                                                opacity, shs, time)
        return means3D, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        # return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
        return self.deformation_net.get_mlp_parameters()
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def poc_fre(input_data, poc_buf):
    """ Encode input data with positional encoding.
    """
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb