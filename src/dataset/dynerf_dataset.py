import glob
import torch
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms as T
from typing import NamedTuple
import torchvision.transforms as transforms
from plyfile import PlyData, PlyElement

from src.dataset.base_dataset import BaseDataset
from src.utils.general_utils import get_spiral
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from src.utils.graphics_utils import BasicPointCloud
from src.utils.camera import Camera


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

class DynerfDataset(BaseDataset):
    """
    """
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.split = split  # 'train' or 'test' or 'val'
        self.downsample = kwargs['downsample']
        self.is_stack = kwargs['is_stack']
        self.N_vis = kwargs['N_vis']
        self.time_scale = kwargs['time_scale']
        self.scene_bbox_min = kwargs['scene_bbox_min']
        self.scene_bbox_max = kwargs['scene_bbox_max']
        self.bd_factor = kwargs['bd_factor']
        self.eval_step = kwargs['eval_step']
        self.eval_index = kwargs['eval_index']

        self.img_wh = (
            int(1352 / self.downsample),
            int(1014 / self.downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.downsample = 2704 / self.img_wh[0]
        self.scene_bbox = torch.tensor([self.scene_bbox_min, self.scene_bbox_max])
        self.world_bound_scale = 1.1
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False
        self.segment_length = 300

        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")

        if self.split == "train":
            self.train_cameras = self.get_train_cameras()
            

        elif self.split == "test":
            self.render_cameras = self.get_render_cameras()

        else:
            raise Exception("Invalid split type")

        
    def load_meta(self):
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.data_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        assert len(videos) == poses_arr.shape[0]
        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

        # Get the validation poses.
        N_views = self.segment_length
        countss = N_views
        self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)  # (60, 4, 4)

        # Get the training split
        poses_i_train = []
        for i in range(len(poses)):
            if i != self.eval_index:
                poses_i_train.append(i)
        self.poses = poses[poses_i_train]
        self.poses_all = poses
        self.image_paths, self.image_poses, self.image_times = self.load_images_path(videos, self.split, countss)


    def load_images_path(self, videos, split, countss):
        image_paths = []
        image_poses = []
        image_times = []
        N_cams = 0
        N_time = 0

        for index, video_path in enumerate(videos):
            if index == self.eval_index:
                if split =="train":
                    continue
            else:
                if split == "test":
                    continue
            N_cams +=1
            video_images_path = video_path.split('.')[0]
            image_path = os.path.join(video_images_path, "images")
            images_path = os.listdir(image_path)
            images_path.sort()
            this_count = 0
            for idx, path in enumerate(images_path):
                if this_count >=countss:break
                image_paths.append(os.path.join(image_path, path))
                pose = np.array(self.poses_all[index])
                R = pose[:3,:3]
                R = -R
                R[:,0] = -R[:,0]
                T = -pose[:3,3].dot(R)
                image_times.append(idx/countss)
                image_poses.append((R,T))
                this_count+=1
            N_time = len(images_path)

        return image_paths, image_poses, image_times

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
    
    def get_train_cameras(self):
        cameras = []
        image = Image.open(self.image_paths[0])
        for idx in range(len(self.image_paths)):
            image_path = None
            image_name = f"{idx}"
            time = self.image_times[idx]
            R, T = self.image_poses[idx]
            FovX = focal2fov(self.focal[0], image.shape[1])
            FovY = focal2fov(self.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

        return cameras
    
    def get_render_cameras(self):
        cameras = []
        len_poses = len(self.image_paths)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        for idx, p in enumerate(len_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = focal2fov(self.focal[0], image.shape[2])
            FovY = focal2fov(self.focal[0], image.shape[1])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        img = Image.open(self.image_paths[index])
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # To tensor

        R,T = self.image_poses[index]
        FovX = focal2fov(self.focal[0], img.shape[2])
        FovY = focal2fov(self.focal[0], img.shape[1])
        time = self.image_times[index]
        mask=None

        return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=img, gt_alpha_mask=None,
                                image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                                mask=mask)