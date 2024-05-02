import os
from src.dataset.dataset_readers import sceneLoadTypeCallbacks
from src.dataset.fourdgs_dataset import FourDGSdataset
from src.model.gaussian_model import GaussianModel
from src.model.deformation_fields import DeformationFields


class Scene:
    def __init__(self, gaussians, deforms, **kwargs):
        self.data_dir = kwargs['data_dir']  # folder tha contains the ply file
        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        # Create the SceneInfo object that includes PCs, NDC dataset, maxtime
        scene_info = sceneLoadTypeCallbacks["dynerf"](kwargs)
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Wrap the NDC dataset into a FourDGSdataset 
        self.train_camera = FourDGSdataset(scene_info.train_cameras)  # scene_info.train_cameras is a NDC dataset
        self.test_camera = FourDGSdataset(scene_info.test_cameras)
        self.video_camera = FourDGSdataset(scene_info.video_cameras)

        # Initialize the 3DGS and deformation fields
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        self.gaussians = gaussians
        self.deforms = deforms
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
        self.deforms.deformation_net.set_aabb(xyz_max, xyz_min)


    def save(self, model_path, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.deforms.save_deformation(point_cloud_path)

    def getTrainCameras(self):
        return self.train_camera

    def getTestCameras(self):
        return self.test_camera
    
    def getVideoCameras(self):
        return self.video_camera