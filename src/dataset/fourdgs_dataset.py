from torch.utils.data import Dataset
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
import torch
from utils.graphics_utils import focal2fov

class FourDGSdataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        try:
            image, w2c, time = self.dataset[index]
            R,T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            mask=None
        except:
            caminfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time

            mask = caminfo.mask
        return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                            image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                            mask=mask)
  
    def __len__(self):
        return len(self.dataset)
