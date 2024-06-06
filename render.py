#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
import logging
import os
from tqdm import tqdm
from os import makedirs
from src.gaussian_renderer import render
import torchvision
from time import time
import concurrent.futures
from box import Box
from pathlib import Path
import random
import src
import argparse
from src import scene

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_scene(scene, config, split):
    if split == "train":
        views = scene.getTrainCameras()
    elif split == "test":
        views = scene.getTestCameras()
    else:
        views = scene.getVideoCameras()

    # Create output directories
    model_path = config.trainer.kwargs.trainer_kwargs.saved_dir
    iteration = config.trainer.kwargs.trainer_kwargs.num_iter
    render_path = os.path.join(model_path, split, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, split, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []

    print("point nums:", scene.gaussians._xyz.shape[0])

    bg_color = [1,1,1] if config.trainer.kwargs.renderer_kwargs.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type = config.dataset.kwargs.cam_type

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, scene.gaussians, scene.deforms, background, cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

        if split in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)

    # Use imageio to write render_images to a video.mp4
    video_writer = imageio.get_writer(os.path.join(model_path, split, "ours_{}".format(iteration), 'video_rgb.mp4'), fps=30)
    for image in render_images:
        video_writer.append_data(image)
    video_writer.close()
    

def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.

    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)

def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.trainer.kwargs.trainer_kwargs.saved_dir)
    if not saved_dir.is_dir():
        raise ValueError(f'The saved directory "{saved_dir}" does not exist.')
    
    # Make the experiment results deterministic.
    seed = 6666
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the trained model
    logging.info('Initialize 3DGS and deformation fields.')
    split = args.split
    saved_dir = config.trainer.kwargs.trainer_kwargs.saved_dir
    iteration = config.trainer.kwargs.trainer_kwargs.num_iter
    model_dir = os.path.join(saved_dir, "point_cloud", f"iteration_{iteration}")
    print("Loading model from:", model_dir)
    with torch.no_grad():
        gaussians = _get_instance(src.model, config.gaussians)
        deforms = _get_instance(src.model, config.net)
        deforms.load_model(model_dir, gaussians._xyz.shape[0])
        _scene = _get_instance(scene, config.dataset, gaussians, deforms)
        
        render_scene(_scene, config, split)

def _parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('--config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--split', type=str, default="test", help='The split to render.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)