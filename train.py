import argparse
import logging
import os
import sys
import torch
import random
import importlib
import yaml
import numpy as np
from box import Box
from pathlib import Path
import src
from src import runner

def main(args):
    # Load config file
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.trainer.kwargs.trainer_kwargs.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)
    logging.info(f'Save the config to "{saved_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Make the experiment results deterministic.
    seed = 6666
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check cuda availability
    logging.info('Create the device.')
    if 'cuda' in config.trainer.kwargs.trainer_kwargs.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
    device = torch.device(config.trainer.kwargs.trainer_kwargs.device)

    # Scene construction
    logging.info('Initialize 3DGS and deformation fields.')
    gaussians = _get_instance(src.model, config.gaussians)
    deforms = _get_instance(src.model, config.net)
    scene = _get_instance(src.scene, config.dataset, gaussians, deforms)
    
    # Create the trainer
    logging.info('Create the trainer.')
    trainer = _get_instance(src.runner, config.trainer, scene)

    # Load the previous checkpoint
    loaded_path = config.trainer.kwargs.trainer_kwargs.get('loaded_path')
    if loaded_path:
        logging.info(f'Load the previous checkpoint from "{loaded_path}".')
        trainer.load(Path(loaded_path))
        logging.info('Resume training.')
    else:
        logging.info('Start training.')
    trainer.train()
    logging.info('End training.')


def _parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('--config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true', help='Perform the testing if specified.')
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    #with ipdb.launch_ipdb_on_exception():
    #    sys.breakpointhook = ipdb.set_trace
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)