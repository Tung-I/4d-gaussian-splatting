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
from src.scene import Scene


def main(args):
    # Load config file
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)
    logging.info(f'Save the config to "{config.main.saved_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Make the experiment results deterministic.
    seed = config.main.random_seed
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check cuda availability
    logging.info('Create the device.')
    if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
    device = torch.device(config.trainer.kwargs.device)

    # Scene construction
    logging.info('Initialize 3DGS and deformation fields.')
    gaussians = _get_instance(src.model, config.gaussians)
    deforms = _get_instance(src.model, config.net)
    scene = _get_instance(src.scene, config.dataset, gaussians, deforms)
    

    logging.info('Create the loss functions and the corresponding weights.')
    loss_fns, loss_weights = [], []
    defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
    for config_loss in config.losses:
        if config_loss.name in defaulted_loss_fns:
            loss_fn = _get_instance(torch.nn, config_loss)
        else:
            loss_fn = _get_instance(src.model.losses, config_loss)
        loss_fns.append(loss_fn)
        loss_weights.append(config_loss.weight)

    logging.info('Create the metric functions.')
    metric_fns = [_get_instance(src.model.metrics, config_metric) for config_metric in config.metrics]

    logging.info('Create the optimizer.')
    optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())

    logging.info('Create the learning rate scheduler.')
    lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer) if config.get('lr_scheduler') else None

    logging.info('Create the logger.')
    config.logger.kwargs.update(log_dir=saved_dir / 'log', net=net, dummy_input=torch.randn(tuple(config.logger.kwargs.dummy_input)))
    logger = _get_instance(src.callbacks.loggers, config.logger)

    logging.info('Create the monitor.')
    config.monitor.kwargs.update(checkpoints_dir=saved_dir / 'checkpoints')
    monitor = _get_instance(src.callbacks.monitor, config.monitor)

    logging.info('Create the trainer.')
    kwargs = {'device': device,
                'train_dataloader': train_dataloader,
                'valid_dataloader': valid_dataloader,
                'net': net,
                'loss_fns': loss_fns,
                'loss_weights': loss_weights,
                'metric_fns': metric_fns,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'logger': logger,
                'monitor': monitor}
    config.trainer.kwargs.update(kwargs)
    trainer = _get_instance(src.runner.trainers, config.trainer)

    loaded_path = config.main.get('loaded_path')
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
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true', help='Perform the training if specified; otherwise perform the testing.')
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