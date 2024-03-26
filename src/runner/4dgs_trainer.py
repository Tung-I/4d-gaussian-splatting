import torch
import numpy as np
from base_trainer import BaseTrainer


class Gaussian4DTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, inputs):
        raise NotImplementedError

    def _training_setup(self):
        """
        Args: 
            percent_dense
            position_lr_init
            position_lr_final
            position_lr_delay_mult
            position_lr_max_steps
            deformation_lr_init
            deformation_lr_final
            deformation_lr_delay_mult
            grid_lr_init
            grid_lr_final
            feature_lr
            opacity_lr
            scaling_lr
            rotation_lr

        """
        self._percent_dense = percent_dense
        self._xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        

        l = [
            {'params': [self._xyz], 'lr': position_lr_init * self._spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': deformation_lr_init * self._spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': grid_lr_init * self._spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=position_lr_init*self._spatial_lr_scale,
                                                    lr_final=position_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=position_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=deformation_lr_init*self._spatial_lr_scale,
                                                    lr_final=deformation_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=deformation_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=grid_lr_init*self._spatial_lr_scale,
                                                    lr_final=grid_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=deformation_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)    


    def _get_inputs_targets(self, batch):
        return batch['image'], batch['label']

    def _compute_losses(self, output, target):
        losses = [loss(output, target) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metric_fns, metrics):
            log[metric.__class__.__name__] += _metric.item() * batch_size