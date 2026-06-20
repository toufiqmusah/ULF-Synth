from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from typing import Union, Tuple, List

from batchgenerators.transforms.abstract_transforms import AbstractTransform

import numpy as np
from nnunetv2.training.loss.kspace import OptimalKSpaceLoss, CombinedImageKSpaceLoss

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D_MRCT
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT

from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs

from time import time, sleep
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from torch import autocast


class dummy_context:
    """Dummy context manager for when autocast is not used"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class nnUNetTrainerMRCT_kspace(nnUNetTrainer):
    """
    nnUNet Trainer using k-space loss for brain MRI enhancement.
    
    Combines frequency-domain supervision with optimal frequency weighting
    and edge preservation for ULF→HF brain MRI translation.
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Args:
            plans: nnUNet training plans
            configuration: configuration name
            fold: fold number
            dataset_json: dataset metadata
            unpack_dataset: whether to unpack the dataset
            device: torch device to use
        """
        # Set to False to use OptimalKSpaceLoss only (for comparison)
        self.use_combined_loss = True
        
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_epochs = 1000

    def _build_loss(self):
        """Build the k-space loss function"""
        if self.use_combined_loss:
            # Recommended: Combined image + k-space + gradient loss
            loss = CombinedImageKSpaceLoss(
                image_weight=1.0,
                kspace_weight=0.15,
                gradient_weight=0.5,
                kspace_config={
                    'band_weights': [1.5, 1.0, 2.0],
                    'use_log_magnitude': True,
                    'use_structural_similarity': True,
                    'structural_weight': 0.3
                },
                image_loss='l1'
            )
        else:
            # k-Space loss only (for comparison)
            loss = OptimalKSpaceLoss(
                band_weights=[1.5, 1.0, 2.0],
                magnitude_loss='l1',
                use_log_magnitude=True,
                use_structural_similarity=True,
                structural_weight=0.3,
                data_consistency=False,
                reduction='mean'
            )
        return loss

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 1,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        """No augmentation for k-space loss training to preserve k-space consistency"""
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Disable mirroring for k-space consistency"""
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch: dict) -> dict:
        """Training step with k-space loss"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            
            # k-space loss returns (loss, loss_dict)
            if isinstance(self.loss, CombinedImageKSpaceLoss):
                l, loss_dict = self.loss(output, target)
            else:
                l, loss_dict = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        """Get data loaders with MRCT-specific configuration"""
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        initial_patch_size = self.configuration_manager.patch_size
        dim = dim

        if dim == 2:
            dl_tr = nnUNetDataLoader2D_MRCT(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D_MRCT(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D_MRCT(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D_MRCT(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    def validation_step(self, batch: dict) -> dict:
        """Validation step with k-space loss"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            
            # Compute loss (returns tuple of loss and loss_dict)
            if isinstance(self.loss, CombinedImageKSpaceLoss):
                l, loss_dict = self.loss(output, target)
            else:
                l, loss_dict = self.loss(output, target)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Log validation metrics"""
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['loss'])
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        """Log epoch-end metrics and handle checkpointing"""
        # Log the end time of the epoch
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Logging train and validation loss
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        # Log the duration of the epoch
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        # Checkpoint handling for best and periodic saves
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        best_metric = 'val_losses'
        if self._best_ema is None or self.logger.my_fantastic_logging[best_metric][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging[best_metric][-1]
            self.print_to_log_file(f"Yayy! New best k-space loss: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Increment the epoch counter
        self.current_epoch += 1
