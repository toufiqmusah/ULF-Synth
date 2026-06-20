from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae import nnUNetTrainerMRCT_mae
import torch
import numpy as np
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerMRCT_mae_fixed(nnUNetTrainerMRCT_mae):
    """
    Fixed version of nnUNetTrainerMRCT_mae that handles NaN losses
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
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Reduce initial learning rate
        self.initial_lr = 1e-4  # Much lower than default
        
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Check for NaN/Inf in input data
        if torch.isnan(data).any() or torch.isinf(data).any():
            self.print_to_log_file("WARNING: NaN/Inf detected in input data! Skipping batch.")
            return {'loss': np.array(0.0)}
        
        # Use FP32 instead of FP16 to avoid overflow
        with autocast(self.device.type, enabled=False):  # Disabled autocast
            output = self.network(data)
            l = self.loss(output, target)
            
            # Check for NaN/Inf in loss
            if torch.isnan(l) or torch.isinf(l):
                self.print_to_log_file(f"WARNING: NaN/Inf loss detected!")
                self.print_to_log_file(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
                self.print_to_log_file(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
                self.print_to_log_file(f"Loss value: {l}")
                return {'loss': np.array(0.0)}

        # No grad scaler - use normal backprop
        l.backward()
        
        # Stricter gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)  # Changed from 12 to 1.0
        
        # Check gradients
        total_norm = 0
        for p in self.network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 10.0:
            self.print_to_log_file(f"WARNING: Large gradient norm: {total_norm:.2f}")
        
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Use FP32 for validation too
        with autocast(self.device.type, enabled=False):
            output = self.network(data)
            del data
            l = self.loss(output, target)
            
            # Check for NaN
            if torch.isnan(l) or torch.isinf(l):
                self.print_to_log_file("WARNING: NaN/Inf in validation loss!")
                return {'loss': np.array(0.0), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}