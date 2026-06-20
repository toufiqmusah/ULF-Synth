import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OptimalKSpaceLoss(nn.Module):
    """
    Optimal k-Space Loss for Maximum Image Quality & Edge Preservation
    
    Designed specifically for ULF→HF brain MRI enhancement.
    Combines multi-scale frequency supervision with edge-preserving properties.
    """
    
    def __init__(
        self,
        # Frequency band weights [low, mid, high]
        band_weights=[1.5, 1.0, 2.0],  # Emphasize high-freq for edges!
        
        # Loss configurations
        magnitude_loss='l1',  # 'l1' or 'l2'
        use_log_magnitude=True,  # Emphasizes high frequencies
        
        # Edge preservation
        use_structural_similarity=True,  # SSIM-like in k-space
        structural_weight=0.3,
        
        # For undersampled data
        data_consistency=False,
        
        # Reduction
        reduction='mean'
    ):
        """
        Parameters optimized for brain MRI edge/fold preservation:
        
        band_weights : [low, mid, high]
            [1.5, 1.0, 2.0] emphasizes high frequencies (edges, folds)
            Low freq = contrast, High freq = edges/details
            
        use_log_magnitude : bool
            True helps recover weak high-frequency components
            
        use_structural_similarity : bool
            Adds SSIM-like structural term in k-space
            Helps preserve coherent brain structures (gyri, sulci)
        """
        super().__init__()
        
        self.band_weights = band_weights
        self.magnitude_loss = magnitude_loss
        self.use_log_magnitude = use_log_magnitude
        self.use_structural_similarity = use_structural_similarity
        self.structural_weight = structural_weight
        self.data_consistency = data_consistency
        self.reduction = reduction
        
        print("="*70)
        print("Optimal k-Space Loss Configuration:")
        print(f"  Band weights [L,M,H]: {band_weights}")
        print(f"  Log magnitude: {use_log_magnitude}")
        print(f"  Structural similarity: {use_structural_similarity}")
        print(f"  → Optimized for edge/fold preservation in brain MRI")
        print("="*70)
    
    def forward(self, pred, target, undersampling_mask=None):
        """
        Compute k-space loss
        
        Args:
            pred: [B, C, D, H, W] predicted high-field image
            target: [B, C, D, H, W] ground truth high-field image
            undersampling_mask: [B, 1, D, H, W] optional mask for data consistency
            
        Returns:
            loss: scalar
            loss_dict: dictionary of loss components (for logging)
        """
        # Transform to k-space (with proper centering)
        k_pred = self._to_kspace(pred)
        k_target = self._to_kspace(target)
        
        # Multi-scale frequency-weighted loss
        loss_freq, freq_losses = self._multiscale_frequency_loss(
            k_pred, k_target, undersampling_mask
        )
        
        total_loss = loss_freq
        loss_dict = {
            'kspace_total': loss_freq.item(),
            'kspace_low': freq_losses['low'],
            'kspace_mid': freq_losses['mid'],
            'kspace_high': freq_losses['high']
        }
        
        # Optional: Structural similarity in k-space
        if self.use_structural_similarity:
            loss_struct = self._structural_loss(k_pred, k_target)
            total_loss = total_loss + self.structural_weight * loss_struct
            loss_dict['kspace_structural'] = loss_struct.item()
        
        return total_loss, loss_dict
    
    def _to_kspace(self, image):
        """Convert image to centered k-space"""
        # cuFFT requires float32 for non-power-of-2 dimensions
        # Convert to float32 if needed
        original_dtype = image.dtype
        if image.dtype == torch.float16:
            image = image.float()
        
        # 3D FFT
        k = torch.fft.fftn(image, dim=(-3, -2, -1))
        # Center DC component
        k = torch.fft.fftshift(k, dim=(-3, -2, -1))
        return k
    
    def _from_kspace(self, kspace):
        """Convert k-space back to image"""
        k = torch.fft.ifftshift(kspace, dim=(-3, -2, -1))
        image = torch.fft.ifftn(k, dim=(-3, -2, -1))
        return torch.abs(image)
    
    def _create_frequency_masks(self, shape, device):
        """
        Create masks for low/mid/high frequency bands
        
        Based on radial distance from k-space center:
        - Low: 0-33% (contrast, tissue intensities)
        - Mid: 33-66% (medium structures)  
        - High: 66-100% (edges, sulci, gyri, fine detail)
        """
        B, C, D, H, W = shape
        
        # Create normalized coordinates centered at DC
        d_coords = (torch.arange(D, device=device) - D // 2).float() / (D / 2)
        h_coords = (torch.arange(H, device=device) - H // 2).float() / (H / 2)
        w_coords = (torch.arange(W, device=device) - W // 2).float() / (W / 2)
        
        dd, hh, ww = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # Radial distance (0 at center, ~1.73 at corners for cube)
        radius = torch.sqrt(dd**2 + hh**2 + ww**2)
        max_radius = radius.max()
        
        # Define band boundaries
        low_thresh = 0.33 * max_radius
        mid_thresh = 0.66 * max_radius
        
        # Create masks
        mask_low = (radius <= low_thresh)
        mask_mid = (radius > low_thresh) & (radius <= mid_thresh)
        mask_high = (radius > mid_thresh)
        
        # Expand to batch/channel dimensions
        mask_low = mask_low.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
        mask_mid = mask_mid.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
        
        return mask_low, mask_mid, mask_high
    
    def _compute_magnitude_loss(self, k_pred, k_target, mask=None):
        """
        Compute loss on k-space magnitude
        
        Supports log-magnitude for better high-frequency emphasis
        """
        mag_pred = torch.abs(k_pred)
        mag_target = torch.abs(k_target)
        
        if self.use_log_magnitude:
            # Log emphasizes small values (high frequencies are typically small)
            mag_pred = torch.log(1 + mag_pred)
            mag_target = torch.log(1 + mag_target)
        
        # Apply frequency mask if provided
        if mask is not None:
            mag_pred = mag_pred * mask.float()
            mag_target = mag_target * mask.float()
        
        # Compute loss
        if self.magnitude_loss == 'l1':
            loss = F.l1_loss(mag_pred, mag_target, reduction='none')
        elif self.magnitude_loss == 'l2':
            loss = F.mse_loss(mag_pred, mag_target, reduction='none')
        else:
            raise ValueError(f"Unknown magnitude_loss: {self.magnitude_loss}")
        
        # Reduce
        if mask is not None:
            # Average only over masked region
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss
    
    def _multiscale_frequency_loss(self, k_pred, k_target, undersampling_mask=None):
        """
        Multi-scale frequency-weighted loss
        
        Computes separate losses for low/mid/high frequency bands
        with different weights to emphasize edge preservation
        """
        # Create frequency masks
        mask_low, mask_mid, mask_high = self._create_frequency_masks(
            k_pred.shape, k_pred.device
        )
        
        # Data consistency: only penalize unsampled k-space
        if self.data_consistency and undersampling_mask is not None:
            dc_mask = ~undersampling_mask.bool()
            mask_low = mask_low & dc_mask
            mask_mid = mask_mid & dc_mask
            mask_high = mask_high & dc_mask
        
        # Compute loss for each band
        loss_low = self._compute_magnitude_loss(k_pred, k_target, mask_low)
        loss_mid = self._compute_magnitude_loss(k_pred, k_target, mask_mid)
        loss_high = self._compute_magnitude_loss(k_pred, k_target, mask_high)
        
        # Weighted combination
        total_loss = (
            self.band_weights[0] * loss_low +
            self.band_weights[1] * loss_mid +
            self.band_weights[2] * loss_high
        ) / sum(self.band_weights)
        
        freq_losses = {
            'low': loss_low.item(),
            'mid': loss_mid.item(),
            'high': loss_high.item()
        }
        
        return total_loss, freq_losses
    
    def _structural_loss(self, k_pred, k_target, window_size=11):
        """
        Structural similarity in k-space (SSIM-like)
        
        Preserves coherent structures (gyri, sulci) by comparing
        local patterns in k-space rather than just point-wise magnitudes
        """
        mag_pred = torch.abs(k_pred)
        mag_target = torch.abs(k_target)
        
        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Local means (using 3D average pooling)
        mu_pred = F.avg_pool3d(mag_pred, kernel_size=window_size, stride=1, 
                               padding=window_size//2)
        mu_target = F.avg_pool3d(mag_target, kernel_size=window_size, stride=1,
                                 padding=window_size//2)
        
        # Local variances and covariance
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool3d(mag_pred ** 2, kernel_size=window_size, 
                                     stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool3d(mag_target ** 2, kernel_size=window_size,
                                       stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool3d(mag_pred * mag_target, kernel_size=window_size,
                                         stride=1, padding=window_size//2) - mu_pred_target
        
        # SSIM formula
        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)
        
        # Loss is 1 - SSIM (we want to minimize dissimilarity)
        loss = 1 - ssim_map.mean()
        
        return loss


class CombinedImageKSpaceLoss(nn.Module):
    """
    Combined loss for optimal image quality:
    Image-space + k-Space + Gradient (edge preservation)
    
    This is the RECOMMENDED loss for ULF-BrainGen
    """
    
    def __init__(
        self,
        # Loss weights (tune these)
        image_weight=1.0,
        kspace_weight=0.15,
        gradient_weight=0.5,
        
        # k-Space configuration (optimized for edges)
        kspace_config=None,
        
        # Image loss type
        image_loss='l1'
    ):
        """
        Recommended configuration for brain MRI:
        
        image_weight=1.0    : Baseline pixel fidelity
        kspace_weight=0.15  : Frequency domain consistency (tune 0.1-0.3)
        gradient_weight=0.5 : Edge preservation (tune 0.3-0.8)
        """
        super().__init__()
        
        self.image_weight = image_weight
        self.kspace_weight = kspace_weight
        self.gradient_weight = gradient_weight
        self.image_loss = image_loss
        
        # k-Space loss with optimal configuration
        default_kspace_config = {
            'band_weights': [1.5, 1.0, 2.0],  # Emphasize high freq (edges)
            'use_log_magnitude': True,
            'use_structural_similarity': True,
            'structural_weight': 0.3
        }
        
        if kspace_config is not None:
            default_kspace_config.update(kspace_config)
        
        self.kspace_loss = OptimalKSpaceLoss(**default_kspace_config)
        
        print("\n" + "="*70)
        print("Combined Loss Configuration:")
        print(f"  Image:    {image_weight:.2f}x {image_loss}")
        print(f"  k-Space:  {kspace_weight:.2f}x (multi-scale + structural)")
        print(f"  Gradient: {gradient_weight:.2f}x (edge preservation)")
        print("="*70 + "\n")
    
    def _gradient_loss(self, pred, target):
        """
        Gradient magnitude loss for edge preservation
        
        Computes gradients in all 3 directions and compares magnitudes
        This is CRITICAL for preserving brain folds, sulci, gyri
        """
        # Compute gradients in each direction
        grad_pred_d = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_pred_h = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_pred_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        grad_target_d = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        grad_target_h = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        grad_target_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        # Gradient magnitude (L2 norm of gradient vector)
        mag_pred = torch.sqrt(
            F.pad(grad_pred_d**2, (0,0,0,0,0,1)) +
            F.pad(grad_pred_h**2, (0,0,0,1,0,0)) +
            F.pad(grad_pred_w**2, (0,1,0,0,0,0)) +
            1e-8
        )
        
        mag_target = torch.sqrt(
            F.pad(grad_target_d**2, (0,0,0,0,0,1)) +
            F.pad(grad_target_h**2, (0,0,0,1,0,0)) +
            F.pad(grad_target_w**2, (0,1,0,0,0,0)) +
            1e-8
        )
        
        # L1 loss on gradient magnitudes
        loss = F.l1_loss(mag_pred, mag_target)
        
        return loss
    
    def forward(self, pred, target, undersampling_mask=None):
        """
        Compute combined loss
        
        Returns:
            total_loss: scalar
            losses_dict: individual components for logging
        """
        # 1. Image-space loss (pixel fidelity)
        if self.image_loss == 'l1':
            loss_image = F.l1_loss(pred, target)
        elif self.image_loss == 'l2':
            loss_image = F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown image_loss: {self.image_loss}")
        
        # 2. k-Space loss (frequency domain consistency)
        loss_kspace, kspace_dict = self.kspace_loss(pred, target, undersampling_mask)
        
        # 3. Gradient loss (edge preservation)
        loss_gradient = self._gradient_loss(pred, target)
        
        # Combined
        total_loss = (
            self.image_weight * loss_image +
            self.kspace_weight * loss_kspace +
            self.gradient_weight * loss_gradient
        )
        
        # Detailed logging dictionary
        losses_dict = {
            'total': total_loss.item(),
            'image': loss_image.item(),
            'kspace': loss_kspace.item(),
            'gradient': loss_gradient.item(),
            **{f'kspace_{k}': v for k, v in kspace_dict.items()}
        }
        
        return total_loss, losses_dict