"""
Loss Functions for Signal Prediction
=====================================

Implementation of various loss functions for training signal
prediction models in Tactile Internet applications.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
import numpy as np


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss.
    
    Standard MSE loss used for signal prediction.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2026
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MSE loss.
        
        Parameters:
        -----------
        reduction : str
            Reduction mode ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions
        targets : torch.Tensor
            Ground truth targets
            
        Returns:
        --------
        torch.Tensor
            Loss value
        """
        return self.mse(predictions, targets)


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss (L1 Loss).
    
    More robust to outliers than MSE.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MAE loss.
        
        Parameters:
        -----------
        reduction : str
            Reduction mode ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MAE loss."""
        return self.l1(predictions, targets)


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss).
    
    Combines MSE and MAE - quadratic for small errors,
    linear for large errors. Good balance between
    sensitivity and robustness.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Huber loss.
        
        Parameters:
        -----------
        delta : float
            Threshold for switching between L1 and L2
        reduction : str
            Reduction mode
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.huber = nn.HuberLoss(delta=delta, reduction=reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Huber loss."""
        return self.huber(predictions, targets)


class PSNRLoss(nn.Module):
    """
    Peak Signal-to-Noise Ratio Loss.
    
    PSNR is used as an evaluation metric in the paper.
    This loss maximizes PSNR by minimizing negative PSNR.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, max_value: float = 1.0):
        """
        Initialize PSNR loss.
        
        Parameters:
        -----------
        max_value : float
            Maximum possible value of signal
        """
        super().__init__()
        self.max_value = max_value
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative PSNR (for minimization).
        
        Returns:
        --------
        torch.Tensor
            Negative PSNR value
        """
        mse = F.mse_loss(predictions, targets)
        
        # Avoid log of zero
        mse = torch.clamp(mse, min=1e-10)
        
        psnr = 10 * torch.log10(self.max_value ** 2 / mse)
        
        # Return negative PSNR for minimization
        return -psnr
    
    @staticmethod
    def compute_psnr(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        max_value: float = 1.0
    ) -> float:
        """
        Compute PSNR value.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions
        targets : torch.Tensor
            Ground truth targets
        max_value : float
            Maximum possible value
            
        Returns:
        --------
        float
            PSNR in dB
        """
        with torch.no_grad():
            mse = F.mse_loss(predictions, targets)
            if mse < 1e-10:
                return float('inf')
            psnr = 10 * torch.log10(max_value ** 2 / mse)
            return psnr.item()


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss for signal prediction.
    
    Combines MSE with frequency domain loss for better
    perceptual quality of predicted signals.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        freq_weight: float = 0.1
    ):
        """
        Initialize perceptual loss.
        
        Parameters:
        -----------
        mse_weight : float
            Weight for MSE component
        freq_weight : float
            Weight for frequency domain component
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.freq_weight = freq_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Combines time-domain MSE with frequency-domain loss.
        """
        # Time domain loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Frequency domain loss
        pred_fft = torch.fft.rfft(predictions, dim=-2)
        target_fft = torch.fft.rfft(targets, dim=-2)
        
        freq_loss = F.mse_loss(
            torch.abs(pred_fft),
            torch.abs(target_fft)
        )
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.freq_weight * freq_loss
        
        return total_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss.
    
    Encourages smooth temporal transitions in predictions,
    important for haptic signal continuity.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        prediction_weight: float = 1.0,
        smoothness_weight: float = 0.1
    ):
        """
        Initialize temporal consistency loss.
        
        Parameters:
        -----------
        prediction_weight : float
            Weight for prediction accuracy term
        smoothness_weight : float
            Weight for temporal smoothness term
        """
        super().__init__()
        self.prediction_weight = prediction_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        L = prediction_weight * MSE(pred, target) 
            + smoothness_weight * MSE(diff(pred), diff(target))
        """
        # Prediction accuracy
        pred_loss = F.mse_loss(predictions, targets)
        
        # Temporal smoothness (first-order difference)
        pred_diff = predictions[:, 1:] - predictions[:, :-1]
        target_diff = targets[:, 1:] - targets[:, :-1]
        smooth_loss = F.mse_loss(pred_diff, target_diff)
        
        # Combined loss
        total_loss = (
            self.prediction_weight * pred_loss +
            self.smoothness_weight * smooth_loss
        )
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss with multiple components.
    
    Allows flexible combination of multiple loss functions
    with configurable weights.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> loss_fn = CombinedLoss({
    ...     'mse': (MSELoss(), 1.0),
    ...     'temporal': (TemporalConsistencyLoss(), 0.1)
    ... })
    >>> loss = loss_fn(predictions, targets)
    """
    
    def __init__(
        self,
        losses: Dict[str, tuple],
        reduction: str = 'mean'
    ):
        """
        Initialize combined loss.
        
        Parameters:
        -----------
        losses : Dict[str, tuple]
            Dictionary mapping loss name to (loss_fn, weight) tuple
        reduction : str
            Final reduction mode
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight
        
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Returns:
        --------
        torch.Tensor
            Weighted sum of all loss components
        """
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            weight = self.weights[name]
            component_loss = loss_fn(predictions, targets)
            total_loss = total_loss + weight * component_loss
        
        return total_loss
    
    def get_loss_components(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get individual loss components.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of loss component values
        """
        components = {}
        
        with torch.no_grad():
            for name, loss_fn in self.losses.items():
                components[name] = loss_fn(predictions, targets).item()
        
        return components


class FeatureWiseLoss(nn.Module):
    """
    Feature-wise Loss for multi-channel signals.
    
    Computes loss separately for each feature (position,
    velocity, force) and combines with configurable weights.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        base_loss: str = 'mse'
    ):
        """
        Initialize feature-wise loss.
        
        Parameters:
        -----------
        feature_weights : Dict[str, float], optional
            Weights for each feature type
        base_loss : str
            Base loss function ('mse', 'mae', 'huber')
        """
        super().__init__()
        
        # Default weights (equal for all features)
        self.feature_weights = feature_weights or {
            'position': 1.0,
            'velocity': 1.0,
            'force': 1.0
        }
        
        # Feature indices (assuming 9 features: 3 pos, 3 vel, 3 force)
        self.feature_indices = {
            'position': slice(0, 3),
            'velocity': slice(3, 6),
            'force': slice(6, 9)
        }
        
        # Base loss function
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='mean')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='mean')
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature-wise weighted loss.
        
        Assumes last dimension has 9 features (3 pos, 3 vel, 3 force).
        """
        total_loss = 0.0
        
        for feature_name, indices in self.feature_indices.items():
            weight = self.feature_weights.get(feature_name, 1.0)
            
            pred_feature = predictions[..., indices]
            target_feature = targets[..., indices]
            
            feature_loss = self.base_loss(pred_feature, target_feature)
            total_loss = total_loss + weight * feature_loss
        
        return total_loss
    
    def get_feature_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Get individual feature losses."""
        losses = {}
        
        with torch.no_grad():
            for feature_name, indices in self.feature_indices.items():
                pred_feature = predictions[..., indices]
                target_feature = targets[..., indices]
                losses[feature_name] = self.base_loss(
                    pred_feature, target_feature
                ).item()
        
        return losses


class AccuracyLoss(nn.Module):
    """
    Loss based on accuracy metric from paper.
    
    Accuracy = 1 - (MAE / target_range)
    Loss = 1 - Accuracy = MAE / target_range
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize accuracy loss.
        
        Parameters:
        -----------
        epsilon : float
            Small constant to prevent division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute accuracy-based loss.
        
        Returns normalized MAE (1 - accuracy).
        """
        mae = torch.mean(torch.abs(predictions - targets))
        target_range = torch.max(targets) - torch.min(targets) + self.epsilon
        
        normalized_error = mae / target_range
        
        return normalized_error


def create_loss_function(
    loss_type: str = 'mse',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Parameters:
    -----------
    loss_type : str
        Type of loss function
    **kwargs : dict
        Additional arguments for the loss function
        
    Returns:
    --------
    nn.Module
        Loss function module
        
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> loss_fn = create_loss_function('mse')
    >>> loss_fn = create_loss_function('combined', losses={
    ...     'mse': (MSELoss(), 1.0),
    ...     'psnr': (PSNRLoss(), 0.1)
    ... })
    """
    loss_type = loss_type.lower()
    
    loss_map = {
        'mse': MSELoss,
        'mae': MAELoss,
        'l1': MAELoss,
        'huber': HuberLoss,
        'smooth_l1': HuberLoss,
        'psnr': PSNRLoss,
        'perceptual': PerceptualLoss,
        'temporal': TemporalConsistencyLoss,
        'combined': CombinedLoss,
        'feature_wise': FeatureWiseLoss,
        'accuracy': AccuracyLoss
    }
    
    if loss_type not in loss_map:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_map.keys())}"
        )
    
    return loss_map[loss_type](**kwargs)


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions")
    print("=" * 50)
    
    # Create test tensors
    predictions = torch.randn(32, 100, 9)
    targets = torch.randn(32, 100, 9)
    
    # Test each loss
    losses_to_test = [
        ('MSE', MSELoss()),
        ('MAE', MAELoss()),
        ('Huber', HuberLoss()),
        ('PSNR', PSNRLoss()),
        ('Perceptual', PerceptualLoss()),
        ('Temporal', TemporalConsistencyLoss()),
        ('Feature-wise', FeatureWiseLoss()),
        ('Accuracy', AccuracyLoss())
    ]
    
    for name, loss_fn in losses_to_test:
        loss_value = loss_fn(predictions, targets)
        print(f"{name} Loss: {loss_value.item():.6f}")
    
    # Test combined loss
    print("\nTesting Combined Loss:")
    combined = CombinedLoss({
        'mse': (MSELoss(), 1.0),
        'temporal': (TemporalConsistencyLoss(), 0.1)
    })
    loss_value = combined(predictions, targets)
    print(f"Combined Loss: {loss_value.item():.6f}")
    
    components = combined.get_loss_components(predictions, targets)
    print(f"Components: {components}")
    
    # Test PSNR computation
    print("\nTesting PSNR:")
    psnr = PSNRLoss.compute_psnr(predictions, targets)
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test factory function
    print("\nTesting Factory Function:")
    loss_fn = create_loss_function('mse')
    print(f"Created loss: {type(loss_fn).__name__}")
