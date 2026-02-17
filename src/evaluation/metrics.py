"""
Evaluation Metrics for Signal Prediction
=========================================

Implementation of evaluation metrics used in the IEEE INFOCOM 2026 paper:
- Accuracy (%)
- Error (%)
- PSNR (dB)
- MSE, MAE, RMSE

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2026: "Discrete Mode Decomposition Meets Shapley Value:
Robust Signal Prediction in Tactile Internet"

Paper Results Summary:
---------------------
- DMD+SMV + Transformer: 98.9% accuracy (W=1), 92.5% accuracy (W=100)
- PSNR: ~29.5 dB (human), ~27.5 dB (robot) at W=1
- Speedup: 820x vs baseline, 3x vs DMD
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import warnings


def compute_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    epsilon: float = 1e-8
) -> float:
    """
    Compute prediction accuracy as percentage.
    
    Accuracy = (1 - MAE / target_range) * 100
    
    This is the primary accuracy metric used in the paper.
    Paper achieves: 98.9% (W=1), 92.5% (W=100) with DMD+SMV+Transformer.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    mae = np.mean(np.abs(predictions - targets))
    target_range = np.max(targets) - np.min(targets) + epsilon
    relative_error = mae / target_range
    accuracy = (1.0 - relative_error) * 100.0
    
    return float(np.clip(accuracy, 0.0, 100.0))


def compute_error(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    epsilon: float = 1e-8
) -> float:
    """Compute prediction error as percentage. Author: Ali Vahedi"""
    return 100.0 - compute_accuracy(predictions, targets, epsilon)


def compute_psnr(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    max_value: Optional[float] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Paper Results: ~29.5 dB (human), ~27.5 dB (robot) at W=1
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    if max_value is None:
        max_value = np.max(np.abs(targets))
    
    mse = np.mean((predictions - targets) ** 2)
    if mse < 1e-10:
        return float('inf')
    
    return float(10.0 * np.log10(max_value ** 2 / mse))


def compute_mse(predictions, targets) -> float:
    """Compute Mean Squared Error. Author: Ali Vahedi"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return float(np.mean((predictions - targets) ** 2))


def compute_mae(predictions, targets) -> float:
    """Compute Mean Absolute Error. Author: Ali Vahedi"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions, targets) -> float:
    """Compute Root Mean Squared Error. Author: Ali Vahedi"""
    return float(np.sqrt(compute_mse(predictions, targets)))


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics. Author: Ali Vahedi, IEEE INFOCOM 2025"""
    accuracy: float = 0.0
    error: float = 0.0
    psnr: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    position_accuracy: Optional[float] = None
    velocity_accuracy: Optional[float] = None
    force_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def __str__(self) -> str:
        return f"EvaluationMetrics(accuracy={self.accuracy:.2f}%, psnr={self.psnr:.2f}dB)"


class MetricsComputer:
    """Comprehensive metrics computation. Author: Ali Vahedi, IEEE INFOCOM 2025"""
    
    def __init__(self, compute_feature_wise: bool = True):
        self.compute_feature_wise = compute_feature_wise
        self.feature_indices = {
            'position': slice(0, 3),
            'velocity': slice(3, 6),
            'force': slice(6, 9)
        }
    
    def compute_all(self, predictions, targets) -> EvaluationMetrics:
        """Compute all metrics."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        metrics = EvaluationMetrics(
            accuracy=compute_accuracy(predictions, targets),
            error=compute_error(predictions, targets),
            psnr=compute_psnr(predictions, targets),
            mse=compute_mse(predictions, targets),
            mae=compute_mae(predictions, targets),
            rmse=compute_rmse(predictions, targets)
        )
        
        if self.compute_feature_wise and predictions.shape[-1] >= 9:
            for feat, idx in self.feature_indices.items():
                setattr(metrics, f'{feat}_accuracy', 
                       compute_accuracy(predictions[..., idx], targets[..., idx]))
        
        return metrics


if __name__ == "__main__":
    print("Testing Evaluation Metrics - Author: Ali Vahedi")
    print("=" * 50)
    
    np.random.seed(42)
    predictions = np.random.randn(32, 100, 9)
    targets = predictions + 0.1 * np.random.randn(32, 100, 9)
    
    print(f"Accuracy: {compute_accuracy(predictions, targets):.2f}%")
    print(f"PSNR: {compute_psnr(predictions, targets):.2f} dB")
    
    computer = MetricsComputer()
    metrics = computer.compute_all(predictions, targets)
    print(f"All metrics: {metrics}")
