"""
Visualization Utilities
=======================

Plotting functions for reproducing figures from the IEEE INFOCOM 2025 paper.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Paper-style colors
COLORS = {
    'dmd_smv_human': '#E20D76',  # myred
    'dmd_smv_robot': '#0070C4',  # myblue
    'dmd_human': '#FF6B6B',
    'dmd_robot': '#4DABF7',
    'baseline_human': '#868E96',
    'baseline_robot': '#ADB5BD',
    'transformer': '#E20D76',
    'resnet': '#0070C4',
    'lstm': '#40C057'
}


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_errors: Optional[List[float]] = None,
    val_errors: Optional[List[float]] = None,
    lr_decay_epochs: List[int] = [80, 120, 170],
    save_path: Optional[str] = None,
    title: str = 'Training Progress'
):
    """
    Plot training curves similar to Figure 2 in the paper.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot errors if provided
    if train_errors is not None:
        ax1.plot(epochs, train_errors, 'b-', label='Train Error', linewidth=2)
        ax1.plot(epochs, val_errors, 'r-', label='Val Error', linewidth=2)
        ax1.set_ylabel('Error (%)', fontsize=12)
    else:
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_ylabel('Loss', fontsize=12)
    
    ax1.set_xlabel('Epochs', fontsize=12)
    
    # Mark LR decay points
    for epoch in lr_decay_epochs:
        if epoch < len(train_losses):
            ax1.axvline(x=epoch, color='g', linestyle='--', alpha=0.5,
                       label='LR Decay' if epoch == lr_decay_epochs[0] else '')
    
    ax1.legend(loc='upper right')
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_accuracy_comparison(
    window_sizes: List[int],
    results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
):
    """
    Plot accuracy comparison across methods (Figure 3 style).
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    architectures = ['Transformer', 'ResNet', 'LSTM']
    
    for idx, arch in enumerate(architectures):
        ax = axes[idx]
        
        if arch.lower() in results:
            data = results[arch.lower()]
            
            # Plot accuracy lines
            if 'dmd_smv_human' in data:
                ax.plot(window_sizes, data['dmd_smv_human'], 'o-',
                       color=COLORS['dmd_smv_human'], label='DMD+SMV Human', linewidth=2)
            if 'dmd_smv_robot' in data:
                ax.plot(window_sizes, data['dmd_smv_robot'], 's-',
                       color=COLORS['dmd_smv_robot'], label='DMD+SMV Robot', linewidth=2)
            if 'dmd_human' in data:
                ax.plot(window_sizes, data['dmd_human'], '^--',
                       color=COLORS['dmd_human'], label='DMD Human', linewidth=1.5)
            if 'baseline_human' in data:
                ax.plot(window_sizes, data['baseline_human'], 'x:',
                       color=COLORS['baseline_human'], label='Baseline Human', linewidth=1.5)
        
        ax.set_xlabel('Window Size', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) {arch}', fontsize=14)
        ax.set_xscale('log')
        ax.set_ylim([50, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_psnr_comparison(
    window_sizes: List[int],
    psnr_dmd_smv: Dict[str, List[float]],
    psnr_dmd: Dict[str, List[float]],
    psnr_baseline: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot PSNR comparison (Figure 4 style).
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(window_sizes, psnr_dmd_smv['human'], 'o-',
           color=COLORS['dmd_smv_human'], label='DMD+SMV Human', linewidth=2)
    ax.plot(window_sizes, psnr_dmd_smv['robot'], 's-',
           color=COLORS['dmd_smv_robot'], label='DMD+SMV Robot', linewidth=2)
    ax.plot(window_sizes, psnr_dmd['human'], '^--',
           color=COLORS['dmd_human'], label='DMD Human', linewidth=1.5)
    ax.plot(window_sizes, psnr_baseline['human'], 'x:',
           color=COLORS['baseline_human'], label='Baseline Human', linewidth=1.5)
    
    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Transformer: PSNR Comparison', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_sliding_window_accuracy(
    accuracies: np.ndarray,
    window_size: int = 5,
    architectures: List[str] = ['LSTM', 'ResNet-32', 'Transformer'],
    save_path: Optional[str] = None
):
    """
    Plot accuracy over sliding windows (Figure 5 style).
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_windows = len(accuracies[0]) if len(accuracies.shape) > 1 else len(accuracies)
    x = np.arange(1, num_windows + 1)
    
    colors = [COLORS['lstm'], COLORS['resnet'], COLORS['transformer']]
    
    for i, arch in enumerate(architectures):
        if len(accuracies.shape) > 1:
            ax.plot(x, accuracies[i], 'o-', color=colors[i],
                   label=f'{arch} DMD+SMV', linewidth=2)
        else:
            ax.plot(x, accuracies, 'o-', color=colors[0], linewidth=2)
            break
    
    ax.set_xlabel('Sliding Window ID', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'DMD+SMV: Accuracy vs Sliding Window Position (W={window_size})', fontsize=14)
    ax.set_ylim([60, 100])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities - Author: Ali Vahedi")
    print("=" * 50)
    
    # Test plotting
    epochs = 200
    train_losses = np.exp(-np.linspace(0, 3, epochs)) + 0.1 * np.random.randn(epochs) * 0.01
    val_losses = np.exp(-np.linspace(0, 2.5, epochs)) + 0.1 * np.random.randn(epochs) * 0.01
    
    plot_training_curves(
        train_losses.tolist(),
        val_losses.tolist(),
        save_path='./test_training_curve.png',
        title='Test Training Curve'
    )
    print("Test plot created!")
