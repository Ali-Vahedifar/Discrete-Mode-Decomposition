"""
Data Module for DMD+SMV Tactile Internet Signal Prediction
===========================================================

This module provides data handling utilities for haptic signal data,
including dataset classes, dataloaders, and preprocessing functions.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

Components:
-----------
- HapticDataset: PyTorch dataset for haptic signals
- HapticDataLoader: Custom dataloader with sliding window support
- DataPreprocessor: Signal preprocessing utilities
"""

from .dataset import (
    HapticDataset,
    HapticMultiChannelDataset,
    TapAndHoldDataset,
    create_sliding_windows
)
from .dataloader import (
    HapticDataLoader,
    create_train_val_test_loaders,
    collate_fn
)
from .preprocessing import (
    DataPreprocessor,
    normalize_signal,
    denormalize_signal,
    add_noise,
    remove_outliers
)

__all__ = [
    'HapticDataset',
    'HapticMultiChannelDataset',
    'TapAndHoldDataset',
    'create_sliding_windows',
    'HapticDataLoader',
    'create_train_val_test_loaders',
    'collate_fn',
    'DataPreprocessor',
    'normalize_signal',
    'denormalize_signal',
    'add_noise',
    'remove_outliers'
]

__author__ = "Ali Vahedi"
__email__ = "av@ece.au.dk"
__institution__ = "Aarhus University, Denmark"
