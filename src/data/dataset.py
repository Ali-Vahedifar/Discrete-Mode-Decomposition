"""
Dataset Classes for Haptic Signal Data
======================================

Implementation of PyTorch dataset classes for haptic signal data,
supporting various haptic interaction scenarios.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

This module implements:
- HapticDataset: Base dataset for single-channel haptic signals
- HapticMultiChannelDataset: Multi-channel (position, velocity, force) dataset
- TapAndHoldDataset: Specific dataset for Tap-and-Hold interactions
- create_sliding_windows: Utility for windowed data preparation

Dataset Structure:
-----------------
The Kinaesthetic Interactions Dataset contains:
- 3D Position (X, Y, Z)
- 3D Velocity (X, Y, Z)  
- 3D Force (X, Y, Z)
- Recorded at 1 kHz sampling rate
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Union
import pandas as pd
from pathlib import Path
import h5py
import warnings
from dataclasses import dataclass, field
import json


@dataclass
class DatasetConfig:
    """Configuration for haptic dataset.
    
    Attributes:
        window_size: Input window size (number of samples)
        prediction_horizon: Number of samples to predict
        stride: Sliding window stride
        features: List of features to include
        normalize: Whether to normalize data
        train_split: Training data fraction (paper: 70%)
        val_split: Validation data fraction (paper: 10%)
        test_split: Test data fraction (paper: 20%)
        seed: Random seed for reproducibility
    
    From paper: "All results are averaged over 10 independent runs, 
    with training using 70% of the data and testing using 20%, 
    and 10% of the data reserved as a validation set."
    """
    window_size: int = 100
    prediction_horizon: int = 100
    stride: int = 1
    features: List[str] = field(default_factory=lambda: [
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'force_x', 'force_y', 'force_z'
    ])
    normalize: bool = True
    train_split: float = 0.7   # Paper: 70% training
    val_split: float = 0.1     # Paper: 10% validation
    test_split: float = 0.2    # Paper: 20% testing
    seed: int = 42
    sampling_rate: int = 1000  # Hz


class HapticDataset(Dataset):
    """
    Base dataset class for haptic signal data.
    
    Supports loading haptic data from various file formats and creating
    sliding window samples for training signal prediction models.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> config = DatasetConfig(window_size=100, prediction_horizon=10)
    >>> dataset = HapticDataset(data_path='data/haptic.csv', config=config)
    >>> x, y = dataset[0]
    >>> print(x.shape, y.shape)  # (100, 9), (10, 9)
    """
    
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        data_path: Optional[str] = None,
        config: Optional[DatasetConfig] = None,
        transform=None,
        target_transform=None,
        side: str = 'human'  # 'human' or 'robot'
    ):
        """
        Initialize haptic dataset.
        
        Parameters:
        -----------
        data : np.ndarray, optional
            Raw data array of shape (N, num_features)
        data_path : str, optional
            Path to data file (csv, npy, or h5)
        config : DatasetConfig, optional
            Dataset configuration
        transform : callable, optional
            Transform for input data
        target_transform : callable, optional
            Transform for target data
        side : str
            'human' for human operator data, 'robot' for robot data
        """
        self.config = config or DatasetConfig()
        self.transform = transform
        self.target_transform = target_transform
        self.side = side
        
        # Load data
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            raise ValueError("Must provide either data array or data_path")
        
        # Validate data
        self._validate_data()
        
        # Normalize if required
        self.mean = None
        self.std = None
        if self.config.normalize:
            self.data, self.mean, self.std = self._normalize(self.data)
        
        # Create sliding windows
        self.windows, self.targets = self._create_windows()
        
        # Store metadata
        self.num_features = self.data.shape[1]
        self.num_samples = len(self.windows)
    
    def _load_data(self, data_path: str) -> np.ndarray:
        """Load data from file."""
        path = Path(data_path)
        
        if path.suffix == '.csv':
            df = pd.read_csv(path)
            # Select only the feature columns
            feature_cols = [c for c in df.columns if any(
                f in c.lower() for f in ['pos', 'vel', 'force']
            )]
            if not feature_cols:
                feature_cols = df.columns.tolist()[:9]  # Assume first 9 columns
            return df[feature_cols].values.astype(np.float32)
        
        elif path.suffix == '.npy':
            return np.load(path).astype(np.float32)
        
        elif path.suffix in ['.h5', '.hdf5']:
            with h5py.File(path, 'r') as f:
                # Try common dataset names
                for name in ['data', 'signals', 'haptic', self.side]:
                    if name in f:
                        return np.array(f[name]).astype(np.float32)
                # Otherwise use first dataset
                first_key = list(f.keys())[0]
                return np.array(f[first_key]).astype(np.float32)
        
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data_dict = json.load(f)
            return np.array(data_dict['signals']).astype(np.float32)
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _validate_data(self):
        """Validate loaded data."""
        if self.data.ndim != 2:
            if self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)
            else:
                raise ValueError(f"Data must be 1D or 2D, got shape {self.data.shape}")
        
        N, C = self.data.shape
        
        # Check for NaN or Inf
        if np.any(np.isnan(self.data)):
            warnings.warn("Data contains NaN values, replacing with interpolation")
            self.data = self._interpolate_nan(self.data)
        
        if np.any(np.isinf(self.data)):
            warnings.warn("Data contains Inf values, clipping to finite range")
            self.data = np.clip(self.data, -1e6, 1e6)
        
        # Check minimum length
        min_length = self.config.window_size + self.config.prediction_horizon
        if N < min_length:
            raise ValueError(
                f"Data length ({N}) must be at least window_size + prediction_horizon ({min_length})"
            )
    
    def _interpolate_nan(self, data: np.ndarray) -> np.ndarray:
        """Interpolate NaN values."""
        result = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            if np.any(mask):
                x = np.arange(len(data))
                valid = ~mask
                result[:, col] = np.interp(x, x[valid], data[valid, col])
        return result
    
    def _normalize(
        self,
        data: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize data to zero mean and unit variance."""
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
            std[std < 1e-8] = 1.0  # Prevent division by zero
        
        normalized = (data - mean) / std
        return normalized, mean, std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        if self.mean is None or self.std is None:
            return data
        return data * self.std + self.mean
    
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window samples."""
        windows = []
        targets = []
        
        N = len(self.data)
        total_length = self.config.window_size + self.config.prediction_horizon
        
        for i in range(0, N - total_length + 1, self.config.stride):
            window = self.data[i:i + self.config.window_size]
            target = self.data[i + self.config.window_size:i + total_length]
            
            windows.append(window)
            targets.append(target)
        
        return np.array(windows), np.array(targets)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample by index.
        
        Parameters:
        -----------
        idx : int
            Sample index
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (input_window, target) pair
        """
        x = self.windows[idx].copy()
        y = self.targets[idx].copy()
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        return x, y
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'num_samples': self.num_samples,
            'num_features': self.num_features,
            'data_length': len(self.data),
            'window_size': self.config.window_size,
            'prediction_horizon': self.config.prediction_horizon,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'side': self.side
        }


class HapticMultiChannelDataset(HapticDataset):
    """
    Multi-channel haptic dataset with position, velocity, and force.
    
    Organizes data into separate channels for each type of measurement.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        position_data: Optional[np.ndarray] = None,
        velocity_data: Optional[np.ndarray] = None,
        force_data: Optional[np.ndarray] = None,
        data_path: Optional[str] = None,
        config: Optional[DatasetConfig] = None,
        **kwargs
    ):
        """
        Initialize multi-channel dataset.
        
        Parameters:
        -----------
        position_data : np.ndarray, optional
            Position data (N, 3)
        velocity_data : np.ndarray, optional
            Velocity data (N, 3)
        force_data : np.ndarray, optional
            Force data (N, 3)
        data_path : str, optional
            Path to data file with all channels
        config : DatasetConfig, optional
            Configuration
        """
        if data_path is not None:
            # Load from file, parent class handles this
            super().__init__(data_path=data_path, config=config, **kwargs)
        else:
            # Combine separate channel data
            if position_data is None or velocity_data is None or force_data is None:
                raise ValueError("Must provide all three channels or data_path")
            
            combined = np.concatenate([position_data, velocity_data, force_data], axis=1)
            super().__init__(data=combined, config=config, **kwargs)
        
        # Store channel indices
        self.channel_indices = {
            'position': slice(0, 3),
            'velocity': slice(3, 6),
            'force': slice(6, 9)
        }
    
    def get_channel(self, channel: str) -> np.ndarray:
        """Get specific channel data."""
        if channel not in self.channel_indices:
            raise ValueError(f"Unknown channel: {channel}")
        return self.data[:, self.channel_indices[channel]]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with separate channels."""
        x, y = super().__getitem__(idx)
        
        return {
            'input': x,
            'target': y,
            'input_position': x[:, :3],
            'input_velocity': x[:, 3:6],
            'input_force': x[:, 6:9],
            'target_position': y[:, :3],
            'target_velocity': y[:, 3:6],
            'target_force': y[:, 6:9]
        }


class TapAndHoldDataset(HapticMultiChannelDataset):
    """
    Dataset for Tap-and-Hold haptic interactions.
    
    This is the specific dataset used for evaluation in the paper,
    containing kinaesthetic interactions from Novint Falcon device.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    Reference:
    ----------
    Rodríguez-Guevara, D. & Hernandez Gobertti, F.A. (2025).
    Kinaesthetic Interactions Dataset. Zenodo.
    https://doi.org/10.5281/zenodo.14924062
    """
    
    def __init__(
        self,
        data_dir: str,
        session: str = 'tap_and_hold',
        config: Optional[DatasetConfig] = None,
        **kwargs
    ):
        """
        Initialize Tap-and-Hold dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing dataset files
        session : str
            Session name to load
        config : DatasetConfig, optional
            Configuration
        """
        self.data_dir = Path(data_dir)
        self.session = session
        
        # Try to load data
        data = self._load_tap_and_hold_data()
        
        super().__init__(data=data, config=config, **kwargs)
    
    def _load_tap_and_hold_data(self) -> np.ndarray:
        """Load Tap-and-Hold session data."""
        # Look for data files
        possible_files = [
            self.data_dir / f'{self.session}.csv',
            self.data_dir / f'{self.session}.npy',
            self.data_dir / f'{self.session}.h5',
            self.data_dir / 'tap_and_hold' / 'data.csv',
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return self._load_data(str(file_path))
        
        # If no file found, generate synthetic data for testing
        warnings.warn("Tap-and-Hold data not found, generating synthetic data")
        return self._generate_synthetic_data()
    
    def _generate_synthetic_data(
        self,
        num_samples: int = 10000
    ) -> np.ndarray:
        """Generate synthetic tap-and-hold data for testing."""
        np.random.seed(42)
        t = np.linspace(0, num_samples / 1000, num_samples)
        
        # Position: smooth trajectory with occasional taps
        pos_x = 0.1 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(num_samples)
        pos_y = 0.1 * np.cos(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(num_samples)
        pos_z = 0.05 * np.sin(2 * np.pi * 1 * t) + 0.03 * np.random.randn(num_samples)
        
        # Add tap events
        tap_indices = np.random.choice(num_samples, size=20, replace=False)
        for idx in tap_indices:
            start = max(0, idx - 50)
            end = min(num_samples, idx + 50)
            pos_z[start:end] -= 0.1 * np.exp(-((np.arange(end-start) - 50)**2) / 200)
        
        # Velocity: derivative of position with noise
        vel_x = np.gradient(pos_x) * 1000 + 0.1 * np.random.randn(num_samples)
        vel_y = np.gradient(pos_y) * 1000 + 0.1 * np.random.randn(num_samples)
        vel_z = np.gradient(pos_z) * 1000 + 0.1 * np.random.randn(num_samples)
        
        # Force: related to position with spring model
        k = 100  # Spring constant
        force_x = -k * pos_x + 5 * np.random.randn(num_samples)
        force_y = -k * pos_y + 5 * np.random.randn(num_samples)
        force_z = -k * np.clip(pos_z, -np.inf, 0) + 5 * np.random.randn(num_samples)
        
        return np.stack([
            pos_x, pos_y, pos_z,
            vel_x, vel_y, vel_z,
            force_x, force_y, force_z
        ], axis=1).astype(np.float32)


class DMDDataset(HapticDataset):
    """
    Dataset that includes pre-computed DMD modes.
    
    Extends HapticDataset to include mode information for
    each sample, enabling mode-aware training.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        modes: np.ndarray,
        center_frequencies: np.ndarray,
        mode_indices: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize DMD-enhanced dataset.
        
        Parameters:
        -----------
        modes : np.ndarray
            Pre-computed DMD modes (num_windows, num_modes, mode_length)
        center_frequencies : np.ndarray
            Center frequencies for each mode
        mode_indices : np.ndarray, optional
            Indices of important modes (from SMV)
        """
        super().__init__(**kwargs)
        
        self.modes = modes
        self.center_frequencies = center_frequencies
        self.mode_indices = mode_indices
        
        # Validate mode count matches window count
        if len(modes) != self.num_samples:
            raise ValueError(
                f"Number of mode sets ({len(modes)}) must match "
                f"number of samples ({self.num_samples})"
            )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with modes."""
        x, y = HapticDataset.__getitem__(self, idx)
        
        modes = torch.from_numpy(self.modes[idx]).float()
        
        # Select important modes if indices provided
        if self.mode_indices is not None:
            modes = modes[self.mode_indices]
        
        return {
            'input': x,
            'target': y,
            'modes': modes,
            'center_frequencies': torch.from_numpy(self.center_frequencies).float()
        }


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    prediction_horizon: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window samples from sequential data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (N, C) or (N,)
    window_size : int
        Size of input window
    prediction_horizon : int
        Size of prediction target
    stride : int
        Stride between consecutive windows
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (windows, targets) arrays
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    N = len(data)
    total_length = window_size + prediction_horizon
    
    windows = []
    targets = []
    
    for i in range(0, N - total_length + 1, stride):
        windows.append(data[i:i + window_size])
        targets.append(data[i + window_size:i + total_length])
    
    return np.array(windows), np.array(targets)


def split_data(
    data: np.ndarray,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    train_split : float
        Fraction for training
    val_split : float
        Fraction for validation
    seed : int
        Random seed
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (train_data, val_data, test_data)
    """
    np.random.seed(seed)
    N = len(data)
    indices = np.random.permutation(N)
    
    train_end = int(N * train_split)
    val_end = train_end + int(N * val_split)
    
    train_data = data[indices[:train_end]]
    val_data = data[indices[train_end:val_end]]
    test_data = data[indices[val_end:]]
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test dataset classes
    print("Testing HapticDataset")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    N = 5000
    data = np.random.randn(N, 9).astype(np.float32)
    
    # Create dataset
    config = DatasetConfig(
        window_size=100,
        prediction_horizon=10,
        stride=5
    )
    
    dataset = HapticDataset(data=data, config=config)
    print(f"Dataset length: {len(dataset)}")
    print(f"Statistics: {dataset.get_statistics()}")
    
    # Test getitem
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Test TapAndHoldDataset
    print("\nTesting TapAndHoldDataset")
    print("=" * 50)
    
    tap_dataset = TapAndHoldDataset(
        data_dir='./data',
        config=config
    )
    print(f"Tap-and-Hold dataset length: {len(tap_dataset)}")
    
    # Test sliding windows
    print("\nTesting create_sliding_windows")
    windows, targets = create_sliding_windows(data, 100, 10, stride=1)
    print(f"Windows shape: {windows.shape}")
    print(f"Targets shape: {targets.shape}")
