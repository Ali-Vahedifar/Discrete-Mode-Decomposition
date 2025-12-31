"""
DataLoader Utilities for Haptic Signal Data
============================================

Custom dataloader implementations with support for sliding windows,
batch processing, and multi-worker loading.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from typing import Tuple, List, Optional, Dict, Union, Callable
from dataclasses import dataclass
import warnings

from .dataset import HapticDataset, DatasetConfig


@dataclass
class DataLoaderConfig:
    """Configuration for dataloaders.
    
    Attributes:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for GPU transfer
        shuffle_train: Whether to shuffle training data
        drop_last: Whether to drop last incomplete batch
        prefetch_factor: Number of batches to prefetch per worker
    """
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for haptic data batches.
    
    Handles both simple (x, y) tuples and dictionary batches.
    
    Parameters:
    -----------
    batch : List[Tuple]
        List of samples from dataset
        
    Returns:
    --------
    Tuple or Dict
        Batched data
    """
    if isinstance(batch[0], dict):
        # Dictionary batch
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result
    else:
        # Tuple batch (x, y)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


def collate_with_modes(
    batch: List[Dict]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batches with DMD modes.
    
    Parameters:
    -----------
    batch : List[Dict]
        List of samples with 'input', 'target', 'modes' keys
        
    Returns:
    --------
    Dict[str, torch.Tensor]
        Batched dictionary
    """
    result = {
        'input': torch.stack([item['input'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
    }
    
    if 'modes' in batch[0]:
        result['modes'] = torch.stack([item['modes'] for item in batch])
    
    if 'center_frequencies' in batch[0]:
        result['center_frequencies'] = torch.stack([
            item['center_frequencies'] for item in batch
        ])
    
    return result


class HapticDataLoader:
    """
    Custom dataloader wrapper for haptic data.
    
    Provides easy-to-use interface for creating train/val/test
    dataloaders with proper configuration.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> loader = HapticDataLoader(dataset, config)
    >>> for batch_x, batch_y in loader:
    >>>     predictions = model(batch_x)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: Optional[DataLoaderConfig] = None,
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None
    ):
        """
        Initialize dataloader wrapper.
        
        Parameters:
        -----------
        dataset : Dataset
            PyTorch dataset
        config : DataLoaderConfig, optional
            Configuration
        shuffle : bool
            Whether to shuffle data
        collate_fn : callable, optional
            Custom collate function
        """
        self.dataset = dataset
        self.config = config or DataLoaderConfig()
        
        # Create PyTorch dataloader
        self.loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            collate_fn=collate_fn,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            persistent_workers=self.config.num_workers > 0
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.config.batch_size
    
    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        return len(self.dataset)


def create_train_val_test_loaders(
    dataset: Dataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    sequential_split: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Parameters:
    -----------
    dataset : Dataset
        Full dataset
    train_split : float
        Fraction for training (default: 0.7)
    val_split : float
        Fraction for validation (default: 0.15)
    batch_size : int
        Batch size
    num_workers : int
        Number of workers
    seed : int
        Random seed
    sequential_split : bool
        If True, split sequentially (for time series).
        If False, split randomly.
        
    Returns:
    --------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    N = len(dataset)
    train_size = int(N * train_split)
    val_size = int(N * val_split)
    test_size = N - train_size - val_size
    
    if sequential_split:
        # Sequential split for time series data
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, N))
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
    else:
        # Random split
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
    
    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_window_loaders(
    data: np.ndarray,
    window_sizes: List[int],
    prediction_horizon: int = 100,
    batch_size: int = 64,
    **kwargs
) -> Dict[int, DataLoader]:
    """
    Create dataloaders for different window sizes.
    
    Used for evaluating model performance across different
    prediction horizons as in the paper experiments.
    
    Parameters:
    -----------
    data : np.ndarray
        Raw data
    window_sizes : List[int]
        List of window sizes to create loaders for
    prediction_horizon : int
        Prediction horizon
    batch_size : int
        Batch size
        
    Returns:
    --------
    Dict[int, DataLoader]
        Dictionary mapping window_size -> DataLoader
    """
    loaders = {}
    
    for window_size in window_sizes:
        config = DatasetConfig(
            window_size=window_size,
            prediction_horizon=prediction_horizon
        )
        dataset = HapticDataset(data=data, config=config)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        loaders[window_size] = loader
    
    return loaders


class InfiniteDataLoader:
    """
    Dataloader that loops infinitely.
    
    Useful for training with a fixed number of iterations
    instead of epochs.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, dataloader: DataLoader):
        """Initialize infinite loader."""
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    def __next__(self):
        """Get next batch, restart if exhausted."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


class BatchSampler:
    """
    Custom batch sampler for haptic data.
    
    Supports various sampling strategies including
    sequential, random, and importance sampling.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize batch sampler.
        
        Parameters:
        -----------
        dataset_size : int
            Total number of samples
        batch_size : int
            Batch size
        drop_last : bool
            Whether to drop last incomplete batch
        shuffle : bool
            Whether to shuffle indices
        seed : int
            Random seed
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
    
    def __iter__(self):
        """Generate batch indices."""
        indices = np.arange(self.dataset_size)
        
        if self.shuffle:
            self.rng.shuffle(indices)
        
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


class SequenceDataLoader:
    """
    Specialized dataloader for sequence-to-sequence prediction.
    
    Handles variable-length sequences and provides
    methods for different prediction modes.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        sequence_length: int = 100,
        prediction_length: int = 100,
        overlap: int = 50,
        num_workers: int = 4
    ):
        """
        Initialize sequence dataloader.
        
        Parameters:
        -----------
        dataset : Dataset
            Dataset to load from
        batch_size : int
            Batch size
        sequence_length : int
            Input sequence length
        prediction_length : int
            Output prediction length
        overlap : int
            Overlap between consecutive sequences
        num_workers : int
            Number of workers
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.overlap = overlap
        
        # Create base dataloader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._sequence_collate
        )
    
    def _sequence_collate(
        self,
        batch: List[Tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate sequences with padding if needed."""
        xs, ys = zip(*batch)
        
        # Pad sequences if they have different lengths
        max_x_len = max(x.shape[0] for x in xs)
        max_y_len = max(y.shape[0] for y in ys)
        
        x_padded = torch.zeros(len(xs), max_x_len, xs[0].shape[1])
        y_padded = torch.zeros(len(ys), max_y_len, ys[0].shape[1])
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            x_padded[i, :x.shape[0]] = x
            y_padded[i, :y.shape[0]] = y
        
        return x_padded, y_padded
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


if __name__ == "__main__":
    # Test dataloader utilities
    print("Testing DataLoader Utilities")
    print("=" * 50)
    
    # Create synthetic dataset
    from .dataset import HapticDataset, DatasetConfig
    
    np.random.seed(42)
    data = np.random.randn(5000, 9).astype(np.float32)
    
    config = DatasetConfig(window_size=100, prediction_horizon=10)
    dataset = HapticDataset(data=data, config=config)
    
    # Test train/val/test split
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        dataset,
        batch_size=32,
        num_workers=0  # For testing
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test iteration
    for x, y in train_loader:
        print(f"Batch X shape: {x.shape}")
        print(f"Batch Y shape: {y.shape}")
        break
    
    # Test window loaders
    window_loaders = create_window_loaders(
        data,
        window_sizes=[1, 5, 10, 25, 50, 100],
        batch_size=32,
        num_workers=0
    )
    
    print(f"\nWindow loaders created for sizes: {list(window_loaders.keys())}")
    
    # Test infinite loader
    infinite_loader = InfiniteDataLoader(train_loader)
    for i, (x, y) in enumerate(infinite_loader):
        if i >= 5:
            break
        print(f"Infinite batch {i}: {x.shape}")
