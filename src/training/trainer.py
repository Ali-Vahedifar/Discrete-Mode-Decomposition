"""
Trainer Class for Signal Prediction Models
==========================================

Implementation of the training loop for neural network models,
following the training procedure described in the IEEE INFOCOM 2025 paper.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025: "Discrete Mode Decomposition Meets Shapley Value:
Robust Signal Prediction in Tactile Internet"

Training Configuration (from paper):
------------------------------------
- Adam optimizer with β1=0.9, β2=0.999
- Initial learning rate: 0.001
- Learning rate decay: factor of 0.005 at epochs 80, 120, 170
- Maximum epochs: 200
- Dropout: 0.1 (ResNet, Transformer), 0.2 (LSTM)
- Results averaged over 10 independent runs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Optional, Dict, List, Callable, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import logging
from tqdm import tqdm
import warnings
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    
    Default values match the paper's experimental setup exactly.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Attributes:
        epochs: Maximum training epochs (paper: 200)
        batch_size: Training batch size (paper: 64)
        learning_rate: Initial learning rate (paper: 0.001)
        lr_decay_epochs: Epochs at which to decay learning rate (paper: [80, 120, 170])
        lr_decay_factor: Factor by which to decay learning rate (paper: 0.005)
        beta1: Adam first moment decay (paper: 0.9)
        beta2: Adam second moment decay (paper: 0.999)
        weight_decay: L2 regularization coefficient
        dropout: Dropout rate (paper: 0.1 for Transformer/ResNet, 0.2 for LSTM)
        optimizer: Optimizer name ('adam', 'adamw', 'sgd')
        loss_function: Loss function name ('mse', 'mae', 'huber')
        gradient_clip: Maximum gradient norm (None to disable)
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Steps between logging
        device: Device to train on ('auto', 'cuda', 'cpu')
        seed: Random seed for reproducibility
        num_runs: Number of independent runs for averaging (paper: 10)
        use_amp: Whether to use automatic mixed precision
        save_best_only: Only save best model checkpoint
    """
    # Training hyperparameters from paper
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    lr_decay_epochs: List[int] = field(default_factory=lambda: [80, 120, 170])
    lr_decay_factor: float = 0.005
    
    # Adam optimizer settings from paper
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Regularization
    dropout: float = 0.1
    
    # Optimizer and loss
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    
    # Training settings
    gradient_clip: Optional[float] = 1.0
    early_stopping_patience: int = 20
    checkpoint_dir: str = './checkpoints'
    log_interval: int = 10
    
    # Device and reproducibility
    device: str = 'auto'
    seed: int = 42
    num_runs: int = 10
    
    # Advanced settings
    use_amp: bool = False  # Automatic mixed precision
    save_best_only: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'lr_decay_epochs': self.lr_decay_epochs,
            'lr_decay_factor': self.lr_decay_factor,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'gradient_clip': self.gradient_clip,
            'early_stopping_patience': self.early_stopping_patience,
            'seed': self.seed,
            'num_runs': self.num_runs
        }


@dataclass
class TrainingResult:
    """
    Result of training.
    
    Stores all metrics and training history for analysis.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Attributes:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_errors: Training error percentages per epoch
        val_errors: Validation error percentages per epoch
        train_accuracies: Training accuracies per epoch
        val_accuracies: Validation accuracies per epoch
        best_epoch: Epoch with best validation loss
        best_val_loss: Best validation loss achieved
        best_val_accuracy: Best validation accuracy achieved
        training_time: Total training time in seconds
        final_model_path: Path to final saved model
        learning_rates: Learning rate at each epoch
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_errors: List[float] = field(default_factory=list)
    val_errors: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    training_time: float = 0.0
    final_model_path: Optional[str] = None
    learning_rates: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_errors': self.train_errors,
            'val_errors': self.val_errors,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'training_time': self.training_time,
            'final_model_path': self.final_model_path,
            'learning_rates': self.learning_rates
        }
    
    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Training results saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingResult':
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        result = cls()
        for key, value in data.items():
            if hasattr(result, key):
                setattr(result, key, value)
        return result
    
    def get_summary(self) -> str:
        """Get a text summary of training results."""
        return (
            f"Training Summary\n"
            f"================\n"
            f"Best Epoch: {self.best_epoch}\n"
            f"Best Validation Loss: {self.best_val_loss:.6f}\n"
            f"Best Validation Accuracy: {self.best_val_accuracy:.2f}%\n"
            f"Final Training Loss: {self.train_losses[-1]:.6f}\n"
            f"Final Training Accuracy: {self.train_accuracies[-1]:.2f}%\n"
            f"Training Time: {self.training_time:.2f} seconds\n"
            f"Model Path: {self.final_model_path}"
        )


class Trainer:
    """
    Trainer class for signal prediction models.
    
    Implements the training procedure from the IEEE INFOCOM 2025 paper
    with support for:
    - Multiple neural network architectures (Transformer, ResNet, LSTM)
    - DMD mode-aware training
    - Learning rate scheduling as specified in paper
    - Early stopping
    - Checkpointing
    - Multi-run averaging (10 runs as per paper)
    - Mixed precision training
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> # Initialize trainer with paper's configuration
    >>> config = TrainingConfig(epochs=200, learning_rate=0.001)
    >>> trainer = Trainer(model, config)
    >>> 
    >>> # Train model
    >>> result = trainer.train(train_loader, val_loader)
    >>> 
    >>> # Print results
    >>> print(f"Best validation accuracy: {result.best_val_accuracy:.2f}%")
    >>> print(f"Best epoch: {result.best_epoch}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        callbacks: Optional[List] = None
    ):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        model : nn.Module
            PyTorch model to train
        config : TrainingConfig, optional
            Training configuration (uses paper defaults if None)
        callbacks : List, optional
            List of callback objects for custom behavior
            
        Author: Ali Vahedi
        """
        self.config = config or TrainingConfig()
        self.callbacks = callbacks or []
        
        # Set device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Create optimizer (Adam with paper's settings)
        self.optimizer = self._create_optimizer()
        
        # Create learning rate scheduler (paper's decay schedule)
        self.scheduler = self._create_scheduler()
        
        # Create loss function
        self.criterion = self._create_loss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.early_stopping_counter = 0
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {num_params:,} trainable parameters")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer based on config.
        
        Uses Adam optimizer with paper's settings:
        - β1 = 0.9, β2 = 0.999
        - Initial learning rate = 0.001
        
        Author: Ali Vahedi
        """
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == 'adam':
            # Paper's exact settings
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Implements paper's schedule:
        - Decay by factor of 0.005 at epochs 80, 120, 170
        
        Author: Ali Vahedi
        """
        def lr_lambda(epoch):
            """Compute learning rate multiplier based on epoch."""
            factor = 1.0
            for decay_epoch in self.config.lr_decay_epochs:
                if epoch >= decay_epoch:
                    factor *= self.config.lr_decay_factor
            return factor
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_loss(self) -> nn.Module:
        """
        Create loss function.
        
        Default is MSE loss for signal prediction.
        
        Author: Ali Vahedi
        """
        loss_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'l1': nn.L1Loss(),
            'huber': nn.HuberLoss(),
            'smooth_l1': nn.SmoothL1Loss()
        }
        
        loss_name = self.config.loss_function.lower()
        if loss_name not in loss_map:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
        
        return loss_map[loss_name]
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Train the model.
        
        Implements the full training loop as described in the paper:
        - 200 epochs maximum
        - Learning rate decay at epochs 80, 120, 170
        - Early stopping with patience
        - Checkpointing of best model
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        TrainingResult
            Training results including losses and metrics
            
        Author: Ali Vahedi
        IEEE INFOCOM 2025
        """
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        result = TrainingResult()
        start_time = time.time()
        
        # Notify callbacks of training start
        self._on_train_begin()
        
        # Training loop
        pbar = tqdm(
            range(self.config.epochs),
            desc="Training",
            disable=not verbose
        )
        
        for epoch in pbar:
            self.current_epoch = epoch
            
            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            result.learning_rates.append(current_lr)
            
            # Notify callbacks of epoch start
            self._on_epoch_begin(epoch)
            
            # Training epoch
            train_loss, train_error, train_accuracy = self._train_epoch(train_loader)
            result.train_losses.append(train_loss)
            result.train_errors.append(train_error)
            result.train_accuracies.append(train_accuracy)
            
            # Validation epoch
            val_loss, val_error, val_accuracy = None, None, None
            if val_loader is not None:
                val_loss, val_error, val_accuracy = self._validate_epoch(val_loader)
                result.val_losses.append(val_loss)
                result.val_errors.append(val_error)
                result.val_accuracies.append(val_accuracy)
                
                # Check for best model
                if val_loss < result.best_val_loss:
                    result.best_val_loss = val_loss
                    result.best_val_accuracy = val_accuracy
                    result.best_epoch = epoch
                    self._save_checkpoint('best_model.pt')
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping check
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update learning rate (paper's schedule)
            self.scheduler.step()
            
            # Update progress bar
            if val_loader is not None:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'train_acc': f'{train_accuracy:.2f}%',
                    'val_acc': f'{val_accuracy:.2f}%',
                    'lr': f'{current_lr:.2e}'
                })
            else:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_accuracy:.2f}%',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Notify callbacks of epoch end
            self._on_epoch_end(epoch, train_loss, val_loss)
        
        # Save final model
        final_path = self._save_checkpoint('final_model.pt')
        result.final_model_path = str(final_path)
        
        result.training_time = time.time() - start_time
        
        # Notify callbacks of training end
        self._on_train_end(result)
        
        # Log final results
        logger.info(result.get_summary())
        
        return result
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Run one training epoch.
        
        Returns:
        --------
        Tuple[float, float, float]
            (loss, error_percentage, accuracy)
            
        Author: Ali Vahedi
        """
        self.model.train()
        
        total_loss = 0.0
        total_error = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch['input'].to(self.device)
                y = batch['target'].to(self.device)
                modes = batch.get('modes', None)
                if modes is not None:
                    modes = modes.to(self.device)
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                modes = None
            
            # Forward pass with optional mixed precision
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    if modes is not None:
                        predictions = self.model(x, modes)
                    else:
                        predictions = self.model(x)
                    loss = self.criterion(predictions, y)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                if modes is not None:
                    predictions = self.model(x, modes)
                else:
                    predictions = self.model(x)
                
                loss = self.criterion(predictions, y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Compute error and accuracy
            with torch.no_grad():
                error, accuracy = self._compute_metrics(predictions, y)
                total_error += error
                total_accuracy += accuracy
            
            num_batches += 1
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        avg_error = total_error / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_error, avg_accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Run one validation epoch.
        
        Returns:
        --------
        Tuple[float, float, float]
            (loss, error_percentage, accuracy)
            
        Author: Ali Vahedi
        """
        self.model.eval()
        
        total_loss = 0.0
        total_error = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    x = batch['input'].to(self.device)
                    y = batch['target'].to(self.device)
                    modes = batch.get('modes', None)
                    if modes is not None:
                        modes = modes.to(self.device)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    modes = None
                
                # Forward pass
                if modes is not None:
                    predictions = self.model(x, modes)
                else:
                    predictions = self.model(x)
                
                # Compute loss
                loss = self.criterion(predictions, y)
                total_loss += loss.item()
                
                # Compute metrics
                error, accuracy = self._compute_metrics(predictions, y)
                total_error += error
                total_accuracy += accuracy
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_error = total_error / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_error, avg_accuracy
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute error percentage and accuracy.
        
        Accuracy is computed as: 1 - (MAE / target_range) * 100
        Error is computed as: MAE / target_range * 100
        
        This follows the paper's evaluation methodology.
        
        Returns:
        --------
        Tuple[float, float]
            (error_percentage, accuracy_percentage)
            
        Author: Ali Vahedi
        """
        # Mean Absolute Error
        mae = torch.mean(torch.abs(predictions - targets))
        
        # Target range for normalization
        target_range = torch.max(targets) - torch.min(targets)
        
        if target_range > 1e-8:
            relative_error = mae / target_range
            error_percentage = relative_error.item() * 100
        else:
            error_percentage = 0.0
        
        # Clip to valid range
        error_percentage = min(max(error_percentage, 0.0), 100.0)
        accuracy = 100.0 - error_percentage
        
        return error_percentage, accuracy
    
    def _save_checkpoint(self, filename: str) -> Path:
        """
        Save model checkpoint.
        
        Saves:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Training state
        - Configuration
        
        Author: Ali Vahedi
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Author: Ali Vahedi
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def _on_train_begin(self):
        """Notify callbacks of training start."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)
    
    def _on_train_end(self, result: TrainingResult):
        """Notify callbacks of training end."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self, result)
    
    def _on_epoch_begin(self, epoch: int):
        """Notify callbacks of epoch start."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(self, epoch)
    
    def _on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ):
        """Notify callbacks of epoch end."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(self, epoch, train_loss, val_loss)


def train_multiple_runs(
    model_class,
    model_config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: Optional[TrainingConfig] = None,
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    Train model over multiple independent runs and average results.
    
    As per the paper: "The prediction accuracy results are averaged 
    over 10 independent runs."
    
    Parameters:
    -----------
    model_class : type
        Model class to instantiate
    model_config : ModelConfig
        Configuration for model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    training_config : TrainingConfig, optional
        Training configuration
    num_runs : int
        Number of independent runs (paper: 10)
        
    Returns:
    --------
    Dict[str, Any]
        Aggregated results across all runs
        
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    config = training_config or TrainingConfig()
    
    all_results = []
    all_accuracies = []
    all_losses = []
    
    logger.info(f"Starting {num_runs} independent training runs")
    
    for run in range(num_runs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Run {run + 1}/{num_runs}")
        logger.info(f"{'='*50}")
        
        # Set different seed for each run
        run_config = TrainingConfig(**config.to_dict())
        run_config.seed = config.seed + run
        
        # Create fresh model instance
        model = model_class(model_config)
        
        # Train
        trainer = Trainer(model, run_config)
        result = trainer.train(train_loader, val_loader, verbose=True)
        
        all_results.append(result)
        all_accuracies.append(result.best_val_accuracy)
        all_losses.append(result.best_val_loss)
    
    # Compute statistics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    
    aggregated = {
        'num_runs': num_runs,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'all_accuracies': all_accuracies,
        'all_losses': all_losses,
        'all_results': all_results
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Multi-Run Summary ({num_runs} runs)")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    logger.info(f"Mean Loss: {mean_loss:.6f} ± {std_loss:.6f}")
    
    return aggregated


if __name__ == "__main__":
    # Test trainer
    print("Testing Trainer")
    print("=" * 50)
    
    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(9, 64),
                nn.ReLU(),
                nn.Linear(64, 9)
            )
        
        def forward(self, x):
            batch_size, seq_len, features = x.shape
            x = x.view(-1, features)
            x = self.fc(x)
            return x.view(batch_size, seq_len, features)
    
    # Create synthetic data
    from torch.utils.data import TensorDataset
    
    X = torch.randn(1000, 100, 9)
    Y = torch.randn(1000, 100, 9)
    
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset[:800], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[800:], batch_size=32)
    
    # Create trainer
    model = SimpleModel()
    config = TrainingConfig(epochs=5, learning_rate=0.001)
    trainer = Trainer(model, config)
    
    # Train
    result = trainer.train(train_loader, val_loader, verbose=True)
    
    print("\nTraining completed!")
    print(result.get_summary())
