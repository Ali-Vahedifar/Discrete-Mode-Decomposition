"""
Training Callbacks for Signal Prediction Models
================================================

Implementation of callback classes for monitoring and controlling
the training process.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

"""

import torch
import numpy as np
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
import json
import time
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Abstract base class for training callbacks.
    
    Callbacks allow custom behavior at various points during training:
    - on_train_begin: Called at the start of training
    - on_train_end: Called at the end of training
    - on_epoch_begin: Called at the start of each epoch
    - on_epoch_end: Called at the end of each epoch
    - on_batch_begin: Called at the start of each batch
    - on_batch_end: Called at the end of each batch
    
    
    """
    
    def on_train_begin(self, trainer) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer, result) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float
    ) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Stops training when validation loss stops improving.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> callback = EarlyStopping(patience=20, min_delta=1e-4)
    >>> trainer = Trainer(model, config, callbacks=[callback])
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Parameters:
        -----------
        patience : int
            Number of epochs with no improvement before stopping
        min_delta : float
            Minimum change to qualify as improvement
        mode : str
            'min' for loss (lower is better), 'max' for accuracy
        restore_best_weights : bool
            Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False
    
    def on_train_begin(self, trainer) -> None:
        """Reset state at training start."""
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ) -> None:
        """Check for improvement at end of epoch."""
        if val_loss is None:
            return
        
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self._save_weights(trainer)
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self._save_weights(trainer)
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: {self.counter}/{self.patience}"
            )
            
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}"
                )
                
                if self.restore_best_weights and self.best_weights:
                    trainer.model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
    
    def _save_weights(self, trainer) -> None:
        """Save current model weights."""
        if self.restore_best_weights:
            self.best_weights = {
                k: v.cpu().clone()
                for k, v in trainer.model.state_dict().items()
            }


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.
    
    Implements the paper's learning rate schedule:
    - Decay by factor of 0.005 at epochs 80, 120, 170
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        decay_epochs: List[int] = [80, 120, 170],
        decay_factor: float = 0.005,
        verbose: bool = True
    ):
        """
        Initialize scheduler.
        
        Parameters:
        -----------
        decay_epochs : List[int]
            Epochs at which to decay learning rate
        decay_factor : float
            Factor by which to multiply learning rate
        verbose : bool
            Whether to log learning rate changes
        """
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor
        self.verbose = verbose
        self.initial_lr = None
    
    def on_train_begin(self, trainer) -> None:
        """Store initial learning rate."""
        self.initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Update learning rate at epoch start."""
        # Compute new learning rate based on paper's schedule
        new_lr = self.initial_lr
        for decay_epoch in self.decay_epochs:
            if epoch >= decay_epoch:
                new_lr *= self.decay_factor
        
        # Update optimizer
        old_lr = trainer.optimizer.param_groups[0]['lr']
        if abs(new_lr - old_lr) > 1e-10:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: Learning rate changed "
                    f"from {old_lr:.2e} to {new_lr:.2e}"
                )


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback.
    
    Saves model checkpoints during training.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        save_best_only: bool = True,
        save_freq: int = 1,
        monitor: str = 'val_loss',
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Parameters:
        -----------
        checkpoint_dir : str
            Directory to save checkpoints
        save_best_only : bool
            Only save when monitored metric improves
        save_freq : int
            Save every N epochs (if not save_best_only)
        monitor : str
            Metric to monitor ('val_loss', 'train_loss')
        mode : str
            'min' or 'max' for the monitored metric
        verbose : bool
            Whether to log saves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ) -> None:
        """Save checkpoint at end of epoch."""
        # Get monitored value
        if self.monitor == 'val_loss':
            current_score = -val_loss if val_loss else float('inf')
        else:
            current_score = -train_loss
        
        if self.mode == 'max':
            current_score = -current_score
        
        should_save = False
        
        if self.save_best_only:
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                should_save = True
        else:
            should_save = (epoch + 1) % self.save_freq == 0
        
        if should_save:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            if self.verbose:
                logger.info(f"Saved checkpoint to {checkpoint_path}")


class ProgressCallback(Callback):
    """
    Progress logging callback.
    
    Logs training progress with configurable verbosity.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        log_to_file: Optional[str] = None
    ):
        """
        Initialize progress callback.
        
        Parameters:
        -----------
        log_interval : int
            Number of epochs between detailed logs
        log_to_file : str, optional
            Path to log file
        """
        self.log_interval = log_interval
        self.log_to_file = log_to_file
        
        self.train_start_time = None
        self.epoch_start_time = None
        self.history = []
        
        if log_to_file:
            Path(log_to_file).parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, trainer) -> None:
        """Record training start time."""
        self.train_start_time = time.time()
        self.history = []
        logger.info("Training started")
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ) -> None:
        """Log epoch results."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Store history
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'lr': trainer.optimizer.param_groups[0]['lr']
        }
        self.history.append(record)
        
        # Log at intervals
        if (epoch + 1) % self.log_interval == 0:
            msg = (
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.6f}, "
            )
            if val_loss is not None:
                msg += f"val_loss={val_loss:.6f}, "
            msg += f"time={epoch_time:.2f}s"
            
            logger.info(msg)
            
            if self.log_to_file:
                with open(self.log_to_file, 'a') as f:
                    f.write(msg + '\n')
    
    def on_train_end(self, trainer, result) -> None:
        """Log training summary."""
        total_time = time.time() - self.train_start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save history to file
        if self.log_to_file:
            history_path = Path(self.log_to_file).with_suffix('.json')
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)


class TensorBoardCallback(Callback):
    """
    TensorBoard logging callback.
    
    Logs training metrics to TensorBoard for visualization.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> callback = TensorBoardCallback(log_dir='./runs/experiment1')
    >>> trainer = Trainer(model, config, callbacks=[callback])
    >>> # View with: tensorboard --logdir=./runs
    """
    
    def __init__(
        self,
        log_dir: str = './runs',
        comment: str = '',
        log_model_graph: bool = True
    ):
        """
        Initialize TensorBoard callback.
        
        Parameters:
        -----------
        log_dir : str
            Directory for TensorBoard logs
        comment : str
            Comment to add to log directory name
        log_model_graph : bool
            Whether to log model architecture graph
        """
        self.log_dir = log_dir
        self.comment = comment
        self.log_model_graph = log_model_graph
        self.writer = None
        self._tensorboard_available = True
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._SummaryWriter = SummaryWriter
        except ImportError:
            logger.warning(
                "TensorBoard not available. "
                "Install with: pip install tensorboard"
            )
            self._tensorboard_available = False
    
    def on_train_begin(self, trainer) -> None:
        """Initialize TensorBoard writer."""
        if not self._tensorboard_available:
            return
        
        self.writer = self._SummaryWriter(
            log_dir=self.log_dir,
            comment=self.comment
        )
        
        # Log model graph
        if self.log_model_graph:
            try:
                dummy_input = torch.randn(1, 100, 9).to(trainer.device)
                self.writer.add_graph(trainer.model, dummy_input)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        # Log losses
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
        
        # Log learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Log gradients histogram
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f'Gradients/{name}',
                    param.grad.cpu().numpy(),
                    epoch
                )
    
    def on_train_end(self, trainer, result) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class GradientMonitor(Callback):
    """
    Gradient monitoring callback.
    
    Monitors gradient norms to detect vanishing/exploding gradients.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        warn_threshold: float = 100.0
    ):
        """
        Initialize gradient monitor.
        
        Parameters:
        -----------
        log_interval : int
            Steps between gradient logging
        warn_threshold : float
            Gradient norm threshold for warning
        """
        self.log_interval = log_interval
        self.warn_threshold = warn_threshold
        self.gradient_history = []
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float
    ) -> None:
        """Monitor gradients after each batch."""
        if batch_idx % self.log_interval != 0:
            return
        
        total_norm = 0.0
        for param in trainer.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_history.append(total_norm)
        
        if total_norm > self.warn_threshold:
            logger.warning(
                f"Large gradient norm detected: {total_norm:.4f}"
            )
        elif total_norm < 1e-7:
            logger.warning(
                f"Very small gradient norm: {total_norm:.8f}"
            )


class MetricLogger(Callback):
    """
    Custom metric logging callback.
    
    Logs custom metrics during training.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        metrics: Dict[str, Callable],
        log_interval: int = 1
    ):
        """
        Initialize metric logger.
        
        Parameters:
        -----------
        metrics : Dict[str, Callable]
            Dictionary of metric name -> function
        log_interval : int
            Epochs between logging
        """
        self.metrics = metrics
        self.log_interval = log_interval
        self.metric_history = {name: [] for name in metrics}
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float]
    ) -> None:
        """Compute and log custom metrics."""
        if (epoch + 1) % self.log_interval != 0:
            return
        
        for name, metric_fn in self.metrics.items():
            try:
                value = metric_fn(trainer)
                self.metric_history[name].append(value)
                logger.info(f"Epoch {epoch}: {name} = {value:.6f}")
            except Exception as e:
                logger.warning(f"Error computing metric {name}: {e}")


if __name__ == "__main__":
    # Test callbacks
    print("Testing Callbacks")
    print("=" * 50)
    
    # Test EarlyStopping
    early_stop = EarlyStopping(patience=5)
    print(f"EarlyStopping created with patience={early_stop.patience}")
    
    # Test LearningRateScheduler
    scheduler = LearningRateScheduler(
        decay_epochs=[80, 120, 170],
        decay_factor=0.005
    )
    print(f"LearningRateScheduler created")
    
    # Test ModelCheckpoint
    checkpoint = ModelCheckpoint(
        checkpoint_dir='./test_checkpoints',
        save_best_only=True
    )
    print(f"ModelCheckpoint created")
    
    # Test ProgressCallback
    progress = ProgressCallback(log_interval=10)
    print(f"ProgressCallback created")
    
    print("\nAll callbacks created successfully!")
