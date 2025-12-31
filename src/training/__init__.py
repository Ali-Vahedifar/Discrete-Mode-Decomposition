"""
Training Module for DMD+SMV Signal Prediction
==============================================

Comprehensive training utilities including trainer classes,
loss functions, optimizers, and callbacks.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    TrainingResult
)
from .losses import (
    MSELoss,
    MAELoss,
    HuberLoss,
    PSNRLoss,
    CombinedLoss,
    create_loss_function
)
from .callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ProgressCallback,
    TensorBoardCallback
)

__all__ = [
    'Trainer',
    'TrainingConfig',
    'TrainingResult',
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'PSNRLoss',
    'CombinedLoss',
    'create_loss_function',
    'Callback',
    'EarlyStopping',
    'LearningRateScheduler',
    'ModelCheckpoint',
    'ProgressCallback',
    'TensorBoardCallback'
]

__author__ = "Ali Vahedi (Mohammad Ali Vahedifar)"
__email__ = "av@ece.au.dk"
__institution__ = "Aarhus University, Denmark"
