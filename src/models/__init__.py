"""
Neural Network Models Module
============================

Implementation of neural network architectures for haptic signal prediction.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2026

Architectures:
--------------
- TransformerPredictor: Transformer with attention mechanism
- ResNetPredictor: ResNet-32 architecture
- LSTMPredictor: LSTM with stacked layers
- BasePredictor: Base class for all predictors
"""

from .base_model import BasePredictor, ModelConfig
from .transformer import TransformerPredictor
from .resnet import ResNetPredictor
from .lstm import LSTMPredictor

__all__ = [
    "BasePredictor",
    "ModelConfig",
    "TransformerPredictor",
    "ResNetPredictor",
    "LSTMPredictor",
]
