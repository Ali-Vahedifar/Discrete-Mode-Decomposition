"""
DMD+SMV Tactile Internet Package
================================

Discrete Mode Decomposition meets Shapley Value for Robust Signal Prediction
in Tactile Internet.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

Modules:
--------
- dmd: Discrete Mode Decomposition algorithms
- smv: Shapley Mode Value computation
- models: Neural network architectures (Transformer, ResNet, LSTM)
- data: Dataset handling and preprocessing
- training: Training utilities
- evaluation: Evaluation metrics and inference
- utils: General utilities
"""

__version__ = "1.0.0"
__author__ = "Ali Vahedi"
__email__ = "av@ece.au.dk"
__license__ = "MIT"
__copyright__ = "Copyright 2025, Ali Vahedi and Qi Zhang, Aarhus University"

from . import dmd
from . import smv
from . import models
from . import data
from . import training
from . import evaluation
from . import utils

__all__ = [
    "dmd",
    "smv",
    "models",
    "data",
    "training",
    "evaluation",
    "utils",
    "__version__",
    "__author__",
    "__email__",
]
