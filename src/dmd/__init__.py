"""
Discrete Mode Decomposition (DMD) Module
========================================

This module implements the Discrete Mode Decomposition algorithm for decomposing
discrete-time signals into fundamental intrinsic modes.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

Key Components:
--------------
- DiscreteModeDcomposition: Main DMD algorithm
- DiscreteWienerFilter: Discrete Wiener filtering for denoising
- DiscreteHilbertTransform: Hilbert transform for analytic signals
- ADMMOptimizer: ADMM-based optimization for mode extraction
"""

from .decomposition import DiscreteModeDcomposition
from .wiener_filter import DiscreteWienerFilter
from .hilbert_transform import DiscreteHilbertTransform
from .optimization import ADMMOptimizer

__all__ = [
    "DiscreteModeDcomposition",
    "DiscreteWienerFilter",
    "DiscreteHilbertTransform",
    "ADMMOptimizer",
]
