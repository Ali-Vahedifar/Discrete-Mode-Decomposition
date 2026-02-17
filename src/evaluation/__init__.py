"""
Evaluation Module for DMD+SMV Signal Prediction
================================================

Comprehensive evaluation utilities including metrics computation,
inference engine, and benchmarking tools.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2026: "Discrete Mode Decomposition Meets Shapley Value:
Robust Signal Prediction in Tactile Internet"
"""

from .metrics import (
    compute_accuracy,
    compute_error,
    compute_psnr,
    compute_mse,
    compute_mae,
    compute_rmse,
    EvaluationMetrics,
    MetricsComputer
)
from .inference import (
    InferenceEngine,
    InferenceConfig,
    InferenceResult,
    measure_inference_time
)
from .benchmarks import (
    Benchmark,
    BenchmarkResult,
    run_window_size_benchmark,
    run_architecture_benchmark,
    compare_methods
)

__all__ = [
    # Metrics
    'compute_accuracy',
    'compute_error',
    'compute_psnr',
    'compute_mse',
    'compute_mae',
    'compute_rmse',
    'EvaluationMetrics',
    'MetricsComputer',
    
    # Inference
    'InferenceEngine',
    'InferenceConfig',
    'InferenceResult',
    'measure_inference_time',
    
    # Benchmarks
    'Benchmark',
    'BenchmarkResult',
    'run_window_size_benchmark',
    'run_architecture_benchmark',
    'compare_methods'
]

__author__ = "Ali Vahedi"
__email__ = "av@ece.au.dk"
__institution__ = "Aarhus University, Denmark"
