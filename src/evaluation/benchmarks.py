"""
Benchmarking Utilities for Signal Prediction
=============================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .metrics import compute_accuracy, compute_psnr, MetricsComputer
from .inference import InferenceEngine, InferenceConfig


@dataclass
class BenchmarkResult:
    """Benchmark result container. Author: Ali Vahedi"""
    method: str
    architecture: str
    window_size: int
    accuracy_human: float
    accuracy_robot: float
    inference_time_ms: float
    psnr_human: float = 0.0
    psnr_robot: float = 0.0
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class Benchmark:
    """
    Benchmarking class for DMD+SMV experiments.
    
    Reproduces the experiments from the IEEE INFOCOM 2025 paper.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, save_dir: str = './results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def run_experiment(
        self,
        model: nn.Module,
        test_data_human: np.ndarray,
        test_data_robot: np.ndarray,
        method: str,
        architecture: str,
        window_size: int
    ) -> BenchmarkResult:
        """Run a single experiment."""
        engine = InferenceEngine(model)
        
        # Predict human side
        x_human = torch.from_numpy(test_data_human[0]).float().unsqueeze(0)
        y_human = test_data_human[1]
        result_human = engine.predict(x_human)
        acc_human = compute_accuracy(result_human.predictions, y_human)
        psnr_human = compute_psnr(result_human.predictions, y_human)
        
        # Predict robot side
        x_robot = torch.from_numpy(test_data_robot[0]).float().unsqueeze(0)
        y_robot = test_data_robot[1]
        result_robot = engine.predict(x_robot)
        acc_robot = compute_accuracy(result_robot.predictions, y_robot)
        psnr_robot = compute_psnr(result_robot.predictions, y_robot)
        
        result = BenchmarkResult(
            method=method,
            architecture=architecture,
            window_size=window_size,
            accuracy_human=acc_human,
            accuracy_robot=acc_robot,
            inference_time_ms=result_human.inference_time_ms,
            psnr_human=psnr_human,
            psnr_robot=psnr_robot
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save results to JSON."""
        path = self.save_dir / filename
        with open(path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"Results saved to {path}")


def run_window_size_benchmark(
    model: nn.Module,
    data: np.ndarray,
    window_sizes: List[int] = [1, 5, 10, 25, 50, 100]
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark model across different window sizes.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    results = {}
    engine = InferenceEngine(model)
    
    for w in window_sizes:
        x = torch.from_numpy(data[:, :w, :]).float()
        result = engine.predict(x)
        
        results[w] = {
            'inference_time_ms': result.inference_time_ms,
            'throughput': result.throughput
        }
        print(f"W={w}: {result.inference_time_ms:.3f} ms")
    
    return results


def run_architecture_benchmark(
    models: Dict[str, nn.Module],
    test_data: np.ndarray,
    targets: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different architectures.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    results = {}
    
    for name, model in models.items():
        engine = InferenceEngine(model)
        x = torch.from_numpy(test_data).float()
        result = engine.predict(x)
        
        results[name] = {
            'accuracy': compute_accuracy(result.predictions, targets),
            'psnr': compute_psnr(result.predictions, targets),
            'inference_time_ms': result.inference_time_ms
        }
        print(f"{name}: Acc={results[name]['accuracy']:.2f}%, "
              f"PSNR={results[name]['psnr']:.2f}dB, "
              f"Time={results[name]['inference_time_ms']:.3f}ms")
    
    return results


def compare_methods(
    baseline_results: Dict,
    dmd_results: Dict,
    dmd_smv_results: Dict
) -> Dict[str, float]:
    """
    Compare methods and compute speedup ratios.
    
    Paper reports:
    - DMD+SMV vs Baseline: 820x speedup (human), 374x (robot)
    - DMD+SMV vs DMD: 3x speedup (human), 2x (robot)
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    comparison = {
        'dmd_smv_vs_baseline_speedup': baseline_results['inference_time_ms'] / 
                                       dmd_smv_results['inference_time_ms'],
        'dmd_smv_vs_dmd_speedup': dmd_results['inference_time_ms'] / 
                                  dmd_smv_results['inference_time_ms'],
        'dmd_smv_accuracy': dmd_smv_results['accuracy'],
        'dmd_accuracy': dmd_results['accuracy'],
        'baseline_accuracy': baseline_results['accuracy'],
        'accuracy_improvement': dmd_smv_results['accuracy'] - baseline_results['accuracy']
    }
    
    print("\n" + "="*50)
    print("Method Comparison - Author: Ali Vahedi")
    print("="*50)
    print(f"DMD+SMV vs Baseline speedup: {comparison['dmd_smv_vs_baseline_speedup']:.1f}x")
    print(f"DMD+SMV vs DMD speedup: {comparison['dmd_smv_vs_dmd_speedup']:.1f}x")
    print(f"Accuracy improvement: {comparison['accuracy_improvement']:.2f}%")
    
    return comparison


if __name__ == "__main__":
    print("Testing Benchmarks - Author: Ali Vahedi")
    print("=" * 50)
    print("Benchmark utilities loaded successfully!")
