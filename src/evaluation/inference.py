"""
Inference Engine for Signal Prediction
=======================================

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Inference configuration. Author: Ali Vahedi"""
    device: str = 'auto'
    warmup_runs: int = 10
    timing_runs: int = 100


@dataclass
class InferenceResult:
    """Inference result. Author: Ali Vahedi"""
    predictions: np.ndarray
    inference_time_ms: float
    throughput: float


class InferenceEngine:
    """
    Inference engine for signal prediction.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, model: nn.Module, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if self.config.device == 'auto' else torch.device(self.config.device)
        self.model = model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, x: Union[torch.Tensor, np.ndarray], 
                modes: Optional[torch.Tensor] = None) -> InferenceResult:
        """Run inference with timing."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)
        
        if modes is not None:
            if isinstance(modes, np.ndarray):
                modes = torch.from_numpy(modes).float()
            modes = modes.to(self.device)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            _ = self.model(x, modes) if modes is not None else self.model(x)
        
        # Timed runs
        times = []
        for _ in range(self.config.timing_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = self.model(x, modes) if modes is not None else self.model(x)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        throughput = (x.shape[0] / avg_time) * 1000
        
        return InferenceResult(
            predictions=predictions.cpu().numpy(),
            inference_time_ms=avg_time,
            throughput=throughput
        )


def measure_inference_time(model: nn.Module, input_shape: Tuple[int, ...],
                           num_runs: int = 100) -> Dict[str, float]:
    """Measure inference time. Author: Ali Vahedi"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    x = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    return {'mean_ms': np.mean(times), 'std_ms': np.std(times)}
