"""
Mode Valuation Utilities
========================

Utilities for analyzing and ranking modes based on their contribution
to prediction performance.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

This module provides:
1. ModeValuation: Analysis of mode importance using various metrics
2. ModeRanker: Ranking modes based on different criteria
3. ModeSelector: Selecting optimal subset of modes
"""

import numpy as np
from typing import List, Optional, Callable, Dict, Tuple, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class ModeStatistics:
    """
    Statistics for a single mode.
    
    Attributes:
        index: Mode index
        energy: Signal energy of the mode
        center_frequency: Center frequency
        bandwidth: Estimated bandwidth
        snr: Signal-to-noise ratio estimate
        entropy: Spectral entropy
        correlation_with_signal: Correlation with original signal
    """
    index: int
    energy: float
    center_frequency: float
    bandwidth: float = 0.0
    snr: float = 0.0
    entropy: float = 0.0
    correlation_with_signal: float = 0.0


class ModeValuation:
    """
    Analysis and valuation of modes from DMD.
    
    This class provides methods for analyzing the importance of modes
    beyond Shapley values, using signal processing metrics.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, fs: float = 1.0):
        """
        Initialize mode valuation.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency
        """
        self.fs = fs
    
    def compute_statistics(
        self,
        modes: np.ndarray,
        center_frequencies: np.ndarray,
        original_signal: Optional[np.ndarray] = None
    ) -> List[ModeStatistics]:
        """
        Compute comprehensive statistics for all modes.
        """
        M = len(modes)
        statistics = []
        
        for i in range(M):
            mode = modes[i]
            omega = center_frequencies[i]
            
            energy = self._compute_energy(mode)
            bandwidth = self._compute_bandwidth(mode, omega)
            snr = self._estimate_snr(mode)
            entropy = self._compute_spectral_entropy(mode)
            
            if original_signal is not None:
                correlation = self._compute_correlation(mode, original_signal)
            else:
                correlation = 0.0
            
            stats = ModeStatistics(
                index=i,
                energy=energy,
                center_frequency=omega * self.fs / (2 * np.pi),
                bandwidth=bandwidth * self.fs / (2 * np.pi),
                snr=snr,
                entropy=entropy,
                correlation_with_signal=correlation
            )
            statistics.append(stats)
        
        return statistics
    
    def _compute_energy(self, mode: np.ndarray) -> float:
        """Compute signal energy of a mode."""
        return np.sum(mode ** 2)
    
    def _compute_bandwidth(self, mode: np.ndarray, center_freq: float) -> float:
        """Estimate bandwidth of a mode."""
        from numpy.fft import fft, fftfreq
        
        N = len(mode)
        X = fft(mode)
        power_spectrum = np.abs(X) ** 2
        freqs = 2 * np.pi * fftfreq(N)
        
        total_power = np.sum(power_spectrum)
        if total_power < 1e-10:
            return 0.0
        
        mean_freq = np.sum(freqs * power_spectrum) / total_power
        rms_bandwidth = np.sqrt(np.sum((freqs - mean_freq) ** 2 * power_spectrum) / total_power)
        
        return rms_bandwidth
    
    def _estimate_snr(self, mode: np.ndarray) -> float:
        """Estimate SNR of a mode."""
        diff = np.diff(mode)
        mad = np.median(np.abs(diff - np.median(diff)))
        noise_std = mad / 0.6745
        
        signal_power = np.var(mode)
        noise_power = noise_std ** 2 + 1e-10
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def _compute_spectral_entropy(self, mode: np.ndarray) -> float:
        """Compute spectral entropy of a mode."""
        from numpy.fft import fft
        
        X = fft(mode)
        power_spectrum = np.abs(X) ** 2
        power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
        
        return entropy
    
    def _compute_correlation(self, mode: np.ndarray, signal: np.ndarray) -> float:
        """Compute normalized correlation."""
        mode_norm = mode / (np.linalg.norm(mode) + 1e-10)
        signal_norm = signal / (np.linalg.norm(signal) + 1e-10)
        return np.abs(np.dot(mode_norm, signal_norm))
    
    def compute_reconstruction_importance(
        self,
        modes: np.ndarray,
        signal: np.ndarray
    ) -> np.ndarray:
        """Compute importance based on reconstruction contribution."""
        M = len(modes)
        importance = np.zeros(M)
        
        full_reconstruction = np.sum(modes, axis=0)
        full_error = np.linalg.norm(signal - full_reconstruction) ** 2
        
        for i in range(M):
            loo_modes = np.delete(modes, i, axis=0)
            if len(loo_modes) > 0:
                loo_reconstruction = np.sum(loo_modes, axis=0)
            else:
                loo_reconstruction = np.zeros_like(signal)
            
            loo_error = np.linalg.norm(signal - loo_reconstruction) ** 2
            importance[i] = loo_error - full_error
        
        importance = importance / (np.sum(np.abs(importance)) + 1e-10)
        return importance


class ModeRanker:
    """
    Rank modes based on various criteria.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, method: str = 'shapley'):
        """Initialize mode ranker."""
        self.method = method
    
    def rank(
        self,
        modes: np.ndarray,
        shapley_values: Optional[np.ndarray] = None,
        statistics: Optional[List[ModeStatistics]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Rank modes from most to least important."""
        if self.method == 'shapley':
            if shapley_values is None:
                raise ValueError("shapley_values required")
            return np.argsort(shapley_values)[::-1]
        
        elif self.method == 'energy':
            energies = np.sum(modes ** 2, axis=1)
            return np.argsort(energies)[::-1]
        
        elif self.method == 'correlation':
            if statistics is None:
                raise ValueError("statistics required")
            correlations = np.array([s.correlation_with_signal for s in statistics])
            return np.argsort(correlations)[::-1]
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ModeSelector:
    """
    Select optimal subset of modes for prediction.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        selection_method: str = 'threshold',
        threshold: float = 0.01,
        max_modes: Optional[int] = None
    ):
        """Initialize mode selector."""
        self.selection_method = selection_method
        self.threshold = threshold
        self.max_modes = max_modes
    
    def select(
        self,
        shapley_values: np.ndarray,
        rankings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select subset of modes based on Shapley values."""
        M = len(shapley_values)
        
        if rankings is None:
            rankings = np.argsort(shapley_values)[::-1]
        
        if self.selection_method == 'threshold':
            selected = np.where(shapley_values > self.threshold)[0]
            
        elif self.selection_method == 'top_k':
            k = self.max_modes or max(1, M // 2)
            selected = rankings[:k]
            
        elif self.selection_method == 'cumulative':
            sorted_values = shapley_values[rankings]
            cumulative = np.cumsum(sorted_values) / (np.sum(sorted_values) + 1e-10)
            num_selected = np.searchsorted(cumulative, self.threshold) + 1
            selected = rankings[:num_selected]
            
        else:
            raise ValueError(f"Unknown method: {self.selection_method}")
        
        if self.max_modes is not None and len(selected) > self.max_modes:
            selected_ranks = [np.where(rankings == s)[0][0] for s in selected]
            sorted_indices = np.argsort(selected_ranks)
            selected = selected[sorted_indices[:self.max_modes]]
        
        if len(selected) == 0:
            selected = rankings[:1]
        
        return selected


if __name__ == "__main__":
    np.random.seed(42)
    
    N = 1000
    M = 5
    t = np.linspace(0, 1, N)
    
    freqs = [5, 10, 20, 35, 50]
    importance = [0.4, 0.3, 0.15, 0.1, 0.05]
    
    modes = np.array([
        importance[i] * np.sin(2 * np.pi * freqs[i] * t)
        for i in range(M)
    ])
    center_freqs = np.array([2 * np.pi * f / N for f in freqs])
    
    signal = np.sum(modes, axis=0) + 0.05 * np.random.randn(N)
    
    valuation = ModeValuation(fs=N)
    statistics = valuation.compute_statistics(modes, center_freqs, signal)
    
    print("Mode Statistics:")
    for stats in statistics:
        print(f"  Mode {stats.index}: Energy={stats.energy:.4f}, SNR={stats.snr:.2f}dB")
    
    recon_importance = valuation.compute_reconstruction_importance(modes, signal)
    print(f"\nReconstruction importance: {recon_importance}")
    print(f"True importance: {importance}")
