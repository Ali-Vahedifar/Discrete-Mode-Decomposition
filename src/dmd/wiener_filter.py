"""
Discrete Wiener Filter Implementation
=====================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
IEEE INFOCOM 2025
This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

Implementation of the Discrete Wiener Filter for signal denoising as part
of the DMD algorithm.

Mathematical Background (Equations 3-5):
---------------------------------------
Consider the observed discrete-time signal y[n], a version of the original
signal x[n] corrupted by additive zero-mean white Gaussian noise η[n]:

    y[n] = x[n] + η[n],  η[n] ~ N(0, α)                    (Eq. 3)

The denoising problem is formulated as discrete Tikhonov regularization:

    min_x { ||x[n] - y[n]||²₂ + α||∂_n x[n]||²₂ }         (Eq. 4)

The solution in the discrete Fourier domain is (Eq. 5):

    X(ω) = Y(ω) / (1 + α|ω|²)

where α represents the variance of white noise.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from typing import Optional, Tuple, Union
from scipy.signal import wiener as scipy_wiener


class DiscreteWienerFilter:
    """
    Discrete Wiener Filter for signal denoising.
    
    This class implements the discrete Wiener filter based on Tikhonov
    regularization, optimized for the DMD algorithm.
    
    The filter acts as a low-pass filter with a power spectrum prior of 1/|ω|².
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    
    Attributes:
        noise_variance: Estimated variance of additive noise (alpha)
        use_adaptive: Whether to adaptively estimate noise variance
        
    Example:
    --------
    >>> wiener = DiscreteWienerFilter(noise_variance=0.01)
    >>> noisy_signal = clean_signal + 0.1 * np.random.randn(len(clean_signal))
    >>> denoised = wiener.filter(noisy_signal)
    """
    
    def __init__(
        self,
        noise_variance: float = 0.01,
        use_adaptive: bool = False,
        min_variance: float = 1e-10
    ):
        """
        Initialize the Wiener filter.
        
        Parameters:
        -----------
        noise_variance : float
            Estimated variance of additive noise (alpha in the paper).
            Default is 0.01.
        use_adaptive : bool
            If True, estimate noise variance from the signal.
            Default is False.
        min_variance : float
            Minimum variance to prevent division by zero.
            Default is 1e-10.
        """
        self.noise_variance = noise_variance
        self.use_adaptive = use_adaptive
        self.min_variance = min_variance
        self._last_estimated_variance = None
        
    def filter(
        self,
        signal: np.ndarray,
        noise_variance: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Wiener filter to denoise a signal.
        
        Implements Equation 4 from the paper:
            X[ω] = Y[ω] / (1 + α|ω|²)
        
        Parameters:
        -----------
        signal : np.ndarray
            Input noisy signal of shape (N,) or (N, C).
        noise_variance : float, optional
            Override noise variance for this call.
            
        Returns:
        --------
        np.ndarray
            Denoised signal with same shape as input.
        """
        if signal.ndim == 1:
            return self._filter_1d(signal, noise_variance)
        elif signal.ndim == 2:
            # Multi-channel: filter each channel
            filtered = np.zeros_like(signal)
            for c in range(signal.shape[1]):
                filtered[:, c] = self._filter_1d(signal[:, c], noise_variance)
            return filtered
        else:
            raise ValueError(f"Signal must be 1D or 2D, got shape {signal.shape}")
    
    def _filter_1d(
        self,
        signal: np.ndarray,
        noise_variance: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Wiener filter to a 1D signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal of shape (N,)
        noise_variance : float, optional
            Override noise variance
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        N = len(signal)
        
        # Use provided variance or estimate
        if noise_variance is not None:
            alpha = noise_variance
        elif self.use_adaptive:
            alpha = self._estimate_noise_variance(signal)
            self._last_estimated_variance = alpha
        else:
            alpha = self.noise_variance
        
        # Ensure minimum variance
        alpha = max(alpha, self.min_variance)
        
        # Compute DFT
        Y = fft(signal)
        
        # Frequency axis (normalized to [0, 2π])
        omega = 2 * np.pi * np.arange(N) / N
        # Center frequencies around 0 for symmetric filter
        omega = np.where(omega > np.pi, omega - 2 * np.pi, omega)
        
        # Wiener filter transfer function - Eq. 4
        # H(ω) = 1 / (1 + α|ω|²)
        H = 1.0 / (1.0 + alpha * omega ** 2)
        
        # Apply filter in frequency domain
        X = Y * H
        
        # Return to time domain
        return np.real(ifft(X))
    
    def _estimate_noise_variance(self, signal: np.ndarray) -> float:
        """
        Estimate noise variance from the signal.
        
        Uses the Median Absolute Deviation (MAD) estimator which is
        robust to outliers.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        float
            Estimated noise variance
        """
        # Use high-frequency components to estimate noise
        # Apply difference operator (high-pass)
        diff = np.diff(signal)
        
        # MAD estimator for Gaussian noise
        # σ = MAD / 0.6745 for Gaussian distribution
        mad = np.median(np.abs(diff - np.median(diff)))
        sigma = mad / 0.6745
        
        return sigma ** 2
    
    def get_transfer_function(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the transfer function of the Wiener filter.
        
        Parameters:
        -----------
        N : int
            Number of frequency points
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Frequency axis and transfer function magnitude
        """
        omega = 2 * np.pi * np.arange(N) / N
        omega = np.where(omega > np.pi, omega - 2 * np.pi, omega)
        
        H = 1.0 / (1.0 + self.noise_variance * omega ** 2)
        
        return omega, np.abs(H)
    
    def compute_optimal_variance(
        self,
        noisy_signal: np.ndarray,
        clean_signal: Optional[np.ndarray] = None,
        search_range: Tuple[float, float] = (1e-6, 1.0),
        num_points: int = 100
    ) -> float:
        """
        Find optimal noise variance by grid search.
        
        If clean signal is provided, minimizes MSE.
        Otherwise, uses cross-validation.
        
        Parameters:
        -----------
        noisy_signal : np.ndarray
            Noisy input signal
        clean_signal : np.ndarray, optional
            Ground truth clean signal
        search_range : Tuple[float, float]
            Range of variance values to search
        num_points : int
            Number of points in grid search
            
        Returns:
        --------
        float
            Optimal noise variance
        """
        variances = np.logspace(
            np.log10(search_range[0]),
            np.log10(search_range[1]),
            num_points
        )
        
        if clean_signal is not None:
            # Minimize MSE with clean signal
            errors = []
            for var in variances:
                filtered = self.filter(noisy_signal, noise_variance=var)
                mse = np.mean((filtered - clean_signal) ** 2)
                errors.append(mse)
            
            optimal_var = variances[np.argmin(errors)]
        else:
            # Use leave-one-out cross-validation
            errors = []
            N = len(noisy_signal)
            
            for var in variances:
                cv_error = 0
                for i in range(0, N, max(1, N // 20)):  # Sample subset for speed
                    # Leave one out
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    
                    # Interpolate missing point
                    signal_cv = noisy_signal.copy()
                    if i > 0 and i < N - 1:
                        signal_cv[i] = (noisy_signal[i-1] + noisy_signal[i+1]) / 2
                    
                    # Filter
                    filtered = self.filter(signal_cv, noise_variance=var)
                    
                    # Prediction error
                    cv_error += (filtered[i] - noisy_signal[i]) ** 2
                
                errors.append(cv_error)
            
            optimal_var = variances[np.argmin(errors)]
        
        return optimal_var


class AdaptiveWienerFilter(DiscreteWienerFilter):
    """
    Adaptive Wiener Filter with local noise estimation.
    
    Estimates noise variance locally using a sliding window approach.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        window_size: int = 32,
        min_variance: float = 1e-10
    ):
        """
        Initialize adaptive Wiener filter.
        
        Parameters:
        -----------
        window_size : int
            Size of sliding window for local noise estimation
        min_variance : float
            Minimum variance to prevent division by zero
        """
        super().__init__(use_adaptive=True, min_variance=min_variance)
        self.window_size = window_size
    
    def filter(
        self,
        signal: np.ndarray,
        noise_variance: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply adaptive Wiener filter using local variance estimation.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input noisy signal
        noise_variance : float, optional
            Global noise variance (if known)
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        N = len(signal)
        
        # Use scipy's built-in adaptive Wiener filter for comparison
        # This uses local mean and variance estimation
        filtered_scipy = scipy_wiener(signal, mysize=self.window_size)
        
        # Also apply our frequency-domain approach with adaptive estimation
        # Estimate global noise variance
        if noise_variance is None:
            noise_variance = self._estimate_noise_variance(signal)
        
        # Apply standard Wiener filter
        filtered_freq = super()._filter_1d(signal, noise_variance)
        
        # Blend both approaches (weighted average)
        # The scipy version handles non-stationary noise better
        # Our version handles periodic components better
        blend_weight = 0.5
        filtered = blend_weight * filtered_scipy + (1 - blend_weight) * filtered_freq
        
        return filtered


class BandpassWienerFilter(DiscreteWienerFilter):
    """
    Bandpass Wiener Filter for mode extraction.
    
    Combines Wiener denoising with bandpass filtering around a center frequency.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        noise_variance: float = 0.01,
        bandwidth: float = 0.1
    ):
        """
        Initialize bandpass Wiener filter.
        
        Parameters:
        -----------
        noise_variance : float
            Noise variance parameter
        bandwidth : float
            Normalized bandwidth (0 to 1)
        """
        super().__init__(noise_variance=noise_variance)
        self.bandwidth = bandwidth
    
    def filter_around_frequency(
        self,
        signal: np.ndarray,
        center_frequency: float
    ) -> np.ndarray:
        """
        Apply bandpass Wiener filter centered at a specific frequency.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        center_frequency : float
            Normalized center frequency (0 to π)
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        N = len(signal)
        
        # Compute DFT
        Y = fft(signal)
        
        # Frequency axis
        omega = 2 * np.pi * np.arange(N) / N
        omega = np.where(omega > np.pi, omega - 2 * np.pi, omega)
        
        # Wiener filter
        H_wiener = 1.0 / (1.0 + self.noise_variance * omega ** 2)
        
        # Bandpass filter (Gaussian)
        sigma = self.bandwidth * np.pi
        H_bandpass = np.exp(-((omega - center_frequency) ** 2) / (2 * sigma ** 2))
        H_bandpass += np.exp(-((omega + center_frequency) ** 2) / (2 * sigma ** 2))
        
        # Combined filter
        H = H_wiener * H_bandpass
        
        # Apply filter
        X = Y * H
        
        return np.real(ifft(X))


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create test signal
    np.random.seed(42)
    N = 1000
    t = np.linspace(0, 1, N)
    
    # Clean signal
    clean = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    
    # Add noise
    noise_level = 0.3
    noisy = clean + noise_level * np.random.randn(N)
    
    # Apply Wiener filter
    wiener = DiscreteWienerFilter(noise_variance=noise_level ** 2)
    filtered = wiener.filter(noisy)
    
    # Compute optimal variance
    optimal_var = wiener.compute_optimal_variance(noisy, clean)
    filtered_optimal = wiener.filter(noisy, noise_variance=optimal_var)
    
    # Compute errors
    mse_noisy = np.mean((noisy - clean) ** 2)
    mse_filtered = np.mean((filtered - clean) ** 2)
    mse_optimal = np.mean((filtered_optimal - clean) ** 2)
    
    print(f"Noise variance: {noise_level ** 2:.4f}")
    print(f"Optimal variance: {optimal_var:.4f}")
    print(f"MSE (noisy): {mse_noisy:.4f}")
    print(f"MSE (filtered): {mse_filtered:.4f}")
    print(f"MSE (optimal): {mse_optimal:.4f}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    axes[0].plot(t, clean, 'b-', label='Clean')
    axes[0].plot(t, noisy, 'r-', alpha=0.5, label='Noisy')
    axes[0].set_title(f'Original Signals (MSE = {mse_noisy:.4f})')
    axes[0].legend()
    
    axes[1].plot(t, clean, 'b-', label='Clean')
    axes[1].plot(t, filtered, 'g-', label='Filtered')
    axes[1].set_title(f'Wiener Filtered (MSE = {mse_filtered:.4f})')
    axes[1].legend()
    
    axes[2].plot(t, clean, 'b-', label='Clean')
    axes[2].plot(t, filtered_optimal, 'm-', label='Optimal Filtered')
    axes[2].set_title(f'Optimal Variance (MSE = {mse_optimal:.4f})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('wiener_filter_example.png', dpi=150)
    print("\nSaved example to 'wiener_filter_example.png'")
