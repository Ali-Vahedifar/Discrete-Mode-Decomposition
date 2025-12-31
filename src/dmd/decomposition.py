"""
Discrete Mode Decomposition (DMD) Algorithm
============================================

Implementation of the Discrete Mode Decomposition algorithm for decomposing
discrete-time signals into fundamental intrinsic modes (IMFs).

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

This module implements the DMD algorithm as described in:
"Discrete Mode Decomposition Meets Shapley Value: Robust Signal Prediction
in Tactile Internet" - IEEE INFOCOM 2025

Mathematical Background:
-----------------------
For a discrete-time signal x[n], DMD decomposes it into M modes:
    x[n] = sum_{k=1}^{M} u_k[n] + x_u[n]

where each mode u_k[n] is an Intrinsic Mode Function (IMF):
    u_k[n] = A_k[n] * cos(phi_k[n])

The decomposition is achieved through an optimization problem:
    min_{u_M, omega_M} T1 + T2 + T3
    s.t. x[n] = sum_{i=1}^{M} u_i[n] + x_u[n]
         ||x_u[n]||_2^2 <= ||u_min[n]||_2^2  (energy-based bound, Eq. 16)

where:
    T1: Spectral compactness of M-th mode
    T2: Minimum overlap with previously extracted modes
    T3: Minimum spectral overlap with unprocessed signal
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import warnings
from tqdm import tqdm

from .wiener_filter import DiscreteWienerFilter
from .hilbert_transform import DiscreteHilbertTransform
from .optimization import ADMMOptimizer


@dataclass
class DMDConfig:
    """Configuration for Discrete Mode Decomposition.
    
    Attributes:
        noise_variance: Variance of additive white Gaussian noise (alpha)
        epsilon1: Regularization constant for previous modes overlap (default: 1e-6)
        epsilon2: Regularization constant for unprocessed signal overlap (default: 1e-6)
        tau1: Step size for equality constraint update (default: 0.1)
        tau2: Step size for inequality constraint update (default: 0.1)
        kappa1: Convergence threshold for inner loop (default: 1e-3)
        kappa2: Convergence threshold for outer loop (default: 1e-3)
        max_modes: Maximum number of modes to extract (default: 10)
        max_inner_iterations: Maximum iterations for inner ADMM loop (default: 500)
        rho_init: Initial penalty parameter (default: 1.0)
        verbose: Whether to print progress (default: False)
    """
    noise_variance: float = 0.01
    epsilon1: float = 1e-6
    epsilon2: float = 1e-6
    tau1: float = 0.1
    tau2: float = 0.1
    kappa1: float = 1e-3
    kappa2: float = 1e-3
    max_modes: int = 10
    max_inner_iterations: int = 500
    rho_init: float = 1.0
    verbose: bool = False


@dataclass
class DMDResult:
    """Result of Discrete Mode Decomposition.
    
    Attributes:
        modes: List of extracted mode signals, shape (M, N)
        center_frequencies: List of center frequencies for each mode
        residual: Remaining unprocessed signal after decomposition
        num_modes: Number of extracted modes
        reconstruction_error: L2 norm of reconstruction error
        convergence_history: List of convergence metrics per mode
    """
    modes: np.ndarray
    center_frequencies: np.ndarray
    residual: np.ndarray
    num_modes: int
    reconstruction_error: float
    convergence_history: List[dict] = field(default_factory=list)


class DiscreteModeDcomposition:
    """
    Discrete Mode Decomposition (DMD) for signal analysis.
    
    This class implements the DMD algorithm that decomposes a discrete-time
    signal into a set of Intrinsic Mode Functions (IMFs), each centered
    around a specific frequency.
    
    The algorithm uses ADMM (Alternating Direction Method of Multipliers)
    to solve the constrained optimization problem that ensures:
    1. Spectral compactness of each mode
    2. Minimum overlap between modes
    3. Complete signal reconstruction
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> dmd = DiscreteModeDcomposition(noise_variance=0.01)
    >>> signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    >>> result = dmd.decompose(signal)
    >>> print(f"Extracted {result.num_modes} modes")
    >>> print(f"Center frequencies: {result.center_frequencies}")
    """
    
    def __init__(
        self,
        noise_variance: float = 0.01,
        config: Optional[DMDConfig] = None,
        **kwargs
    ):
        """
        Initialize the DMD decomposer.
        
        Parameters:
        -----------
        noise_variance : float
            Estimated variance of additive noise in the signal.
            This is the alpha parameter in the paper.
        config : DMDConfig, optional
            Configuration object with all parameters.
        **kwargs : dict
            Additional parameters to override config.
        """
        if config is not None:
            self.config = config
        else:
            self.config = DMDConfig(noise_variance=noise_variance, **kwargs)
        
        # Initialize helper classes
        self.wiener_filter = DiscreteWienerFilter(self.config.noise_variance)
        self.hilbert = DiscreteHilbertTransform()
        self.optimizer = ADMMOptimizer(
            tau1=self.config.tau1,
            tau2=self.config.tau2,
            rho_init=self.config.rho_init
        )
        
    def decompose(
        self,
        signal: np.ndarray,
        num_modes: Optional[int] = None
    ) -> DMDResult:
        """
        Decompose a signal into intrinsic modes.
        
        This implements Algorithm 1 from the paper: Discrete Mode Decomposition.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input discrete-time signal of shape (N,) or (N, C) for multi-channel.
        num_modes : int, optional
            Number of modes to extract. If None, extract until convergence.
            
        Returns:
        --------
        DMDResult
            Object containing modes, center frequencies, and metadata.
        """
        # Ensure signal is 1D for now (handle multi-channel separately)
        if signal.ndim == 1:
            return self._decompose_1d(signal, num_modes)
        elif signal.ndim == 2:
            # Multi-channel: decompose each channel independently
            return self._decompose_multichannel(signal, num_modes)
        else:
            raise ValueError(f"Signal must be 1D or 2D, got shape {signal.shape}")
    
    def _decompose_1d(
        self,
        signal: np.ndarray,
        num_modes: Optional[int] = None
    ) -> DMDResult:
        """
        Decompose a 1D signal into intrinsic modes.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal of shape (N,)
        num_modes : int, optional
            Number of modes to extract
            
        Returns:
        --------
        DMDResult
            Decomposition result
        """
        N = len(signal)
        max_modes = num_modes if num_modes else self.config.max_modes
        
        # Initialize storage
        modes = []
        center_frequencies = []
        convergence_history = []
        
        # Current unprocessed signal
        residual = signal.copy()
        
        # Frequency axis for FFT
        freqs = fftfreq(N)
        omega_axis = 2 * np.pi * freqs
        
        # Extract modes iteratively
        M = 0  # Current mode index
        
        pbar = tqdm(range(max_modes), desc="Extracting modes", disable=not self.config.verbose)
        
        for _ in pbar:
            M += 1
            
            # Initialize mode and center frequency
            u_M, omega_M = self._initialize_mode(residual, M)
            
            # Previous modes for overlap constraint
            prev_modes_freq = [fft(m) for m in modes] if modes else []
            prev_omegas = center_frequencies.copy()
            
            # Inner ADMM loop
            mode_convergence = []
            for n_iter in range(self.config.max_inner_iterations):
                # Store previous mode for convergence check
                u_M_prev = u_M.copy()
                
                # Update mode in frequency domain (Eq. 23)
                U_M, omega_M = self._update_mode_frequency(
                    residual, u_M, omega_M, prev_modes_freq, prev_omegas, N
                )
                
                # Convert back to time domain
                u_M = np.real(ifft(U_M))
                
                # Update unprocessed signal magnitude (Eq. 29)
                X_u = self._update_unprocessed(
                    signal, modes + [u_M], omega_M, N
                )
                
                # Update Lagrangian multipliers (Eq. 30, 31)
                self.optimizer.update_multipliers(
                    signal, modes + [u_M], X_u, np.min(np.abs(u_M)) if modes else 1.0
                )
                
                # Check convergence (inner loop)
                mode_diff = np.linalg.norm(u_M - u_M_prev) ** 2
                mode_norm = np.linalg.norm(u_M_prev) ** 2 + 1e-10
                convergence_ratio = mode_diff / mode_norm
                
                mode_convergence.append(convergence_ratio)
                
                if convergence_ratio < self.config.kappa1:
                    break
            
            # Store extracted mode
            modes.append(u_M)
            center_frequencies.append(omega_M)
            convergence_history.append({
                'mode_index': M,
                'iterations': n_iter + 1,
                'final_convergence': convergence_ratio,
                'history': mode_convergence
            })
            
            # Update residual
            residual = signal - np.sum(modes, axis=0)
            
            # Check outer loop convergence
            reconstruction_error = np.linalg.norm(residual) ** 2 / N
            relative_error = np.abs(self.config.noise_variance - reconstruction_error)
            relative_error /= (self.config.noise_variance + 1e-10)
            
            if self.config.verbose:
                pbar.set_postfix({
                    'modes': M,
                    'recon_error': f'{reconstruction_error:.6f}',
                    'rel_error': f'{relative_error:.4f}'
                })
            
            if relative_error < self.config.kappa2:
                break
        
        # Convert to arrays
        modes = np.array(modes)
        center_frequencies = np.array(center_frequencies)
        
        # Final reconstruction error
        reconstruction = np.sum(modes, axis=0) if len(modes) > 0 else np.zeros_like(signal)
        final_residual = signal - reconstruction
        reconstruction_error = np.linalg.norm(final_residual)
        
        return DMDResult(
            modes=modes,
            center_frequencies=center_frequencies,
            residual=final_residual,
            num_modes=len(modes),
            reconstruction_error=reconstruction_error,
            convergence_history=convergence_history
        )
    
    def _decompose_multichannel(
        self,
        signal: np.ndarray,
        num_modes: Optional[int] = None
    ) -> List[DMDResult]:
        """
        Decompose a multi-channel signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal of shape (N, C) where C is number of channels
        num_modes : int, optional
            Number of modes to extract per channel
            
        Returns:
        --------
        List[DMDResult]
            List of decomposition results for each channel
        """
        results = []
        for c in range(signal.shape[1]):
            result = self._decompose_1d(signal[:, c], num_modes)
            results.append(result)
        return results
    
    def _initialize_mode(
        self,
        signal: np.ndarray,
        mode_index: int
    ) -> Tuple[np.ndarray, float]:
        """
        Initialize a mode and its center frequency.
        
        Uses spectral analysis to find a good initial estimate.
        
        Parameters:
        -----------
        signal : np.ndarray
            Current residual signal
        mode_index : int
            Index of the mode being extracted (1-indexed)
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Initial mode estimate and center frequency
        """
        N = len(signal)
        
        # Compute spectrum
        X = fft(signal)
        power_spectrum = np.abs(X) ** 2
        
        # Find dominant frequency (excluding DC)
        positive_freqs = np.arange(1, N // 2)
        peak_idx = positive_freqs[np.argmax(power_spectrum[positive_freqs])]
        
        # Center frequency
        omega_init = 2 * np.pi * peak_idx / N
        
        # Initial mode: bandpass filtered version of signal
        bandwidth = np.pi / (2 * mode_index)  # Narrower for later modes
        freq_axis = 2 * np.pi * np.arange(N) / N
        
        # Create bandpass filter
        bp_filter = np.exp(-((freq_axis - omega_init) ** 2) / (2 * bandwidth ** 2))
        bp_filter += np.exp(-((freq_axis - (2 * np.pi - omega_init)) ** 2) / (2 * bandwidth ** 2))
        
        # Apply filter
        U_init = X * bp_filter
        u_init = np.real(ifft(U_init))
        
        return u_init, omega_init
    
    def _update_mode_frequency(
        self,
        residual: np.ndarray,
        u_M: np.ndarray,
        omega_M: float,
        prev_modes_freq: List[np.ndarray],
        prev_omegas: List[float],
        N: int
    ) -> Tuple[np.ndarray, float]:
        """
        Update mode in frequency domain using ADMM.
        
        Implements Equations 23 and 26 from the paper.
        
        Parameters:
        -----------
        residual : np.ndarray
            Current residual signal
        u_M : np.ndarray
            Current mode estimate
        omega_M : float
            Current center frequency estimate
        prev_modes_freq : List[np.ndarray]
            FFT of previously extracted modes
        prev_omegas : List[float]
            Center frequencies of previous modes
        N : int
            Signal length
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Updated mode in frequency domain and center frequency
        """
        # Frequency axis
        omega_axis = 2 * np.pi * np.arange(N) / N
        
        # Get current multipliers
        rho = self.optimizer.rho
        theta = self.optimizer.theta
        
        # Compute auxiliary variable Q(omega) - Eq. 22
        X = fft(residual)
        sum_prev = np.zeros(N, dtype=complex)
        if prev_modes_freq:
            sum_prev = np.sum(prev_modes_freq, axis=0)
        
        # Note: X_u is computed from residual directly
        X_u = fft(residual - np.sum([np.real(ifft(U)) for U in prev_modes_freq], axis=0)) if prev_modes_freq else fft(residual)
        Theta = fft(theta) if theta is not None else np.zeros(N, dtype=complex)
        
        Q = X - sum_prev - X_u + Theta
        
        # Compute beta_i filters for previous modes - Eq. 12
        beta_sum = np.zeros(N)
        for omega_i in prev_omegas:
            beta_i = 1.0 / (self.config.noise_variance * (omega_axis - omega_i) ** 2 + self.config.epsilon1)
            beta_sum += beta_i ** 2
        
        # Spectral compactness term - sin^2(omega - omega_M)
        sin_sq = np.sin(omega_axis - omega_M) ** 2
        
        # Update U_M - Eq. 23
        numerator = (rho / 2) * Q
        denominator = (rho / 2) + beta_sum + (2 / np.pi) * sin_sq
        U_M = numerator / (denominator + 1e-10)
        
        # Update omega_M - Eq. 26
        U_M_abs_sq = np.abs(U_M[:N//2]) ** 2
        omega_positive = omega_axis[:N//2]
        
        numerator_omega = np.sum(omega_positive * U_M_abs_sq)
        denominator_omega = np.sum(U_M_abs_sq) + 1e-10
        omega_M_new = numerator_omega / denominator_omega
        
        return U_M, omega_M_new
    
    def _update_unprocessed(
        self,
        signal: np.ndarray,
        modes: List[np.ndarray],
        omega_M: float,
        N: int
    ) -> np.ndarray:
        """
        Update unprocessed signal component.
        
        Implements Equation 29 from the paper.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        modes : List[np.ndarray]
            All extracted modes including current
        omega_M : float
            Center frequency of current mode
        N : int
            Signal length
            
        Returns:
        --------
        np.ndarray
            Updated unprocessed signal in frequency domain
        """
        # Compute residual
        residual = signal - np.sum(modes, axis=0)
        X_residual = fft(residual)
        
        # Frequency axis
        omega_axis = 2 * np.pi * np.arange(N) / N
        
        # Beta_M filter - Eq. 14
        beta_M = 1.0 / (self.config.noise_variance * (omega_axis - omega_M) ** 2 + self.config.epsilon2)
        
        # Get multipliers
        rho = self.optimizer.rho
        mu = self.optimizer.mu
        
        # Compute Q_e - Eq. 28
        sum_modes = np.sum([fft(m) for m in modes], axis=0)
        X = fft(signal)
        Theta = fft(self.optimizer.theta) if self.optimizer.theta is not None else np.zeros(N, dtype=complex)
        Q_e = X - sum_modes + Theta
        
        # Minimum mode magnitude
        u_j_mag = np.min([np.max(np.abs(m)) for m in modes]) if modes else 1.0
        
        # Update magnitude - Eq. 30 (updated: includes 2*mu term)
        X_u_mag = (rho * np.abs(Q_e)) / (2 * np.abs(beta_M) ** 2 + 2 * mu + rho + 1e-10)
        
        # Preserve phase from residual
        X_u_phase = np.angle(X_residual)
        X_u = X_u_mag * np.exp(1j * X_u_phase)
        
        return X_u
    
    def reconstruct(self, result: DMDResult) -> np.ndarray:
        """
        Reconstruct signal from DMD result.
        
        Parameters:
        -----------
        result : DMDResult
            Decomposition result
            
        Returns:
        --------
        np.ndarray
            Reconstructed signal
        """
        if result.num_modes == 0:
            return result.residual
        return np.sum(result.modes, axis=0) + result.residual
    
    def get_mode_energy(self, result: DMDResult) -> np.ndarray:
        """
        Compute energy of each mode.
        
        Parameters:
        -----------
        result : DMDResult
            Decomposition result
            
        Returns:
        --------
        np.ndarray
            Energy of each mode
        """
        return np.sum(result.modes ** 2, axis=1)
    
    def get_instantaneous_frequency(self, mode: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous frequency of a mode using Hilbert transform.
        
        Parameters:
        -----------
        mode : np.ndarray
            Mode signal
            
        Returns:
        --------
        np.ndarray
            Instantaneous frequency
        """
        analytic = self.hilbert.transform(mode)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2 * np.pi)
        return np.concatenate([[inst_freq[0]], inst_freq])
    
    def get_instantaneous_amplitude(self, mode: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous amplitude of a mode using Hilbert transform.
        
        Parameters:
        -----------
        mode : np.ndarray
            Mode signal
            
        Returns:
        --------
        np.ndarray
            Instantaneous amplitude
        """
        analytic = self.hilbert.transform(mode)
        return np.abs(analytic)


class DMDBatch:
    """
    Batch processing wrapper for DMD.
    
    Efficiently processes multiple signals in parallel.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, config: Optional[DMDConfig] = None, num_workers: int = 4):
        """
        Initialize batch processor.
        
        Parameters:
        -----------
        config : DMDConfig, optional
            Configuration for DMD
        num_workers : int
            Number of parallel workers
        """
        self.config = config or DMDConfig()
        self.num_workers = num_workers
        self.dmd = DiscreteModeDcomposition(config=self.config)
    
    def decompose_batch(
        self,
        signals: np.ndarray,
        num_modes: Optional[int] = None
    ) -> List[DMDResult]:
        """
        Decompose a batch of signals.
        
        Parameters:
        -----------
        signals : np.ndarray
            Batch of signals, shape (B, N) or (B, N, C)
        num_modes : int, optional
            Number of modes to extract
            
        Returns:
        --------
        List[DMDResult]
            List of decomposition results
        """
        results = []
        for signal in tqdm(signals, desc="Processing batch"):
            result = self.dmd.decompose(signal, num_modes)
            results.append(result)
        return results


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create test signal: sum of sinusoids with noise
    np.random.seed(42)
    N = 1000
    t = np.linspace(0, 1, N)
    
    # Signal components
    f1, f2, f3 = 10, 25, 50  # Hz
    signal = (np.sin(2 * np.pi * f1 * t) + 
              0.5 * np.sin(2 * np.pi * f2 * t) + 
              0.3 * np.sin(2 * np.pi * f3 * t))
    signal += 0.1 * np.random.randn(N)  # Add noise
    
    # Decompose
    print("Decomposing signal...")
    dmd = DiscreteModeDcomposition(noise_variance=0.01, verbose=True)
    result = dmd.decompose(signal, num_modes=5)
    
    print(f"\nExtracted {result.num_modes} modes")
    print(f"Center frequencies: {result.center_frequencies * N / (2 * np.pi):.2f} Hz")
    print(f"Reconstruction error: {result.reconstruction_error:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(result.num_modes + 2, 1, figsize=(12, 3 * (result.num_modes + 2)))
    
    axes[0].plot(t, signal, 'b-', label='Original')
    axes[0].set_title('Original Signal')
    axes[0].legend()
    
    for i, mode in enumerate(result.modes):
        axes[i + 1].plot(t, mode, 'g-')
        freq = result.center_frequencies[i] * N / (2 * np.pi)
        axes[i + 1].set_title(f'Mode {i + 1} (f ≈ {freq:.1f} Hz)')
    
    reconstructed = dmd.reconstruct(result)
    axes[-1].plot(t, signal, 'b-', alpha=0.5, label='Original')
    axes[-1].plot(t, reconstructed, 'r--', label='Reconstructed')
    axes[-1].set_title('Reconstruction')
    axes[-1].legend()
    
    plt.tight_layout()
    plt.savefig('dmd_example.png', dpi=150)
    print("\nSaved example plot to 'dmd_example.png'")
