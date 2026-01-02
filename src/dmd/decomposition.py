"""
Discrete Mode Decomposition (DMD) Algorithm - EXACT Paper Implementation
=========================================================================

EXACT implementation of the DMD algorithm as described in:
"Discrete Mode Decomposition Meets Shapley Value: Robust Signal Prediction
in Tactile Internet" - IEEE INFOCOM 2025

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

Key Equations from Paper:
-------------------------
Eq. 10 (T1): Spectral compactness - (2/π) ∫₀^π sin²(ω-ω_M) |U_M(ω)|² dω
Eq. 11-12 (T2): Minimum overlap - Σ∫₀^π |β_i(ω) U_M(ω)|² dω
Eq. 13-14 (T3): Unprocessed overlap - ∫₀^π |β_M(ω) X_u(ω)|² dω
Eq. 15: Reconstruction - x[n] = Σu_k[n] + x_u[n]
Eq. 16: Energy bound - ||x_u||² ≤ ||u_min||²

Eq. 19: Frequency domain Lagrangian
Eq. 22: Q(ω) = X(ω) - Σ_{i=1}^{M-1} U_i(ω) - X_u(ω) + Θ(ω)
Eq. 24: U_M update
Eq. 27: ω_M update  
Eq. 28: Q̃(ω) = X(ω) - Σ_{i=1}^M U_i(ω) + Θ(ω)
Eq. 30: X_u update (with 2μ term)
Eq. 31: ρ update
Eq. 32: μ update (with max and integral over [0,π])

Algorithm 1: DMD convergence criteria
- Inner: ||U_M^{n+1} - U_M^n||² / ||U_M^n||² ≤ κ₁
- Outer: |α - (1/N)||x - Σu_i||²| / α ≤ κ₂
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class DMDConfig:
    """Configuration for Discrete Mode Decomposition.
    
    All parameters match the paper notation exactly.
    """
    noise_variance: float = 0.01      # α: Noise variance (Eq. 2)
    epsilon1: float = 1e-6            # ε₁: Regularization for β_i (Eq. 12)
    epsilon2: float = 1e-6            # ε₂: Regularization for β_M (Eq. 14)
    tau1: float = 0.1                 # τ₁: Step size for ρ update (Eq. 31)
    tau2: float = 0.1                 # τ₂: Step size for μ update (Eq. 32)
    kappa1: float = 1e-3              # κ₁: Inner convergence (Algorithm 1)
    kappa2: float = 1e-3              # κ₂: Outer convergence (Algorithm 1)
    max_modes: int = 10               # Maximum number of modes
    max_inner_iterations: int = 500   # Max inner ADMM iterations
    rho_init: float = 1.0             # Initial ρ
    mu_init: float = 0.0              # Initial μ
    verbose: bool = False


@dataclass
class DMDResult:
    """Result of Discrete Mode Decomposition.
    
    Output matches Algorithm 1: U = {u_k}_{k=1}^M, W = {ω_k}_{k=1}^M
    """
    modes: np.ndarray                 # U: Extracted modes (M x N)
    center_frequencies: np.ndarray    # W: Center frequencies (M,)
    residual: np.ndarray              # Final residual signal
    x_u: np.ndarray                   # Unprocessed signal component
    num_modes: int                    # M: Number of modes
    reconstruction_error: float       # Final reconstruction error
    convergence_history: List[dict] = field(default_factory=list)


class DiscreteModeDcomposition:
    """
    Discrete Mode Decomposition (DMD) - Exact Paper Implementation.
    
    Implements Algorithm 1 from the IEEE INFOCOM 2025 paper exactly.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
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
            Estimated variance of additive noise (α in paper)
        config : DMDConfig, optional
            Configuration object
        """
        if config is not None:
            self.config = config
        else:
            self.config = DMDConfig(noise_variance=noise_variance, **kwargs)
    
    def decompose(
        self,
        signal: np.ndarray,
        num_modes: Optional[int] = None
    ) -> DMDResult:
        """
        Decompose a signal into intrinsic modes.
        
        Implements Algorithm 1 from the paper exactly.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input discrete-time signal x[n] of shape (N,)
        num_modes : int, optional
            Number of modes to extract. If None, extract until convergence.
            
        Returns:
        --------
        DMDResult
            Object containing U (modes), W (center frequencies), and metadata.
        """
        signal = np.asarray(signal).flatten()
        N = len(signal)
        max_modes = num_modes if num_modes else self.config.max_modes
        alpha = self.config.noise_variance
        
        # Frequency axis: ω ∈ [0, 2π)
        omega = 2 * np.pi * np.arange(N) / N
        
        # For integrals over [0, π] (positive frequencies only, as per paper)
        N_half = N // 2 + 1
        omega_positive = omega[:N_half]
        d_omega = 2 * np.pi / N  # Frequency resolution for numerical integration
        
        # Signal spectrum
        X = fft(signal)
        
        # Storage
        modes = []                    # U = {u_k}
        center_frequencies = []       # W = {ω_k}
        convergence_history = []
        
        # Initialize unprocessed signal x_u[n]
        x_u = np.zeros(N)
        
        if self.config.verbose:
            print("=" * 60)
            print("Discrete Mode Decomposition (DMD)")
            print("Author: Ali Vahedi (Mohammad Ali Vahedifar)")
            print("IEEE INFOCOM 2025")
            print("=" * 60)
            print(f"Signal length: N = {N}")
            print(f"Noise variance (α): {alpha}")
            print("-" * 60)
        
        # =====================================================================
        # MAIN LOOP - Algorithm 1: REPEAT until outer convergence
        # =====================================================================
        M = 0  # Mode counter
        
        pbar = tqdm(range(max_modes), desc="Extracting modes", 
                    disable=not self.config.verbose)
        
        for _ in pbar:
            M += 1
            
            # -----------------------------------------------------------------
            # Algorithm 1, line 3: Initialize u¹_M, ω¹_M, ρ¹, μ¹, n←0
            # -----------------------------------------------------------------
            U_M, omega_M = self._initialize_mode(signal, modes, N, omega)
            
            rho = self.config.rho_init * np.ones(N)  # ρ(ω) - frequency dependent
            mu = self.config.mu_init                  # μ (scalar)
            Theta = np.zeros(N, dtype=complex)        # Θ(ω) scaled dual variable
            
            # =================================================================
            # INNER ADMM LOOP - Algorithm 1, lines 6-11
            # REPEAT until ||U_M^{n+1} - U_M^n||² / ||U_M^n||² ≤ κ₁
            # =================================================================
            for n_iter in range(self.config.max_inner_iterations):
                U_M_prev = U_M.copy()
                
                # -------------------------------------------------------------
                # Step 1: Update U_M(ω) with Eq. 24
                # U_M^{n+1}(ω) = [ρ(ω)/2 · Q(ω)] / [ρ(ω)/2 + Σ|β_i(ω)|² + (2/π)sin²(ω-ω_M)]
                # -------------------------------------------------------------
                
                # Sum of previous modes in frequency domain: Σ_{i=1}^{M-1} U_i(ω)
                sum_prev_U = np.zeros(N, dtype=complex)
                for mode in modes:
                    sum_prev_U += fft(mode)
                
                # X_u(ω)
                X_u = fft(x_u)
                
                # Q(ω) - Eq. 22
                Q = X - sum_prev_U - X_u + Theta
                
                # Sum of |β_i(ω)|² for i = 1, ..., M-1 (Eq. 12)
                # β_i(ω) = 1 / [α(ω - ω_i)² + ε₁]
                beta_sq_sum = np.zeros(N)
                for omega_i in center_frequencies:
                    beta_i = 1.0 / (alpha * (omega - omega_i) ** 2 + self.config.epsilon1)
                    beta_sq_sum += np.abs(beta_i) ** 2
                
                # Spectral compactness term: (2/π) sin²(ω - ω_M)
                sin_sq_term = (2.0 / np.pi) * np.sin(omega - omega_M) ** 2
                
                # Update U_M (Eq. 24)
                numerator = (rho / 2) * Q
                denominator = (rho / 2) + beta_sq_sum + sin_sq_term
                U_M = numerator / (denominator + 1e-10)
                
                # -------------------------------------------------------------
                # Step 2: Update ω_M with Eq. 27
                # ω_M^{n+1} = ∫₀^π ω|U_M(ω)|² dω / ∫₀^π |U_M(ω)|² dω
                # -------------------------------------------------------------
                U_M_positive = U_M[:N_half]
                U_M_abs_sq = np.abs(U_M_positive) ** 2
                
                # Numerical integration over [0, π]
                numerator_omega = np.sum(omega_positive * U_M_abs_sq) * d_omega
                denominator_omega = np.sum(U_M_abs_sq) * d_omega + 1e-10
                omega_M = numerator_omega / denominator_omega
                
                # -------------------------------------------------------------
                # Step 3: Update X_u(ω) with Eq. 30
                # X_u^{n+1}(ω) = [ρ(ω) · Q̃(ω)] / [2|β_M(ω)|² + 2μ + ρ(ω)]
                # -------------------------------------------------------------
                
                # Q̃(ω) - Eq. 28: X(ω) - Σ_{i=1}^M U_i(ω) + Θ(ω)
                Q_tilde = X - sum_prev_U - U_M + Theta
                
                # β_M(ω) = 1 / [α(ω - ω_M)² + ε₂] (Eq. 14)
                beta_M = 1.0 / (alpha * (omega - omega_M) ** 2 + self.config.epsilon2)
                
                # Update X_u (Eq. 30) - NOTE: includes 2μ in denominator
                numerator_Xu = rho * Q_tilde
                denominator_Xu = 2 * np.abs(beta_M) ** 2 + 2 * mu + rho
                X_u = numerator_Xu / (denominator_Xu + 1e-10)
                
                # Convert to time domain
                x_u = np.real(ifft(X_u))
                
                # -------------------------------------------------------------
                # Step 4: Update ρ(ω) with Eq. 31
                # ρ^{n+1}(ω) = ρ^n(ω) + τ₁ · (X(ω) - Σ_{i=1}^M U_i^{n+1}(ω))
                # -------------------------------------------------------------
                sum_all_U = sum_prev_U + U_M
                rho = rho + self.config.tau1 * np.abs(X - sum_all_U)
                
                # -------------------------------------------------------------
                # Step 5: Update μ with Eq. 32
                # μ^{n+1} = max(0, μ^n + τ₂ · ∫₀^π (||X_u(ω)||² - ||U_min(ω)||²) dω)
                # -------------------------------------------------------------
                
                # Energy of X_u over [0, π]
                X_u_energy = np.sum(np.abs(X_u[:N_half]) ** 2) * d_omega
                
                # Find U_min = argmin_{u ∈ U} ||u||₂
                if modes:
                    mode_energies = []
                    for mode in modes:
                        U_i = fft(mode)
                        mode_energies.append(np.sum(np.abs(U_i[:N_half]) ** 2) * d_omega)
                    U_min_energy = min(mode_energies)
                else:
                    # If no previous modes, use current mode energy
                    U_min_energy = np.sum(np.abs(U_M[:N_half]) ** 2) * d_omega
                
                # Update μ (Eq. 32) - with max(0, ...)
                mu = max(0, mu + self.config.tau2 * (X_u_energy - U_min_energy))
                
                # -------------------------------------------------------------
                # Step 6: Update Θ(ω) - scaled dual variable
                # Θ^{n+1}(ω) = Θ^n(ω) + (X(ω) - Σ U_i(ω) - X_u(ω))
                # -------------------------------------------------------------
                Theta = Theta + (X - sum_all_U - X_u)
                
                # -------------------------------------------------------------
                # Check inner convergence (Algorithm 1, line 11)
                # ||U_M^{n+1} - U_M^n||² / ||U_M^n||² ≤ κ₁
                # -------------------------------------------------------------
                conv_metric = np.sum(np.abs(U_M - U_M_prev) ** 2) / (np.sum(np.abs(U_M_prev) ** 2) + 1e-10)
                
                if conv_metric < self.config.kappa1:
                    if self.config.verbose:
                        print(f"  Inner converged: iter {n_iter+1}, metric = {conv_metric:.2e} < κ₁ = {self.config.kappa1:.2e}")
                    break
            
            # -----------------------------------------------------------------
            # Store extracted mode
            # -----------------------------------------------------------------
            u_M = np.real(ifft(U_M))
            modes.append(u_M)
            center_frequencies.append(omega_M)
            
            # -----------------------------------------------------------------
            # Check outer convergence (Algorithm 1, line 12)
            # |α - (1/N)||x - Σ u_i||²| / α ≤ κ₂
            # -----------------------------------------------------------------
            reconstruction = np.sum(modes, axis=0)
            reconstruction_error = np.sum((signal - reconstruction) ** 2) / N
            outer_metric = np.abs(alpha - reconstruction_error) / alpha
            
            convergence_history.append({
                'mode': M,
                'inner_iterations': n_iter + 1,
                'omega_M': omega_M,
                'reconstruction_error': reconstruction_error,
                'outer_metric': outer_metric
            })
            
            if self.config.verbose:
                freq_normalized = omega_M / (2 * np.pi)
                pbar.set_postfix({
                    'M': M,
                    'ω_M': f'{omega_M:.4f}',
                    'outer': f'{outer_metric:.4f}'
                })
                print(f"  Mode {M}: ω_M = {omega_M:.4f} rad (f = {freq_normalized:.4f} normalized)")
                print(f"  Reconstruction error: {reconstruction_error:.6f}")
                print(f"  Outer metric: |α - MSE|/α = {outer_metric:.4f} (κ₂ = {self.config.kappa2})")
            
            if outer_metric < self.config.kappa2:
                if self.config.verbose:
                    print(f"\nOuter loop converged after M = {M} modes.")
                break
        
        # =====================================================================
        # Final output: U = {u_k}_{k=1}^M, W = {ω_k}_{k=1}^M
        # =====================================================================
        modes = np.array(modes)
        center_frequencies = np.array(center_frequencies)
        residual = signal - np.sum(modes, axis=0)
        final_reconstruction_error = np.sum(residual ** 2) / N
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("DMD Complete")
            print(f"Extracted M = {M} modes")
            print(f"Final reconstruction error: {final_reconstruction_error:.6f}")
            print(f"Output: U = {{u_k}}_{{k=1}}^{M}, W = {{ω_k}}_{{k=1}}^{M}")
            print("=" * 60)
        
        return DMDResult(
            modes=modes,
            center_frequencies=center_frequencies,
            residual=residual,
            x_u=x_u,
            num_modes=M,
            reconstruction_error=final_reconstruction_error,
            convergence_history=convergence_history
        )
    
    def _initialize_mode(
        self,
        signal: np.ndarray,
        modes: List[np.ndarray],
        N: int,
        omega: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Initialize new mode from residual signal.
        
        Strategy: Find dominant frequency in residual spectrum.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        modes : List[np.ndarray]
            Previously extracted modes
        N : int
            Signal length
        omega : np.ndarray
            Frequency axis
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Initial U_M(ω) and ω_M
        """
        # Compute residual
        if modes:
            residual = signal - np.sum(modes, axis=0)
        else:
            residual = signal
        
        # FFT of residual
        R = fft(residual)
        
        # Find dominant frequency in positive spectrum [0, π]
        N_half = N // 2 + 1
        R_positive = np.abs(R[:N_half])
        
        # Exclude DC component
        R_positive[0] = 0
        
        idx = np.argmax(R_positive)
        omega_M = omega[idx]
        
        # Initialize U_M as bandpass filtered version centered around dominant frequency
        bandwidth = 0.5  # Initial bandwidth parameter
        weight = np.exp(-((omega - omega_M) ** 2) / (2 * bandwidth ** 2))
        # Handle conjugate symmetry for negative frequencies
        weight += np.exp(-((omega - (2 * np.pi - omega_M)) ** 2) / (2 * bandwidth ** 2))
        U_M = R * weight
        
        return U_M, omega_M
    
    def reconstruct(self, result: DMDResult) -> np.ndarray:
        """
        Reconstruct signal from DMD result.
        
        x[n] = Σ_{k=1}^M u_k[n] + x_u[n]
        
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
            return result.x_u
        return np.sum(result.modes, axis=0) + result.x_u
    
    def get_mode_energy(self, result: DMDResult) -> np.ndarray:
        """
        Compute energy of each mode: ||u_k||₂²
        
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


# =============================================================================
# Batch processing wrapper
# =============================================================================
class DMDBatch:
    """
    Batch processing wrapper for DMD.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, config: Optional[DMDConfig] = None):
        self.config = config or DMDConfig()
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
            Batch of signals, shape (B, N)
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


# =============================================================================
# Demo / Test
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing DMD implementation against paper equations...")
    
    # Create test signal: sum of sinusoids with noise
    np.random.seed(42)
    N = 1000
    fs = 1000  # 1 kHz sampling rate (TI requirement)
    t = np.arange(N) / fs
    
    # Signal components (matching paper's TI context)
    f1, f2, f3 = 5, 15, 35  # Hz
    signal = (1.0 * np.sin(2 * np.pi * f1 * t) + 
              0.6 * np.sin(2 * np.pi * f2 * t) + 
              0.3 * np.sin(2 * np.pi * f3 * t))
    noise_std = 0.1
    signal += noise_std * np.random.randn(N)
    
    # Decompose with paper parameters
    config = DMDConfig(
        noise_variance=noise_std ** 2,  # α = 0.01
        epsilon1=1e-6,
        epsilon2=1e-6,
        tau1=0.1,
        tau2=0.1,
        kappa1=1e-3,
        kappa2=1e-3,
        max_modes=5,
        verbose=True
    )
    
    dmd = DiscreteModeDcomposition(config=config)
    result = dmd.decompose(signal)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Extracted M = {result.num_modes} modes")
    print(f"Center frequencies (rad/sample):")
    for i, omega in enumerate(result.center_frequencies):
        freq_hz = omega * fs / (2 * np.pi)
        print(f"  Mode {i+1}: ω = {omega:.4f} rad/sample ({freq_hz:.1f} Hz)")
    print(f"Final reconstruction error: {result.reconstruction_error:.6f}")
    
    # Verify reconstruction
    reconstructed = dmd.reconstruct(result)
    mse = np.mean((signal - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Compute accuracy (paper metric)
    mae = np.mean(np.abs(signal - reconstructed))
    signal_range = np.max(signal) - np.min(signal)
    accuracy = (1 - mae / signal_range) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Plot results
    fig, axes = plt.subplots(result.num_modes + 2, 1, figsize=(12, 3 * (result.num_modes + 2)))
    
    axes[0].plot(t, signal, 'b-', alpha=0.7, label='Original')
    axes[0].set_title('Original Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].legend()
    
    for i, mode in enumerate(result.modes):
        freq_hz = result.center_frequencies[i] * fs / (2 * np.pi)
        axes[i + 1].plot(t, mode, 'g-')
        axes[i + 1].set_title(f'Mode {i + 1} (ω = {result.center_frequencies[i]:.3f} rad, f ≈ {freq_hz:.1f} Hz)')
        axes[i + 1].set_xlabel('Time (s)')
    
    axes[-1].plot(t, signal, 'b-', alpha=0.5, label='Original')
    axes[-1].plot(t, reconstructed, 'r--', label='Reconstructed')
    axes[-1].set_title(f'Reconstruction (Accuracy: {accuracy:.2f}%)')
    axes[-1].set_xlabel('Time (s)')
    axes[-1].legend()
    
    plt.tight_layout()
    plt.savefig('dmd_exact_implementation_test.png', dpi=150)
    print(f"\nSaved test plot to 'dmd_exact_implementation_test.png'")
