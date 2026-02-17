"""
Discrete Mode Decomposition (DMD)
=========================================================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
IEEE INFOCOM 2026
This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

Key Equations from Paper:
-------------------------
Eq. (10) - T1: Spectral compactness of Z-th mode
    T1 = |∂_n[(δ[n] + j(1-(-1)^n)/(πn)) * m_Z[n]] e^{-jω_Z n}|²₂

Eq. (11)-(12) - T2: Minimum overlap with previously extracted modes
    T2 = Σ_{k=1}^{Z-1} ||β_k[n] * m_Z[n]||²₂
    β_k(ω) = 1 / (α(ω - ω_k)² + ε₁)

Eq. (13)-(14) - T3: Minimum spectral overlap with unprocessed signal
    T3 = ||β_Z[n] * x_u[n]||²₂
    β_Z(ω) = 1 / (α(ω - ω_Z)² + ε₂)

Eq. (15): Mode update - M_Z^{n+1}(ω)
    M_Z^{n+1}(ω) = [ρ(ω)/2 · Q(ω)] / [ρ(ω)/2 + Σ|β_k(ω)|² + (2/π)sin²(ω-ω_Z)]

Eq. (18): Center frequency update - ω_Z^{n+1}
    ω_Z^{n+1} = ∫₀^π ω|M_Z^{n+1}(ω)|² dω / ∫₀^π |M_Z^{n+1}(ω)|² dω

Eq. (19): Auxiliary variable Q̃(ω)
    Q̃(ω) = X(ω) - Σ_{k=1}^Z M_k(ω) + Θ(ω)

Eq. (20): Unprocessed signal update - X_u^{n+1}(ω)
    X_u^{n+1}(ω) = [ρ(ω)·Q̃(ω)] / [2|β_Z(ω)|² + 2μ + ρ(ω)]

Eq. (21): Scaled dual variable update - Θ^{n+1}(ω)
    Θ^{n+1}(ω) = Θ^n(ω) + τ₁(X(ω) - Σ_{k=1}^Z M_k^{n+1}(ω) - X_u^{n+1}(ω))

Eq. (22): Inequality constraint multiplier update - μ^{n+1}
    μ^{n+1} = max(0, μ^n + τ₂ ∫₀^π (||X_u^{n+1}(ω)||² - ||M_min(ω)||²) dω)

Algorithm 1: DMD convergence criteria
- Inner: ||M_Z^{n+1} - M_Z^n||²₂ / ||M_Z^n||²₂ ≤ κ₁
- Outer: |α - (1/Z)||x - Σ_{k=1}^Z m_k||²₂| / α ≤ κ₂
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
    epsilon1: float = 1e-6            # ε₁: Regularization for β_k (Eq. 12)
    epsilon2: float = 1e-6            # ε₂: Regularization for β_Z (Eq. 14)
    tau1: float = 0.1                 # τ₁: Step size for Θ update (Eq. 21)
    tau2: float = 0.1                 # τ₂: Step size for μ update (Eq. 22)
    kappa1: float = 1e-3              # κ₁: Inner convergence (Algorithm 1)
    kappa2: float = 1e-3              # κ₂: Outer convergence (Algorithm 1)
    max_modes: int = 10               # Maximum number of modes (Z)
    max_inner_iterations: int = 500   # Max inner ADMM iterations
    rho_init: float = 1.0             # Initial ρ
    mu_init: float = 0.0              # Initial μ
    verbose: bool = False


@dataclass
class DMDResult:
    """Result of Discrete Mode Decomposition.
    
    Output matches Algorithm 1: M = {m_k}_{k=1}^Z, W = {ω_k}_{k=1}^Z
    """
    modes: np.ndarray                 # M: Extracted modes (Z x N)
    center_frequencies: np.ndarray    # W: Center frequencies (Z,)
    residual: np.ndarray              # Final residual signal
    x_u: np.ndarray                   # Unprocessed signal component
    num_modes: int                    # Z: Number of modes
    reconstruction_error: float       # Final reconstruction error
    convergence_history: List[dict] = field(default_factory=list)


class DiscreteModeDcomposition:
    """
    Discrete Mode Decomposition (DMD) - Exact Paper Implementation.
    
    Implements Algorithm 1 from the paper exactly.
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
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
            Object containing M (modes), W (center frequencies), and metadata.
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
        modes = []                    # M = {m_k}
        center_frequencies = []       # W = {ω_k}
        convergence_history = []
        
        # Initialize unprocessed signal x_u[n]
        x_u = np.zeros(N)
        
        if self.config.verbose:
            print("=" * 60)
            print("Discrete Mode Decomposition (DMD)")
            print("Author: Ali Vahedi")
            print("IEEE INFOCOM 2025")
            print("=" * 60)
            print(f"Signal length: N = {N}")
            print(f"Noise variance (α): {alpha}")
            print("-" * 60)
        
        # =====================================================================
        # MAIN LOOP - Algorithm 1: REPEAT until outer convergence
        # =====================================================================
        Z = 0  # Mode counter
        
        pbar = tqdm(range(max_modes), desc="Extracting modes", 
                    disable=not self.config.verbose)
        
        for _ in pbar:
            Z += 1
            
            # -----------------------------------------------------------------
            # Algorithm 1, line 3: Initialize m¹_Z, ω¹_Z, ρ¹, μ¹, n←0
            # -----------------------------------------------------------------
            M_Z, omega_Z = self._initialize_mode(signal, modes, N, omega)
            
            rho = self.config.rho_init * np.ones(N)  # ρ(ω) - frequency dependent
            mu = self.config.mu_init                  # μ (scalar)
            Theta = np.zeros(N, dtype=complex)        # Θ(ω) scaled dual variable
            
            # =================================================================
            # INNER ADMM LOOP - Algorithm 1, lines 6-11
            # REPEAT until ||M_Z^{n+1} - M_Z^n||²₂ / ||M_Z^n||²₂ ≤ κ₁
            # =================================================================
            for n_iter in range(self.config.max_inner_iterations):
                M_Z_prev = M_Z.copy()
                
                # -------------------------------------------------------------
                # Step 1: Update M_Z(ω) with Eq. (15)
                # M_Z^{n+1}(ω) = [ρ(ω)/2 · Q(ω)] / [ρ(ω)/2 + Σ|β_k(ω)|² + (2/π)sin²(ω-ω_Z)]
                # -------------------------------------------------------------
                
                # Sum of previous modes in frequency domain: Σ_{k=1}^{Z-1} M_k(ω)
                sum_prev_M = np.zeros(N, dtype=complex)
                for mode in modes:
                    sum_prev_M += fft(mode)
                
                # X_u(ω)
                X_u = fft(x_u)
                
                # Q(ω) - Auxiliary variable
                Q = X - sum_prev_M - X_u + Theta
                
                # Sum of |β_k(ω)|² for k = 1, ..., Z-1 (Eq. 12)
                # β_k(ω) = 1 / [α(ω - ω_k)² + ε₁]
                beta_sq_sum = np.zeros(N)
                for omega_k in center_frequencies:
                    beta_k = 1.0 / (alpha * (omega - omega_k) ** 2 + self.config.epsilon1)
                    beta_sq_sum += np.abs(beta_k) ** 2
                
                # Spectral compactness term: (2/π) sin²(ω - ω_Z)
                sin_sq_term = (2.0 / np.pi) * np.sin(omega - omega_Z) ** 2
                
                # Update M_Z (Eq. 15)
                numerator = (rho / 2) * Q
                denominator = (rho / 2) + beta_sq_sum + sin_sq_term
                M_Z = numerator / (denominator + 1e-10)
                
                # -------------------------------------------------------------
                # Step 2: Update ω_Z with Eq. (18)
                # ω_Z^{n+1} = ∫₀^π ω|M_Z(ω)|² dω / ∫₀^π |M_Z(ω)|² dω
                # -------------------------------------------------------------
                M_Z_positive = M_Z[:N_half]
                M_Z_abs_sq = np.abs(M_Z_positive) ** 2
                
                # Numerical integration over [0, π]
                numerator_omega = np.sum(omega_positive * M_Z_abs_sq) * d_omega
                denominator_omega = np.sum(M_Z_abs_sq) * d_omega + 1e-10
                omega_Z = numerator_omega / denominator_omega
                
                # -------------------------------------------------------------
                # Step 3: Update X_u(ω) with Eq. (20)
                # X_u^{n+1}(ω) = [ρ(ω) · Q̃(ω)] / [2|β_Z(ω)|² + 2μ + ρ(ω)]
                # -------------------------------------------------------------
                
                # Q̃(ω) - Eq. (19): X(ω) - Σ_{k=1}^Z M_k(ω) + Θ(ω)
                Q_tilde = X - sum_prev_M - M_Z + Theta
                
                # β_Z(ω) = 1 / [α(ω - ω_Z)² + ε₂] (Eq. 14)
                beta_Z = 1.0 / (alpha * (omega - omega_Z) ** 2 + self.config.epsilon2)
                
                # Update X_u (Eq. 20) - NOTE: includes 2μ in denominator
                numerator_Xu = rho * Q_tilde
                denominator_Xu = 2 * np.abs(beta_Z) ** 2 + 2 * mu + rho
                X_u = numerator_Xu / (denominator_Xu + 1e-10)
                
                # Convert to time domain
                x_u = np.real(ifft(X_u))
                
                # -------------------------------------------------------------
                # Step 4: Update Θ(ω) with Eq. (21)
                # Θ^{n+1}(ω) = Θ^n(ω) + τ₁ · (X(ω) - Σ_{k=1}^Z M_k^{n+1}(ω) - X_u^{n+1}(ω))
                # -------------------------------------------------------------
                sum_all_M = sum_prev_M + M_Z
                Theta = Theta + self.config.tau1 * (X - sum_all_M - X_u)
                
                # -------------------------------------------------------------
                # Step 5: Update μ with Eq. (22)
                # μ^{n+1} = max(0, μ^n + τ₂ · ∫₀^π (||X_u(ω)||² - ||M_min(ω)||²) dω)
                # -------------------------------------------------------------
                
                # Energy of X_u over [0, π]
                X_u_energy = np.sum(np.abs(X_u[:N_half]) ** 2) * d_omega
                
                # Find M_min = argmin_{m ∈ M} ||m||₂
                if modes:
                    mode_energies = []
                    for mode in modes:
                        M_k = fft(mode)
                        mode_energies.append(np.sum(np.abs(M_k[:N_half]) ** 2) * d_omega)
                    M_min_energy = min(mode_energies)
                else:
                    # If no previous modes, use current mode energy
                    M_min_energy = np.sum(np.abs(M_Z[:N_half]) ** 2) * d_omega
                
                # Update μ (Eq. 22) - with max(0, ...)
                mu = max(0, mu + self.config.tau2 * (X_u_energy - M_min_energy))
                
                # -------------------------------------------------------------
                # Check inner convergence (Algorithm 1, line 11)
                # ||M_Z^{n+1} - M_Z^n||²₂ / ||M_Z^n||²₂ ≤ κ₁
                # -------------------------------------------------------------
                conv_metric = np.sum(np.abs(M_Z - M_Z_prev) ** 2) / (np.sum(np.abs(M_Z_prev) ** 2) + 1e-10)
                
                if conv_metric < self.config.kappa1:
                    if self.config.verbose:
                        print(f"  Inner converged: iter {n_iter+1}, metric = {conv_metric:.2e} < κ₁ = {self.config.kappa1:.2e}")
                    break
            
            # -----------------------------------------------------------------
            # Store extracted mode
            # -----------------------------------------------------------------
            m_Z = np.real(ifft(M_Z))
            modes.append(m_Z)
            center_frequencies.append(omega_Z)
            
            # -----------------------------------------------------------------
            # Check outer convergence (Algorithm 1, line 12)
            # |α - (1/Z)||x - Σ_{k=1}^Z m_k||²₂| / α ≤ κ₂
            # -----------------------------------------------------------------
            reconstruction = np.sum(modes, axis=0)
            reconstruction_error = np.sum((signal - reconstruction) ** 2) / N
            outer_metric = np.abs(alpha - reconstruction_error) / alpha
            
            convergence_history.append({
                'mode': Z,
                'inner_iterations': n_iter + 1,
                'omega_Z': omega_Z,
                'reconstruction_error': reconstruction_error,
                'outer_metric': outer_metric
            })
            
            if self.config.verbose:
                freq_normalized = omega_Z / (2 * np.pi)
                pbar.set_postfix({
                    'Z': Z,
                    'ω_Z': f'{omega_Z:.4f}',
                    'outer': f'{outer_metric:.4f}'
                })
                print(f"  Mode {Z}: ω_Z = {omega_Z:.4f} rad (f = {freq_normalized:.4f} normalized)")
                print(f"  Reconstruction error: {reconstruction_error:.6f}")
                print(f"  Outer metric: |α - MSE|/α = {outer_metric:.4f} (κ₂ = {self.config.kappa2})")
            
            if outer_metric < self.config.kappa2:
                if self.config.verbose:
                    print(f"\nOuter loop converged after Z = {Z} modes.")
                break
        
        # =====================================================================
        # Final output: M = {m_k}_{k=1}^Z, W = {ω_k}_{k=1}^Z
        # =====================================================================
        modes = np.array(modes)
        center_frequencies = np.array(center_frequencies)
        residual = signal - np.sum(modes, axis=0)
        final_reconstruction_error = np.sum(residual ** 2) / N
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("DMD Complete")
            print(f"Extracted Z = {Z} modes")
            print(f"Final reconstruction error: {final_reconstruction_error:.6f}")
            print(f"Output: M = {{m_k}}_{{k=1}}^{Z}, W = {{ω_k}}_{{k=1}}^{Z}")
            print("=" * 60)
        
        return DMDResult(
            modes=modes,
            center_frequencies=center_frequencies,
            residual=residual,
            x_u=x_u,
            num_modes=Z,
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
            Initial M_Z(ω) and ω_Z
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
        omega_Z = omega[idx]
        
        # Initialize M_Z as bandpass filtered version centered around dominant frequency
        bandwidth = 0.5  # Initial bandwidth parameter
        weight = np.exp(-((omega - omega_Z) ** 2) / (2 * bandwidth ** 2))
        # Handle conjugate symmetry for negative frequencies
        weight += np.exp(-((omega - (2 * np.pi - omega_Z)) ** 2) / (2 * bandwidth ** 2))
        M_Z = R * weight
        
        return M_Z, omega_Z
    
    def reconstruct(self, result: DMDResult) -> np.ndarray:
        """
        Reconstruct signal from DMD result.
        
        x[n] = Σ_{k=1}^Z m_k[n] + x_u[n]
        
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
        Compute energy of each mode: ||m_k||₂²
        
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
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
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
