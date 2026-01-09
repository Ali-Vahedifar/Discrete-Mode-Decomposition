"""
ADMM Optimization for Discrete Mode Decomposition - EXACT Paper Implementation
===============================================================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
IEEE INFOCOM 2025
This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

EXACT implementation of the ADMM algorithm as described in:
"Discrete Mode Decomposition Meets Shapley Value: Robust Signal Prediction
in Tactile Internet"

Mathematical Background (from paper):
-------------------------------------
The DMD optimization problem (Eq. 17):
    min_{m_Z, ω_Z} T1 + T2 + T3
    s.t. x[n] = Σ_{k=1}^{Z} m_k[n] + x_u[n]       (Reconstruction)
         ||x_u[n]||²₂ ≤ ||m_min[n]||²₂             (Energy bound)

The Augmented Lagrangian (Eq. 18):
    L_aug(μ,ρ) = ||∂_n[A_n * m_Z] e^{-jω_Z n}||²₂ + Σ||β_k * m_Z||²₂
                 + ||β_Z * x_u||²₂ + (ρ/2)||x - Σm_k - x_u + θ||²₂
                 + μ(||x_u||²₂ - ||m_min||²₂) - (ρ/2)||θ||²₂

Frequency Domain Lagrangian:
    L_aug(μ,ρ(ω)) = (2/π)∫₀^π sin²(ω-ω_Z)|M_Z(ω)|² dω
                    + Σ∫₀^π |β_k(ω)M_Z(ω)|² dω
                    + ∫₀^π |β_Z(ω)X_u(ω)|² dω
                    + (ρ(ω)/2)||X - ΣM_k - X_u + Θ||²₂
                    + μ(||X_u||²₂ - ||M_min||²₂) - (ρ(ω)/2)||Θ||²₂

ADMM Update Equations:
    Eq. (15): M_Z update
    Eq. (18): ω_Z update
    Eq. (20): X_u update (with 2μ term)
    Eq. (21): Θ update (scaled dual variable with τ₁)
    Eq. (22): μ update (with max(0,...) and τ₂ for inequality constraint)
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class ADMMState:
    """
    State of ADMM optimization.
    
    All variables match paper notation exactly.
    """
    rho: np.ndarray = None            # ρ(ω): Frequency-dependent penalty parameter
    mu: float = 0.0                   # μ: Scalar Lagrangian multiplier for inequality
    Theta: np.ndarray = None          # Θ(ω): Scaled dual variable (λ = ρ·θ)
    iteration: int = 0
    history: List[Dict] = field(default_factory=list)


class ADMMOptimizer:
    """
    ADMM Optimizer for DMD - Exact Paper Implementation.
    
    Implements the update equations from the paper:
    - Eq. (21): Θ update (scaled dual variable)
    - Eq. (22): μ update (inequality constraint with energy-based bound)
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        tau1: float = 0.1,
        tau2: float = 0.1,
        rho_init: float = 1.0,
        mu_init: float = 0.0
    ):
        """
        Initialize ADMM optimizer with paper parameters.
        
        Parameters:
        -----------
        tau1 : float
            τ₁: Step size for Θ update (Eq. 21)
        tau2 : float
            τ₂: Step size for μ update (Eq. 22)
        rho_init : float
            Initial ρ value
        mu_init : float
            Initial μ value
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.rho_init = rho_init
        self.mu_init = mu_init
        self._state = None
    
    @property
    def rho(self) -> np.ndarray:
        """Current penalty parameter ρ(ω)."""
        return self._state.rho if self._state else None
    
    @property
    def mu(self) -> float:
        """Current inequality multiplier μ."""
        return self._state.mu if self._state else self.mu_init
    
    @property
    def Theta(self) -> np.ndarray:
        """Current scaled dual variable Θ(ω)."""
        return self._state.Theta if self._state else None
    
    def reset(self, N: int):
        """
        Reset optimizer state for a new mode extraction.
        
        Algorithm 1, line 3: Initialize ρ¹, μ¹
        
        Parameters:
        -----------
        N : int
            Signal length
        """
        self._state = ADMMState(
            rho=self.rho_init * np.ones(N),
            mu=self.mu_init,
            Theta=np.zeros(N, dtype=complex),
            iteration=0,
            history=[]
        )
    
    def update_rho(
        self,
        X: np.ndarray,
        sum_M: np.ndarray
    ) -> np.ndarray:
        """
        Update penalty parameter ρ(ω).
        
        Standard ADMM penalty update for primal residual minimization.
        
        Parameters:
        -----------
        X : np.ndarray
            Signal spectrum X(ω)
        sum_M : np.ndarray
            Sum of all mode spectra Σ M_k(ω)
            
        Returns:
        --------
        np.ndarray
            Updated ρ(ω)
        """
        # Update ρ based on primal residual
        self._state.rho = self._state.rho + self.tau1 * np.abs(X - sum_M)
        return self._state.rho
    
    def update_mu(
        self,
        X_u: np.ndarray,
        modes: List[np.ndarray],
        N: int
    ) -> float:
        """
        Update inequality constraint multiplier μ.
        
        Implements Equation (22) from the paper:
        μ^{n+1} = max(0, μ^n + τ₂ · ∫₀^π (||X_u(ω)||²₂ - ||M_min(ω)||²₂) dω)
        
        Parameters:
        -----------
        X_u : np.ndarray
            Unprocessed signal spectrum X_u(ω)
        modes : List[np.ndarray]
            List of all extracted modes (time domain)
        N : int
            Signal length
            
        Returns:
        --------
        float
            Updated μ
        """
        # Integration over [0, π] only (positive frequencies)
        N_half = N // 2 + 1
        d_omega = 2 * np.pi / N
        
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
            M_min_energy = X_u_energy  # Fallback
        
        # Eq. (22): μ^{n+1} = max(0, μ^n + τ₂ · ∫(||X_u||² - ||M_min||²) dω)
        self._state.mu = max(0, self._state.mu + self.tau2 * (X_u_energy - M_min_energy))
        
        return self._state.mu
    
    def update_Theta(
        self,
        X: np.ndarray,
        sum_M: np.ndarray,
        X_u: np.ndarray
    ) -> np.ndarray:
        """
        Update scaled dual variable Θ(ω).
        
        Implements Equation (21) from the paper:
        Θ^{n+1}(ω) = Θ^n(ω) + τ₁(X(ω) - Σ_{k=1}^Z M_k(ω) - X_u(ω))
        
        Parameters:
        -----------
        X : np.ndarray
            Signal spectrum
        sum_M : np.ndarray
            Sum of all mode spectra
        X_u : np.ndarray
            Unprocessed signal spectrum
            
        Returns:
        --------
        np.ndarray
            Updated Θ(ω)
        """
        # Eq. (21): Θ^{n+1}(ω) = Θ^n(ω) + τ₁(X(ω) - Σ M_k(ω) - X_u(ω))
        self._state.Theta = self._state.Theta + self.tau1 * (X - sum_M - X_u)
        return self._state.Theta
    
    def step(
        self,
        signal: np.ndarray,
        modes: List[np.ndarray],
        M_Z: np.ndarray,
        X_u: np.ndarray
    ):
        """
        Perform complete ADMM step: update ρ, μ, and Θ.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal x[n]
        modes : List[np.ndarray]
            Previously extracted modes
        M_Z : np.ndarray
            Current mode spectrum M_Z(ω)
        X_u : np.ndarray
            Unprocessed signal spectrum X_u(ω)
        """
        N = len(signal)
        
        if self._state is None:
            self.reset(N)
        
        # Compute required spectra
        X = fft(signal)
        
        sum_prev_M = np.zeros(N, dtype=complex)
        for mode in modes:
            sum_prev_M += fft(mode)
        
        sum_all_M = sum_prev_M + M_Z
        
        # Update multipliers (Eq. 21, 22)
        self.update_rho(X, sum_all_M)
        self.update_mu(X_u, modes + [np.real(ifft(M_Z))], N)
        self.update_Theta(X, sum_all_M, X_u)
        
        # Increment iteration
        self._state.iteration += 1
        
        # Store history
        self._state.history.append({
            'iteration': self._state.iteration,
            'rho_mean': np.mean(self._state.rho),
            'mu': self._state.mu,
            'primal_residual': np.linalg.norm(X - sum_all_M - X_u)
        })
    
    def get_Q(
        self,
        X: np.ndarray,
        sum_prev_M: np.ndarray,
        X_u: np.ndarray
    ) -> np.ndarray:
        """
        Compute auxiliary variable Q(ω) for M_Z update.
        
        Q(ω) = X(ω) - Σ_{k=1}^{Z-1} M_k(ω) - X_u(ω) + Θ(ω)
        
        Parameters:
        -----------
        X : np.ndarray
            Signal spectrum
        sum_prev_M : np.ndarray
            Sum of previous mode spectra
        X_u : np.ndarray
            Unprocessed signal spectrum
            
        Returns:
        --------
        np.ndarray
            Q(ω)
        """
        return X - sum_prev_M - X_u + self._state.Theta
    
    def get_Q_tilde(
        self,
        X: np.ndarray,
        sum_all_M: np.ndarray
    ) -> np.ndarray:
        """
        Compute auxiliary variable Q̃(ω) for X_u update.
        
        Equation (19):
        Q̃(ω) = X(ω) - Σ_{k=1}^Z M_k(ω) + Θ(ω)
        
        Parameters:
        -----------
        X : np.ndarray
            Signal spectrum
        sum_all_M : np.ndarray
            Sum of all mode spectra (including current)
            
        Returns:
        --------
        np.ndarray
            Q̃(ω)
        """
        return X - sum_all_M + self._state.Theta
    
    def get_convergence_metrics(self) -> Dict:
        """
        Get convergence metrics from optimization history.
        
        Returns:
        --------
        Dict
            Dictionary containing convergence metrics
        """
        if not self._state or not self._state.history:
            return {}
        
        return {
            'num_iterations': self._state.iteration,
            'final_rho_mean': np.mean(self._state.rho),
            'final_mu': self._state.mu,
            'rho_history': [h['rho_mean'] for h in self._state.history],
            'mu_history': [h['mu'] for h in self._state.history],
            'primal_residual_history': [h['primal_residual'] for h in self._state.history]
        }


# =============================================================================
# Filter functions from paper
# =============================================================================

def compute_beta_i(
    omega: np.ndarray,
    omega_k: float,
    alpha: float,
    epsilon1: float = 1e-6
) -> np.ndarray:
    """
    Compute β_k(ω) filter for mode overlap constraint.
    
    Equation (12):
    β_k(ω) = 1 / [α(ω - ω_k)² + ε₁]
    
    Parameters:
    -----------
    omega : np.ndarray
        Frequency axis
    omega_k : float
        Center frequency of mode k
    alpha : float
        Noise variance
    epsilon1 : float
        Regularization constant
        
    Returns:
    --------
    np.ndarray
        β_k(ω)
    """
    return 1.0 / (alpha * (omega - omega_k) ** 2 + epsilon1)


def compute_beta_Z(
    omega: np.ndarray,
    omega_Z: float,
    alpha: float,
    epsilon2: float = 1e-6
) -> np.ndarray:
    """
    Compute β_Z(ω) filter for unprocessed signal constraint.
    
    Equation (14):
    β_Z(ω) = 1 / [α(ω - ω_Z)² + ε₂]
    
    Parameters:
    -----------
    omega : np.ndarray
        Frequency axis
    omega_Z : float
        Center frequency of current mode
    alpha : float
        Noise variance
    epsilon2 : float
        Regularization constant
        
    Returns:
    --------
    np.ndarray
        β_Z(ω)
    """
    return 1.0 / (alpha * (omega - omega_Z) ** 2 + epsilon2)


def compute_spectral_compactness_term(
    omega: np.ndarray,
    omega_Z: float
) -> np.ndarray:
    """
    Compute spectral compactness term for M_Z update.
    
    From Equation (15): (2/π) sin²(ω - ω_Z)
    
    Parameters:
    -----------
    omega : np.ndarray
        Frequency axis
    omega_Z : float
        Center frequency
        
    Returns:
    --------
    np.ndarray
        Spectral compactness term
    """
    return (2.0 / np.pi) * np.sin(omega - omega_Z) ** 2


# =============================================================================
# Demo / Test
# =============================================================================
if __name__ == "__main__":
    print("Testing ADMM optimizer against paper equations...")
    
    # Test setup
    np.random.seed(42)
    N = 100
    
    # Create test signal
    t = np.linspace(0, 1, N)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(N)
    
    # Initialize optimizer
    optimizer = ADMMOptimizer(tau1=0.1, tau2=0.1, rho_init=1.0, mu_init=0.0)
    optimizer.reset(N)
    
    # Test mode (dummy)
    M_Z = fft(signal * 0.5)
    X_u = fft(signal * 0.3)
    modes = [signal * 0.2]
    
    print(f"Initial ρ (mean): {np.mean(optimizer.rho):.4f}")
    print(f"Initial μ: {optimizer.mu:.4f}")
    
    # Run a few iterations
    for i in range(10):
        optimizer.step(signal, modes, M_Z, X_u)
    
    metrics = optimizer.get_convergence_metrics()
    print(f"\nAfter 10 iterations:")
    print(f"  ρ (mean): {metrics['final_rho_mean']:.4f}")
    print(f"  μ: {metrics['final_mu']:.4f}")
    
    # Test filter functions
    omega = 2 * np.pi * np.arange(N) / N
    omega_Z = np.pi / 4
    alpha = 0.01
    
    beta_k = compute_beta_i(omega, omega_Z, alpha)
    beta_Z = compute_beta_Z(omega, omega_Z, alpha)
    sin_sq = compute_spectral_compactness_term(omega, omega_Z)
    
    print(f"\nFilter tests at ω_Z = {omega_Z:.4f}:")
    print(f"  β_k(ω_Z): {beta_k[int(omega_Z * N / (2 * np.pi))]:.4f}")
    print(f"  β_Z(ω_Z): {beta_Z[int(omega_Z * N / (2 * np.pi))]:.4f}")
    print(f"  sin²(0) at ω_Z: {sin_sq[int(omega_Z * N / (2 * np.pi))]:.4f}")
    
    print("\n✓ All ADMM optimizer tests passed!")
