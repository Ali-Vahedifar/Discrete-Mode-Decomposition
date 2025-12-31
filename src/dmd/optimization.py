"""
ADMM Optimization for Discrete Mode Decomposition
==================================================

Implementation of the Alternating Direction Method of Multipliers (ADMM)
for solving the DMD optimization problem.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

Mathematical Background:
-----------------------
The DMD optimization problem is:

    min_{u_M, ω_M} T1 + T2 + T3
    s.t. x[n] = Σ_{i=1}^{M} u_i[n] + x_u[n]
         ||x_u[n]||_2^2 ≤ ||u_min[n]||_2^2  (energy-based bound, Eq. 16)

The Augmented Lagrangian (Eq. 18) is:

    L_aug = T1 + T2 + T3 + (ρ/2)||x - Σu_i - x_u + θ||²
            + μ(||x_u||² - ||u_min||²) - (ρ/2)||θ||²

ADMM breaks this into sub-problems solved iteratively:
1. Update U_M(ω) - Eq. 24
2. Update ω_M - Eq. 27
3. Update X_u(ω) - Eq. 30
4. Update ρ(ω) - Eq. 31
5. Update μ - Eq. 32 (with max(0, ...) projection)
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class ADMMState:
    """
    State of ADMM optimization.
    
    Attributes:
        rho: Penalty parameter for equality constraint
        mu: Lagrangian multiplier for inequality constraint
        theta: Scaled dual variable (λ = ρ * θ)
        iteration: Current iteration number
        primal_residual: Primal residual norm
        dual_residual: Dual residual norm
    """
    rho: float = 1.0
    mu: float = 0.0
    theta: Optional[np.ndarray] = None
    iteration: int = 0
    primal_residual: float = float('inf')
    dual_residual: float = float('inf')
    history: List[dict] = field(default_factory=list)


class ADMMOptimizer:
    """
    ADMM Optimizer for Discrete Mode Decomposition.
    
    This class implements the ADMM algorithm for solving the constrained
    optimization problem in DMD. The algorithm iteratively updates the
    primal and dual variables until convergence.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    References:
    -----------
    [1] Boyd et al., "Distributed Optimization and Statistical Learning via
        the Alternating Direction Method of Multipliers", 2011
    [2] Bertsekas, "Constrained Optimization and Lagrange Multiplier Methods", 1996
    
    Example:
    --------
    >>> optimizer = ADMMOptimizer(tau1=0.1, tau2=0.1)
    >>> optimizer.reset(signal_length=1000)
    >>> for iteration in range(max_iter):
    ...     # Update mode and other variables
    ...     optimizer.update_multipliers(signal, modes, X_u, min_mode_magnitude)
    ...     if optimizer.check_convergence():
    ...         break
    """
    
    def __init__(
        self,
        tau1: float = 0.1,
        tau2: float = 0.1,
        rho_init: float = 1.0,
        rho_min: float = 1e-6,
        rho_max: float = 1e6,
        adaptive_rho: bool = True,
        mu_init: float = 0.0,
        abstol: float = 1e-4,
        reltol: float = 1e-3
    ):
        """
        Initialize ADMM optimizer.
        
        Parameters:
        -----------
        tau1 : float
            Step size for equality constraint update (Eq. 30)
        tau2 : float
            Step size for inequality constraint update (Eq. 31)
        rho_init : float
            Initial penalty parameter
        rho_min : float
            Minimum allowed penalty parameter
        rho_max : float
            Maximum allowed penalty parameter
        adaptive_rho : bool
            Whether to adaptively adjust rho based on residuals
        mu_init : float
            Initial Lagrangian multiplier for inequality constraint
        abstol : float
            Absolute tolerance for convergence
        reltol : float
            Relative tolerance for convergence
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.rho_init = rho_init
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.adaptive_rho = adaptive_rho
        self.mu_init = mu_init
        self.abstol = abstol
        self.reltol = reltol
        
        # Initialize state
        self._state = ADMMState(rho=rho_init, mu=mu_init)
    
    @property
    def rho(self) -> float:
        """Current penalty parameter."""
        return self._state.rho
    
    @property
    def mu(self) -> float:
        """Current inequality multiplier."""
        return self._state.mu
    
    @property
    def theta(self) -> Optional[np.ndarray]:
        """Current scaled dual variable."""
        return self._state.theta
    
    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._state.iteration
    
    def reset(self, signal_length: int):
        """
        Reset optimizer state for a new optimization.
        
        Parameters:
        -----------
        signal_length : int
            Length of the signal being processed
        """
        self._state = ADMMState(
            rho=self.rho_init,
            mu=self.mu_init,
            theta=np.zeros(signal_length),
            iteration=0,
            primal_residual=float('inf'),
            dual_residual=float('inf'),
            history=[]
        )
    
    def update_multipliers(
        self,
        signal: np.ndarray,
        modes: List[np.ndarray],
        X_u: np.ndarray,
        min_mode_magnitude: float
    ):
        """
        Update Lagrangian multipliers using dual ascent.
        
        Implements Equations 30 and 31 from the paper.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal x[n]
        modes : List[np.ndarray]
            List of all extracted modes including current
        X_u : np.ndarray
            Unprocessed signal component (frequency domain)
        min_mode_magnitude : float
            Magnitude of smallest mode for inequality constraint
        """
        N = len(signal)
        
        # Ensure theta is initialized
        if self._state.theta is None:
            self._state.theta = np.zeros(N)
        
        # Compute residuals
        mode_sum = np.sum(modes, axis=0) if modes else np.zeros(N)
        x_u_time = np.real(ifft(X_u)) if np.iscomplexobj(X_u) else X_u
        
        # Primal residual: x - Σu_i - x_u
        primal_residual = signal - mode_sum - x_u_time
        
        # Update equality constraint multiplier - Eq. 30
        # ρ^{n+1}(ω) = ρ^n(ω) + τ_1 * (X(ω) - Σ U_i(ω))
        X = fft(signal)
        sum_U = fft(mode_sum) if len(modes) > 0 else np.zeros(N, dtype=complex)
        
        self._state.rho = self._state.rho + self.tau1 * np.mean(np.abs(X - sum_U))
        
        # Clip rho to valid range
        self._state.rho = np.clip(self._state.rho, self.rho_min, self.rho_max)
        
        # Update theta (scaled dual variable)
        self._state.theta = self._state.theta + primal_residual
        
        # Update inequality constraint multiplier - Eq. 32 (energy-based)
        # μ^{n+1} = max(0, μ^n + τ_2 * ∫(||X_u(ω)||² - ||U_min(ω)||²) dω)
        x_u_energy = np.sum(np.abs(X_u) ** 2)
        min_mode_energy = min_mode_magnitude ** 2 * len(X_u)  # Energy of min mode
        inequality_violation = x_u_energy - min_mode_energy
        
        self._state.mu = max(0, self._state.mu + self.tau2 * inequality_violation)
        
        # Store residuals
        self._state.primal_residual = np.linalg.norm(primal_residual)
        
        # Increment iteration
        self._state.iteration += 1
        
        # Store history
        self._state.history.append({
            'iteration': self._state.iteration,
            'rho': self._state.rho,
            'mu': self._state.mu,
            'primal_residual': self._state.primal_residual,
            'inequality_violation': inequality_violation
        })
        
        # Adaptive rho adjustment
        if self.adaptive_rho:
            self._adapt_rho()
    
    def _adapt_rho(self):
        """
        Adaptively adjust penalty parameter based on residuals.
        
        Uses the standard ADMM adaptive scheme:
        - Increase rho if primal residual >> dual residual
        - Decrease rho if dual residual >> primal residual
        """
        if len(self._state.history) < 2:
            return
        
        current = self._state.history[-1]
        previous = self._state.history[-2]
        
        # Estimate dual residual from rho change
        rho_change = current['rho'] - previous['rho']
        
        # Adaptation parameters
        mu_adapt = 10  # Factor for residual comparison
        tau_adapt = 2  # Adaptation step
        
        primal = current['primal_residual']
        dual_estimate = abs(rho_change) * 100  # Rough estimate
        
        if primal > mu_adapt * dual_estimate:
            self._state.rho *= tau_adapt
        elif dual_estimate > mu_adapt * primal:
            self._state.rho /= tau_adapt
        
        # Clip to valid range
        self._state.rho = np.clip(self._state.rho, self.rho_min, self.rho_max)
    
    def check_convergence(
        self,
        modes: Optional[List[np.ndarray]] = None,
        signal: Optional[np.ndarray] = None,
        tol: Optional[float] = None
    ) -> bool:
        """
        Check if ADMM has converged.
        
        Convergence is determined by the primal and dual residuals
        being below specified tolerances.
        
        Parameters:
        -----------
        modes : List[np.ndarray], optional
            Current modes for computing residuals
        signal : np.ndarray, optional
            Original signal
        tol : float, optional
            Tolerance override
            
        Returns:
        --------
        bool
            True if converged, False otherwise
        """
        if tol is None:
            tol = self.reltol
        
        # Simple convergence check based on residual
        if self._state.primal_residual < tol:
            return True
        
        # Check if residuals are decreasing
        if len(self._state.history) >= 10:
            recent = self._state.history[-10:]
            residuals = [h['primal_residual'] for h in recent]
            
            # Converged if residuals are stable and small
            if np.std(residuals) < 0.1 * np.mean(residuals):
                if np.mean(residuals) < tol * 10:
                    return True
        
        return False
    
    def get_convergence_metrics(self) -> dict:
        """
        Get convergence metrics from the optimization history.
        
        Returns:
        --------
        dict
            Dictionary containing convergence metrics
        """
        if not self._state.history:
            return {}
        
        history = self._state.history
        
        return {
            'num_iterations': len(history),
            'final_rho': history[-1]['rho'],
            'final_mu': history[-1]['mu'],
            'final_primal_residual': history[-1]['primal_residual'],
            'rho_history': [h['rho'] for h in history],
            'mu_history': [h['mu'] for h in history],
            'primal_residual_history': [h['primal_residual'] for h in history]
        }


class ProximalOperators:
    """
    Collection of proximal operators for ADMM sub-problems.
    
    Proximal operators are key building blocks for ADMM algorithms,
    providing closed-form solutions for various regularization terms.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    @staticmethod
    def prox_l1(x: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Proximal operator for L1 norm (soft thresholding).
        
        prox_{λ||·||_1}(x) = sign(x) * max(|x| - λ, 0)
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
        lambda_ : float
            Regularization parameter
            
        Returns:
        --------
        np.ndarray
            Result of proximal operator
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    @staticmethod
    def prox_l2(x: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Proximal operator for L2 norm (shrinkage).
        
        prox_{λ||·||_2}(x) = max(1 - λ/||x||_2, 0) * x
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
        lambda_ : float
            Regularization parameter
            
        Returns:
        --------
        np.ndarray
            Result of proximal operator
        """
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            return np.zeros_like(x)
        return np.maximum(1 - lambda_ / norm_x, 0) * x
    
    @staticmethod
    def prox_box(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """
        Proximal operator for box constraints.
        
        prox_{I_{[l,u]}}(x) = clip(x, l, u)
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
        lower : float
            Lower bound
        upper : float
            Upper bound
            
        Returns:
        --------
        np.ndarray
            Projected vector
        """
        return np.clip(x, lower, upper)
    
    @staticmethod
    def prox_quadratic(
        x: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        rho: float
    ) -> np.ndarray:
        """
        Proximal operator for quadratic function (1/2)x^TAx + b^Tx.
        
        prox_{f}(x) = (A + ρI)^{-1}(ρx - b)
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
        A : np.ndarray
            Quadratic term matrix
        b : np.ndarray
            Linear term vector
        rho : float
            Penalty parameter
            
        Returns:
        --------
        np.ndarray
            Result of proximal operator
        """
        n = len(x)
        return np.linalg.solve(A + rho * np.eye(n), rho * x - b)


class ConsensusADMM:
    """
    Consensus ADMM for distributed optimization.
    
    Useful for parallel processing of multiple signals or modes.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        num_agents: int,
        rho: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ):
        """
        Initialize consensus ADMM.
        
        Parameters:
        -----------
        num_agents : int
            Number of agents (e.g., signals or modes)
        rho : float
            Penalty parameter
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
        """
        self.num_agents = num_agents
        self.rho = rho
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self.z = None  # Consensus variable
        self.u = None  # Dual variables
    
    def solve(
        self,
        local_solvers: List[callable],
        initial_z: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve distributed optimization using consensus ADMM.
        
        Parameters:
        -----------
        local_solvers : List[callable]
            List of functions that solve local sub-problems.
            Each takes (z - u) and returns local solution x_i.
        initial_z : np.ndarray
            Initial consensus variable
            
        Returns:
        --------
        Tuple[np.ndarray, List[np.ndarray]]
            Consensus solution and list of local solutions
        """
        n = len(initial_z)
        
        # Initialize
        self.z = initial_z.copy()
        self.u = [np.zeros(n) for _ in range(self.num_agents)]
        x = [np.zeros(n) for _ in range(self.num_agents)]
        
        for iteration in range(self.max_iterations):
            z_old = self.z.copy()
            
            # Local updates (can be parallelized)
            for i in range(self.num_agents):
                x[i] = local_solvers[i](self.z - self.u[i])
            
            # Consensus update: z = mean(x_i + u_i)
            self.z = np.mean([x[i] + self.u[i] for i in range(self.num_agents)], axis=0)
            
            # Dual update
            for i in range(self.num_agents):
                self.u[i] = self.u[i] + x[i] - self.z
            
            # Check convergence
            primal_residual = np.sqrt(sum(np.linalg.norm(x[i] - self.z)**2 
                                         for i in range(self.num_agents)))
            dual_residual = np.linalg.norm(self.rho * (self.z - z_old))
            
            if primal_residual < self.tolerance and dual_residual < self.tolerance:
                break
        
        return self.z, x


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a simple test problem
    np.random.seed(42)
    N = 100
    
    # Original signal
    t = np.linspace(0, 1, N)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(N)
    
    # Initialize optimizer
    optimizer = ADMMOptimizer(tau1=0.1, tau2=0.1, rho_init=1.0)
    optimizer.reset(N)
    
    # Simulate optimization iterations
    modes = [signal * 0.5]  # Dummy mode
    X_u = fft(signal * 0.5)
    min_mode_mag = 0.1
    
    print("Running ADMM optimization...")
    for i in range(50):
        optimizer.update_multipliers(signal, modes, X_u, min_mode_mag)
        
        if optimizer.check_convergence():
            print(f"Converged at iteration {i+1}")
            break
    
    # Get convergence metrics
    metrics = optimizer.get_convergence_metrics()
    print(f"\nOptimization completed:")
    print(f"  Iterations: {metrics['num_iterations']}")
    print(f"  Final rho: {metrics['final_rho']:.4f}")
    print(f"  Final mu: {metrics['final_mu']:.4f}")
    print(f"  Final primal residual: {metrics['final_primal_residual']:.6f}")
    
    # Plot convergence
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    iterations = range(1, len(metrics['rho_history']) + 1)
    
    axes[0].plot(iterations, metrics['rho_history'], 'b-')
    axes[0].set_title('Penalty Parameter ρ')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('ρ')
    axes[0].set_yscale('log')
    
    axes[1].plot(iterations, metrics['mu_history'], 'g-')
    axes[1].set_title('Inequality Multiplier μ')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('μ')
    
    axes[2].plot(iterations, metrics['primal_residual_history'], 'r-')
    axes[2].set_title('Primal Residual')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('||r||')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('admm_convergence.png', dpi=150)
    print("\nSaved convergence plot to 'admm_convergence.png'")
