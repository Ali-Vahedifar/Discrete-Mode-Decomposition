"""
Monte Carlo Approximation for Shapley Values
=============================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
IEEE INFOCOM 2025
This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

Implementation of Monte Carlo methods for efficient Shapley value computation.

Mathematical Background:
-----------------------
Computing exact Shapley values requires 2^Z evaluations, which is intractable
for large Z. Monte Carlo approximation rewrites the Shapley value as:

    X_k = E_U[V(S_k^U ∪ {k}) - V(S_k^U)]

where U is a uniform permutation of D and S_k^U is the set of elements
before k in permutation U.

The Monte Carlo algorithm:
1. Sample permutations uniformly
2. Scan through each permutation to compute marginal contributions
3. Average contributions across permutations

This gives an unbiased estimator with variance decreasing as O(1/T).

Convergence criterion (Eq. 24):
    (1/Z)Σ_{k=1}^Z |X_k^t - X_k^{t-100}|/|X_k^t| < 0.01
"""

import numpy as np
from typing import Callable, List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
import warnings
from tqdm import tqdm


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo Shapley approximation.
    
    Attributes:
        max_iterations: Maximum number of permutation samples
        tolerance: Convergence tolerance (Eq. 24)
        check_interval: Iterations between convergence checks
        use_antithetic: Use antithetic variates for variance reduction
        use_stratified: Use stratified sampling
        early_stopping_patience: Number of non-improving iterations before stop
        seed: Random seed
        verbose: Print progress
    """
    max_iterations: int = 1000
    tolerance: float = 0.01
    check_interval: int = 100
    use_antithetic: bool = True
    use_stratified: bool = False
    early_stopping_patience: int = 3
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class MonteCarloResult:
    """
    Result of Monte Carlo Shapley computation.
    
    Attributes:
        values: Estimated Shapley values
        std_errors: Standard errors of estimates
        num_iterations: Number of iterations performed
        converged: Whether estimation converged
        convergence_history: History of estimates
    """
    values: np.ndarray
    std_errors: np.ndarray
    num_iterations: int
    converged: bool
    convergence_history: List[dict] = field(default_factory=list)


class MonteCarloShapley:
    """
    Monte Carlo approximation for Shapley Mode Values.
    
    This class implements Algorithm 2 from the paper with several
    variance reduction techniques for faster convergence.
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    
    Variance Reduction Techniques:
    -----------------------------
    1. Antithetic variates: For each permutation, also use its reverse
    2. Stratified sampling: Ensure coverage of all mode positions
    3. Early stopping: Stop when marginal contributions become negligible
    
    Example:
    --------
    >>> mc = MonteCarloShapley(config)
    >>> result = mc.compute(Z=5, value_function=eval_coalition)
    >>> print(f"Shapley values: {result.values}")
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo Shapley estimator.
        
        Parameters:
        -----------
        config : MonteCarloConfig, optional
            Configuration for the estimator
        """
        self.config = config or MonteCarloConfig()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Storage for online statistics
        self._means: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None  # For Welford's algorithm
        self._counts: Optional[np.ndarray] = None
    
    def compute(
        self,
        Z: int,
        value_function: Callable[[Set[int]], float],
        performance_tolerance: Optional[float] = None
    ) -> MonteCarloResult:
        """
        Compute Shapley values using Monte Carlo approximation.
        
        Implements Algorithm 2: Monte Carlo Shapley.
        
        Parameters:
        -----------
        Z : int
            Number of modes
        value_function : Callable
            Function V(S) that returns performance score for coalition S.
            Takes a set of indices and returns a float.
        performance_tolerance : float, optional
            Tolerance for early stopping (ε₃ in Algorithm 2)
            
        Returns:
        --------
        MonteCarloResult
            Estimation result with Shapley values and statistics
        """
        # Initialize statistics
        self._means = np.zeros(Z)
        self._m2 = np.zeros(Z)
        self._counts = np.zeros(Z)
        
        # Get V(D) for early stopping
        V_full = value_function(set(range(Z)))
        epsilon_3 = performance_tolerance or self.config.tolerance * V_full
        
        convergence_history = []
        converged = False
        non_improving_count = 0
        prev_values = None
        
        pbar = tqdm(
            range(self.config.max_iterations),
            desc="Monte Carlo Shapley",
            disable=not self.config.verbose
        )
        
        for t in pbar:
            # Generate permutation(s)
            permutations = self._generate_permutations(Z, t)
            
            for perm in permutations:
                # Scan through permutation
                self._process_permutation(
                    perm, value_function, V_full, epsilon_3
                )
            
            # Check convergence
            if (t + 1) % self.config.check_interval == 0:
                current_values = self._means.copy()
                std_errors = self._compute_std_errors()
                
                # Store history
                convergence_history.append({
                    'iteration': t + 1,
                    'values': current_values.copy(),
                    'std_errors': std_errors.copy()
                })
                
                # Check convergence criterion (Eq. 24)
                if prev_values is not None:
                    rel_errors = np.abs(current_values - prev_values) / (np.abs(current_values) + 1e-10)
                    mean_rel_error = np.mean(rel_errors[np.isfinite(rel_errors)])
                    
                    if self.config.verbose:
                        pbar.set_postfix({
                            'rel_error': f'{mean_rel_error:.4f}',
                            'max_std': f'{np.max(std_errors):.4f}'
                        })
                    
                    if mean_rel_error < self.config.tolerance:
                        non_improving_count += 1
                        if non_improving_count >= self.config.early_stopping_patience:
                            converged = True
                            break
                    else:
                        non_improving_count = 0
                
                prev_values = current_values
        
        return MonteCarloResult(
            values=self._means,
            std_errors=self._compute_std_errors(),
            num_iterations=t + 1,
            converged=converged,
            convergence_history=convergence_history
        )
    
    def _generate_permutations(self, Z: int, iteration: int) -> List[np.ndarray]:
        """
        Generate permutation(s) for this iteration.
        
        Parameters:
        -----------
        Z : int
            Number of modes
        iteration : int
            Current iteration number
            
        Returns:
        --------
        List[np.ndarray]
            List of permutations to process
        """
        permutations = []
        
        # Main permutation
        perm = np.random.permutation(Z)
        permutations.append(perm)
        
        # Antithetic variate (reverse permutation)
        if self.config.use_antithetic:
            permutations.append(perm[::-1].copy())
        
        return permutations
    
    def _process_permutation(
        self,
        permutation: np.ndarray,
        value_function: Callable,
        V_full: float,
        epsilon_3: float
    ):
        """
        Process a single permutation and update Shapley estimates.
        
        Implements the inner loop of Algorithm 2.
        
        Parameters:
        -----------
        permutation : np.ndarray
            Permutation of mode indices
        value_function : Callable
            Coalition value function
        V_full : float
            Value of grand coalition
        epsilon_3 : float
            Early stopping threshold
        """
        M = len(permutation)
        v_prev = value_function(set())  # V(∅)
        
        coalition = set()
        
        for j, mode_idx in enumerate(permutation):
            # Add mode to coalition
            coalition.add(mode_idx)
            
            # Early stopping: if remaining value is small, skip
            if abs(V_full - v_prev) < epsilon_3:
                v_curr = v_prev
            else:
                v_curr = value_function(coalition.copy())
            
            # Marginal contribution
            marginal = v_curr - v_prev
            
            # Update statistics using Welford's online algorithm
            self._update_statistics(mode_idx, marginal)
            
            v_prev = v_curr
    
    def _update_statistics(self, mode_idx: int, value: float):
        """
        Update running statistics using Welford's algorithm.
        
        This computes online mean and variance without storing all values.
        
        Parameters:
        -----------
        mode_idx : int
            Index of mode to update
        value : float
            New marginal contribution value
        """
        self._counts[mode_idx] += 1
        n = self._counts[mode_idx]
        
        delta = value - self._means[mode_idx]
        self._means[mode_idx] += delta / n
        delta2 = value - self._means[mode_idx]
        self._m2[mode_idx] += delta * delta2
    
    def _compute_std_errors(self) -> np.ndarray:
        """
        Compute standard errors of Shapley estimates.
        
        Returns:
        --------
        np.ndarray
            Standard errors for each mode
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            variances = self._m2 / (self._counts - 1)
            variances = np.where(np.isfinite(variances), variances, 0)
            std_errors = np.sqrt(variances / self._counts)
            std_errors = np.where(np.isfinite(std_errors), std_errors, 0)
        
        return std_errors
    
    def compute_confidence_intervals(
        self,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for Shapley estimates.
        
        Parameters:
        -----------
        confidence : float
            Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds of confidence intervals
        """
        from scipy import stats
        
        if self._means is None:
            raise ValueError("Must call compute() first")
        
        z = stats.norm.ppf((1 + confidence) / 2)
        std_errors = self._compute_std_errors()
        
        lower = self._means - z * std_errors
        upper = self._means + z * std_errors
        
        return lower, upper


class StratifiedMonteCarloShapley(MonteCarloShapley):
    """
    Stratified Monte Carlo for improved Shapley estimation.
    
    Uses stratified sampling to ensure each mode appears in
    each position of the permutation with equal frequency.
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    """
    
    def _generate_permutations(self, Z: int, iteration: int) -> List[np.ndarray]:
        """
        Generate stratified permutation.
        
        Ensures uniform coverage of all positions over iterations.
        """
        permutations = []
        
        # Stratified: ensure position coverage
        if self.config.use_stratified:
            # Latin hypercube-like approach
            base_perm = np.random.permutation(Z)
            shift = iteration % Z
            perm = np.roll(base_perm, shift)
            permutations.append(perm)
        else:
            perm = np.random.permutation(Z)
            permutations.append(perm)
        
        if self.config.use_antithetic:
            permutations.append(perm[::-1].copy())
        
        return permutations


class KernelSHAP:
    """
    Kernel SHAP approximation for Shapley values.
    
    Uses weighted linear regression in feature space for faster
    Shapley value estimation.
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    
    Reference:
    ---------
    Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", 2017
    """
    
    def __init__(self, num_samples: int = 1000, l1_reg: float = 0.0):
        """
        Initialize Kernel SHAP.
        
        Parameters:
        -----------
        num_samples : int
            Number of coalition samples
        l1_reg : float
            L1 regularization strength
        """
        self.num_samples = num_samples
        self.l1_reg = l1_reg
    
    def compute(
        self,
        Z: int,
        value_function: Callable[[Set[int]], float]
    ) -> np.ndarray:
        """
        Compute Shapley values using Kernel SHAP.
        
        Parameters:
        -----------
        Z : int
            Number of modes
        value_function : Callable
            Coalition value function
            
        Returns:
        --------
        np.ndarray
            Estimated Shapley values
        """
        # Sample coalitions with Shapley kernel weights
        X, weights = self._sample_coalitions(Z)
        
        # Evaluate coalitions
        y = np.array([value_function(self._mask_to_set(x)) for x in X])
        
        # Weighted least squares
        shapley_values = self._weighted_regression(X, y, weights)
        
        return shapley_values
    
    def _sample_coalitions(self, Z: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample coalitions with Shapley kernel weights.
        
        Parameters:
        -----------
        Z : int
            Number of modes
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Coalition masks and weights
        """
        X = []
        weights = []
        
        # Always include empty and full coalition
        X.append(np.zeros(Z))
        weights.append(1e6)  # High weight
        X.append(np.ones(Z))
        weights.append(1e6)
        
        # Sample other coalitions
        for _ in range(self.num_samples - 2):
            # Random coalition size (favor small and large)
            s = np.random.choice(range(1, Z))
            
            # Random coalition of size s
            mask = np.zeros(Z)
            indices = np.random.choice(Z, s, replace=False)
            mask[indices] = 1
            
            X.append(mask)
            
            # Shapley kernel weight
            weight = self._shapley_kernel_weight(Z, s)
            weights.append(weight)
        
        return np.array(X), np.array(weights)
    
    def _shapley_kernel_weight(self, Z: int, s: int) -> float:
        """
        Compute Shapley kernel weight.
        
        weight(s) = (Z-1) / (C(Z,s) * s * (Z-s))
        
        Parameters:
        -----------
        Z : int
            Total number of modes
        s : int
            Coalition size
            
        Returns:
        --------
        float
            Kernel weight
        """
        from scipy.special import comb
        
        if s == 0 or s == Z:
            return 1e6  # Infinity for full/empty
        
        return (Z - 1) / (comb(Z, s) * s * (Z - s))
    
    def _mask_to_set(self, mask: np.ndarray) -> Set[int]:
        """Convert binary mask to set of indices."""
        return set(np.where(mask > 0.5)[0])
    
    def _weighted_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Perform weighted least squares regression.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (coalition masks)
        y : np.ndarray
            Target values
        weights : np.ndarray
            Sample weights
            
        Returns:
        --------
        np.ndarray
            Regression coefficients (Shapley values)
        """
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted least squares
        W = np.diag(weights)
        
        if self.l1_reg > 0:
            # Ridge regression with L1
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(alpha=self.l1_reg, l1_ratio=0.5, fit_intercept=False)
            model.fit(X, y, sample_weight=weights)
            return model.coef_
        else:
            # Standard weighted least squares
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            
            # Regularize for stability
            XtWX += 1e-6 * np.eye(X.shape[1])
            
            return np.linalg.solve(XtWX, XtWy)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create a simple value function
    Z = 6  # Number of modes
    
    # True Shapley values (for ground truth comparison)
    true_importance = np.array([0.4, 0.25, 0.15, 0.1, 0.07, 0.03])
    
    def value_function(coalition: Set[int]) -> float:
        """Simple additive value function."""
        if len(coalition) == 0:
            return 0.0
        return sum(true_importance[i] for i in coalition)
    
    # Compare different methods
    print("=" * 60)
    print("Monte Carlo Shapley Comparison")
    print("=" * 60)
    print(f"\nTrue Shapley values: {true_importance}")
    
    # Standard Monte Carlo
    print("\n1. Standard Monte Carlo:")
    config = MonteCarloConfig(
        max_iterations=500,
        tolerance=0.005,
        verbose=True
    )
    mc = MonteCarloShapley(config)
    result = mc.compute(Z, value_function)
    print(f"   Estimated: {result.values}")
    print(f"   Converged: {result.converged}, Iterations: {result.num_iterations}")
    
    # Kernel SHAP
    print("\n2. Kernel SHAP:")
    kshap = KernelSHAP(num_samples=200)
    kshap_values = kshap.compute(Z, value_function)
    print(f"   Estimated: {kshap_values}")
    
    # Compute errors
    mc_error = np.mean(np.abs(result.values - true_importance))
    kshap_error = np.mean(np.abs(kshap_values - true_importance))
    
    print(f"\nMean Absolute Errors:")
    print(f"   Monte Carlo: {mc_error:.6f}")
    print(f"   Kernel SHAP: {kshap_error:.6f}")
