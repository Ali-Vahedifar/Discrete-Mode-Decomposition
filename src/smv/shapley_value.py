"""
Shapley Mode Value (SMV) Implementation
=======================================

Implementation of the Shapley Mode Value algorithm for quantifying
the contribution of each mode to the prediction task.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

Mathematical Background:
-----------------------
The Shapley value provides a fair attribution of contributions in cooperative
game theory. For mode valuation, we define:

Let D = {(u_i, ω_i)}_{i=1}^M be the training set containing modes and their
center frequencies. Let Z denote the DMD algorithm, and S ⊆ D.

The Shapley Mode Value X_i satisfies:
1. Transferability: Σ_{i∈D} X_i = V(D) (sum of values equals total utility)
2. Monotonicity: Combines null contribution, symmetry, and linearity axioms

The unique solution is (Eq. 33):
    X_i = Σ_{S⊆D\{i}} [|S|!(|D|-|S|-1)!/|D|!] * [V(S∪{i}) - V(S)]

Convergence criterion (Eq. 36): Average change in Shapley values over 100
iterations is less than 1%.

Algorithm returns: top K modes {X_i}_{i=1}^K, where K ≤ M
"""

import numpy as np
from typing import List, Optional, Callable, Dict, Tuple, Union
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import warnings


@dataclass
class ShapleyConfig:
    """
    Configuration for Shapley Mode Value computation.
    
    Attributes:
        tolerance: Convergence tolerance for Monte Carlo approximation
        max_iterations: Maximum number of Monte Carlo iterations
        performance_tolerance: Threshold for early stopping (Eq. 36)
        bootstrap_samples: Number of bootstrap samples for variance estimation
        use_antithetic: Use antithetic sampling for variance reduction
        seed: Random seed for reproducibility
        verbose: Whether to print progress
    """
    tolerance: float = 0.01
    max_iterations: int = 1000
    performance_tolerance: float = 0.01
    bootstrap_samples: int = 100
    use_antithetic: bool = True
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class ShapleyResult:
    """
    Result of Shapley Mode Value computation.
    
    Attributes:
        values: Shapley values for each mode
        rankings: Sorted indices from most to least important
        num_iterations: Number of iterations used
        converged: Whether computation converged
        convergence_history: History of convergence metrics
        selected_modes: Indices of selected important modes
    """
    values: np.ndarray
    rankings: np.ndarray
    num_iterations: int
    converged: bool
    convergence_history: List[dict] = field(default_factory=list)
    selected_modes: Optional[np.ndarray] = None


class ShapleyModeValue:
    """
    Shapley Mode Value (SMV) for mode importance quantification.
    
    This class implements the SMV algorithm that determines the contribution
    of each mode to the overall prediction performance. It enables:
    1. Accelerating inference by retaining only task-relevant modes
    2. Enhancing accuracy by prioritizing meaningful signal modes
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    The implementation follows Algorithm 2 from the paper, using Monte Carlo
    approximation for tractable computation.
    
    Example:
    --------
    >>> smv = ShapleyModeValue(config)
    >>> result = smv.compute(modes, center_freqs, predictor, validation_data)
    >>> important_modes = modes[result.selected_modes]
    """
    
    def __init__(
        self,
        config: Optional[ShapleyConfig] = None,
        performance_metric: str = 'accuracy'
    ):
        """
        Initialize Shapley Mode Value computer.
        
        Parameters:
        -----------
        config : ShapleyConfig, optional
            Configuration for SMV computation
        performance_metric : str
            Metric to use for evaluating mode contribution.
            Options: 'accuracy', 'mse', 'mae', 'psnr'
        """
        self.config = config or ShapleyConfig()
        self.performance_metric = performance_metric
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Performance oracle cache
        self._performance_cache: Dict[frozenset, float] = {}
    
    def compute(
        self,
        modes: np.ndarray,
        center_frequencies: np.ndarray,
        predictor: Callable,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_selected: Optional[int] = None
    ) -> ShapleyResult:
        """
        Compute Shapley Mode Values.
        
        Implements Algorithm 2: Monte Carlo Shapley approximation.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of modes, shape (M, N)
        center_frequencies : np.ndarray
            Center frequencies of each mode
        predictor : Callable
            Prediction function that takes modes and returns predictions
        X_val : np.ndarray
            Validation input data
        y_val : np.ndarray
            Validation target data
        num_selected : int, optional
            Number of top modes to select. If None, select all non-negative.
            
        Returns:
        --------
        ShapleyResult
            Result containing Shapley values and mode rankings
        """
        M = len(modes)
        
        if M == 0:
            return ShapleyResult(
                values=np.array([]),
                rankings=np.array([]),
                num_iterations=0,
                converged=True
            )
        
        # Use exact computation for small M, Monte Carlo for large M
        if M <= 8:
            shapley_values, converged = self._compute_exact(
                modes, center_frequencies, predictor, X_val, y_val
            )
            num_iterations = 2 ** M
        else:
            shapley_values, num_iterations, converged = self._compute_monte_carlo(
                modes, center_frequencies, predictor, X_val, y_val
            )
        
        # Rank modes by Shapley value (descending)
        rankings = np.argsort(shapley_values)[::-1]
        
        # Select top modes
        if num_selected is None:
            # Select modes with positive contribution
            selected_modes = np.where(shapley_values > 0)[0]
            if len(selected_modes) == 0:
                selected_modes = rankings[:max(1, M // 2)]
        else:
            selected_modes = rankings[:num_selected]
        
        return ShapleyResult(
            values=shapley_values,
            rankings=rankings,
            num_iterations=num_iterations,
            converged=converged,
            selected_modes=selected_modes
        )
    
    def _compute_exact(
        self,
        modes: np.ndarray,
        center_frequencies: np.ndarray,
        predictor: Callable,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute exact Shapley values (for small M).
        
        Implements Equation 34:
            X_i = Σ_{S⊆D\{i}} [|S|!(|D|-|S|-1)!/|D|!] * [V(S∪{i}) - V(S)]
        
        Parameters:
        -----------
        modes, center_frequencies, predictor, X_val, y_val : as in compute()
        
        Returns:
        --------
        Tuple[np.ndarray, bool]
            Shapley values and convergence flag
        """
        M = len(modes)
        shapley_values = np.zeros(M)
        
        # Precompute factorial
        factorial = np.array([np.math.factorial(k) for k in range(M + 1)])
        
        # Iterate over all subsets
        all_indices = set(range(M))
        
        for i in range(M):
            other_indices = all_indices - {i}
            
            for subset_size in range(M):
                for subset in itertools.combinations(other_indices, subset_size):
                    subset = set(subset)
                    
                    # Compute weight: |S|!(M-|S|-1)!/M!
                    s = len(subset)
                    weight = (factorial[s] * factorial[M - s - 1]) / factorial[M]
                    
                    # V(S ∪ {i})
                    v_with_i = self._evaluate_coalition(
                        modes, list(subset | {i}), predictor, X_val, y_val
                    )
                    
                    # V(S)
                    v_without_i = self._evaluate_coalition(
                        modes, list(subset), predictor, X_val, y_val
                    )
                    
                    # Marginal contribution
                    marginal = v_with_i - v_without_i
                    
                    shapley_values[i] += weight * marginal
        
        return shapley_values, True
    
    def _compute_monte_carlo(
        self,
        modes: np.ndarray,
        center_frequencies: np.ndarray,
        predictor: Callable,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Compute Shapley values using Monte Carlo approximation.
        
        Implements Algorithm 2: Monte Carlo Shapley.
        
        The algorithm samples random permutations and computes marginal
        contributions, which gives an unbiased estimate of Shapley values.
        
        Parameters:
        -----------
        modes, center_frequencies, predictor, X_val, y_val : as in compute()
        
        Returns:
        --------
        Tuple[np.ndarray, int, bool]
            Shapley values, number of iterations, and convergence flag
        """
        M = len(modes)
        shapley_values = np.zeros(M)
        value_counts = np.zeros(M)
        
        # Value of full coalition (for early stopping)
        V_full = self._evaluate_coalition(
            modes, list(range(M)), predictor, X_val, y_val
        )
        
        converged = False
        convergence_history = []
        
        pbar = tqdm(
            range(self.config.max_iterations),
            desc="SMV Monte Carlo",
            disable=not self.config.verbose
        )
        
        for t in pbar:
            # Sample random permutation
            permutation = np.random.permutation(M)
            
            # Antithetic sampling for variance reduction
            if self.config.use_antithetic:
                permutations = [permutation, permutation[::-1]]
            else:
                permutations = [permutation]
            
            for perm in permutations:
                # Initialize
                v_prev = self._evaluate_coalition(modes, [], predictor, X_val, y_val)
                
                for j, mode_idx in enumerate(perm):
                    # Current coalition (modes before j in permutation)
                    coalition = list(perm[:j + 1])
                    
                    # Check early stopping condition
                    if abs(V_full - v_prev) < self.config.performance_tolerance:
                        v_curr = v_prev
                    else:
                        v_curr = self._evaluate_coalition(
                            modes, coalition, predictor, X_val, y_val
                        )
                    
                    # Marginal contribution
                    marginal = v_curr - v_prev
                    
                    # Update Shapley value estimate (online mean)
                    value_counts[mode_idx] += 1
                    shapley_values[mode_idx] += (marginal - shapley_values[mode_idx]) / value_counts[mode_idx]
                    
                    v_prev = v_curr
            
            # Check convergence every 100 iterations (Eq. 36)
            if (t + 1) % 100 == 0 and t > 0:
                if len(convergence_history) > 0:
                    prev_values = convergence_history[-1]['values']
                    
                    # Mean relative absolute error
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rel_errors = np.abs(shapley_values - prev_values) / (np.abs(shapley_values) + 1e-10)
                    mean_rel_error = np.mean(rel_errors[np.isfinite(rel_errors)])
                    
                    if self.config.verbose:
                        pbar.set_postfix({'rel_error': f'{mean_rel_error:.4f}'})
                    
                    if mean_rel_error < self.config.tolerance:
                        converged = True
                        break
                
                convergence_history.append({
                    'iteration': t + 1,
                    'values': shapley_values.copy()
                })
        
        return shapley_values, t + 1, converged
    
    def _evaluate_coalition(
        self,
        modes: np.ndarray,
        coalition: List[int],
        predictor: Callable,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Evaluate performance of a coalition of modes.
        
        This is the performance oracle V(S, Z) from the paper.
        
        Parameters:
        -----------
        modes : np.ndarray
            All modes
        coalition : List[int]
            Indices of modes in the coalition
        predictor : Callable
            Prediction function
        X_val, y_val : np.ndarray
            Validation data
            
        Returns:
        --------
        float
            Performance score of the coalition
        """
        # Check cache
        coalition_key = frozenset(coalition)
        if coalition_key in self._performance_cache:
            return self._performance_cache[coalition_key]
        
        # Empty coalition
        if len(coalition) == 0:
            # Baseline performance (e.g., mean prediction)
            if self.performance_metric == 'accuracy':
                performance = 0.0
            else:
                performance = -np.mean((y_val - np.mean(y_val)) ** 2)
        else:
            # Select modes in coalition
            selected_modes = modes[coalition]
            
            try:
                # Get predictions
                predictions = predictor(selected_modes, X_val)
                
                # Compute performance metric
                performance = self._compute_metric(predictions, y_val)
            except Exception as e:
                warnings.warn(f"Evaluation failed for coalition {coalition}: {e}")
                performance = 0.0
        
        # Cache result
        self._performance_cache[coalition_key] = performance
        
        return performance
    
    def _compute_metric(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute performance metric.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Ground truth targets
            
        Returns:
        --------
        float
            Performance score (higher is better)
        """
        if self.performance_metric == 'accuracy':
            # Classification accuracy
            if predictions.ndim > 1:
                pred_labels = np.argmax(predictions, axis=1)
            else:
                pred_labels = (predictions > 0.5).astype(int)
            
            if targets.ndim > 1:
                true_labels = np.argmax(targets, axis=1)
            else:
                true_labels = targets.astype(int)
            
            return np.mean(pred_labels == true_labels)
        
        elif self.performance_metric == 'mse':
            # Negative MSE (so higher is better)
            return -np.mean((predictions - targets) ** 2)
        
        elif self.performance_metric == 'mae':
            # Negative MAE
            return -np.mean(np.abs(predictions - targets))
        
        elif self.performance_metric == 'psnr':
            # PSNR
            mse = np.mean((predictions - targets) ** 2)
            if mse == 0:
                return 100.0
            max_val = np.max(targets)
            return 10 * np.log10(max_val ** 2 / mse)
        
        else:
            raise ValueError(f"Unknown metric: {self.performance_metric}")
    
    def clear_cache(self):
        """Clear the performance cache."""
        self._performance_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the performance cache."""
        return {
            'size': len(self._performance_cache),
            'memory_bytes': sum(
                len(str(k)) + 8 for k in self._performance_cache.keys()
            )
        }


class ShapleyModeValueFast:
    """
    Fast approximation of Shapley Mode Value using gradient-based methods.
    
    This is a faster alternative for scenarios where the predictor is
    differentiable and we can use gradient information.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, num_samples: int = 100):
        """
        Initialize fast SMV.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples for gradient estimation
        """
        self.num_samples = num_samples
    
    def compute(
        self,
        modes: np.ndarray,
        predictor: Callable,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> np.ndarray:
        """
        Compute approximate Shapley values using gradient-based method.
        
        Uses integrated gradients as an approximation.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of modes
        predictor : Callable
            Differentiable prediction function
        X_val, y_val : np.ndarray
            Validation data
            
        Returns:
        --------
        np.ndarray
            Approximate Shapley values
        """
        M = len(modes)
        
        # Baseline (all zeros)
        baseline = np.zeros_like(modes)
        
        # Integrated gradients
        shapley_values = np.zeros(M)
        
        for alpha in np.linspace(0, 1, self.num_samples):
            interpolated = baseline + alpha * (modes - baseline)
            
            # Numerical gradient
            eps = 1e-5
            for i in range(M):
                modes_plus = interpolated.copy()
                modes_minus = interpolated.copy()
                modes_plus[i] += eps
                modes_minus[i] -= eps
                
                pred_plus = predictor(modes_plus, X_val)
                pred_minus = predictor(modes_minus, X_val)
                
                gradient = (np.mean(pred_plus) - np.mean(pred_minus)) / (2 * eps)
                shapley_values[i] += gradient * (modes[i].sum() - baseline[i].sum())
        
        shapley_values /= self.num_samples
        
        return shapley_values


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create synthetic modes
    N = 100  # Signal length
    M = 5    # Number of modes
    
    t = np.linspace(0, 1, N)
    modes = np.array([
        np.sin(2 * np.pi * (i + 1) * 5 * t)
        for i in range(M)
    ])
    center_freqs = np.array([5 * (i + 1) for i in range(M)])
    
    # Add varying importance
    importance = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    modes = modes * importance[:, np.newaxis]
    
    # Simple predictor function
    def simple_predictor(selected_modes, X):
        """Sum of modes as prediction."""
        if len(selected_modes) == 0:
            return np.zeros(len(X))
        return np.sum(selected_modes, axis=0)[:len(X)]
    
    # Validation data
    X_val = np.arange(N)
    y_val = np.sum(modes, axis=0)
    
    # Compute Shapley values
    config = ShapleyConfig(verbose=True, max_iterations=500)
    smv = ShapleyModeValue(config=config, performance_metric='mse')
    
    print("Computing Shapley Mode Values...")
    result = smv.compute(modes, center_freqs, simple_predictor, X_val, y_val)
    
    print(f"\nShapley values: {result.values}")
    print(f"Rankings: {result.rankings}")
    print(f"True importance: {importance}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.num_iterations}")
    print(f"Selected modes: {result.selected_modes}")
    
    # Verify that Shapley values roughly correspond to true importance
    correlation = np.corrcoef(result.values, importance)[0, 1]
    print(f"\nCorrelation with true importance: {correlation:.4f}")
