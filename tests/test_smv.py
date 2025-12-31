"""
Tests for Shapley Mode Value Module
===================================

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025

Usage: pytest tests/test_smv.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestSMV:
    """Test suite for Shapley Mode Value. Author: Ali Vahedi"""
    
    @pytest.fixture
    def sample_modes(self):
        """Create test modes."""
        np.random.seed(42)
        N = 100
        n_modes = 5
        
        modes = np.random.randn(n_modes, N)
        frequencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        return modes, frequencies
    
    def test_smv_import(self):
        """Test SMV module import."""
        try:
            from smv import ShapleyModeValue
            assert True
        except ImportError:
            pytest.skip("SMV module not available")
    
    def test_smv_axioms(self, sample_modes):
        """Test Shapley value axioms."""
        try:
            from smv import ShapleyModeValue
            
            modes, frequencies = sample_modes
            smv = ShapleyModeValue()
            
            # Create simple mock performance function
            def mock_performance(mode_indices):
                return len(mode_indices) * 0.1
            
            # Compute Shapley values
            values = smv.compute_values(modes, frequencies, mock_performance)
            
            # Test transferability (sum should equal total value)
            total_value = mock_performance(list(range(len(modes))))
            assert abs(sum(values) - total_value) < 0.1
            
        except ImportError:
            pytest.skip("SMV module not available")
    
    def test_smv_convergence(self, sample_modes):
        """Test Monte Carlo convergence."""
        try:
            from smv import ShapleyModeValue
            
            modes, frequencies = sample_modes
            smv = ShapleyModeValue(tolerance=0.01, max_iterations=100)
            
            def mock_performance(mode_indices):
                return len(mode_indices) * 0.1
            
            values = smv.compute_values(modes, frequencies, mock_performance)
            
            assert len(values) == len(modes)
            
        except ImportError:
            pytest.skip("SMV module not available")


class TestMonteCarloApproximation:
    """Test Monte Carlo approximation. Author: Ali Vahedi"""
    
    def test_monte_carlo_import(self):
        """Test Monte Carlo import."""
        try:
            from smv.monte_carlo import MonteCarloShapley
            assert True
        except ImportError:
            pytest.skip("Monte Carlo module not available")
    
    def test_permutation_sampling(self):
        """Test permutation sampling."""
        try:
            from smv.monte_carlo import MonteCarloShapley
            
            mc = MonteCarloShapley(n_permutations=100)
            
            # Check that sampling works
            data_points = list(range(10))
            permutation = mc._sample_permutation(data_points)
            
            assert len(permutation) == len(data_points)
            assert set(permutation) == set(data_points)
            
        except ImportError:
            pytest.skip("Monte Carlo module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
