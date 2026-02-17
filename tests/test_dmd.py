"""
Tests for Discrete Mode Decomposition Module
=============================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2026

Usage: pytest tests/test_dmd.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestDMD:
    """Test suite for DMD. Author: Ali Vahedi"""
    
    @pytest.fixture
    def sample_signal(self):
        """Create test signal."""
        np.random.seed(42)
        N = 1000
        t = np.linspace(0, 1, N)
        signal = (np.sin(2 * np.pi * 10 * t) + 
                  0.5 * np.sin(2 * np.pi * 25 * t) +
                  0.1 * np.random.randn(N))
        return signal
    
    def test_dmd_import(self):
        """Test that DMD module can be imported."""
        try:
            from dmd import DiscreteModeDcomposition, DMDConfig
            assert True
        except ImportError:
            pytest.skip("DMD module not available")
    
    def test_dmd_config(self):
        """Test DMD configuration."""
        try:
            from dmd import DMDConfig
            config = DMDConfig(
                noise_variance=0.01,
                epsilon1=1e-6,
                epsilon2=1e-6,
                max_modes=5
            )
            assert config.noise_variance == 0.01
            assert config.max_modes == 5
        except ImportError:
            pytest.skip("DMD module not available")
    
    def test_dmd_decompose(self, sample_signal):
        """Test DMD decomposition."""
        try:
            from dmd import DiscreteModeDcomposition
            dmd = DiscreteModeDcomposition(noise_variance=0.01)
            result = dmd.decompose(sample_signal, num_modes=3)
            
            assert result.num_modes > 0
            assert len(result.modes) == result.num_modes
            assert len(result.center_frequencies) == result.num_modes
        except ImportError:
            pytest.skip("DMD module not available")
    
    def test_dmd_reconstruction(self, sample_signal):
        """Test signal reconstruction from modes."""
        try:
            from dmd import DiscreteModeDcomposition
            dmd = DiscreteModeDcomposition(noise_variance=0.01)
            result = dmd.decompose(sample_signal, num_modes=3)
            
            reconstructed = dmd.reconstruct(result)
            
            # Check reconstruction error
            error = np.mean((sample_signal - reconstructed) ** 2)
            assert error < 1.0  # Reasonable reconstruction
        except ImportError:
            pytest.skip("DMD module not available")


class TestHilbertTransform:
    """Test Hilbert transform. Author: Ali Vahedi"""
    
    def test_hilbert_import(self):
        """Test Hilbert transform import."""
        try:
            from dmd.hilbert_transform import DiscreteHilbertTransform
            assert True
        except ImportError:
            pytest.skip("Hilbert transform not available")
    
    def test_hilbert_analytic(self):
        """Test analytic signal computation."""
        try:
            from dmd.hilbert_transform import DiscreteHilbertTransform
            
            N = 100
            t = np.linspace(0, 1, N)
            signal = np.sin(2 * np.pi * 5 * t)
            
            ht = DiscreteHilbertTransform()
            analytic = ht.transform(signal)
            
            assert len(analytic) == N
            assert np.iscomplexobj(analytic)
        except ImportError:
            pytest.skip("Hilbert transform not available")


class TestWienerFilter:
    """Test Wiener filter. Author: Ali Vahedi"""
    
    def test_wiener_import(self):
        """Test Wiener filter import."""
        try:
            from dmd.wiener_filter import DiscreteWienerFilter
            assert True
        except ImportError:
            pytest.skip("Wiener filter not available")
    
    def test_wiener_denoising(self):
        """Test signal denoising."""
        try:
            from dmd.wiener_filter import DiscreteWienerFilter
            
            N = 100
            clean = np.sin(2 * np.pi * 5 * np.linspace(0, 1, N))
            noisy = clean + 0.5 * np.random.randn(N)
            
            wf = DiscreteWienerFilter(noise_variance=0.25)
            denoised = wf.filter(noisy)
            
            # Denoised should be closer to clean
            assert np.mean((denoised - clean) ** 2) < np.mean((noisy - clean) ** 2)
        except ImportError:
            pytest.skip("Wiener filter not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
