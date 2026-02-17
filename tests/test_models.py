"""
Tests for Neural Network Models
===============================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2026

Usage: pytest tests/test_models.py -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestTransformer:
    """Test Transformer model. Author: Ali Vahedi"""
    
    @pytest.fixture
    def model_config(self):
        """Create model config."""
        try:
            from models import ModelConfig
            return ModelConfig(
                input_dim=9,
                output_dim=9,
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                window_size=100,
                prediction_horizon=10
            )
        except ImportError:
            pytest.skip("Models module not available")
    
    def test_transformer_import(self):
        """Test Transformer import."""
        try:
            from models import TransformerPredictor
            assert True
        except ImportError:
            pytest.skip("Transformer not available")
    
    def test_transformer_forward(self, model_config):
        """Test forward pass."""
        try:
            from models import TransformerPredictor
            
            model = TransformerPredictor(model_config)
            x = torch.randn(4, 100, 9)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (4, 10, 9)
        except ImportError:
            pytest.skip("Transformer not available")
    
    def test_transformer_parameters(self, model_config):
        """Test parameter count."""
        try:
            from models import TransformerPredictor
            
            model = TransformerPredictor(model_config)
            n_params = sum(p.numel() for p in model.parameters())
            
            assert n_params > 0
        except ImportError:
            pytest.skip("Transformer not available")


class TestResNet:
    """Test ResNet model. Author: Ali Vahedi"""
    
    @pytest.fixture
    def model_config(self):
        try:
            from models import ModelConfig
            return ModelConfig(
                input_dim=9,
                output_dim=9,
                hidden_dim=64,
                num_layers=8,
                dropout=0.1,
                window_size=100,
                prediction_horizon=10
            )
        except ImportError:
            pytest.skip("Models module not available")
    
    def test_resnet_forward(self, model_config):
        """Test ResNet forward pass."""
        try:
            from models import ResNetPredictor
            
            model = ResNetPredictor(model_config)
            x = torch.randn(4, 100, 9)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[0] == 4
            assert output.shape[-1] == 9
        except ImportError:
            pytest.skip("ResNet not available")


class TestLSTM:
    """Test LSTM model. Author: Ali Vahedi"""
    
    @pytest.fixture
    def model_config(self):
        try:
            from models import ModelConfig
            return ModelConfig(
                input_dim=9,
                output_dim=9,
                hidden_dim=128,  # Paper: 128 units
                num_layers=2,    # Paper: 2 layers
                dropout=0.2,     # Paper: 0.2 for LSTM
                window_size=100,
                prediction_horizon=10
            )
        except ImportError:
            pytest.skip("Models module not available")
    
    def test_lstm_forward(self, model_config):
        """Test LSTM forward pass."""
        try:
            from models import LSTMPredictor
            
            model = LSTMPredictor(model_config)
            x = torch.randn(4, 100, 9)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[0] == 4
            assert output.shape[-1] == 9
        except ImportError:
            pytest.skip("LSTM not available")


class TestModelConfig:
    """Test model configuration. Author: Ali Vahedi"""
    
    def test_config_defaults(self):
        """Test default configuration."""
        try:
            from models import ModelConfig
            
            config = ModelConfig()
            
            assert config.input_dim == 9
            assert config.output_dim == 9
            assert config.hidden_dim == 128
        except ImportError:
            pytest.skip("ModelConfig not available")
    
    def test_config_paper_values(self):
        """Test paper configuration values."""
        try:
            from models import ModelConfig
            
            # Transformer/ResNet config from paper
            config = ModelConfig(
                input_dim=9,
                hidden_dim=128,
                num_layers=4,
                dropout=0.1  # Paper: P-drop = 0.1
            )
            
            assert config.dropout == 0.1
            
            # LSTM config from paper
            lstm_config = ModelConfig(
                input_dim=9,
                hidden_dim=128,  # Paper: 128 units
                num_layers=2,    # Paper: 2 stacked layers
                dropout=0.2      # Paper: P-drop = 0.2
            )
            
            assert lstm_config.num_layers == 2
            assert lstm_config.dropout == 0.2
        except ImportError:
            pytest.skip("ModelConfig not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
