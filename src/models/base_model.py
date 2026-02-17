"""
Base Model Class for Signal Prediction
======================================

Base class defining the interface for all prediction models.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ModelConfig:
    """
    Configuration for prediction models.
    
    Attributes:
        input_dim: Dimension of input features (e.g., 9 for 3D position, velocity, force)
        output_dim: Dimension of output predictions
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        dropout: Dropout probability
        window_size: Input window size
        prediction_horizon: Number of future samples to predict
        use_modes: Whether to use DMD modes as input
        num_modes: Number of DMD modes to use
    """
    input_dim: int = 9  # 3 features (P, V, F) x 3 dimensions (X, Y, Z)
    output_dim: int = 9
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    window_size: int = 100
    prediction_horizon: int = 100
    use_modes: bool = True
    num_modes: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BasePredictor(nn.Module, ABC):
    """
    Abstract base class for signal prediction models.
    
    All predictor models should inherit from this class and implement
    the forward method.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Methods:
    --------
    forward(x, modes=None): Forward pass
    predict(x, modes=None): Prediction with numpy input/output
    get_num_parameters(): Get total number of parameters
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base predictor.
        
        Parameters:
        -----------
        config : ModelConfig
            Model configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        modes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim)
        modes : torch.Tensor, optional
            DMD modes tensor of shape (batch, num_modes, seq_len)
            
        Returns:
        --------
        torch.Tensor
            Predictions of shape (batch, prediction_horizon, output_dim)
        """
        pass
    
    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        modes: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """
        Make predictions with numpy input/output.
        
        Parameters:
        -----------
        x : np.ndarray or torch.Tensor
            Input signal
        modes : np.ndarray or torch.Tensor, optional
            DMD modes
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        self.eval()
        
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if modes is not None and isinstance(modes, np.ndarray):
            modes = torch.FloatTensor(modes)
        
        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if modes is not None and modes.dim() == 2:
            modes = modes.unsqueeze(0)
        
        # Move to device
        x = x.to(self.device)
        if modes is not None:
            modes = modes.to(self.device)
        
        with torch.no_grad():
            predictions = self.forward(x, modes)
        
        return predictions.cpu().numpy()
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate FLOPs for a single forward pass.
        
        Parameters:
        -----------
        input_shape : Tuple[int, ...]
            Shape of input tensor
            
        Returns:
        --------
        int
            Estimated FLOPs
        """
        # This is a rough estimate; for accurate FLOPs, use profiling tools
        total_flops = 0
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                total_flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.Conv1d):
                # Approximate
                total_flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0]
        
        return total_flops
    
    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        if device is not None:
            config.device = device
        
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(model.device)
        
        return model


class ModeEncoder(nn.Module):
    """
    Encoder for DMD modes.
    
    Transforms DMD modes into feature representations that can be
    concatenated with signal features.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        num_modes: int,
        mode_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize mode encoder.
        
        Parameters:
        -----------
        num_modes : int
            Number of input modes
        mode_dim : int
            Dimension of each mode
        output_dim : int
            Output feature dimension
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.num_modes = num_modes
        
        # Per-mode encoding
        self.mode_embed = nn.Sequential(
            nn.Linear(mode_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Mode attention
        self.mode_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Encode modes into features.
        
        Parameters:
        -----------
        modes : torch.Tensor
            Shape (batch, num_modes, mode_dim)
            
        Returns:
        --------
        torch.Tensor
            Shape (batch, output_dim)
        """
        batch_size = modes.shape[0]
        
        # Embed each mode
        mode_features = self.mode_embed(modes)  # (batch, num_modes, output_dim)
        
        # Self-attention over modes
        attended, _ = self.mode_attention(
            mode_features, mode_features, mode_features
        )
        
        # Residual connection
        attended = self.layer_norm(attended + mode_features)
        
        # Aggregate modes (mean pooling)
        aggregated = attended.mean(dim=1)  # (batch, output_dim)
        
        return self.output_proj(aggregated)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds sinusoidal position information to input embeddings.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Parameters:
        -----------
        d_model : int
            Model dimension
        max_len : int
            Maximum sequence length
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
        --------
        torch.Tensor
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    # Test base classes
    config = ModelConfig(
        input_dim=9,
        hidden_dim=64,
        window_size=100,
        prediction_horizon=10
    )
    
    print("ModelConfig:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Device: {config.device}")
    
    # Test mode encoder
    mode_encoder = ModeEncoder(
        num_modes=5,
        mode_dim=100,
        output_dim=64
    )
    
    modes = torch.randn(2, 5, 100)
    encoded = mode_encoder(modes)
    print(f"\nMode encoder output shape: {encoded.shape}")
    
    # Test positional encoding
    pos_enc = PositionalEncoding(d_model=64)
    x = torch.randn(2, 50, 64)
    x_with_pos = pos_enc(x)
    print(f"Positional encoding output shape: {x_with_pos.shape}")
