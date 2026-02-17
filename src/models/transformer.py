"""
Transformer Model for Signal Prediction
=======================================

Implementation of Transformer architecture for haptic signal prediction.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2026

This implementation includes:
- Multi-head self-attention mechanism
- Masked attention for autoregressive prediction
- Mode-aware attention for DMD integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

from .base_model import BasePredictor, ModelConfig, PositionalEncoding, ModeEncoder


class TransformerPredictor(BasePredictor):
    """
    Transformer-based predictor for haptic signals.
    
    Uses self-attention mechanism to capture feature-mode relationships
    and masked multi-head attention for sequence prediction.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Key features:
    - Attention mechanism for capturing contextual dependencies
    - Masked attention for autoregressive generation
    - Mode-aware encoding for DMD integration
    
    Example:
    --------
    >>> config = ModelConfig(input_dim=9, hidden_dim=128, num_layers=4)
    >>> model = TransformerPredictor(config)
    >>> x = torch.randn(32, 100, 9)  # (batch, seq_len, features)
    >>> modes = torch.randn(32, 5, 100)  # (batch, num_modes, mode_len)
    >>> pred = model(x, modes)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Transformer predictor.
        
        Parameters:
        -----------
        config : ModelConfig
            Model configuration
        """
        super().__init__(config)
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = max(1, config.hidden_dim // 32)  # Ensure at least 1 head
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.hidden_dim,
            max_len=config.window_size + config.prediction_horizon,
            dropout=config.dropout
        )
        
        # Mode encoder (if using modes)
        if config.use_modes:
            self.mode_encoder = ModeEncoder(
                num_modes=config.num_modes,
                mode_dim=config.window_size,
                output_dim=config.hidden_dim,
                dropout=config.dropout
            )
            self.mode_fusion = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Transformer decoder layers (for autoregressive prediction)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Prediction head for different window sizes
        self.prediction_head = nn.Linear(
            config.window_size * config.hidden_dim,
            config.prediction_horizon * config.output_dim
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.
        
        Creates a mask where position i can only attend to positions <= i.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)
    
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
            Input signal of shape (batch, seq_len, input_dim)
        modes : torch.Tensor, optional
            DMD modes of shape (batch, num_modes, mode_len)
            
        Returns:
        --------
        torch.Tensor
            Predictions of shape (batch, prediction_horizon, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Fuse with mode information if available
        if modes is not None and self.config.use_modes:
            mode_features = self.mode_encoder(modes)  # (batch, hidden_dim)
            mode_features = mode_features.unsqueeze(1).expand(-1, seq_len, -1)
            x = self.mode_fusion(torch.cat([x, mode_features], dim=-1))
        
        # Encode input sequence
        memory = self.transformer_encoder(x)
        
        # Use the encoded sequence directly for prediction
        # Flatten and project to prediction horizon
        memory_flat = memory.reshape(batch_size, -1)
        
        # Handle variable sequence lengths
        if memory_flat.shape[1] != self.config.window_size * self.config.hidden_dim:
            # Adaptive projection
            adaptive_proj = nn.Linear(
                memory_flat.shape[1],
                self.config.prediction_horizon * self.config.output_dim
            ).to(self.device)
            predictions = adaptive_proj(memory_flat)
        else:
            predictions = self.prediction_head(memory_flat)
        
        # Reshape to (batch, prediction_horizon, output_dim)
        predictions = predictions.reshape(
            batch_size,
            self.config.prediction_horizon,
            self.config.output_dim
        )
        
        return predictions
    
    def forward_autoregressive(
        self,
        x: torch.Tensor,
        modes: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation for longer predictions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input signal
        modes : torch.Tensor, optional
            DMD modes
        num_steps : int, optional
            Number of steps to generate
            
        Returns:
        --------
        torch.Tensor
            Generated sequence
        """
        if num_steps is None:
            num_steps = self.config.prediction_horizon
        
        batch_size = x.shape[0]
        
        # Encode input
        x_proj = self.input_projection(x)
        x_proj = self.pos_encoder(x_proj)
        
        if modes is not None and self.config.use_modes:
            mode_features = self.mode_encoder(modes)
            mode_features = mode_features.unsqueeze(1).expand(-1, x_proj.shape[1], -1)
            x_proj = self.mode_fusion(torch.cat([x_proj, mode_features], dim=-1))
        
        memory = self.transformer_encoder(x_proj)
        
        # Initialize decoder input (start token)
        decoder_input = x_proj[:, -1:, :]  # Use last input as start
        
        predictions = []
        
        for step in range(num_steps):
            # Add positional encoding to decoder input
            decoder_input_pos = self.pos_encoder(decoder_input)
            
            # Generate mask
            tgt_mask = self._generate_square_subsequent_mask(decoder_input.shape[1])
            
            # Decode
            decoder_output = self.transformer_decoder(
                decoder_input_pos,
                memory,
                tgt_mask=tgt_mask
            )
            
            # Project to output
            step_pred = self.output_projection(decoder_output[:, -1:, :])
            predictions.append(step_pred)
            
            # Update decoder input
            next_input = self.input_projection(step_pred)
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
        
        return torch.cat(predictions, dim=1)


class TransformerEncoderOnly(BasePredictor):
    """
    Encoder-only Transformer for faster inference.
    
    Uses only the encoder part of the transformer with a projection
    head for prediction. Faster than encoder-decoder architecture.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize encoder-only transformer."""
        super().__init__(config)
        
        self.hidden_dim = config.hidden_dim
        self.num_heads = max(1, config.hidden_dim // 32)
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(config.hidden_dim, dropout=config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Prediction MLP
        self.pred_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.output_dim)
        )
        
        # For multi-step prediction
        self.horizon_proj = nn.Linear(
            config.window_size,
            config.prediction_horizon
        )
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        modes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        x = self.pos_enc(x)
        
        # Encode
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim)
        
        # Project each timestep to output
        output = self.pred_mlp(encoded)  # (batch, seq_len, output_dim)
        
        # Reshape for horizon projection
        output = output.permute(0, 2, 1)  # (batch, output_dim, seq_len)
        output = self.horizon_proj(output)  # (batch, output_dim, pred_horizon)
        output = output.permute(0, 2, 1)  # (batch, pred_horizon, output_dim)
        
        return output


class AttentionLayer(nn.Module):
    """
    Custom attention layer with mode conditioning.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """Initialize attention layer."""
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)
        
        return output, attn_weights


if __name__ == "__main__":
    # Test Transformer model
    print("Testing Transformer Predictor")
    print("=" * 50)
    
    config = ModelConfig(
        input_dim=9,
        output_dim=9,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        window_size=100,
        prediction_horizon=10,
        use_modes=True,
        num_modes=5
    )
    
    model = TransformerPredictor(config)
    print(f"Number of parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 100, 9)
    modes = torch.randn(batch_size, 5, 100)
    
    with torch.no_grad():
        output = model(x, modes)
    
    print(f"Input shape: {x.shape}")
    print(f"Modes shape: {modes.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test autoregressive generation
    print("\nTesting autoregressive generation...")
    with torch.no_grad():
        output_ar = model.forward_autoregressive(x, modes, num_steps=10)
    print(f"Autoregressive output shape: {output_ar.shape}")
    
    # Test encoder-only variant
    print("\nTesting Encoder-Only Transformer...")
    model_enc = TransformerEncoderOnly(config)
    print(f"Encoder-only parameters: {model_enc.get_num_parameters():,}")
    
    with torch.no_grad():
        output_enc = model_enc(x)
    print(f"Encoder-only output shape: {output_enc.shape}")
