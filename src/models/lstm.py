"""
LSTM Model for Signal Prediction
================================

Implementation of LSTM architecture for haptic signal prediction.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025: "Discrete Mode Decomposition Meets Shapley Value:
Robust Signal Prediction in Tactile Internet"

Paper Configuration:
-------------------
- Two stacked LSTM layers with 128 units each
- Dense output layer with linear activation
- Dropout rate: 0.2 (higher than Transformer/ResNet)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base_model import BasePredictor, ModelConfig


class LSTMPredictor(BasePredictor):
    """
    LSTM-based predictor for haptic signals.
    
    Paper's LSTM configuration:
    - Two stacked LSTM layers with 128 units each
    - Dense output layer with linear activation
    - Dropout rate: P-drop = 0.2
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LSTM predictor.
        
        Parameters:
        -----------
        config : ModelConfig
            Model configuration
        """
        super().__init__(config)
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        # LSTM layers (paper: 2 layers, 128 units each)
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Output projection (paper: dense output layer with linear activation)
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # For multi-step prediction
        self.prediction_head = nn.Linear(
            config.window_size * config.hidden_dim,
            config.prediction_horizon * config.output_dim
        )
        
        # Mode encoder (optional)
        if config.use_modes:
            self.mode_encoder = nn.Sequential(
                nn.Linear(config.num_modes * config.window_size, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
            self.mode_fusion = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Fuse with mode information if available
        if modes is not None and self.config.use_modes:
            modes_flat = modes.view(batch_size, -1)
            mode_features = self.mode_encoder(modes_flat)
            mode_features = mode_features.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_out = self.mode_fusion(torch.cat([lstm_out, mode_features], dim=-1))
        
        # Flatten and project to prediction
        lstm_flat = lstm_out.reshape(batch_size, -1)
        
        if lstm_flat.shape[1] != self.config.window_size * self.config.hidden_dim:
            adaptive_proj = nn.Linear(
                lstm_flat.shape[1],
                self.config.prediction_horizon * self.config.output_dim
            ).to(self.device)
            predictions = adaptive_proj(lstm_flat)
        else:
            predictions = self.prediction_head(lstm_flat)
        
        # Reshape
        predictions = predictions.reshape(
            batch_size,
            self.config.prediction_horizon,
            self.config.output_dim
        )
        
        return predictions
    
    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step forward for autoregressive generation.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input of shape (batch, 1, input_dim)
        hidden : tuple, optional
            Previous hidden state (h, c)
            
        Returns:
        --------
        Tuple[torch.Tensor, Tuple]
            (output, (h_n, c_n))
        """
        if hidden is None:
            batch_size = x.shape[0]
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=self.device
            )
            c_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=self.device
            )
            hidden = (h_0, c_0)
        
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        output = self.output_projection(lstm_out)
        
        return output, (h_n, c_n)
    
    def forward_autoregressive(
        self,
        x: torch.Tensor,
        modes: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Parameters:
        -----------
        x : torch.Tensor
            Initial input sequence
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
        
        # Encode initial sequence
        _, (h, c) = self.lstm(x)
        
        # Get last input as starting point
        current_input = x[:, -1:, :]
        
        predictions = []
        for _ in range(num_steps):
            output, (h, c) = self.forward_step(current_input, (h, c))
            predictions.append(output)
            current_input = output
        
        return torch.cat(predictions, dim=1)


if __name__ == "__main__":
    print("Testing LSTM Predictor")
    print("=" * 50)
    
    config = ModelConfig(
        input_dim=9,
        output_dim=9,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        window_size=100,
        prediction_horizon=10
    )
    
    model = LSTMPredictor(config)
    print(f"Number of parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 100, 9)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test autoregressive
    with torch.no_grad():
        output_ar = model.forward_autoregressive(x, num_steps=10)
    print(f"Autoregressive output shape: {output_ar.shape}")
