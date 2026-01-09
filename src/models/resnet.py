"""
ResNet Model for Signal Prediction
==================================

Implementation of ResNet-32 architecture for haptic signal prediction.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025

This implementation includes:
- Residual blocks with skip connections
- He initialization for stable training
- 1D convolutions adapted for time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

from .base_model import BasePredictor, ModelConfig, ModeEncoder


class ResidualBlock1D(nn.Module):
    """
    1D Residual block for time series.
    
    Uses two convolutional layers with a skip connection.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        """
        Initialize residual block.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int
            Stride for downsampling
        downsample : nn.Module, optional
            Downsampling layer for skip connection
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, seq_len)
            
        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class BottleneckBlock1D(nn.Module):
    """
    Bottleneck residual block for deeper networks.
    
    Uses 1x1 -> 3x3 -> 1x1 convolution pattern for efficiency.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        """Initialize bottleneck block."""
        super().__init__()
        
        # 1x1 conv (reduce)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 1x1 conv (expand)
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.dropout(out)
        
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNetPredictor(BasePredictor):
    """
    ResNet-32 predictor for haptic signals.
    
    Uses residual connections to preserve and transmit signal features
    across layers, improving ability to model long-term temporal patterns.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    
    Architecture:
    - Initial convolution
    - 5 stages of residual blocks (following ResNet-32 structure)
    - Global average pooling
    - Fully connected prediction head
    
    Example:
    --------
    >>> config = ModelConfig(input_dim=9, hidden_dim=64, num_layers=32)
    >>> model = ResNetPredictor(config)
    >>> x = torch.randn(32, 100, 9)
    >>> pred = model(x)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize ResNet predictor.
        
        Parameters:
        -----------
        config : ModelConfig
            Model configuration
        """
        super().__init__(config)
        
        self.in_channels = config.hidden_dim
        
        # Initial convolution
        self.conv1 = nn.Conv1d(
            config.input_dim, config.hidden_dim,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        # ResNet-32 has [5, 5, 5] blocks for 32 layers (3*5*2 + 2 = 32)
        num_blocks = [5, 5, 5]
        channels = [config.hidden_dim, config.hidden_dim * 2, config.hidden_dim * 4]
        
        self.layer1 = self._make_layer(
            ResidualBlock1D, channels[0], num_blocks[0],
            stride=1, dropout=config.dropout
        )
        self.layer2 = self._make_layer(
            ResidualBlock1D, channels[1], num_blocks[1],
            stride=2, dropout=config.dropout
        )
        self.layer3 = self._make_layer(
            ResidualBlock1D, channels[2], num_blocks[2],
            stride=2, dropout=config.dropout
        )
        
        # Mode encoder
        if config.use_modes:
            self.mode_encoder = ModeEncoder(
                num_modes=config.num_modes,
                mode_dim=config.window_size,
                output_dim=channels[2],
                dropout=config.dropout
            )
            self.mode_fusion = nn.Sequential(
                nn.Linear(channels[2] * 2, channels[2]),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(channels[2], channels[2]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(channels[2], config.prediction_horizon * config.output_dim)
        )
        
        # Initialize weights using He initialization
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _make_layer(
        self,
        block: type,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dropout: float = 0.1
    ) -> nn.Sequential:
        """
        Create a layer with multiple residual blocks.
        
        Parameters:
        -----------
        block : type
            Block class (ResidualBlock1D or BottleneckBlock1D)
        out_channels : int
            Output channels for this layer
        num_blocks : int
            Number of blocks in this layer
        stride : int
            Stride for first block (for downsampling)
        dropout : float
            Dropout probability
            
        Returns:
        --------
        nn.Sequential
            Layer containing all blocks
        """
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(block(
            self.in_channels, out_channels, stride, downsample, dropout
        ))
        
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, dropout=dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """
        Initialize weights using He initialization.
        
        Recommended by Kaiming He et al. for networks with ReLU activation.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
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
        batch_size = x.shape[0]
        
        # Transpose for 1D convolution: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.flatten(1)  # (batch, channels)
        
        # Fuse with mode information
        if modes is not None and self.config.use_modes:
            mode_features = self.mode_encoder(modes)
            x = self.mode_fusion(torch.cat([x, mode_features], dim=-1))
        
        # Prediction head
        predictions = self.fc(x)
        
        # Reshape to (batch, prediction_horizon, output_dim)
        predictions = predictions.reshape(
            batch_size,
            self.config.prediction_horizon,
            self.config.output_dim
        )
        
        return predictions
    
    def get_feature_maps(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input signal
            
        Returns:
        --------
        List[torch.Tensor]
            Feature maps from each layer
        """
        feature_maps = []
        
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        feature_maps.append(x)
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        feature_maps.append(x)
        
        x = self.layer2(x)
        feature_maps.append(x)
        
        x = self.layer3(x)
        feature_maps.append(x)
        
        return feature_maps


class ResNet1D(nn.Module):
    """
    Flexible ResNet-1D with configurable depth.
    
    Author: Ali Vahedi
    IEEE INFOCOM 2025
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layers: List[int] = [3, 4, 6, 3],
        base_channels: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize flexible ResNet.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels
        num_classes : int
            Number of output classes/features
        layers : List[int]
            Number of blocks in each stage
        base_channels : int
            Base channel width
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.in_channels = base_channels
        
        self.conv1 = nn.Conv1d(
            in_channels, base_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, layers[0], dropout=dropout)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2, dropout=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dropout: float = 0.1
    ) -> nn.Sequential:
        """Create a layer with residual blocks."""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = [ResidualBlock1D(
            self.in_channels, out_channels, stride, downsample, dropout
        )]
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, dropout=dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    # Test ResNet model
    print("Testing ResNet Predictor")
    print("=" * 50)
    
    config = ModelConfig(
        input_dim=9,
        output_dim=9,
        hidden_dim=64,
        num_layers=32,
        dropout=0.1,
        window_size=100,
        prediction_horizon=10,
        use_modes=True,
        num_modes=5
    )
    
    model = ResNetPredictor(config)
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
    
    # Test feature extraction
    print("\nFeature maps:")
    feature_maps = model.get_feature_maps(x)
    for i, fm in enumerate(feature_maps):
        print(f"  Layer {i}: {fm.shape}")
