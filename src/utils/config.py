"""
Configuration Management
========================

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """
    Configuration class for DMD+SMV experiments.
    
    Default values match the paper's experimental setup.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    # Training parameters (from paper)
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    lr_decay_epochs: list = field(default_factory=lambda: [80, 120, 170])
    lr_decay_factor: float = 0.005
    
    # Adam optimizer (from paper)
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Model parameters
    hidden_dim: int = 128
    num_layers: int = 4
    dropout_transformer: float = 0.1
    dropout_lstm: float = 0.2
    
    # DMD parameters
    noise_variance: float = 0.01
    epsilon1: float = 1e-6
    epsilon2: float = 1e-6
    kappa1: float = 1e-3
    kappa2: float = 1e-3
    
    # SMV parameters
    smv_tolerance: float = 0.01
    smv_max_iterations: int = 1000
    
    # Data parameters
    window_sizes: list = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])
    prediction_horizon: int = 100
    mode_update_interval: int = 100
    
    # Evaluation
    num_runs: int = 10
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        return cls(**d)


def load_config(path: str) -> Config:
    """Load configuration from YAML or JSON file."""
    path = Path(path)
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return Config.from_dict(data)


def save_config(config: Config, path: str):
    """Save configuration to YAML or JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            json.dump(config.to_dict(), f, indent=2)
