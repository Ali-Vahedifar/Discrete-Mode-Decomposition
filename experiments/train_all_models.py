#!/usr/bin/env python3
"""
Train All Models - Main Experiment Script
==========================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025

Usage: python experiments/train_all_models.py --config configs/default_config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    return parser.parse_args()


def create_synthetic_data(n_samples=10000, seed=42):
    """Create synthetic haptic data for testing. Author: Ali Vahedi"""
    np.random.seed(seed)
    t = np.linspace(0, n_samples/1000, n_samples)
    
    data = np.zeros((n_samples, 9), dtype=np.float32)
    
    # Position (smooth trajectories)
    for i in range(3):
        data[:, i] = 0.1 * np.sin(2 * np.pi * (i+1) * t) + 0.02 * np.random.randn(n_samples)
    
    # Velocity (derivative + noise)
    for i in range(3):
        data[:, 3+i] = np.gradient(data[:, i]) * 1000 + 0.1 * np.random.randn(n_samples)
    
    # Force (spring model)
    k = 100
    for i in range(3):
        data[:, 6+i] = -k * data[:, i] + 5 * np.random.randn(n_samples)
    
    return data


def create_windows(data, window_size=100, prediction_horizon=100, stride=10):
    """Create sliding windows. Author: Ali Vahedi"""
    X, Y = [], []
    for i in range(0, len(data) - window_size - prediction_horizon, stride):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size:i+window_size+prediction_horizon])
    return np.array(X), np.array(Y)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("DMD+SMV Training Script")
    print("Author: Ali Vahedi")
    print("IEEE INFOCOM 2025")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data
    print("\nCreating synthetic data...")
    data = create_synthetic_data(50000, args.seed)
    X, Y = create_windows(data)
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Import models
    try:
        from models import TransformerPredictor, ModelConfig
        from training import Trainer, TrainingConfig
        
        # Create model
        print("\nCreating Transformer model...")
        config = ModelConfig(
            input_dim=9,
            output_dim=9,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            window_size=100,
            prediction_horizon=100
        )
        
        model = TransformerPredictor(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        print("\nTraining...")
        train_config = TrainingConfig(
            epochs=min(args.epochs, 10),  # Limit for demo
            batch_size=64,
            learning_rate=0.001
        )
        
        trainer = Trainer(model, train_config)
        result = trainer.train(train_loader, val_loader, verbose=True)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation accuracy: {result.best_val_accuracy:.2f}%")
        print(f"Best epoch: {result.best_epoch}")
        print("=" * 60)
        
        # Save results
        result.save(str(output_dir / 'training_result.json'))
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Running basic training demo...")
        
        # Basic demo without custom modules
        import torch.nn as nn
        import torch.optim as optim
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(9, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 9)
                )
            
            def forward(self, x):
                b, s, f = x.shape
                x = x.view(-1, f)
                x = self.net(x)
                return x.view(b, s, f)
        
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print("\nTraining simple model...")
        for epoch in range(5):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        print("\nDemo training complete!")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
