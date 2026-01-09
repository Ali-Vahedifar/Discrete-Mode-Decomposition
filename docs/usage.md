# Usage Guide

## DMD+SMV: Discrete Mode Decomposition Meets Shapley Value

**Author:** Ali Vahedi  
**Affiliation:** DIGIT and Department of ECE, Aarhus University, Denmark  
**Email:** av@ece.au.dk  
**IEEE INFOCOM 2025**

---

## Quick Start

### 1. Basic DMD Decomposition

```python
from src.dmd import DiscreteModeDcomposition, DMDConfig
import numpy as np

# Create signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t)

# Configure DMD (paper parameters)
config = DMDConfig(
    noise_variance=0.01,  # α parameter
    epsilon1=1e-6,
    epsilon2=1e-6,
    max_modes=10
)

# Decompose
dmd = DiscreteModeDcomposition(config=config)
result = dmd.decompose(signal, num_modes=5)

print(f"Extracted {result.num_modes} modes")
print(f"Center frequencies: {result.center_frequencies}")
```

### 2. Shapley Mode Value Computation

```python
from src.smv import ShapleyModeValue

# Initialize SMV
smv = ShapleyModeValue(
    tolerance=0.01,      # Convergence: average change < 1%
    max_iterations=1000
)

# Compute mode values
mode_values = smv.compute(
    modes=result.modes,
    center_frequencies=result.center_frequencies,
    model=trained_model,
    validation_data=val_loader
)

# Get sorted modes by importance
sorted_indices = np.argsort(mode_values)[::-1]
important_modes = result.modes[sorted_indices[:5]]  # Top 5
```

### 3. Training a Prediction Model

```python
from src.models import TransformerPredictor, ModelConfig
from src.training import Trainer, TrainingConfig
from src.data import HapticDataset, DatasetConfig

# Model configuration (paper settings)
model_config = ModelConfig(
    input_dim=9,           # 3D position + velocity + force
    output_dim=9,
    hidden_dim=128,
    num_layers=4,
    dropout=0.1,           # Paper: P-drop = 0.1
    window_size=100,
    prediction_horizon=100,
    use_modes=True,
    num_modes=10
)

# Training configuration (paper settings)
train_config = TrainingConfig(
    epochs=200,
    batch_size=64,
    learning_rate=0.001,
    lr_decay_epochs=[80, 120, 170],  # Paper schedule
    lr_decay_factor=0.005,
    num_runs=10  # Paper: "averaged over 10 runs"
)

# Create model and trainer
model = TransformerPredictor(model_config)
trainer = Trainer(model, train_config)

# Train
result = trainer.train(train_loader, val_loader)
print(f"Best accuracy: {result.best_val_accuracy:.2f}%")
```

### 4. Full Pipeline Example

```python
# Complete DMD+SMV pipeline
import numpy as np
import torch
from src.dmd import DiscreteModeDcomposition
from src.smv import ShapleyModeValue
from src.models import TransformerPredictor, ModelConfig
from src.training import Trainer, TrainingConfig
from src.evaluation import MetricsComputer, InferenceEngine

# 1. Load and preprocess data
from src.data import HapticDataset, DatasetConfig

data_config = DatasetConfig(
    window_size=100,
    prediction_horizon=100,
    stride=10
)
dataset = HapticDataset(data_path='./data/tap_and_hold/data.csv', config=data_config)

# 2. Apply DMD decomposition
dmd = DiscreteModeDcomposition(noise_variance=0.01)
modes_list = []
for i in range(len(dataset)):
    x, _ = dataset[i]
    result = dmd.decompose(x.numpy(), num_modes=10)
    modes_list.append(result.modes)

# 3. Train model with modes
model_config = ModelConfig(input_dim=9, hidden_dim=128, num_layers=4)
model = TransformerPredictor(model_config)

train_config = TrainingConfig(epochs=200, learning_rate=0.001)
trainer = Trainer(model, train_config)
train_result = trainer.train(train_loader, val_loader)

# 4. Compute SMV
smv = ShapleyModeValue()
mode_values = smv.compute(modes_list[0], model, val_loader)

# 5. Evaluate
metrics = MetricsComputer()
engine = InferenceEngine(model)

eval_metrics = metrics.compute_all(predictions, targets)
timing = engine.predict_with_timing(test_input)

print(f"Accuracy: {eval_metrics.accuracy:.2f}%")
print(f"PSNR: {eval_metrics.psnr:.2f} dB")
print(f"Inference time: {timing.inference_time_ms:.3f} ms")
```

## Command Line Usage

### Train All Models
```bash
python experiments/train_all_models.py \
    --config configs/default_config.yaml \
    --epochs 200 \
    --seed 42
```

### Generate Figures
```bash
python experiments/generate_figures.py \
    --results_dir ./results \
    --output_dir ./figures
```

### Run Full Experiment
```bash
bash scripts/run_experiments.sh
```

## Configuration

### Training Parameters (from paper)
| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 200 | Maximum training epochs |
| batch_size | 64 | Training batch size |
| learning_rate | 0.001 | Initial learning rate |
| lr_decay_epochs | [80, 120, 170] | LR decay schedule |
| lr_decay_factor | 0.005 | LR decay multiplier |
| β1, β2 | 0.9, 0.999 | Adam optimizer parameters |
| dropout | 0.1/0.2 | 0.1 for Transformer/ResNet, 0.2 for LSTM |

### Expected Results
| Method | W=1 | W=100 | Time (W=100) |
|--------|-----|-------|--------------|
| DMD+SMV | 98.9% | 92.5% | 2.05 ms |
| DMD | 96.9% | 90.0% | 6.91 ms |
| Baseline | 73.6% | 67.3% | 1640 ms |

---

**Author:** Ali Vahedi  
**IEEE INFOCOM 2025**
