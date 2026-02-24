# Discrete Mode Decomposition Meets Shapley Value for Robust Signal Prediction in Tactile Internet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![IEEE INFOCOM 2026](https://img.shields.io/badge/IEEE-INFOCOM%202026-green.svg)](https://infocom2026.ieee-infocom.org/)

## 🏆 Accepted at IEEE INFOCOM 2026

**Authors:** Ali Vahedi and Qi Zhang

**Affiliation:** DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

**Acknowledgments:** This research was supported by:
- TOAST project, funded by the European Union's Horizon Europe research and innovation program under the Marie Skłodowska-Curie Actions Doctoral Network (Grant Agreement No. 101073465)
- Danish Council for Independent Research project eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

---

## 📖 Abstract

The Tactile Internet (TI) requires ultra-low latency and high reliability to ensure stability and transparency in touch-enabled teleoperation. However, variable delays and packet loss present significant challenges to maintaining immersive haptic communication. This work proposes a novel predictive framework that integrates **Discrete Mode Decomposition (DMD)** with **Shapley Mode Value (SMV)** for accurate and timely haptic signal prediction.

<img width="911" height="1039" alt="Screenshot 2026-01-09 at 14 52 28" src="https://github.com/user-attachments/assets/f423f40d-35ac-4e47-a33a-1c040f1cfb5d" />

- **DMD** decomposes haptic signals into interpretable intrinsic modes
- **SMV** evaluates each mode's contribution to prediction accuracy, aligned with goal-oriented semantic communication
- Combined **DMD+SMV** accelerates inference, enabling efficient communication and smooth teleoperation

### Key Results
| Method | 1-Sample Accuracy | 100-Sample Accuracy | 1-Sample Latency | 100-Sample Latency |
|--------|-------------------|---------------------|------------------|---------------------|
| **DMD+SMV (Ours)** | **98.9%** | **92.5%** | **0.056ms** | **2ms** |
| DMD | 96.9% | 90.0% | 0.05ms | 6.91ms |
| Baseline | 73.6% | 67.3% | 0.04ms | 1640.76ms |

---

## 🏗️ Project Structure

```
dmd-smv-tactile-internet/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── pyproject.toml              # Modern Python packaging
│
├── configs/                     # Configuration files
│   ├── default_config.yaml     # Default training configuration
│   ├── transformer_config.yaml # Transformer-specific settings
│   ├── resnet_config.yaml      # ResNet-specific settings
│   └── lstm_config.yaml        # LSTM-specific settings
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── dmd/                     # Discrete Mode Decomposition
│   │   ├── __init__.py
│   │   ├── decomposition.py    # Core DMD algorithm
│   │   ├── wiener_filter.py    # Discrete Wiener filtering
│   │   ├── hilbert_transform.py # Discrete Hilbert transform
│   │   └── optimization.py     # ADMM optimization
│   │
│   ├── smv/                     # Shapley Mode Value
│   │   ├── __init__.py
│   │   ├── shapley_value.py    # Shapley value computation
│   │   ├── monte_carlo.py      # Monte Carlo approximation
│   │   └── mode_valuation.py   # Mode valuation utilities
│   │
│   ├── models/                  # Neural Network architectures
│   │   ├── __init__.py
│   │   ├── transformer.py      # Transformer architecture
│   │   ├── resnet.py           # ResNet-32 architecture
│   │   ├── lstm.py             # LSTM architecture
│   │   └── base_model.py       # Base model class
│   │
│   ├── data/                    # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes
│   │   ├── dataloader.py       # DataLoader utilities
│   │   └── preprocessing.py    # Data preprocessing
│   │
│   ├── training/                # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop
│   │   ├── losses.py           # Loss functions
│   │   └── callbacks.py        # Training callbacks
│   │
│   ├── evaluation/              # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py          # Accuracy, PSNR, etc.
│   │   ├── inference.py        # Inference engine
│   │   └── benchmarks.py       # Benchmarking utilities
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── logger.py           # Logging utilities
│       ├── visualization.py    # Plotting and visualization
│       ├── config.py           # Configuration management
│
├── experiments/                 # Experiment scripts
│   ├── train_all_models.py     # Train all architectures
│   ├── evaluate_accuracy.py    # Accuracy evaluation
│   ├── evaluate_inference.py   # Inference time evaluation
│   ├── ablation_study.py       # Ablation studies
│   └── generate_figures.py     # Generate paper figures
│
├── scripts/                     # Utility scripts
│   ├── download_data.py        # Download dataset
│   ├── prepare_data.py         # Prepare data for training
│   └── run_experiments.sh      # Run all experiments
│
├── tests/                       # Unit tests
│   ├── test_dmd.py             # Test DMD module
│   ├── test_smv.py             # Test SMV module
│   ├── test_models.py          # Test neural networks
│   └── test_metrics.py         # Test evaluation metrics
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_dmd_visualization.ipynb
│   ├── 03_smv_analysis.ipynb
│   └── 04_results_analysis.ipynb
│
├── results/                     # Results storage
│   ├── figures/                # Generated figures
│   └── logs/                   # Training logs
│
└── docs/                        # Documentation
    ├── installation.md
    ├── usage.md
    └── api_reference.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alivahedi/dmd-smv-tactile-internet.git
cd dmd-smv-tactile-internet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Download Dataset

```bash
# Download the Kinaesthetic Interactions Dataset
python scripts/download_data.py
```

### Quick Training Example

```python
from src.dmd import DiscreteModeDcomposition
from src.smv import ShapleyModeValue
from src.models import TransformerPredictor
from src.training import Trainer

# Initialize DMD
dmd = DiscreteModeDcomposition(noise_variance=0.01)

# Decompose signal
modes, center_frequencies = dmd.decompose(signal)

# Compute Shapley Mode Values
smv = ShapleyModeValue()
mode_values = smv.compute(modes, model, validation_data)

# Train Transformer with important modes
trainer = Trainer(model=TransformerPredictor(), config=config)
trainer.train(train_data, val_data)
```

### Run Full Experiment Pipeline

```bash
# Run all experiments
bash scripts/run_experiments.sh

# Or run individual experiments
python experiments/train_all_models.py --config configs/default_config.yaml
python experiments/evaluate_accuracy.py --checkpoint best_model.pt
python experiments/generate_figures.py --output results/figures/
```

---

## 📊 Reproducing Paper Results

### Training All Models

```bash
python experiments/train_all_models.py \
    --architectures transformer resnet lstm \
    --methods baseline dmd dmd_smv \
    --window_sizes 1 5 10 25 50 100 \
    --epochs 200 \
```

### Generate Main Results Figure (Figure 3)

```bash
python experiments/generate_figures.py \
    --figure accuracy_inference \
    --output results/figures/figure3.pdf
```

### Ablation Study (Mode Update Frequency)

```bash
python experiments/ablation_study.py \
    --update_intervals 40 90 100 \
    --architecture transformer \
    --output results/ablation/
```

---

## 📈 Results

### Accuracy Comparison (Figure 3)

Our DMD+SMV framework achieves:
- **98.9%** accuracy for 1-sample prediction (Transformer)
- **92.5%** accuracy for 100-sample prediction (Transformer)
- **820× speedup** compared to baseline methods

### PSNR Results (Figure 4)

- **29.5 dB** PSNR at W=1 for human side
- **27.5 dB** PSNR at W=1 for robot side
- Consistent **9-10 dB improvement** over baseline across all horizons

### Inference Time (Table I & II)

| Architecture | DMD+SMV | DMD | Baseline |
|--------------|---------|-----|----------|
| Transformer | 2.85ms | 7.30ms | 1640ms |
| ResNet-32 | 4.30ms | 10.45ms | 2059ms |
| LSTM | 6.00ms | 15.2ms | 2616ms |

### FLOPs and FLOPS Comparison (Table III)

| Architecture | FLOPs (DMD) | FLOPs (DMD+SMV) | FLOPS (DMD) | FLOPS (DMD+SMV) |
|--------------|-------------|-----------------|-------------|-----------------|
| LSTM | 3.4×10⁷ | 2.1×10⁶ | 2.24×10⁹ | 0.35×10⁹ |
| ResNet-32 | 11.2×10⁷ | 8.6×10⁶ | 10.72×10⁹ | 2×10⁹ |
| Transformer | 19.3×10⁷ | 14.8×10⁶ | **26.4×10⁹** | **5.19×10⁹** |

---

## 🔗 Links

- **Paper:** IEEE INFOCOM 2025 Proceedings
- **Code:** [github.com/Ali-Vahedifar/Discrete-Mode-Decomposition](https://github.com/Ali-Vahedifar/Discrete-Mode-Decomposition.git)
- **Dataset:** [Kinaesthetic Interactions Dataset (Zenodo)](https://doi.org/10.5281/zenodo.14924062)

---

## 🔧 Configuration

### Default Configuration

```yaml
# configs/default_config.yaml
training:
  epochs: 200
  batch_size: 64
  learning_rate: 0.001
  lr_decay_epochs: [80, 120, 170]
  lr_decay_factor: 0.005
  dropout: 0.1

dmd:
  noise_variance: 0.01
  epsilon1: 1e-6
  epsilon2: 1e-6
  kappa1: 1e-3
  kappa2: 1e-3

smv:
  tolerance: 0.01
  max_iterations: 1000

evaluation:
  window_sizes: [1, 5, 10, 25, 50, 100]
  num_runs: 10
```

---

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{vahedifar2025dmd,
  title={Discrete Mode Decomposition Meets Shapley Value: Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={IEEE INFOCOM 2026 - IEEE Conference on Computer Communications},
  year={2026},
  organization={IEEE}
}
```

---

## 📧 Contact

- **Ali Vahedi**: av@ece.au.dk

**Institution:** DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [TOAST Doctoral Network](https://toast-dn.eu/) - EU Horizon Europe Program
- [Danish Council for Independent Research](https://dff.dk/) - eTouch Project
- [NordForsk](https://www.nordforsk.org/) - Nordic Edge Intelligence Cooperation
- [Kinaesthetic Interactions Dataset](https://doi.org/10.5281/zenodo.14924062)

---

## 📜 Changelog

### v1.0.0 (2026)
- Initial release
- Full implementation of DMD and SMV algorithms
- Support for Transformer, ResNet-32, and LSTM architectures
- Comprehensive evaluation framework
- Paper reproduction scripts
