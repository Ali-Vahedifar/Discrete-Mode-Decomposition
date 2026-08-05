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

## 🔗 Links

- **Paper:** IEEE INFOCOM 2026 Proceedings
- **Code:** [github.com/Ali-Vahedifar/Discrete-Mode-Decomposition](https://github.com/Ali-Vahedifar/Discrete-Mode-Decomposition.git)
- **Dataset:** [Kinaesthetic Interactions Dataset (Zenodo)](https://doi.org/10.5281/zenodo.14924062)


---

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{vahedifar2026dmd,
  title={Discrete Mode Decomposition Meets Shapley Value: Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={IEEE INFOCOM 2026 - IEEE Conference on Computer Communications},
  year={2026},
  organization={IEEE}
}
```

