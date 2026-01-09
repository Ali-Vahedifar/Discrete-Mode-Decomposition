# Installation Guide

## DMD+SMV: Discrete Mode Decomposition Meets Shapley Value

**Author:** Ali Vahedi  
**Affiliation:** DIGIT and Department of ECE, Aarhus University, Denmark  
**Email:** av@ece.au.dk  
**IEEE INFOCOM 2025**

---

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/alivahedi/dmd-smv-tactile-internet.git
cd dmd-smv-tactile-internet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Requirements

### Python Version
- Python 3.8 or higher

### Hardware Requirements
- **Recommended:** NVIDIA GPU with CUDA support
- **Paper setup:** 2x NVIDIA RTX A6000 GPUs
- **Minimum:** 16GB RAM, 4 CPU cores

### Dependencies
Core dependencies (see `requirements.txt` for full list):
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## GPU Setup

### CUDA Installation
1. Install NVIDIA drivers
2. Install CUDA Toolkit (11.7 or higher recommended)
3. Install cuDNN

### PyTorch with CUDA
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

```python
import torch
import numpy as np

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Check DMD+SMV installation
from src.dmd import DiscreteModeDcomposition
from src.smv import ShapleyModeValue
from src.models import TransformerPredictor

print("DMD+SMV installation successful!")
```

## Download Dataset

```bash
# Download Kinaesthetic Interactions Dataset
python scripts/download_data.py --output_dir ./data
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in config
   - Use gradient accumulation

2. **Import errors**
   - Ensure you're in the correct virtual environment
   - Run `pip install -e .`

3. **Data not found**
   - Run `python scripts/download_data.py`
   - Check `./data` directory

### Getting Help
- Open an issue on GitHub
- Contact: av@ece.au.dk

---

**Citation:**
```bibtex
@inproceedings{vahedifar2025dmd,
  title={Discrete Mode Decomposition Meets Shapley Value: 
         Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={IEEE INFOCOM 2025},
  year={2025}
}
```
