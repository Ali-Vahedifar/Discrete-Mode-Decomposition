#!/bin/bash
# =============================================================================
# Run All Experiments Script
# =============================================================================
# 
# Author: Ali Vahedi (Mohammad Ali Vahedifar)
# Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
# Email: av@ece.au.dk
# 
# IEEE INFOCOM 2025: "Discrete Mode Decomposition Meets Shapley Value:
# Robust Signal Prediction in Tactile Internet"
#
# This script runs all experiments to reproduce the paper results.
#
# Usage: bash scripts/run_experiments.sh
# =============================================================================

echo "=============================================================="
echo "DMD+SMV Experiment Runner"
echo "Author: Ali Vahedi (Mohammad Ali Vahedifar)"
echo "IEEE INFOCOM 2025"
echo "=============================================================="

# Configuration
RESULTS_DIR="./results"
FIGURES_DIR="./results/figures"
CHECKPOINTS_DIR="./checkpoints"
DATA_DIR="./data"

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR
mkdir -p $CHECKPOINTS_DIR
mkdir -p $DATA_DIR

echo ""
echo "Step 1: Downloading dataset..."
echo "--------------------------------------------------------------"
python scripts/download_data.py --output_dir $DATA_DIR

echo ""
echo "Step 2: Training all models..."
echo "--------------------------------------------------------------"
echo "This may take several hours on GPU, or days on CPU."
echo "Paper uses: 2x NVIDIA RTX A6000 GPUs"
echo ""

python experiments/train_all_models.py \
    --config configs/default_config.yaml \
    --epochs 200 \
    --seed 42 \
    --output_dir $RESULTS_DIR

echo ""
echo "Step 3: Generating figures..."
echo "--------------------------------------------------------------"
python experiments/generate_figures.py \
    --results_dir $RESULTS_DIR \
    --output_dir $FIGURES_DIR

echo ""
echo "=============================================================="
echo "Experiments completed!"
echo "Results saved to: $RESULTS_DIR"
echo "Figures saved to: $FIGURES_DIR"
echo "=============================================================="
echo ""
echo "Paper Results Summary (expected):"
echo "  - Transformer + DMD+SMV: 98.9% accuracy (W=1)"
echo "  - Inference time: 0.056ms (W=1), 2ms (W=100)"
echo "  - Speedup: 820x vs baseline"
echo ""
