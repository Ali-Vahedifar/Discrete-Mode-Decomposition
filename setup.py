#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for DMD+SMV Tactile Internet package.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
IEEE INFOCOM 2025

This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            # Remove version specifiers for basic setup
            requirements.append(line.split(">=")[0].split("==")[0])

setup(
    name="dmd-smv-tactile-internet",
    version="1.0.0",
    author="Ali Vahedi",
    author_email="av@ece.au.dk",
    description="Discrete Mode Decomposition meets Shapley Value for Robust Signal Prediction in Tactile Internet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alivahedi/dmd-smv-tactile-internet",
    project_urls={
        "Bug Tracker": "https://github.com/alivahedi/dmd-smv-tactile-internet/issues",
        "Documentation": "https://github.com/alivahedi/dmd-smv-tactile-internet/docs",
        "Source Code": "https://github.com/alivahedi/dmd-smv-tactile-internet",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Signal Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "tracking": [
            "tensorboard>=2.10.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dmd-train=experiments.train_all_models:main",
            "dmd-eval=experiments.evaluate_accuracy:main",
            "dmd-infer=experiments.evaluate_inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "tactile internet",
        "haptic signals",
        "discrete mode decomposition",
        "shapley value",
        "signal prediction",
        "deep learning",
        "transformer",
        "teleoperation",
    ],
)
