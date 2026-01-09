#!/usr/bin/env python3
"""
Download Dataset Script
=======================

Downloads the Kinaesthetic Interactions Dataset from Zenodo.

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025

Dataset Reference:
-----------------
Rodríguez-Guevara, D. & Hernandez Gobertti, F.A. (2025).
Kinaesthetic Interactions Dataset. Zenodo.
https://doi.org/10.5281/zenodo.14924062

Usage: python scripts/download_data.py --output_dir ./data
"""

import argparse
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import hashlib
from tqdm import tqdm


# Dataset information
DATASET_URL = "https://zenodo.org/records/14924062/files/kinaesthetic_interactions.zip"
DATASET_DOI = "10.5281/zenodo.14924062"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads. Author: Ali Vahedi"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file with progress bar. Author: Ali Vahedi"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Kinaesthetic Interactions Dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./data',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download even if exists'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Kinaesthetic Interactions Dataset Downloader")
    print("Author: Ali Vahedi")
    print("IEEE INFOCOM 2025")
    print("=" * 60)
    print(f"\nDataset DOI: {DATASET_DOI}")
    print(f"URL: {DATASET_URL}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "kinaesthetic_interactions.zip"
    
    # Check if already downloaded
    if zip_path.exists() and not args.force:
        print(f"\nDataset already downloaded at {zip_path}")
        print("Use --force to re-download")
    else:
        print(f"\nDownloading dataset to {zip_path}...")
        try:
            download_file(DATASET_URL, zip_path, "Downloading dataset")
            print("Download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nPlease download manually from:")
            print(f"  {DATASET_URL}")
            print(f"And place in: {output_dir}")
            
            # Create placeholder data for testing
            print("\nCreating synthetic placeholder data for testing...")
            create_placeholder_data(output_dir)
            return
    
    # Extract
    print("\nExtracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Extraction failed: {e}")
        create_placeholder_data(output_dir)
    
    print(f"\nDataset ready at: {output_dir}")
    print("\nDataset contains:")
    for item in output_dir.iterdir():
        if item.is_file():
            size = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}: {size:.2f} MB")


def create_placeholder_data(output_dir: Path):
    """Create synthetic placeholder data. Author: Ali Vahedi"""
    import numpy as np
    
    print("Creating synthetic Tap-and-Hold data...")
    
    np.random.seed(42)
    n_samples = 50000
    t = np.linspace(0, n_samples/1000, n_samples)
    
    # Position
    pos_x = 0.1 * np.sin(2 * np.pi * 0.5 * t) + 0.02 * np.random.randn(n_samples)
    pos_y = 0.1 * np.cos(2 * np.pi * 0.5 * t) + 0.02 * np.random.randn(n_samples)
    pos_z = 0.05 * np.sin(2 * np.pi * 1.0 * t) + 0.01 * np.random.randn(n_samples)
    
    # Velocity
    vel_x = np.gradient(pos_x) * 1000 + 0.1 * np.random.randn(n_samples)
    vel_y = np.gradient(pos_y) * 1000 + 0.1 * np.random.randn(n_samples)
    vel_z = np.gradient(pos_z) * 1000 + 0.05 * np.random.randn(n_samples)
    
    # Force
    k = 100
    force_x = -k * pos_x + 5 * np.random.randn(n_samples)
    force_y = -k * pos_y + 5 * np.random.randn(n_samples)
    force_z = -k * np.clip(pos_z, -np.inf, 0) + 5 * np.random.randn(n_samples)
    
    data = np.column_stack([
        pos_x, pos_y, pos_z,
        vel_x, vel_y, vel_z,
        force_x, force_y, force_z
    ]).astype(np.float32)
    
    # Save
    tap_hold_dir = output_dir / "tap_and_hold"
    tap_hold_dir.mkdir(exist_ok=True)
    
    np.save(tap_hold_dir / "human_data.npy", data)
    np.save(tap_hold_dir / "robot_data.npy", data * 0.95)  # Slightly different
    
    # Save CSV version
    import csv
    with open(tap_hold_dir / "data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z',
            'force_x', 'force_y', 'force_z'
        ])
        writer.writerows(data[:10000])  # First 10k samples
    
    print(f"Placeholder data created at: {tap_hold_dir}")
    print(f"  human_data.npy: {data.nbytes / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
