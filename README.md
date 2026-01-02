# DMD+SMV MATLAB Implementation

## Discrete Mode Decomposition Meets Shapley Value for Tactile Internet

**Author:** Ali Vahedi (Mohammad Ali Vahedifar)  
**Affiliation:** DIGIT and Department of ECE, Aarhus University, Denmark  
**Email:** av@ece.au.dk  
**IEEE INFOCOM 2025**

---

## Overview

This folder contains the MATLAB implementation of the DMD+SMV framework for robust signal prediction in Tactile Internet systems.

## Files

| File | Description |
|------|-------------|
| `DMD.m` | Main Discrete Mode Decomposition algorithm (Algorithm 1) |
| `SMV.m` | Shapley Mode Value computation with Monte Carlo approximation (Algorithm 2) |
| `hilbert_transform.m` | Discrete Hilbert Transform (Equations 6-8) |
| `wiener_filter.m` | Discrete Wiener Filtering (Equations 2-5) |
| `demo_dmd_smv.m` | Complete demonstration script |


## Requirements

- MATLAB R2019b or later
- Signal Processing Toolbox (optional, for comparison)

## Citation

```bibtex
@inproceedings{vahedifar2025dmd,
  title={Discrete Mode Decomposition Meets Shapley Value: 
         Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={IEEE INFOCOM 2026},
  year={2025}
}
```

## Acknowledgments

This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

## Contact

For questions or issues, contact: av@ece.au.dk

GitHub: https://github.com/Ali-Vahedifar/Discrete-Mode-Decomposition.git
