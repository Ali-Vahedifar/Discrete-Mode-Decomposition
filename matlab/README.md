# DMD+SMV MATLAB Implementation

## Discrete Mode Decomposition Meets Shapley Value for Tactile Internet

**Author:** Ali Vahedi  
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

## Quick Start

```matlab
% Run the demo
demo_dmd_smv

% Or use individual functions:
% 1. Apply DMD
[modes, center_freqs, residual] = DMD(signal, alpha);

% 2. Compute Shapley values
[shapley_values, rankings] = SMV(modes, center_freqs, predictor, X_val, y_val);
```

## Algorithm Parameters

### DMD Parameters (from paper)
```matlab
params.epsilon1 = 1e-6;      % Regularization (Eq. 12)
params.epsilon2 = 1e-6;      % Regularization (Eq. 14)
params.tau1 = 0.1;           % Step size for ρ update (Eq. 31)
params.tau2 = 0.1;           % Step size for μ update (Eq. 32)
params.kappa1 = 1e-3;        % Inner convergence threshold
params.kappa2 = 1e-3;        % Outer convergence threshold
params.max_modes = 10;       % Maximum modes to extract
```

### SMV Parameters (from paper)
```matlab
params.tolerance = 0.01;     % 1% convergence tolerance (Eq. 36)
params.max_iterations = 1000;
params.epsilon3 = 0.01;      % Performance tolerance
params.metric = 'accuracy';  % Performance metric
```

## Key Equations

### DMD Update Equations

**Mode Update (Eq. 24):**
```
U_M^{n+1}(ω) = [ρ(ω)/2 * Q(ω)] / [ρ(ω)/2 + Σ|β_i(ω)|² + (2/π)sin²(ω-ω_M)]
```

**Center Frequency Update (Eq. 27):**
```
ω_M^{n+1} = ∫ω|U_M(ω)|²dω / ∫|U_M(ω)|²dω
```

**Unprocessed Signal Update (Eq. 30):**
```
X_u^{n+1}(ω) = [ρ(ω) * Q̃(ω)] / [2|β_M(ω)|² + 2μ + ρ(ω)]
```

**Multiplier Updates (Eq. 31-32):**
```
ρ^{n+1}(ω) = ρ^n(ω) + τ_1 * (X(ω) - Σ U_i(ω))
μ^{n+1} = max(0, μ^n + τ_2 * ∫(||X_u||² - ||U_min||²)dω)
```

### Shapley Mode Value (Eq. 33)
```
X_i = Σ_{S⊆D\{i}} [|S|!(|D|-|S|-1)!/|D|!] * [V(S∪{i}) - V(S)]
```

## Expected Results (from paper)

| Metric | Value |
|--------|-------|
| Accuracy (W=1) | 98.9% |
| Accuracy (W=100) | 92.5% |
| PSNR (W=1) | ~29.5 dB |
| Inference time (W=1) | 0.056 ms |
| Inference time (W=100) | 2 ms |
| Speedup vs baseline | 820× |

## Requirements

- MATLAB R2019b or later
- Signal Processing Toolbox (optional, for comparison)

## Citation

```bibtex
@inproceedings{vahedifar2025dmd,
  title={Discrete Mode Decomposition Meets Shapley Value: 
         Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={IEEE INFOCOM 2025},
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
