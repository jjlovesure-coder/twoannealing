---
name: twosteps-workflow
description: Two-step (high→low T) annealing analysis — KWW phenom fit + TNM direct fit to experiment + 3-way comparison plot
model: haiku
---

# Two-Step Annealing — Full Analysis Workflow

从双步退火（高温→低温）DSC 焓变数据出发，执行完整的分析流水线。

## Protocol

- T1 = 90°C, T2 = 80°C (down-jump, standard enthalpy relaxation)
- T1 hold times: 50s (Group A) / 500s (Group B)
- T2 hold times: 0.1 – 1000s
- Cooling rate: 60 K/min throughout

## Workflow Stages

```
enthalpy_twosteps.csv
    │
    ▼  Stage 1: twosteps_fit_phenom.py
KWW Phenomenological Fit + Phenom Curve
    │
    ▼  Stage 2: twosteps_fit_tnm.py
TNM Direct Fit to Experiment
    │
    ▼  Stage 3: Final Comparison
3-way Plot (Exp + KWW + TNM)
```

## Execution

### Full pipeline
```bash
python3 src/run_twosteps_workflow.py
```

### Individual stages
```bash
python3 src/twosteps_fit_phenom.py     # Stage 1: KWW phenom. fit
python3 src/twosteps_fit_tnm.py        # Stage 2: TNM direct fit
```

## Stage Details

### Stage 1 — KWW Fit (`src/twosteps_fit_phenom.py`)
- **Input:** `results/enthalpy/enthalpy_twosteps.csv`
- **Model:** Time-shifted KWW with shared asymptote
- **Output:** `results/twosteps_phenom_curve.csv`, `results/twosteps_phenom_fit.png`

### Stage 2 — TNM Fit (`src/twosteps_fit_tnm.py`)
- **Critical:** Fits TNM directly to experimental data, NOT to KWW phenom curve
  - KWW τ_50 > τ_500 contradicts TNM physics (higher Tf → smaller τ)
  - Direct fit to experiment avoids this contradiction
- **Model:** Instantaneous-quench TNM
- **Bounds:** Wider x range (0.05–0.7) than Kovacs, to allow Tf coupling
- **Output:** `results/tnm/tnm_twosteps_params.csv`

### Stage 3 — Comparison (`run_twosteps_workflow.py`)
- 3-way overlay: Experiment × KWW fit × TNM prediction
- Semilog axes with zoom inset (0.06–20s), connection arrows
- **Output:** `results/twosteps_full_comparison.png`, `results/twosteps_comparison_table.csv`

## Output Files

| File | Description |
|------|-------------|
| `results/twosteps_phenom_curve.csv` | KWW fitted curve (dense, 150 pts) |
| `results/twosteps_phenom_fit.png` | KWW fit plot |
| `results/tnm/tnm_twosteps_params.csv` | TNM best-fit parameters |
| `results/twosteps_full_comparison.png` | Final 3-way comparison |
| `results/twosteps_comparison_table.csv` | Numerical comparison table |

## Best-Fit TNM Parameters

| Parameter | Value |
|-----------|-------|
| log A | −24.17 |
| H* | 187.6 kJ/mol |
| x | 0.98 |
| β | 0.98 |
| T0 | 450 K (177°C) |
| ΔH_max | 6.94 J/g |

## Key Differences from Kovacs Workflow

| Aspect | Kovacs (up-jump) | Two-Step (down-jump) |
|--------|:---:|:---:|
| TNM target | KWW phenom curve ✓ | Experimental data directly |
| x bounds | [0.1, 0.95] | [0.2, 0.98] |
| T0 bounds | [375, 420] | [380, 450] |
| R² (typical) | 0.95 / 0.99 | 0.82 / 0.78 |

> **Note:** For down-jump, KWW phenom curve has physically impossible shape for TNM
> (τ_50s > τ_500s). Always fit TNM directly to experimental data for down-jump protocols.

## Applicability

This workflow applies to any **high-temperature → low-temperature** two-step
annealing protocol. If the protocol uses different temperatures, update the
`T1` and `T2` constants in the scripts. The core methodology (KWW phenom +
TNM direct fit) is protocol-agnostic.
