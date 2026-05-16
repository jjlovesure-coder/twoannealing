---
name: kovacs-workflow
description: Full Kovacs DSC analysis pipeline — process raw DSC data → ΔH → KWW phenomenological fit → TNM reverse engineering → final comparison plot
model: haiku
---

# Kovacs Analysis — Full Experiment Workflow

从原始 DSC 实验数据出发，执行完整的 Kovacs 退火分析流水线。

## Workflow Stages

```
Raw DSC Data (.xlsx)
    │
    ▼  Stage 1: export_enthalpy.py
ΔH data (.csv)
    │
    ▼  Stage 2: kovacs_fit_phenom.py
KWW Phenomenological Fit
    │
    ▼  Stage 3: kovacs_reverse_tnm.py
TNM Model Parameters
    │
    ▼  Stage 4: Final Comparison
3-way Plot (Exp + KWW + TNM)
```

## Execution

### Quick — run full pipeline
```bash
python3 src/run_full_workflow.py
```

### Individual stages
```bash
python3 src/export_enthalpy.py          # Stage 1: DSC → ΔH
python3 src/kovacs_fit_phenom.py        # Stage 2: KWW fit
python3 src/kovacs_reverse_tnm.py       # Stage 3: TNM reverse
```

## Stage Details

### Stage 1 — DSC Processing (`src/export_enthalpy.py`)
- **Input**: Raw DSC Excel files in `data/` (pskovacs.xlsx, PS-onestep-*.xlsx, twosteps.xlsx, ps-empty-01.xlsx)
- **Process**: Baseline subtraction, liquid onset detection, ΔH integration (40°C → T_onset), unit conversion (μW·K → J/g)
- **Output**: `results/enthalpy/enthalpy_*.csv` (kovacs, onestep_50C, onestep_70C, twosteps)
- **Reference state**: As-cooled without annealing (first heating ramp)

### Stage 2 — KWW Phenomenological Fit (`src/kovacs_fit_phenom.py`)
- **Model**: Time-shifted KWW with shared asymptote
  ```
  ΔH(t; g) = H_max - H_max · exp(−((t + c_g) / τ_g)^β_g)
  c_g = τ_g · (−ln(1 − ΔH₀_g / H_max))^(1/β_g)
  ```
- **Parameters**: H_max (shared), ΔH₀_g, τ_g, β_g per group (50s / 500s)
- **Method**: differential_evolution + least_squares refinement
- **Output**: `results/kovacs_phenom_curve.csv` (150 dense sample points), `results/kovacs_phenom_fit.png`

### Stage 3 — TNM Reverse Engineering (`src/kovacs_reverse_tnm.py`)
- **Model**: Instantaneous-quench TNM (standard in literature, see D'Amore 2006, Grassi 2018)
  ```
  τ(T, Tf) = A · exp[x·H*/(RT) + (1−x)·H*/(RTf)]
  Tf = T_hold + (Tf_start − T_hold) · exp(−(∫dt/τ)^β)
  ```
- **Protocol**: T0 → T1=80°C (hold t1=50s or 500s) → T2=90°C (hold t2)
- **Parameters**: logA, H*, x, β, T0, scale
- **Method**: Multi-start L-BFGS-B with physically-constrained bounds
- **Bounds**: x∈[0.1,0.95], β∈[0.1,0.7], T0∈[375,420]K (PS Tg≈373K)
- **Output**: `results/tnm/tnm_reverse_params.csv`, `results/tnm/kovacs_reverse_tnm.png`

### Stage 4 — Final Comparison
- 3-way overlay: Experiment × KWW fit × TNM prediction
- Semilog axes, zoom inset for short-time details (0.06–20s)
- Connection arrows from zoom region to inset
- Parameter annotation box
- **Output**: `results/kovacs_full_comparison.png`, `results/kovacs_comparison_table.csv`

## Key Parameters (Best Fit)

| Parameter | Value | Source |
|-----------|-------|--------|
| H_max | 6.62 J/g | Phenom KWW fit |
| log A | −30.55 | TNM reverse |
| H* | 229.9 kJ/mol | TNM reverse |
| x | 0.950 | TNM reverse (cf. D'Amore 2006: x=0.9) |
| β | 0.700 | TNM reverse |
| T0 | 380.2 K (107°C) | TNM reverse (just above PS Tg) |
| scale | 6.58 J/g | TNM reverse (≈ H_max) |

## Fit Quality

| Model | R² (50s) | R² (500s) | RMSE (J/g) |
|-------|----------|-----------|------------|
| KWW phenom. | 0.979 | 0.995 | 0.195 |
| TNM vs experiment | 0.946 | 0.992 | — |
| TNM vs phenom. curve | 0.991 | 0.999 | 0.173 |

## Data Files

| File | Description |
|------|-------------|
| `data/pskovacs.xlsx` | Kovacs up-jump DSC raw data |
| `data/PS-onestep-01.xlsx` | One-step 50°C DSC |
| `data/PS-onestep-02.xlsx` | One-step 70°C DSC |
| `data/twosteps.xlsx` | Two-step 90→80°C DSC |
| `data/ps-empty-01.xlsx` | Empty crucible baseline |
| `results/enthalpy/enthalpy_kovacs.csv` | Processed Kovacs ΔH |
| `results/kovacs_phenom_curve.csv` | KWW fitted curve (dense) |
| `results/tnm/tnm_reverse_params.csv` | TNM best-fit parameters |

## Literature References

- D'Amore et al. (2006), *Composites Part A* — PS TNM parameters: x=0.9
- Grassi, Koh, Simon (2018), *Macromolecules* — Flash DSC, modified TNM for Kovacs signatures
- Tropin et al. (2015), *J. Non-Cryst. Solids* — PS heat capacity vs cooling rate
