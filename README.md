# Two-Step Annealing DSC Analysis for Polystyrene (PS)

Differential scanning calorimetry (DSC) data processing for two-step annealing experiments on polystyrene, using the three-step specific heat capacity method.

## Background

Physical aging of amorphous polymers is studied by annealing below the glass transition temperature (Tg). This project investigates the effects of **two-step annealing** (T1 at 90°C, T2 at 80°C) with varying hold times on the enthalpy recovery of polystyrene measured during subsequent heating.

The **three-step DSC method** provides accurate specific heat capacity cp(T) by calibrating the sample signal against an empty crucible (baseline) and a sapphire reference:

```
cp_sample(T) = cp_ref(T) × [DSC_sample(T) − DSC_empty(T)] / [DSC_ref(T) − DSC_empty(T)] × (m_ref / m_sample)
```

Enthalpy change is obtained by integrating cp(T) over the temperature range of interest:

```
ΔH = ∫ cp(T) dT    (30°C → 100°C)
```

## Directory Structure

```
├── data/           # Raw DSC measurement files
│   ├── twosteps.xlsx
│   ├── ps-empty-01.xlsx
│   ├── ps-ref-01.xlsx
│   ├── PS-onestep-01.xlsx
│   ├── PS-onestep-02.xlsx
│   └── pskovacs.xlsx
├── results/        # Generated output
│   ├── twosteps_enthalpy_results.csv
│   └── twosteps_enthalpy_results.png
├── src/            # Processing code
│   └── process_twosteps.py
└── README.md
```

## Data Files

| File | Description |
|------|-------------|
| `data/twosteps.xlsx` | Main experiment: 20 annealing cycles with varying T1/T2 hold times |
| `data/ps-empty-01.xlsx` | Empty crucible baseline (30–200°C ramp, stopped at ~168°C) |
| `data/ps-ref-01.xlsx` | Sapphire (Al₂O₃) reference, 10 mg (30–200°C) |
| `data/PS-onestep-01.xlsx` | One-step annealing data (supplementary) |
| `data/PS-onestep-02.xlsx` | One-step annealing data (supplementary) |
| `data/pskovacs.xlsx` | Kovacs-type experiment data |

## Experimental Parameters

- **Sample**: Polystyrene (PS), 4.7 mg
- **Reference**: Synthetic sapphire, 10.0 mg
- **Temperature range**: 30–200°C
- **Heating rate**: 10°C/min
- **Annealing conditions**:
  - T1 at 90°C: 0.833 min (Group A) or 8.333 min (Group B)
  - T2 at 80°C: 0.0017 to 16.667 min (10 levels per group)

## Results

### Enthalpy Change (ΔH, 30–100°C)

| Group | T1 @90°C | ΔH range (J/g) | ΔH mean ± σ (J/g) |
|-------|----------|-----------------|---------------------|
| A | 0.833 min | 83.81 – 84.83 | 84.28 ± 0.40 |
| B | 8.333 min | 83.32 – 83.52 | 83.41 ± 0.06 |

- **Inter-group difference**: Δ(ΔH) = 0.87 J/g (A − B)
- Longer T1 annealing → more enthalpy relaxation → lower ΔH
- Within each group, longer T2 hold time → monotonic decrease in ΔH

### Output Files

- `results/twosteps_enthalpy_results.png` — Six-panel comprehensive figure (cp curves, DSC curves, ΔH vs T2, bar chart, cp_avg comparison, results table)
- `results/twosteps_enthalpy_results.csv` — Numerical results for all 20 annealing conditions

## Usage

```bash
cd src
pip install numpy pandas scipy matplotlib openpyxl
python process_twosteps.py
```

## Requirements

- Python ≥ 3.8
- numpy, pandas, scipy, matplotlib, openpyxl

## Reference

The sapphire specific heat data is based on NIST SRM 720 certificate values.
