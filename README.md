# Two-Annealing DSC Analysis for Polystyrene (PS)

DSC data processing for one-step, two-step, and Kovacs-type annealing experiments on polystyrene.

## Directory Structure

```
├── data/                   # Raw DSC measurement files (.xlsx)
├── results/enthalpy/       # Exported enthalpy data (.csv)
├── src/
│   └── export_enthalpy.py  # Enthalpy calculation + export
└── README.md
```

## Experimental Parameters

- **Sample**: Polystyrene (PS), 4.7 mg, Mw = 280,000
- **Heating rate**: 10°C/min
- **Integration**: 30°C to liquid onset temperature (condition-dependent)

## Experiments

| Experiment | T1 | T2 | Type |
|-----------|-----|-----|------|
| One-step 50°C | 50°C | — | Single annealing |
| One-step 70°C | 70°C | — | Single annealing |
| Two-step | 90°C | 80°C | Down-jump (cooling) |
| Kovacs | 80°C | 90°C | Up-jump (memory) |

## Usage

```bash
cd src
pip install numpy pandas scipy openpyxl
python export_enthalpy.py
```

Output: `results/enthalpy/enthalpy_*.csv`
