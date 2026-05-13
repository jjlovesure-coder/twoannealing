# Empty-Crucible Subtracted Direct Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace reference-ramp difference with empty-crucible subtraction + direct integration, outputting two ΔH schemes (dynamic T_onset vs fixed 100°C upper bound) per experiment.

**Architecture:** Add three shared functions to `process_shared.py` (load empty crucible, interpolate+subtract, detect T_onset). Refactor `process_onestep.py`, `process_twosteps.py`, `process_kovacs.py` to use empty-crucible subtraction and produce dual-scheme outputs. Each script runs independently and saves plots + CSVs to `results/dsc/`.

**Tech Stack:** Python 3, numpy, scipy, pandas, matplotlib, openpyxl

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/process_shared.py` | Modify | Add `load_empty_crucible()`, `subtract_empty_crucible()`, `detect_t_onset()` |
| `src/process_onestep.py` | Modify | Refactor to empty-crucible subtraction + dual-scheme integration |
| `src/process_twosteps.py` | Modify | Refactor to empty-crucible subtraction + dual-scheme integration |
| `src/process_kovacs.py` | Modify | Refactor to empty-crucible subtraction + dual-scheme integration |

---

### Task 1: Add shared functions to `process_shared.py`

**Files:**
- Modify: `src/process_shared.py`

- [ ] **Step 1: Add `load_empty_crucible` function**

Add after the existing `load_dsc_simple` function (after line 82). The empty crucible CSV has columns `Time/min,Temp/Cel,DSC/uW,DDSC/(uW/min)` and contains a single 30→~168°C heating ramp.

```python
def load_empty_crucible(csv_path):
    """Load empty crucible baseline CSV (single heating ramp 30→~168°C).

    Returns T (°C) and DSC (uW) arrays for the heating portion.
    """
    df = pd.read_csv(csv_path)
    T = df['Temp/Cel'].values.astype(float)
    DSC = df['DSC/uW'].values.astype(float)
    # Filter to heating ramp: 30°C → ~168°C (monotonic increase)
    mask = T >= 30.0
    return T[mask], DSC[mask]
```

- [ ] **Step 2: Add `subtract_empty_crucible` function**

Add after `load_empty_crucible`:

```python
def subtract_empty_crucible(T_sample, DSC_sample, T_empty, DSC_empty):
    """Interpolate empty crucible DSC to sample T grid and subtract.

    Returns DSC_corrected = DSC_sample - DSC_empty_interp.
    """
    f_empty = interp1d(T_empty, DSC_empty, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    DSC_empty_interp = f_empty(T_sample)
    return DSC_sample - DSC_empty_interp
```

- [ ] **Step 3: Add `detect_t_onset` function**

Add after `subtract_empty_crucible`. The function scans downward from 150°C to find where the corrected DSC signal deviates from the post-Tg linear baseline, marking the entry to the supercooled liquid regime.

```python
def detect_t_onset(T, DSC_corrected, post_Tg_range=(120, 150)):
    """Detect supercooled-liquid onset temperature from corrected DSC.

    Fits a linear baseline to the post-Tg region, then scans downward
    from the high-T end to find where the corrected signal deviates
    beyond 3x the post-Tg residual noise level.

    Returns T_onset in °C, or 100.0 if detection fails.
    """
    post_mask = (T >= post_Tg_range[0]) & (T <= post_Tg_range[1])
    if np.sum(post_mask) < 5:
        return 100.0

    post_T = T[post_mask]
    post_DSC = DSC_corrected[post_mask]
    coeffs = np.polyfit(post_T, post_DSC, 1)
    baseline = np.polyval(coeffs, post_T)
    residuals = post_DSC - baseline
    noise_std = np.std(residuals)
    threshold = 3.0 * max(noise_std, 0.01)

    # Scan downward from 150°C
    scan_mask = (T >= 80) & (T <= 150)
    scan_idx = np.where(scan_mask)[0]
    if len(scan_idx) == 0:
        return 100.0

    full_baseline = np.polyval(coeffs, T)
    excess = np.abs(DSC_corrected - full_baseline)

    # Find first index (highest T) where excess exceeds threshold
    for i in range(len(scan_idx) - 1, -1, -1):
        idx = scan_idx[i]
        if excess[idx] > threshold:
            T_candidate = T[idx]
            return float(np.clip(T_candidate, 95, 140))

    return 100.0
```

- [ ] **Step 4: Verify the module imports work**

```bash
cd /root/twoannealing/src && python -c "
from process_shared import load_empty_crucible, subtract_empty_crucible, detect_t_onset
print('All three functions imported successfully')
print('load_empty_crucible:', load_empty_crucible)
print('subtract_empty_crucible:', subtract_empty_crucible)
print('detect_t_onset:', detect_t_onset)
"
```
Expected: prints success message with function references.

- [ ] **Step 5: Quick smoke test**

```bash
cd /root/twoannealing/src && python -c "
import numpy as np
from process_shared import load_empty_crucible, subtract_empty_crucible, detect_t_onset
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(ROOT, 'data', 'csv', 'baseline-01.csv')

# Test load
T_e, DSC_e = load_empty_crucible(csv_path)
print(f'Loaded empty: {len(T_e)} pts, T range {T_e.min():.1f}-{T_e.max():.1f} C')

# Test subtract with synthetic sample data
T_sample = np.arange(30, 150, 0.5)
DSC_sample = np.sin(T_sample / 20) * 100 + 500  # synthetic
DSC_corr = subtract_empty_crucible(T_sample, DSC_sample, T_e, DSC_e)
print(f'Subtracted: {len(DSC_corr)} pts, range {DSC_corr.min():.1f}-{DSC_corr.max():.1f} uW')

# Test T_onset detection
T_onset = detect_t_onset(T_sample, DSC_corr)
print(f'T_onset: {T_onset:.1f} C')
print('Smoke test passed')
"
```
Expected: prints data ranges and T_onset, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/process_shared.py
git commit -m "feat: add empty-crucible loading, subtraction, and T_onset detection to process_shared"
```

---

### Task 2: Refactor `process_onestep.py` to use empty-crucible subtraction

**Files:**
- Modify: `src/process_onestep.py`

The core change: replace the "per-run shortest-hold reference" logic with empty-crucible subtraction, then integrate over both Scheme ① (dynamic T_onset) and Scheme ② (fixed 30-100°C).

- [ ] **Step 1: Update imports**

Replace the import block (lines 17-21) to also import the new shared functions:

```python
from process_shared import (
    load_dsc_experiment, detect_heating_ramps, compute_enthalpy,
    load_empty_crucible, subtract_empty_crucible, detect_t_onset,
    HEATING_T_START, HEATING_T_END,
)
```

- [ ] **Step 2: Update constants and add empty-crucible path**

Replace `T_INT_LOW`, `T_INT_HIGH`, `T_GRID_STEP` block (lines 27-30) and add the empty crucible path:

```python
T_INT_LOW = 30       # lower bound for both schemes
T_INT_HIGH_FIXED = 100  # Scheme ② fixed upper bound
T_GRID_STEP = 0.2

EMPTY_CRUCIBLE_CSV = os.path.join(DATA_DIR, 'csv', 'baseline-01.csv')
```

- [ ] **Step 3: Rewrite `process_one_step` function**

Replace lines 82-255 (the entire `process_one_step` function). Key changes:
- Load empty crucible data once
- For each ramp: subtract empty, detect T_onset (Scheme ①), integrate 30→T_onset and 30→100°C (Scheme ②)
- Output plots comparing both schemes

```python
def process_one_step(data_file, sheet, T_anneal, label, out_name):
    """Process one-step annealing with empty-crucible subtraction.

    Two schemes:
      ① Dynamic: integrate 30°C → T_onset (per-curve supercooled liquid onset)
      ② Fixed:   integrate 30°C → 100°C
    """
    print(f"\n{'='*70}")
    print(f"  One-step annealing ({label}) — T_anneal = {T_anneal}°C")
    print(f"  Method: empty-crucible subtraction + direct integration")
    print(f"{'='*70}")

    # Load empty crucible
    T_empty, DSC_empty = load_empty_crucible(EMPTY_CRUCIBLE_CSV)
    print(f"  Loaded empty crucible: {len(T_empty)} pts, "
          f"{T_empty.min():.1f}–{T_empty.max():.1f}°C")

    # Load experiment and detect ramps
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, exp_data['Time'].values)
    print(f"  Found {len(ramps)} heating ramps")

    conditions = map_program_to_ramps(program, ramps)
    print(f"  Mapped {len(conditions)} annealed ramps per run")

    T_grid = np.arange(T_INT_LOW, T_INT_HIGH_FIXED + 20 + T_GRID_STEP, T_GRID_STEP)

    # Process each ramp
    results = []
    run_colors = {'R1': '#2166AC', 'R2': '#B2182B'}
    run_markers = {'R1': 'o', 'R2': 's'}

    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_ramp = T_exp[s:e+1]
        DSC_ramp = DSC_exp[s:e+1]

        # Subtract empty crucible
        DSC_corr = subtract_empty_crucible(T_ramp, DSC_ramp, T_empty, DSC_empty)

        # Scheme ①: dynamic T_onset
        T_onset = detect_t_onset(T_ramp, DSC_corr)
        mask1 = (T_ramp >= T_INT_LOW) & (T_ramp <= T_onset)
        area1 = trapezoid(DSC_corr[mask1], T_ramp[mask1])
        dH1 = compute_enthalpy(area1)

        # Scheme ②: fixed 30–100°C
        mask2 = (T_ramp >= T_INT_LOW) & (T_ramp <= T_INT_HIGH_FIXED)
        area2 = trapezoid(DSC_corr[mask2], T_ramp[mask2])
        dH2 = compute_enthalpy(area2)

        results.append({
            'ramp': cond['ramp_idx'],
            'run': cond['run'],
            'hold_s': cond['hold_s'],
            'hold_min': cond['hold_min'],
            'T_onset': T_onset,
            'delta_H_dyn_kJmol': dH1,
            'delta_H_fixed_kJmol': dH2,
        })
        print(f"    Ramp {cond['ramp_idx']:2d}: hold={cond['hold_s']:8.1f}s, "
              f"T_onset={T_onset:.1f}°C → "
              f"ΔH_dyn={dH1:+.2f}, ΔH_fixed={dH2:+.2f} kJ/mol")

    results_df = pd.DataFrame(results)

    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: ΔH vs hold time — both schemes
    ax1 = axes[0, 0]
    for rl in ['R1', 'R2']:
        rd = results_df[results_df['run'] == rl].sort_values('hold_s')
        ax1.plot(rd['hold_s'], rd['delta_H_dyn_kJmol'],
                 marker=run_markers[rl], color=run_colors[rl],
                 linewidth=1.8, markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=f'{rl} dyn', linestyle='-')
        ax1.plot(rd['hold_s'], rd['delta_H_fixed_kJmol'],
                 marker=run_markers[rl], color=run_colors[rl],
                 linewidth=1.2, markersize=6, markerfacecolor='white',
                 markeredgewidth=1.0, label=f'{rl} fixed', linestyle='--')
    ax1.set_xlabel(f'Hold time at {T_anneal}°C (s)')
    ax1.set_ylabel('ΔH (kJ/mol)')
    ax1.set_title(f'{label}: ΔH vs annealing time (both schemes)')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: Raw DSC curves (selected)
    ax2 = axes[0, 1]
    sample_idx = [0, 2, 4, 6, 8]
    for idx in sample_idx:
        if idx >= len(conditions):
            continue
        cond = conditions[idx]
        s, e = cond['start_idx'], cond['end_idx']
        ax2.plot(T_exp[s:e+1], DSC_exp[s:e+1],
                 alpha=0.7, linewidth=0.7,
                 color=run_colors[cond['run']],
                 label=f"{cond['run']}: {cond['hold_s']:.1f}s")
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('DSC (µW)')
    ax2.set_title('Raw DSC heating curves (selected)')
    ax2.legend(fontsize=6, loc='lower right')
    ax2.set_xlim(T_INT_LOW - 10, T_INT_HIGH_FIXED + 30)

    # Panel 3: Corrected DSC with integration ranges
    ax3 = axes[0, 2]
    for idx in sample_idx:
        if idx >= len(conditions):
            continue
        cond = conditions[idx]
        s, e = cond['start_idx'], cond['end_idx']
        T_ramp = T_exp[s:e+1]
        DSC_ramp = DSC_exp[s:e+1]
        DSC_corr = subtract_empty_crucible(T_ramp, DSC_ramp, T_empty, DSC_empty)
        ax3.plot(T_ramp, DSC_corr, alpha=0.7, linewidth=0.7,
                 color=run_colors[cond['run']])
    ax3.axvline(T_INT_HIGH_FIXED, color='green', linestyle='--', alpha=0.6,
                label=f'Scheme ②: {T_INT_HIGH_FIXED}°C')
    ax3.axvline(T_INT_LOW, color='gray', linestyle=':', alpha=0.4,
                label=f'Low: {T_INT_LOW}°C')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('DSC_corrected (µW)')
    ax3.set_title('Corrected DSC (sample − empty)')
    ax3.legend(fontsize=7)
    ax3.set_xlim(T_INT_LOW - 10, T_INT_HIGH_FIXED + 30)

    # Panel 4: ΔH bar comparison
    ax4 = axes[1, 0]
    n = len(results_df)
    x_pos = np.arange(n)
    ax4.bar(x_pos - 0.15, results_df['delta_H_dyn_kJmol'], 0.3,
            color='#2166AC', alpha=0.7, label='Scheme ① dyn')
    ax4.bar(x_pos + 0.15, results_df['delta_H_fixed_kJmol'], 0.3,
            color='#B2182B', alpha=0.7, label='Scheme ② fixed')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{r['run']}\n{r['hold_s']:.1f}s"
                         for _, r in results_df.iterrows()],
                        fontsize=5.5, rotation=45)
    ax4.set_ylabel('ΔH (kJ/mol)')
    ax4.set_title(f'{label}: ΔH distribution — both schemes')
    ax4.legend(fontsize=7)

    # Panel 5: T_onset distribution
    ax5 = axes[1, 1]
    for rl in ['R1', 'R2']:
        rd = results_df[results_df['run'] == rl]
        ax5.plot(rd['hold_s'], rd['T_onset'],
                 marker=run_markers[rl], color=run_colors[rl],
                 linewidth=1.5, markersize=8, markerfacecolor='white',
                 markeredgewidth=1.5, label=rl)
    ax5.set_xlabel(f'Hold time at {T_anneal}°C (s)')
    ax5.set_ylabel('T_onset (°C)')
    ax5.set_title(f'{label}: Detected T_onset per ramp')
    ax5.set_xscale('log')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, which='both')

    # Panel 6: Results table
    ax6 = axes[1, 2]
    ax6.axis('off')
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}", r['run'],
            f"{r['hold_s']:.1f}",
            f"{r['T_onset']:.1f}",
            f"{r['delta_H_dyn_kJmol']:.2f}",
            f"{r['delta_H_fixed_kJmol']:.2f}",
        ])
    col_labels = ['Ramp', 'Run', 'Hold(s)', 'T_onset', 'ΔH_dyn', 'ΔH_fixed']
    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.06, 0.05, 0.11, 0.09, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.2)
    ax6.set_title('Results Summary', fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    fig.suptitle(f'{label}: Empty-Crucible Subtracted Direct Integration',
                 fontsize=12, fontweight='bold', y=1.01)
    png_path = os.path.join(RESULTS_DIR, f'{out_name}_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {png_path}")

    csv_path = os.path.join(RESULTS_DIR, f'{out_name}_results.csv')
    save_cols = ['ramp', 'run', 'hold_s', 'hold_min', 'T_onset',
                 'delta_H_dyn_kJmol', 'delta_H_fixed_kJmol']
    results_df[save_cols].to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Summary
    print(f"\n  Results ({label}):")
    for rl in ['R1', 'R2']:
        rd = results_df[results_df['run'] == rl]
        if len(rd) == 0:
            continue
        print(f"  {rl}: ΔH_dyn = {rd['delta_H_dyn_kJmol'].min():.2f} – "
              f"{rd['delta_H_dyn_kJmol'].max():.2f} kJ/mol, "
              f"ΔH_fixed = {rd['delta_H_fixed_kJmol'].min():.2f} – "
              f"{rd['delta_H_fixed_kJmol'].max():.2f} kJ/mol, "
              f"n={len(rd)} ramps")

    return results_df
```

- [ ] **Step 4: Run the script**

```bash
cd /root/twoannealing/src && python process_onestep.py
```
Expected: processes both one-step datasets, saves plots and CSVs, prints summary.

- [ ] **Step 5: Verify both CSVs have the new columns**

```bash
head -3 /root/twoannealing/results/dsc/onestep_50C_results.csv && echo "---" && head -3 /root/twoannealing/results/dsc/onestep_70C_results.csv
```
Expected: columns include `ramp,run,hold_s,hold_min,T_onset,delta_H_dyn_kJmol,delta_H_fixed_kJmol`.

- [ ] **Step 6: Commit**

```bash
git add src/process_onestep.py results/dsc/onestep_50C_results.csv results/dsc/onestep_50C_results.png results/dsc/onestep_70C_results.csv results/dsc/onestep_70C_results.png
git commit -m "refactor: one-step processing to empty-crucible subtraction with dual-scheme integration"
```

---

### Task 3: Refactor `process_twosteps.py` to use empty-crucible subtraction

**Files:**
- Modify: `src/process_twosteps.py`

Same pattern as Task 2 but for two-step annealing protocol. The ramp mapping (`map_two_step`) stays the same.

- [ ] **Step 1: Update imports**

Replace lines 17-20:

```python
from process_shared import (
    load_dsc_experiment, detect_heating_ramps, compute_enthalpy,
    load_empty_crucible, subtract_empty_crucible, detect_t_onset,
    HEATING_T_START, HEATING_T_END,
)
```

- [ ] **Step 2: Update constants**

Replace lines 26-28:

```python
T_INT_LOW = 30
T_INT_HIGH_FIXED = 100
T_GRID_STEP = 0.2

EMPTY_CRUCIBLE_CSV = os.path.join(DATA_DIR, 'csv', 'baseline-01.csv')
```

- [ ] **Step 3: Rewrite `main` function**

Replace lines 73-269 (the entire `main` function). The `map_two_step` function stays unchanged. Key changes:
- Load empty crucible
- For each ramp: subtract empty, detect T_onset, integrate both schemes
- Plot dual-scheme comparison

```python
def main():
    print("=" * 70)
    print("  Two-step annealing DSC — Empty-crucible subtraction + direct integration")
    print("=" * 70)

    # Load empty crucible
    T_empty, DSC_empty = load_empty_crucible(EMPTY_CRUCIBLE_CSV)
    print(f"  Loaded empty crucible: {len(T_empty)} pts, "
          f"{T_empty.min():.1f}–{T_empty.max():.1f}°C")

    exp_data, program = load_dsc_experiment(
        os.path.join(DATA_DIR, 'twosteps.xlsx'), sheet='PS-02')
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, exp_data['Time'].values)
    print(f"  Found {len(ramps)} heating ramps")

    conditions = map_two_step(program, ramps)
    print(f"  Mapped {len(conditions)} annealed ramps:")
    for c in conditions:
        print(f"    Ramp {c['ramp_idx']:2d} [Grp {c['group']}]: "
              f"T1@90°C={c['T1_hold_s']:8.1f}s, T2@80°C={c['T2_hold_s']:8.1f}s")

    T_grid = np.arange(T_INT_LOW, T_INT_HIGH_FIXED + 30 + T_GRID_STEP, T_GRID_STEP)

    # Process each ramp
    results = []
    colors = {'A': '#2166AC', 'B': '#B2182B'}
    markers = {'A': 'o', 'B': 's'}

    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_ramp = T_exp[s:e+1]
        DSC_ramp = DSC_exp[s:e+1]

        DSC_corr = subtract_empty_crucible(T_ramp, DSC_ramp, T_empty, DSC_empty)

        # Scheme ①: dynamic T_onset
        T_onset = detect_t_onset(T_ramp, DSC_corr)
        mask1 = (T_ramp >= T_INT_LOW) & (T_ramp <= T_onset)
        area1 = trapezoid(DSC_corr[mask1], T_ramp[mask1])
        dH1 = compute_enthalpy(area1)

        # Scheme ②: fixed 30–100°C
        mask2 = (T_ramp >= T_INT_LOW) & (T_ramp <= T_INT_HIGH_FIXED)
        area2 = trapezoid(DSC_corr[mask2], T_ramp[mask2])
        dH2 = compute_enthalpy(area2)

        results.append({
            'ramp': cond['ramp_idx'],
            'group': cond['group'],
            'T1_hold_s': cond['T1_hold_s'],
            'T2_hold_s': cond['T2_hold_s'],
            'T1_hold_min': cond['T1_hold_min'],
            'T2_hold_min': cond['T2_hold_min'],
            'T_onset': T_onset,
            'delta_H_dyn_kJmol': dH1,
            'delta_H_fixed_kJmol': dH2,
        })
        print(f"    Ramp {cond['ramp_idx']:2d}: T1={cond['T1_hold_s']:8.1f}s, "
              f"T2={cond['T2_hold_s']:8.1f}s, T_onset={T_onset:.1f}°C → "
              f"ΔH_dyn={dH1:+.2f}, ΔH_fixed={dH2:+.2f} kJ/mol")

    results_df = pd.DataFrame(results)

    # Generate plots
    fig = plt.figure(figsize=(20, 14))

    # Panel 1: ΔH vs T2 hold time — both schemes per group
    ax1 = fig.add_subplot(2, 3, 1)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp].sort_values('T2_hold_s')
        ax1.plot(rd['T2_hold_s'], rd['delta_H_dyn_kJmol'],
                 marker=markers[grp], color=colors[grp], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, linestyle='-',
                 label=f'Grp {grp} dyn')
        ax1.plot(rd['T2_hold_s'], rd['delta_H_fixed_kJmol'],
                 marker=markers[grp], color=colors[grp], linewidth=1.2,
                 markersize=6, markerfacecolor='white',
                 markeredgewidth=1.0, linestyle='--',
                 label=f'Grp {grp} fixed')
    ax1.set_xlabel('T2 hold time at 80°C (s)')
    ax1.set_ylabel('ΔH (kJ/mol)')
    ax1.set_title('ΔH vs T2 annealing time (both schemes)')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: Corrected DSC curves
    ax2 = fig.add_subplot(2, 3, 2)
    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_ramp = T_exp[s:e+1]
        DSC_ramp = DSC_exp[s:e+1]
        DSC_corr = subtract_empty_crucible(T_ramp, DSC_ramp, T_empty, DSC_empty)
        ax2.plot(T_ramp, DSC_corr, alpha=0.35, linewidth=0.5,
                 color=colors[cond['group']])
    ax2.axvline(T_INT_HIGH_FIXED, color='green', linestyle='--', alpha=0.5,
                label=f'Fixed: {T_INT_HIGH_FIXED}°C')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('DSC_corrected (µW)')
    ax2.set_title('Corrected DSC — all ramps by group')
    ax2.set_xlim(T_INT_LOW - 10, T_INT_HIGH_FIXED + 30)
    ax2.legend(fontsize=7)

    # Panel 3: T_onset distribution
    ax3 = fig.add_subplot(2, 3, 3)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp].sort_values('T2_hold_s')
        ax3.plot(rd['T2_hold_s'], rd['T_onset'],
                 marker=markers[grp], color=colors[grp], linewidth=1.5,
                 markersize=8, markerfacecolor='white',
                 markeredgewidth=1.5, label=f'Grp {grp}')
    ax3.set_xlabel('T2 hold time at 80°C (s)')
    ax3.set_ylabel('T_onset (°C)')
    ax3.set_title('Detected T_onset per ramp')
    ax3.set_xscale('log')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')

    # Panel 4: ΔH bar chart comparison
    ax4 = fig.add_subplot(2, 3, 4)
    n = len(results_df)
    x_pos = np.arange(n)
    bar_colors_dyn = [colors[r['group']] for _, r in results_df.iterrows()]
    bar_colors_fixed = ['#92C5DE' if r['group'] == 'A' else '#F4A582'
                        for _, r in results_df.iterrows()]
    ax4.bar(x_pos - 0.15, results_df['delta_H_dyn_kJmol'], 0.3,
            color=bar_colors_dyn, edgecolor='black', linewidth=0.3, alpha=0.85,
            label='Scheme ① dyn')
    ax4.bar(x_pos + 0.15, results_df['delta_H_fixed_kJmol'], 0.3,
            color=bar_colors_fixed, edgecolor='black', linewidth=0.3, alpha=0.85,
            label='Scheme ② fixed')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"G{r['group']}\nT2={r['T2_hold_s']:.1f}s"
                         for _, r in results_df.iterrows()],
                        fontsize=5.5, rotation=45)
    ax4.set_ylabel('ΔH (kJ/mol)')
    ax4.set_title('ΔH distribution — both schemes')
    ax4.legend(fontsize=7)
    mid = len(results_df[results_df['group'] == 'A']) - 0.5
    ax4.axvline(mid, color='gray', linestyle='--', alpha=0.7)

    # Panel 5: Raw DSC curves on original grid (selected)
    ax5 = fig.add_subplot(2, 3, 5)
    highlight = [0, 2, 4, 6, 8]
    for idx in highlight:
        if idx >= len(conditions):
            continue
        cond = conditions[idx]
        s, e = cond['start_idx'], cond['end_idx']
        ax5.plot(T_exp[s:e+1], DSC_exp[s:e+1],
                 alpha=0.7, linewidth=0.7,
                 color=colors[cond['group']],
                 label=f"G{cond['group']}: T1={cond['T1_hold_s']:.0f}s, T2={cond['T2_hold_s']:.1f}s")
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('DSC (µW)')
    ax5.set_title('Raw DSC heating curves (selected)')
    ax5.legend(fontsize=5.5, loc='lower right')
    ax5.set_xlim(T_INT_LOW - 10, T_INT_HIGH_FIXED + 30)

    # Panel 6: Results table (top 12 rows for readability)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}", r['group'],
            f"{r['T1_hold_s']:.1f}", f"{r['T2_hold_s']:.1f}",
            f"{r['T_onset']:.1f}",
            f"{r['delta_H_dyn_kJmol']:.2f}",
            f"{r['delta_H_fixed_kJmol']:.2f}",
        ])
    col_labels = ['Ramp', 'Grp', 'T1(s)', 'T2(s)', 'T_onset',
                  'ΔH_dyn', 'ΔH_fixed']
    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.05, 0.05, 0.10, 0.10, 0.08, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.15)
    for row_idx in range(len(table_data)):
        for col_idx in range(7):
            cell = table[row_idx + 1, col_idx]
            cell.set_facecolor('#E3EDF8' if row_idx < 10 else '#FDE0DD')
    ax6.set_title('Results Summary', fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    fig.suptitle('Two-Step Annealing: Empty-Crucible Subtracted Direct Integration',
                 fontsize=12, fontweight='bold', y=1.01)
    png_path = os.path.join(RESULTS_DIR, 'twosteps_enthalpy_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {png_path}")

    csv_path = os.path.join(RESULTS_DIR, 'twosteps_enthalpy_results.csv')
    save_cols = ['ramp', 'group', 'T1_hold_s', 'T2_hold_s',
                 'T1_hold_min', 'T2_hold_min', 'T_onset',
                 'delta_H_dyn_kJmol', 'delta_H_fixed_kJmol']
    results_df[save_cols].to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Two-step annealing")
    print("=" * 70)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp]
        t1_val = rd['T1_hold_s'].iloc[0]
        print(f"\n  Group {grp} (T1@90°C = {t1_val:.0f}s):")
        print(f"    ΔH_dyn:   {rd['delta_H_dyn_kJmol'].min():.2f} – "
              f"{rd['delta_H_dyn_kJmol'].max():.2f} kJ/mol")
        print(f"    ΔH_fixed: {rd['delta_H_fixed_kJmol'].min():.2f} – "
              f"{rd['delta_H_fixed_kJmol'].max():.2f} kJ/mol")
        print(f"    T_onset:  {rd['T_onset'].min():.0f} – "
              f"{rd['T_onset'].max():.0f} °C, n={len(rd)}")

    return results_df
```

- [ ] **Step 4: Run the script**

```bash
cd /root/twoannealing/src && python process_twosteps.py
```
Expected: processes two-step data, saves plot and CSV, prints summary.

- [ ] **Step 5: Verify CSV columns**

```bash
head -3 /root/twoannealing/results/dsc/twosteps_enthalpy_results.csv
```
Expected: columns include `T_onset,delta_H_dyn_kJmol,delta_H_fixed_kJmol`.

- [ ] **Step 6: Commit**

```bash
git add src/process_twosteps.py results/dsc/twosteps_enthalpy_results.csv results/dsc/twosteps_enthalpy_results.png
git commit -m "refactor: two-step processing to empty-crucible subtraction with dual-scheme integration"
```

---

### Task 4: Refactor `process_kovacs.py` to use empty-crucible subtraction

**Files:**
- Modify: `src/process_kovacs.py`

Same pattern as Task 3 but for Kovacs up-jump protocol. The ramp mapping (`map_kovacs`) stays the same.

- [ ] **Step 1: Update imports**

Replace lines 19-22:

```python
from process_shared import (
    load_dsc_experiment, detect_heating_ramps, compute_enthalpy,
    load_empty_crucible, subtract_empty_crucible, detect_t_onset,
    HEATING_T_START, HEATING_T_END,
)
```

- [ ] **Step 2: Update constants**

Replace lines 28-30:

```python
T_INT_LOW = 30
T_INT_HIGH_FIXED = 100
T_GRID_STEP = 0.2

EMPTY_CRUCIBLE_CSV = os.path.join(DATA_DIR, 'csv', 'baseline-01.csv')
```

- [ ] **Step 3: Rewrite `main` function**

Replace lines 75-275 (the entire `main` function). The `map_kovacs` function stays unchanged.

```python
def main():
    print("=" * 70)
    print("  Kovacs annealing DSC — Empty-crucible subtraction + direct integration")
    print("=" * 70)

    # Load empty crucible
    T_empty, DSC_empty = load_empty_crucible(EMPTY_CRUCIBLE_CSV)
    print(f"  Loaded empty crucible: {len(T_empty)} pts, "
          f"{T_empty.min():.1f}–{T_empty.max():.1f}°C")

    exp_data, program = load_dsc_experiment(
        os.path.join(DATA_DIR, 'pskovacs.xlsx'), sheet='PS-kovacs-01')
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, exp_data['Time'].values)
    print(f"  Found {len(ramps)} heating ramps")

    conditions = map_kovacs(program, ramps)
    print(f"  Mapped {len(conditions)} annealed ramps:")
    for c in conditions:
        print(f"    Ramp {c['ramp_idx']:2d} [Grp {c['group']}]: "
              f"T1@80°C={c['T1_hold_s']:8.1f}s, T2@90°C={c['T2_hold_s']:8.1f}s")

    T_grid = np.arange(T_INT_LOW, T_INT_HIGH_FIXED + 30 + T_GRID_STEP, T_GRID_STEP)

    results = []
    colors = {'A': '#2166AC', 'B': '#B2182B'}
    markers = {'A': 'o', 'B': 's'}

    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_ramp = T_exp[s:e+1]
        DSC_ramp = DSC_exp[s:e+1]

        DSC_corr = subtract_empty_crucible(T_ramp, DSC_ramp, T_empty, DSC_empty)

        # Scheme ①: dynamic T_onset
        T_onset = detect_t_onset(T_ramp, DSC_corr)
        mask1 = (T_ramp >= T_INT_LOW) & (T_ramp <= T_onset)
        area1 = trapezoid(DSC_corr[mask1], T_ramp[mask1])
        dH1 = compute_enthalpy(area1)

        # Scheme ②: fixed 30–100°C
        mask2 = (T_ramp >= T_INT_LOW) & (T_ramp <= T_INT_HIGH_FIXED)
        area2 = trapezoid(DSC_corr[mask2], T_ramp[mask2])
        dH2 = compute_enthalpy(area2)

        results.append({
            'ramp': cond['ramp_idx'],
            'group': cond['group'],
            'T1_hold_s': cond['T1_hold_s'],
            'T2_hold_s': cond['T2_hold_s'],
            'T1_hold_min': cond['T1_hold_min'],
            'T2_hold_min': cond['T2_hold_min'],
            'T_onset': T_onset,
            'delta_H_dyn_kJmol': dH1,
            'delta_H_fixed_kJmol': dH2,
        })
        print(f"    Ramp {cond['ramp_idx']:2d}: T1={cond['T1_hold_s']:8.1f}s, "
              f"T2={cond['T2_hold_s']:8.1f}s, T_onset={T_onset:.1f}°C → "
              f"ΔH_dyn={dH1:+.2f}, ΔH_fixed={dH2:+.2f} kJ/mol")

    results_df = pd.DataFrame(results)

    # Generate plots
    fig = plt.figure(figsize=(20, 14))

    # Panel 1: ΔH vs T2 hold time — Kovacs hump (both schemes)
    ax1 = fig.add_subplot(2, 3, 1)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp].sort_values('T2_hold_s')
        ax1.plot(rd['T2_hold_s'], rd['delta_H_dyn_kJmol'],
                 marker=markers[grp], color=colors[grp], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, linestyle='-',
                 label=f'Grp {grp} dyn')
        ax1.plot(rd['T2_hold_s'], rd['delta_H_fixed_kJmol'],
                 marker=markers[grp], color=colors[grp], linewidth=1.2,
                 markersize=6, markerfacecolor='white',
                 markeredgewidth=1.0, linestyle='--',
                 label=f'Grp {grp} fixed')
    ax1.set_xlabel('T2 hold time at 90°C (s)')
    ax1.set_ylabel('ΔH (kJ/mol)')
    ax1.set_title('Kovacs hump: ΔH vs up-jump time (both schemes)')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: Corrected DSC curves
    ax2 = fig.add_subplot(2, 3, 2)
    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_ramp = T_exp[s:e+1]
        DSC_ramp = DSC_exp[s:e+1]
        DSC_corr = subtract_empty_crucible(T_ramp, DSC_ramp, T_empty, DSC_empty)
        ax2.plot(T_ramp, DSC_corr, alpha=0.35, linewidth=0.5,
                 color=colors[cond['group']])
    ax2.axvline(T_INT_HIGH_FIXED, color='green', linestyle='--', alpha=0.5,
                label=f'Fixed: {T_INT_HIGH_FIXED}°C')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('DSC_corrected (µW)')
    ax2.set_title('Corrected DSC — all ramps by group')
    ax2.set_xlim(T_INT_LOW - 10, T_INT_HIGH_FIXED + 30)
    ax2.legend(fontsize=7)

    # Panel 3: T_onset distribution
    ax3 = fig.add_subplot(2, 3, 3)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp].sort_values('T2_hold_s')
        ax3.plot(rd['T2_hold_s'], rd['T_onset'],
                 marker=markers[grp], color=colors[grp], linewidth=1.5,
                 markersize=8, markerfacecolor='white',
                 markeredgewidth=1.5, label=f'Grp {grp}')
    ax3.set_xlabel('T2 hold time at 90°C (s)')
    ax3.set_ylabel('T_onset (°C)')
    ax3.set_title('Detected T_onset per ramp')
    ax3.set_xscale('log')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')

    # Panel 4: ΔH bar chart comparison
    ax4 = fig.add_subplot(2, 3, 4)
    n = len(results_df)
    x_pos = np.arange(n)
    bar_colors_dyn = [colors[r['group']] for _, r in results_df.iterrows()]
    bar_colors_fixed = ['#92C5DE' if r['group'] == 'A' else '#F4A582'
                        for _, r in results_df.iterrows()]
    ax4.bar(x_pos - 0.15, results_df['delta_H_dyn_kJmol'], 0.3,
            color=bar_colors_dyn, edgecolor='black', linewidth=0.3, alpha=0.85,
            label='Scheme ① dyn')
    ax4.bar(x_pos + 0.15, results_df['delta_H_fixed_kJmol'], 0.3,
            color=bar_colors_fixed, edgecolor='black', linewidth=0.3, alpha=0.85,
            label='Scheme ② fixed')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"G{r['group']}\nT2={r['T2_hold_s']:.1f}s"
                         for _, r in results_df.iterrows()],
                        fontsize=5.5, rotation=45)
    ax4.set_ylabel('ΔH (kJ/mol)')
    ax4.set_title('Kovacs ΔH distribution — both schemes')
    ax4.legend(fontsize=7)
    mid = len(results_df[results_df['group'] == 'A']) - 0.5
    ax4.axvline(mid, color='gray', linestyle='--', alpha=0.7)

    # Panel 5: Raw DSC curves — Tg region zoom
    ax5 = fig.add_subplot(2, 3, 5)
    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        ax5.plot(T_exp[s:e+1], DSC_exp[s:e+1],
                 alpha=0.45, linewidth=0.5,
                 color=colors[cond['group']])
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('DSC (µW)')
    ax5.set_title('Raw DSC curves — Tg region')
    ax5.set_xlim(T_INT_LOW, T_INT_HIGH_FIXED + 30)

    # Panel 6: Results table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}", r['group'],
            f"{r['T1_hold_s']:.1f}", f"{r['T2_hold_s']:.1f}",
            f"{r['T_onset']:.1f}",
            f"{r['delta_H_dyn_kJmol']:.2f}",
            f"{r['delta_H_fixed_kJmol']:.2f}",
        ])
    col_labels = ['Ramp', 'Grp', 'T1(s)', 'T2(s)', 'T_onset',
                  'ΔH_dyn', 'ΔH_fixed']
    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.05, 0.05, 0.10, 0.10, 0.08, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.15)
    for row_idx in range(len(table_data)):
        for col_idx in range(7):
            cell = table[row_idx + 1, col_idx]
            cell.set_facecolor('#E3EDF8' if row_idx < 10 else '#FDE0DD')
    ax6.set_title('Results Summary', fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    fig.suptitle('Kovacs Annealing: Empty-Crucible Subtracted Direct Integration',
                 fontsize=12, fontweight='bold', y=1.01)
    png_path = os.path.join(RESULTS_DIR, 'kovacs_enthalpy_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {png_path}")

    csv_path = os.path.join(RESULTS_DIR, 'kovacs_enthalpy_results.csv')
    save_cols = ['ramp', 'group', 'T1_hold_s', 'T2_hold_s',
                 'T1_hold_min', 'T2_hold_min', 'T_onset',
                 'delta_H_dyn_kJmol', 'delta_H_fixed_kJmol']
    results_df[save_cols].to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Summary with Kovacs hump check
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Kovacs annealing")
    print("=" * 70)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp]
        t1_val = rd['T1_hold_s'].iloc[0]
        print(f"\n  Group {grp} (T1@80°C = {t1_val:.0f}s):")
        print(f"    ΔH_dyn:   {rd['delta_H_dyn_kJmol'].min():.2f} – "
              f"{rd['delta_H_dyn_kJmol'].max():.2f} kJ/mol")
        print(f"    ΔH_fixed: {rd['delta_H_fixed_kJmol'].min():.2f} – "
              f"{rd['delta_H_fixed_kJmol'].max():.2f} kJ/mol")
        # Check for Kovacs hump pattern
        dH_vals_dyn = rd['delta_H_dyn_kJmol'].values
        dH_vals_fixed = rd['delta_H_fixed_kJmol'].values
        for scheme_name, dH_vals in [('dyn', dH_vals_dyn), ('fixed', dH_vals_fixed)]:
            if len(dH_vals) >= 3:
                mid_min = np.min(dH_vals[1:-1])
                edge_min = min(dH_vals[0], dH_vals[-1])
                if mid_min < edge_min * 0.8:
                    print(f"    ✓ {scheme_name}: Kovacs hump detected (mid-range dip!)")
                else:
                    print(f"    {scheme_name}: No clear hump pattern")

    return results_df
```

- [ ] **Step 4: Run the script**

```bash
cd /root/twoannealing/src && python process_kovacs.py
```
Expected: processes Kovacs data, saves plot and CSV, prints summary with hump check.

- [ ] **Step 5: Verify CSV columns**

```bash
head -3 /root/twoannealing/results/dsc/kovacs_enthalpy_results.csv
```
Expected: columns include `T_onset,delta_H_dyn_kJmol,delta_H_fixed_kJmol`.

- [ ] **Step 6: Commit**

```bash
git add src/process_kovacs.py results/dsc/kovacs_enthalpy_results.csv results/dsc/kovacs_enthalpy_results.png
git commit -m "refactor: Kovacs processing to empty-crucible subtraction with dual-scheme integration"
```

---

## Validation Checkpoints

After all 4 tasks complete, run all three scripts end-to-end and verify:

```bash
cd /root/twoannealing/src && python process_onestep.py && python process_twosteps.py && python process_kovacs.py
```

Then check:
1. All three scripts run without errors
2. All 6 output files (3 PNG + 3 CSV) exist in `results/dsc/`
3. CSVs contain both `delta_H_dyn_kJmol` and `delta_H_fixed_kJmol` columns
4. Kovacs script reports hump detection
5. One-step ΔH values show monotonic increase with log(hold time)
