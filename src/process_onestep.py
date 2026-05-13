"""
One-step annealing DSC data processing — Direct difference method.

Uses the shortest-hold DSC heating curve as reference per cooling-rate group.
Excess DSC = DSC(T, t_hold) - DSC(T, t_ref), integrated over the Tg recovery
region to give the physical aging enthalpy.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from process_shared import (
    load_dsc_experiment, detect_heating_ramps, compute_enthalpy,
    HEATING_T_START, HEATING_T_END,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dsc')

# Integration range for Tg recovery peak
T_INT_LOW = 70    # °C — well before Tg onset
T_INT_HIGH = 130  # °C — well after recovery peak
T_GRID_STEP = 0.2  # °C — fine grid for interpolation


def map_program_to_ramps(program, ramps):
    """Map each detected heating ramp to its preceding annealing hold step.

    Pattern: step n = heating (30→200), step n-1 = cool (T_anneal→30),
    step n-2 = anneal (200→T_anneal with hold time).

    Returns list of dicts with ramp_idx, hold_s, hold_min, run_label.
    """
    heating_steps = [p for p in program
                     if p['T_start'] == HEATING_T_START and
                     p['T_end'] == HEATING_T_END]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            anneal_step = None
            for p in program:
                if p['step'] == step_num - 2:
                    anneal_step = p
                    break
            t_hold_min = anneal_step['time_min'] if anneal_step else None
            t_hold_s = t_hold_min * 60.0 if t_hold_min else None
        else:
            t_hold_min = None
            t_hold_s = None

        conditions.append({
            'ramp_idx': idx + 1,
            'hold_s': t_hold_s,
            'hold_min': t_hold_min,
            'start_idx': s,
            'end_idx': e,
        })

    # Keep only ramps with valid hold times
    conditions = [c for c in conditions if c['hold_s'] is not None]
    for i, c in enumerate(conditions):
        c['ramp_idx'] = i + 1

    n = len(conditions)
    mid = n // 2
    for i, c in enumerate(conditions):
        c['run'] = 'R1' if i < mid else 'R2'

    return conditions


def process_one_step(data_file, sheet, T_anneal, label, out_name):
    """Process one-step annealing DSC data with direct difference method.

    The shortest-hold ramp in each run serves as the reference (zero-aging, or
    the closest approximation available). All other ramps in the same run are
    compared against it.
    """
    print(f"\n{'='*70}")
    print(f"  One-step annealing ({label}) — T_anneal = {T_anneal}°C")
    print(f"  Method: direct difference vs shortest-hold reference")
    print(f"{'='*70}")

    # Load experiment and detect ramps
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, exp_data['Time'].values)
    print(f"  Found {len(ramps)} heating ramps")

    conditions = map_program_to_ramps(program, ramps)
    print(f"  Mapped {len(conditions)} annealed ramps per run")

    # Build common temperature grid for inter-ramp comparison
    T_grid = np.arange(T_INT_LOW, T_INT_HIGH + T_GRID_STEP, T_GRID_STEP)

    # Interpolate all ramp DSC to common grid
    dsc_interp = {}
    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]
        DSC_seg = DSC_exp[s:e+1]
        f = interp1d(T_seg, DSC_seg, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        dsc_interp[cond['ramp_idx']] = f(T_grid)

    # Per-run reference: shortest hold ramp
    results = []
    for run_label in ['R1', 'R2']:
        run_conds = [c for c in conditions if c['run'] == run_label]
        if len(run_conds) < 2:
            continue

        # Reference = shortest hold
        ref_cond = min(run_conds, key=lambda c: c['hold_s'])
        ref_dsc = dsc_interp[ref_cond['ramp_idx']]

        print(f"\n  [{run_label}] Reference: ramp {ref_cond['ramp_idx']}, "
              f"hold = {ref_cond['hold_s']:.1f} s")

        for cond in run_conds:
            excess = dsc_interp[cond['ramp_idx']] - ref_dsc
            # Integrate positive excess (recovery peak is endothermic → positive in DSC)
            excess_area = trapezoid(excess, T_grid)
            delta_H = compute_enthalpy(excess_area)
            results.append({
                'ramp': cond['ramp_idx'],
                'run': cond['run'],
                'hold_s': cond['hold_s'],
                'hold_min': cond['hold_min'],
                'excess_area_uWC': excess_area,
                'delta_H_kJmol': delta_H,
                'is_ref': (cond['ramp_idx'] == ref_cond['ramp_idx']),
            })
            marker = ' *REF*' if cond['ramp_idx'] == ref_cond['ramp_idx'] else ''
            print(f"    Ramp {cond['ramp_idx']:2d}: hold={cond['hold_s']:8.1f}s → "
                  f"excess={excess_area:.1f} uW·°C, ΔH={delta_H:+.2f} kJ/mol{marker}")

    results_df = pd.DataFrame(results)

    # ── Generate plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    run_colors = {'R1': '#2166AC', 'R2': '#B2182B'}
    run_markers = {'R1': 'o', 'R2': 's'}

    # Panel 1: ΔH vs hold time (log scale, inverted y for aging convention)
    ax1 = axes[0, 0]
    for rl in ['R1', 'R2']:
        rd = results_df[results_df['run'] == rl].sort_values('hold_s')
        ax1.plot(rd['hold_s'], rd['delta_H_kJmol'],
                 marker=run_markers[rl], color=run_colors[rl],
                 linewidth=1.8, markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=rl)
    ax1.set_xlabel(f'Hold time at {T_anneal}°C (s)')
    ax1.set_ylabel('ΔH (kJ/mol)')
    ax1.set_title(f'{label}: Released enthalpy vs annealing time')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: DSC curves on common grid with ref highlighted
    ax2 = axes[0, 1]
    sample_idx = [0, 2, 4, 6, 8]
    for idx in sample_idx:
        if idx >= len(conditions):
            continue
        cond = conditions[idx]
        ridx = cond['ramp_idx']
        excess_vals = dsc_interp[ridx]
        ax2.plot(T_grid, excess_vals, alpha=0.6, linewidth=0.7,
                 color=run_colors[cond['run']])
    # Highlight references
    for rl in ['R1', 'R2']:
        rcs = [c for c in conditions if c['run'] == rl]
        if rcs:
            ref = min(rcs, key=lambda c: c['hold_s'])
            ax2.plot(T_grid, dsc_interp[ref['ramp_idx']],
                     '-', linewidth=2.0, color=run_colors[rl],
                     label=f'{rl} ref ({ref["hold_s"]:.1f}s hold)')
    ax2.axvspan(T_INT_LOW, T_INT_HIGH, alpha=0.06, color='green')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('DSC (µW)')
    ax2.set_title('DSC curves on common grid')
    ax2.legend(fontsize=7)
    ax2.set_xlim(T_INT_LOW - 10, T_INT_HIGH + 10)

    # Panel 3: Excess DSC (relative to ref)
    ax3 = axes[1, 0]
    ref_indices = {}
    for rl in ['R1', 'R2']:
        rcs = [c for c in conditions if c['run'] == rl]
        ref_indices[rl] = min(rcs, key=lambda c: c['hold_s'])['ramp_idx']
    for cond in conditions:
        ridx = cond['ramp_idx']
        ref_idx = ref_indices[cond['run']]
        if ridx == ref_idx:
            continue
        excess = dsc_interp[ridx] - dsc_interp[ref_idx]
        ax3.plot(T_grid, excess, alpha=0.45, linewidth=0.6,
                 color=run_colors[cond['run']])
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.4)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Excess DSC (µW)')
    ax3.set_title('Excess DSC relative to shortest-hold reference')
    ax3.set_xlim(T_INT_LOW, T_INT_HIGH)

    # Panel 4: ΔH bar chart
    ax4 = axes[1, 1]
    n = len(results_df)
    x_pos = np.arange(n)
    bar_colors = [run_colors[r['run']] for _, r in results_df.iterrows()]
    ax4.bar(x_pos, results_df['delta_H_kJmol'], color=bar_colors,
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{r['run']}\n{r['hold_s']:.1f}s"
                         for _, r in results_df.iterrows()],
                        fontsize=5.5, rotation=45)
    ax4.set_ylabel('ΔH (kJ/mol)')
    ax4.set_title(f'{label}: ΔH distribution')
    mid = len(results_df[results_df['run'] == 'R1'])
    ax4.axvline(mid - 0.5, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f'{out_name}_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {png_path}")

    csv_path = os.path.join(RESULTS_DIR, f'{out_name}_results.csv')
    save_cols = ['ramp', 'run', 'hold_s', 'hold_min', 'delta_H_kJmol']
    results_df[save_cols].to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Summary
    print(f"\n  Results ({label}):")
    for rl in ['R1', 'R2']:
        rd = results_df[results_df['run'] == rl]
        if len(rd) == 0:
            continue
        print(f"  {rl}: ΔH = {rd['delta_H_kJmol'].min():.2f} – "
              f"{rd['delta_H_kJmol'].max():.2f} kJ/mol, "
              f"n={len(rd)} ramps")

    return results_df


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    r1 = process_one_step(
        os.path.join(DATA_DIR, 'PS-onestep-01.xlsx'),
        sheet='PS-onestep-01',
        T_anneal=50,
        label='One-step @50°C',
        out_name='onestep_50C',
    )
    r2 = process_one_step(
        os.path.join(DATA_DIR, 'PS-onestep-02.xlsx'),
        sheet='PS-onestep-02',
        T_anneal=70,
        label='One-step @70°C',
        out_name='onestep_70C',
    )
