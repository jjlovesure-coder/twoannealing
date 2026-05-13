"""
One-step annealing DSC data processing — Direct difference method.

Uses the shortest-hold DSC heating curve as reference per cooling-rate group.
Excess DSC = DSC(T, t_hold) - DSC(T, t_ref), integrated over the Tg recovery
region to give the physical aging enthalpy.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from process_shared import (
    load_dsc_experiment, detect_heating_ramps, compute_enthalpy,
    load_empty_crucible, subtract_empty_crucible, detect_t_onset,
    HEATING_T_START, HEATING_T_END,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dsc')

T_INT_LOW = 30       # lower bound for both schemes
T_INT_HIGH_FIXED = 100  # Scheme ② fixed upper bound
T_GRID_STEP = 0.2

EMPTY_CRUCIBLE_CSV = os.path.join(DATA_DIR, 'csv', 'baseline-01.csv')


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

    # ── Generate plots ──────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    r1 = process_one_step(
        os.path.join(DATA_DIR, 'raw', 'onestep-01.xlsx'),
        sheet='PS-onestep-01',
        T_anneal=50,
        label='One-step @50°C',
        out_name='onestep_50C',
    )
    r2 = process_one_step(
        os.path.join(DATA_DIR, 'raw', 'onestep-02.xlsx'),
        sheet='PS-onestep-02',
        T_anneal=70,
        label='One-step @70°C',
        out_name='onestep_70C',
    )
