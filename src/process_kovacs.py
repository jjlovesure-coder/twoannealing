"""
Kovacs-type (up-jump) annealing DSC data processing — Direct difference method.

Kovacs: 200→80°C(hold T1) → 80→90°C(hold T2, UP-JUMP) → 90→30°C → 30→200°C(measure)
Per-group reference: shortest T2-hold ramp in each group.
The classic "Kovacs hump" — ΔH decreases then increases with T2 hold time —
should emerge naturally from the data.
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

T_INT_LOW = 30
T_INT_HIGH = 130
T_GRID_STEP = 0.2


def map_kovacs(program, ramps):
    """Map ramps to Kovacs annealing conditions.

    Pattern: 4n+1=200→80(hold T1), 4n+2=80→90(hold T2, up-jump),
    4n+3=90→30, 4n+4=30→200(heating).
    """
    heating_steps = [p for p in program
                     if p['T_start'] == HEATING_T_START and
                     p['T_end'] == HEATING_T_END]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            t1_hold_min, t2_hold_min = None, None
            for p in program:
                if p['step'] == step_num - 2:
                    t2_hold_min = p['time_min']  # 80→90 hold (up-jump)
                elif p['step'] == step_num - 3:
                    t1_hold_min = p['time_min']  # 200→80 hold
        else:
            t1_hold_min = t2_hold_min = None

        conditions.append({
            'ramp_idx': idx + 1,
            'T1_hold_s': t1_hold_min * 60.0 if t1_hold_min else None,
            'T2_hold_s': t2_hold_min * 60.0 if t2_hold_min else None,
            'T1_hold_min': t1_hold_min,
            'T2_hold_min': t2_hold_min,
            'start_idx': s, 'end_idx': e,
        })

    conditions = [c for c in conditions
                  if c['T1_hold_s'] is not None and c['T2_hold_s'] is not None]
    for i, c in enumerate(conditions):
        c['ramp_idx'] = i + 1
        c['group'] = 'A' if c['T1_hold_s'] < 60 else 'B'

    return conditions


def main():
    print("=" * 70)
    print("  Kovacs annealing DSC — Direct difference method")
    print("=" * 70)

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

    T_grid = np.arange(T_INT_LOW, T_INT_HIGH + T_GRID_STEP, T_GRID_STEP)

    # Interpolate all ramps
    dsc_interp = {}
    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        f = interp1d(T_exp[s:e+1], DSC_exp[s:e+1], kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        dsc_interp[cond['ramp_idx']] = f(T_grid)

    # Per-group processing
    results = []
    colors = {'A': '#2166AC', 'B': '#B2182B'}
    markers = {'A': 'o', 'B': 's'}

    for grp_label in ['A', 'B']:
        grp_conds = [c for c in conditions if c['group'] == grp_label]
        if len(grp_conds) < 2:
            continue

        ref_cond = min(grp_conds, key=lambda c: c['T2_hold_s'])
        ref_dsc = dsc_interp[ref_cond['ramp_idx']]

        print(f"\n  [Group {grp_label}] Reference: ramp {ref_cond['ramp_idx']}, "
              f"T2 hold = {ref_cond['T2_hold_s']:.1f}s")

        for cond in grp_conds:
            excess = dsc_interp[cond['ramp_idx']] - ref_dsc
            excess_area = trapezoid(excess, T_grid)
            delta_H = compute_enthalpy(excess_area)
            results.append({
                'ramp': cond['ramp_idx'],
                'group': cond['group'],
                'T1_hold_s': cond['T1_hold_s'],
                'T2_hold_s': cond['T2_hold_s'],
                'T1_hold_min': cond['T1_hold_min'],
                'T2_hold_min': cond['T2_hold_min'],
                'delta_H_kJmol': delta_H,
                'is_ref': (cond['ramp_idx'] == ref_cond['ramp_idx']),
            })
            marker = ' *REF*' if cond['ramp_idx'] == ref_cond['ramp_idx'] else ''
            print(f"    Ramp {cond['ramp_idx']:2d}: "
                  f"T1={cond['T1_hold_s']:8.1f}s, T2={cond['T2_hold_s']:8.1f}s → "
                  f"ΔH={delta_H:+.2f} kJ/mol{marker}")

    results_df = pd.DataFrame(results)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))

    # Panel 1: ΔH vs T2 hold time — THE KOVACS HUMP
    ax1 = fig.add_subplot(2, 3, 1)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp].sort_values('T2_hold_s')
        ax1.plot(rd['T2_hold_s'], rd['delta_H_kJmol'],
                 marker=markers[grp], color=colors[grp], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5,
                 label=f'Group {grp} (T1@80°C={50 if grp=="A" else 500}s)')
    ax1.set_xlabel('T2 hold time at 90°C (s)')
    ax1.set_ylabel('ΔH (kJ/mol)')
    ax1.set_title('Kovacs hump: Enthalpy vs up-jump time')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: All DSC curves
    ax2 = fig.add_subplot(2, 3, 2)
    for cond in conditions:
        ax2.plot(T_grid, dsc_interp[cond['ramp_idx']],
                 alpha=0.35, linewidth=0.5,
                 color=colors[cond['group']])
    ax2.axvspan(T_INT_LOW, T_INT_HIGH, alpha=0.06, color='green')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('DSC (µW)')
    ax2.set_title('All DSC curves by group — Kovacs')
    ax2.set_xlim(T_INT_LOW - 10, T_INT_HIGH + 10)

    # Panel 3: Excess DSC
    ax3 = fig.add_subplot(2, 3, 3)
    ref_idxs = {}
    for grp in ['A', 'B']:
        gcs = [c for c in conditions if c['group'] == grp]
        ref_idxs[grp] = min(gcs, key=lambda c: c['T2_hold_s'])['ramp_idx']
    for cond in conditions:
        ridx = cond['ramp_idx']
        ref_idx = ref_idxs[cond['group']]
        if ridx == ref_idx:
            continue
        excess = dsc_interp[ridx] - dsc_interp[ref_idx]
        ax3.plot(T_grid, excess, alpha=0.45, linewidth=0.6,
                 color=colors[cond['group']])
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.4)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Excess DSC (µW)')
    ax3.set_title('Excess DSC — Kovacs overshoot')
    ax3.set_xlim(T_INT_LOW, T_INT_HIGH)

    # Panel 4: ΔH bar chart
    ax4 = fig.add_subplot(2, 3, 4)
    n = len(results_df)
    x_pos = np.arange(n)
    bar_colors = [colors[r['group']] for _, r in results_df.iterrows()]
    ax4.bar(x_pos, results_df['delta_H_kJmol'], color=bar_colors,
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"G{r['group']}\nT2={r['T2_hold_s']:.1f}s"
                         for _, r in results_df.iterrows()],
                        fontsize=5.5, rotation=45)
    ax4.set_ylabel('ΔH (kJ/mol)')
    ax4.set_title('Kovacs ΔH distribution')
    mid = len(results_df[results_df['group'] == 'A']) - 0.5
    ax4.axvline(mid, color='gray', linestyle='--', alpha=0.7)

    # Panel 5: DSC curves on original grid (Tg region zoom)
    ax5 = fig.add_subplot(2, 3, 5)
    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        ax5.plot(T_exp[s:e+1], DSC_exp[s:e+1],
                 alpha=0.45, linewidth=0.5,
                 color=colors[cond['group']])
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('DSC (µW)')
    ax5.set_title('Raw DSC curves — Tg region')
    ax5.set_xlim(T_INT_LOW, T_INT_HIGH)

    # Panel 6: Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}", r['group'],
            f"{r['T1_hold_s']:.1f}", f"{r['T2_hold_s']:.1f}",
            f"{r['delta_H_kJmol']:.2f}",
        ])
    col_labels = ['Ramp', 'Grp', 'T1(s)', 'T2(s)', 'ΔH(kJ/mol)']
    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.08, 0.06, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for row_idx in range(len(table_data)):
        for col_idx in range(5):
            cell = table[row_idx + 1, col_idx]
            cell.set_facecolor('#E3EDF8' if row_idx < 10 else '#FDE0DD')
    ax6.set_title('Results Summary', fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    png_path = os.path.join(RESULTS_DIR, 'kovacs_enthalpy_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {png_path}")

    csv_path = os.path.join(RESULTS_DIR, 'kovacs_enthalpy_results.csv')
    save_cols = ['ramp', 'group', 'T1_hold_s', 'T2_hold_s',
                 'T1_hold_min', 'T2_hold_min', 'delta_H_kJmol']
    results_df[save_cols].to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Kovacs annealing")
    print("=" * 70)
    for grp in ['A', 'B']:
        rd = results_df[results_df['group'] == grp]
        t1_val = rd['T1_hold_s'].iloc[0]
        print(f"\n  Group {grp} (T1@80°C = {t1_val:.0f}s):")
        print(f"    ΔH: {rd['delta_H_kJmol'].min():.2f} – "
              f"{rd['delta_H_kJmol'].max():.2f} kJ/mol, n={len(rd)}")
        dH_vals = rd['delta_H_kJmol'].values
        if len(dH_vals) >= 3:
            # Check for Kovacs hump pattern: decrease then increase
            mid_min = np.min(dH_vals[1:-1])
            edge_min = min(dH_vals[0], dH_vals[-1])
            if mid_min < edge_min * 0.8:
                print(f"    ✓ Kovacs hump detected (mid-range dip!)")
            else:
                print(f"    No clear hump pattern")

    return results_df


if __name__ == '__main__':
    results = main()
