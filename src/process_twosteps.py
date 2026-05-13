"""
Two-step annealing DSC data processing — Empty-crucible subtraction + direct integration.

Two-step: 200→90°C(hold T1)→80°C(hold T2)→30°C→200°C(measure)
Empty crucible baseline subtracted; dual-scheme integration:
  Scheme (1): 30°C → T_onset (dynamic)
  Scheme (2): 30°C → 100°C (fixed)
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

T_INT_LOW = 30
T_INT_HIGH_FIXED = 100
T_GRID_STEP = 0.2

EMPTY_CRUCIBLE_CSV = os.path.join(DATA_DIR, 'csv', 'baseline-01.csv')


def map_two_step(program, ramps):
    """Map ramps to two-step annealing conditions.

    Pattern: step 4n+1=200→90(hold T1), 4n+2=90→80(hold T2),
    4n+3=80→30, 4n+4=30→200(heating).
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
                    t2_hold_min = p['time_min']  # 90→80 hold
                elif p['step'] == step_num - 3:
                    t1_hold_min = p['time_min']  # 200→90 hold
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


if __name__ == '__main__':
    results = main()
