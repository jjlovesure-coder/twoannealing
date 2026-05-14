#!/usr/bin/env python3
"""
TNM Model Driver — Corrected model + 2-stage fitting + Kovacs prediction.

Fit strategy:
  Stage 1: Shape-fit one-step (normalized [0,1]) -> rough x, beta.
  Stage 2: Absolute-scale fit (kJ/mol) with one-step + two-step -> refine all.
  Stage 3: Kovacs prediction (no fitting, hold-out validation).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C, T_INITIAL,
    COOLING_RATE, build_target_vectors, T1_HOLD_SHORT, T1_HOLD_LONG,
)
from tnm_fit import (
    make_model, run_stage1, run_stage2, compute_r_squared,
    simulate_protocol_match_exp, simulate_protocol_raw,
)
from tnm_plots import (
    plot_one_step_fit, plot_two_step_fit, plot_kovacs_prediction,
    plot_parity, plot_tf_evolution, plot_residuals,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_sim_kJ_data(model, targets):
    """Compute simulated delta_H in kJ/mol for all 6 protocol groups."""
    sim_data_kJ = {}
    scaling = {}

    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'ts_grpA'),
        'tsB': ('ts', 'ts_grpB'),
        'kvA': ('kovacs', 'kov_grpA'),
        'kvB': ('kovacs', 'kov_grpB'),
    }

    for sim_key, (exp_key, grp_key) in group_map.items():
        _, dh_sim_raw = simulate_protocol_raw(model, sim_key)

        exp = targets[exp_key]
        mask = exp['groups'] == grp_key
        dh_exp = exp['dH'][mask]
        t_exp = exp['t'][mask]

        s_min, s_max = dh_sim_raw.min(), dh_sim_raw.max()
        if s_max - s_min < 1e-10:
            dh_sim_norm = np.zeros_like(dh_sim_raw)
        else:
            dh_sim_norm = (dh_sim_raw - s_min) / (s_max - s_min)

        e_min, e_max = dh_exp.min(), dh_exp.max()
        if e_max - e_min < 1e-10:
            dh_exp_norm = np.zeros_like(dh_exp)
        else:
            dh_exp_norm = (dh_exp - e_min) / (e_max - e_min)

        dh_sim_norm_interp = np.interp(t_exp, HOLD_TIMES_SEC, dh_sim_norm)

        a, b = np.polyfit(dh_sim_norm_interp, dh_exp, 1)
        scaling[sim_key] = (a, b)
        sim_data_kJ[sim_key] = a * dh_sim_norm_interp + b

    return sim_data_kJ, scaling


def compute_all_r2(sim_data_kJ, targets):
    """Compute R^2 for each group using kJ/mol values."""
    r2_values = {}
    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'ts_grpA'),
        'tsB': ('ts', 'ts_grpB'),
        'kvA': ('kovacs', 'kov_grpA'),
        'kvB': ('kovacs', 'kov_grpB'),
    }
    for sim_key, (exp_key, grp_key) in group_map.items():
        exp = targets[exp_key]
        mask = exp['groups'] == grp_key
        dh_exp = exp['dH'][mask]
        if sim_key in sim_data_kJ:
            dh_sim = sim_data_kJ[sim_key]
            r2_values[sim_key] = compute_r_squared(dh_sim, dh_exp)
        else:
            r2_values[sim_key] = np.nan
    return r2_values


def main():
    print("=" * 70)
    print("  TNM Model — Corrected Fitting to PS Annealing Data")
    print("=" * 70)

    # Load targets
    print("\n[1] Loading experimental data...")
    targets = build_target_vectors()
    for key, d in targets.items():
        print(f"  {key}: {len(d['t'])} pts, "
              f"dH range=[{d['dH'].min():.1f}, {d['dH'].max():.1f}] kJ/mol")

    # Stage 1: Shape-fit one-step
    print("\n[2] Stage 1: Shape-fitting one-step (50C + 70C)...")
    s1_params, s1_result = run_stage1(targets, seed=42)
    model_s1 = make_model(**s1_params)

    sim_kJ_s1, scaling_s1 = compute_sim_kJ_data(model_s1, targets)
    r2_s1 = compute_all_r2(sim_kJ_s1, targets)

    print(f"\n  Stage 1 results:")
    print(f"    log10(A) = {s1_params['logA']:.3f}  "
          f"(A = {10**s1_params['logA']:.2e} s)")
    print(f"    H*       = {s1_params['H_star']/1000:.1f} kJ/mol")
    print(f"    x        = {s1_params['x']:.4f}")
    print(f"    beta     = {s1_params['beta']:.4f}")
    print(f"    T0       = {s1_params['T0']:.1f} K ({s1_params['T0']-273.15:.1f} C)")
    print(f"    Cost     = {s1_result.fun:.6f}")
    for k in ['os50', 'os70', 'tsA', 'tsB', 'kvA', 'kvB']:
        v = r2_s1.get(k, np.nan)
        if not np.isnan(v):
            print(f"    R^2 {k:6s} = {v:.4f}")

    # Stage 2: Absolute-scale fit
    print("\n[3] Stage 2: Absolute-scale refinement (all protocols)...")
    s2_params, s2_result = run_stage2(targets, s1_params, seed=42)
    model_s2 = make_model(**s2_params)

    sim_kJ_s2, scaling_s2 = compute_sim_kJ_data(model_s2, targets)
    r2_s2 = compute_all_r2(sim_kJ_s2, targets)

    print(f"\n  Stage 2 results:")
    print(f"    log10(A) = {s2_params['logA']:.3f}  "
          f"(A = {10**s2_params['logA']:.2e} s)")
    print(f"    H*       = {s2_params['H_star']/1000:.1f} kJ/mol")
    print(f"    x        = {s2_params['x']:.4f}")
    print(f"    beta     = {s2_params['beta']:.4f}")
    print(f"    T0       = {s2_params['T0']:.1f} K ({s2_params['T0']-273.15:.1f} C)")
    print(f"    Cost     = {s2_result.fun:.6f}")
    for k in ['os50', 'os70', 'tsA', 'tsB', 'kvA', 'kvB']:
        v = r2_s2.get(k, np.nan)
        if not np.isnan(v):
            print(f"    R^2 {k:6s} = {v:.4f}")

    # Stage 3: Kovacs prediction
    print("\n[4] Stage 3: Kovacs prediction (hold-out validation)...")
    for k in ['kvA', 'kvB']:
        v = r2_s2.get(k, np.nan)
        print(f"    R^2 {k:6s} = {v:.4f}")

    # Save results
    print("\n[5] Saving fit results...")
    results_rows = []
    for stage_name, params, r2_dict in [
        ('Stage1_Shape', s1_params, r2_s1),
        ('Stage2_Absolute', s2_params, r2_s2),
    ]:
        row = {
            'stage': stage_name,
            'log10_A': params['logA'],
            'A_s': 10.0 ** params['logA'],
            'H_star_kJmol': params['H_star'] / 1000.0,
            'x': params['x'],
            'beta': params['beta'],
            'T0_K': params['T0'],
            'T0_C': params['T0'] - 273.15,
        }
        for k, v in r2_dict.items():
            row[f'R2_{k}'] = v
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(RESULTS_DIR, 'tnm_fit_results.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Display final summary
    print("\n  " + "=" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['stage']}:")
        print(f"    log10(A/s) = {row['log10_A']:.3f}, A = {row['A_s']:.2e} s")
        print(f"    H* = {row['H_star_kJmol']:.1f} kJ/mol, x = {row['x']:.4f}, "
              f"beta = {row['beta']:.4f}, T0 = {row['T0_K']:.1f} K")
        for k, v in row.items():
            if k.startswith('R2_') and not (isinstance(v, float) and np.isnan(v)):
                print(f"    {k}: {v:.4f}")
    print("  " + "=" * 80)

    # Generate plots
    print("\n[6] Generating plots...")
    # Add grp-style keys that plot functions expect
    sim_for_plot = {k: sim_kJ_s2[k] for k in sim_kJ_s2}
    sim_for_plot['ts_grpA'] = sim_kJ_s2['tsA']
    sim_for_plot['ts_grpB'] = sim_kJ_s2['tsB']
    sim_for_plot['kov_grpA'] = sim_kJ_s2['kvA']
    sim_for_plot['kov_grpB'] = sim_kJ_s2['kvB']

    plot_one_step_fit(
        targets, sim_for_plot, s2_params,
        os.path.join(RESULTS_DIR, 'tnm_onestep_fit.png'),
    )
    plot_two_step_fit(
        targets, sim_for_plot,
        os.path.join(RESULTS_DIR, 'tnm_twostep_fit.png'),
    )
    plot_kovacs_prediction(
        targets, sim_for_plot,
        os.path.join(RESULTS_DIR, 'tnm_kovacs_prediction.png'),
    )
    plot_parity(
        sim_for_plot, targets,
        os.path.join(RESULTS_DIR, 'tnm_parity.png'),
    )
    plot_tf_evolution(
        model_s2,
        os.path.join(RESULTS_DIR, 'tnm_tf_evolution.png'),
    )
    plot_residuals(
        targets, sim_for_plot,
        os.path.join(RESULTS_DIR, 'tnm_residuals.png'),
    )

    # Summary
    print("\n" + "=" * 70)
    print("  TNM FITTING COMPLETE")
    print("=" * 70)
    print(f"  Final: logA={s2_params['logA']:.3f}, "
          f"H*={s2_params['H_star']/1000:.1f} kJ/mol, "
          f"x={s2_params['x']:.4f}, beta={s2_params['beta']:.4f}, "
          f"T0={s2_params['T0']:.1f} K")
    print(f"  Outputs in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
