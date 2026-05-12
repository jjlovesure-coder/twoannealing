#!/usr/bin/env python3
"""
TNM Model Driver — Shape-based staged fitting and comprehensive analysis.

Fit strategy: normalize both simulated and experimental delta_H to [0,1],
then minimize the squared error in normalized space. This focuses on
the curve SHAPE regardless of absolute scale.
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
    COOLING_RATE, load_experimental_data, build_target_vectors,
)
from tnm_fit import (
    make_model, run_stage1, run_stage2, compute_r_squared,
    simulate_protocol_dh,
)
from tnm_plots import (
    plot_one_step_fit, plot_two_step_fit, plot_kovacs_prediction,
    plot_parity, plot_tf_evolution, plot_residuals,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')


def make_scaled_sim_data(model, targets):
    """Compute simulated delta_H for all 6 protocol groups.

    Returns normalized [0,1] values plus fitted scaling to kJ/mol.
    """
    sim_data_norm = {}
    sim_data_kJ = {}
    scaling = {}

    for key in ['os50', 'os70', 'tsA', 'tsB', 'kvA', 'kvB']:
        dh_sim_norm, dh_exp_norm = simulate_protocol_dh(model, key, targets)
        if dh_sim_norm is not None:
            sim_data_norm[key] = dh_sim_norm

    # Fit scaling to kJ/mol for each protocol
    # Group mapping for each protocol
    proto_groups = {
        ('os50', 'os50'): ('os50', 'run1'),
        ('os70', 'os70'): ('os70', 'run1'),
        ('ts_grpA', 'tsA'): ('ts', 'ts_grpA'),
        ('ts_grpB', 'tsB'): ('ts', 'ts_grpB'),
        ('kov_grpA', 'kvA'): ('kovacs', 'kov_grpA'),
        ('kov_grpB', 'kvB'): ('kovacs', 'kov_grpB'),
    }

    for (proto_name, sim_key), (exp_key, grp_key) in proto_groups.items():
        exp = targets[exp_key]
        if exp_key in ('os50', 'os70'):
            dh_e = exp['dH']
            t_e = exp['t']
        else:
            mask = exp['groups'] == grp_key
            dh_e = exp['dH'][mask]
            t_e = exp['t'][mask]

        if sim_key in sim_data_norm:
            dh_s = sim_data_norm[sim_key]
            n_sim_raw = 10  # sim has 10 unique hold times
            if len(dh_s) == n_sim_raw:
                # Sim is 10 raw values; interp to match exp
                dh_s_interp = np.interp(t_e, HOLD_TIMES_SEC, dh_s)
            else:
                dh_s_interp = dh_s
            a, b = np.polyfit(dh_s_interp, dh_e, 1)
            scaling[sim_key] = (a, b)
            sim_data_kJ[sim_key] = a * dh_s_interp + b

    return sim_data_norm, sim_data_kJ, scaling


def compute_all_r2(sim_data_kJ, targets):
    """Compute R^2 for each group using kJ/mol values."""
    r2_values = {}
    group_map = {
        'os50': ('os50', 'os50'), 'os70': ('os70', 'os70'),
        'ts_grpA': ('ts', 'ts_grpA'), 'ts_grpB': ('ts', 'ts_grpB'),
        'kov_grpA': ('kovacs', 'kov_grpA'), 'kov_grpB': ('kovacs', 'kov_grpB'),
    }
    for proto_name, (exp_key, grp_key) in group_map.items():
        exp = targets[exp_key]
        if exp_key in ('os50', 'os70'):
            dh_e = exp['dH']
            t_e = exp['t']
        else:
            mask = exp['groups'] == grp_key
            dh_e = exp['dH'][mask]
            t_e = exp['t'][mask]

        sim_key = proto_name.replace('_grpA', 'A').replace('_grpB', 'B').replace('os', 'os')
        # Fix: map proto_name to correct sim_data_kJ key
        sim_key_map = {
            'os50': 'os50', 'os70': 'os70',
            'ts_grpA': 'tsA', 'ts_grpB': 'tsB',
            'kov_grpA': 'kvA', 'kov_grpB': 'kvB',
        }
        sk = sim_key_map[proto_name]
        if sk in sim_data_kJ:
            dh_s = sim_data_kJ[sk]
            r2_values[proto_name] = compute_r_squared(dh_s, dh_e)
        else:
            r2_values[proto_name] = np.nan
    return r2_values


def main():
    print("=" * 70)
    print("  TNM Model — Shape-Based Fitting to PS Annealing Data")
    print("=" * 70)

    T0 = 393.15  # 120 C, equilibrium reference for PS

    # Load targets
    print("\n[1] Loading experimental data...")
    targets = build_target_vectors()
    print(f"  Loaded: os50={len(targets['os50']['dH'])}pts, "
          f"os70={len(targets['os70']['dH'])}pts, "
          f"ts={len(targets['ts']['dH'])}pts, "
          f"kovacs={len(targets['kovacs']['dH'])}pts")

    # Stage 1: Shape-fit one-step
    print("\n[2] Stage 1: Shape-fitting one-step (50C + 70C)...")
    s1_params, s1_result = run_stage1(targets, T0, seed=42)
    model_s1 = make_model(**s1_params, T0=T0)
    sim_norm_s1, sim_kJ_s1, scaling_s1 = make_scaled_sim_data(model_s1, targets)
    r2_s1 = compute_all_r2(sim_kJ_s1, targets)

    print(f"\n  Stage 1 results:")
    print(f"    log10(A) = {s1_params['logA']:.3f}  "
          f"(A = {10**s1_params['logA']:.2e} s)")
    print(f"    H*       = {s1_params['H_star']/1000:.1f} kJ/mol")
    print(f"    x        = {s1_params['x']:.4f}")
    print(f"    beta     = {s1_params['beta']:.4f}")
    print(f"    Cost     = {s1_result.fun:.6f}")
    for k, v in r2_s1.items():
        if not np.isnan(v):
            print(f"    R^2 {k:12s} = {v:.4f}")

    # Stage 2: Add two-step
    print("\n[3] Stage 2: Refining with two-step shape...")
    s2_params, s2_result = run_stage2(targets, T0, s1_params, seed=42)
    model_s2 = make_model(**s2_params, T0=T0)
    sim_norm_s2, sim_kJ_s2, scaling_s2 = make_scaled_sim_data(model_s2, targets)
    r2_s2 = compute_all_r2(sim_kJ_s2, targets)

    print(f"\n  Stage 2 results:")
    print(f"    log10(A) = {s2_params['logA']:.3f}  "
          f"(A = {10**s2_params['logA']:.2e} s)")
    print(f"    H*       = {s2_params['H_star']/1000:.1f} kJ/mol")
    print(f"    x        = {s2_params['x']:.4f}")
    print(f"    beta     = {s2_params['beta']:.4f}")
    print(f"    Cost     = {s2_result.fun:.6f}")
    for k, v in r2_s2.items():
        if not np.isnan(v):
            print(f"    R^2 {k:12s} = {v:.4f}")

    # Stage 3: Kovacs prediction
    print("\n[4] Stage 3: Kovacs prediction (no fitting)...")
    r2_kv = {k: v for k, v in r2_s2.items() if k.startswith('kov')}
    for k, v in r2_kv.items():
        print(f"    R^2 {k:12s} = {v:.4f}")

    # Save results
    print("\n[5] Saving fit results...")
    results_rows = []
    for stage_name, params, r2_dict in [
        ('Stage1_OneStep', s1_params, r2_s1),
        ('Stage2_TwoStep', s2_params, r2_s2),
    ]:
        row = {
            'stage': stage_name,
            'log10_A': params['logA'],
            'A_s': 10.0 ** params['logA'],
            'H_star_kJmol': params['H_star'] / 1000.0,
            'x': params['x'],
            'beta': params['beta'],
            **{f'R2_{k}': v for k, v in r2_dict.items()},
        }
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(RESULTS_DIR, 'tnm_fit_results.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    print("\n  " + "=" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['stage']}:")
        print(f"    log10(A/s) = {row['log10_A']:.3f}, "
              f"A = {row['A_s']:.2e} s")
        print(f"    H* = {row['H_star_kJmol']:.1f} kJ/mol, "
              f"x = {row['x']:.4f}, beta = {row['beta']:.4f}")
        for k, v in row.items():
            if k.startswith('R2_') and not (isinstance(v, float) and np.isnan(v)):
                print(f"    {k}: {v:.4f}")
    print("  " + "=" * 80)

    # Generate plots
    print("\n[6] Generating plots...")
    # Build sim_data dict for plot functions
    sim_for_plot = {}
    for proto_name in ['os50', 'os70', 'ts_grpA', 'ts_grpB',
                        'kov_grpA', 'kov_grpB']:
        if proto_name in sim_kJ_s2:
            sim_for_plot[proto_name] = sim_kJ_s2[proto_name]
        else:
            sim_for_plot[proto_name] = np.zeros(10)

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
          f"x={s2_params['x']:.4f}, beta={s2_params['beta']:.4f}")
    print(f"  Outputs in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
