#!/usr/bin/env python3
"""
Comprehensive error analysis for TNM model fitting.

Generates:
  1. Per-protocol residual analysis
  2. Bootstrap parameter uncertainty (200 resamples)
  3. Parameter correlation matrix
  4. Sensitivity to cooling rate and T0
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C, T_INITIAL,
    COOLING_RATE, build_target_vectors, T1_HOLD_SHORT, T1_HOLD_LONG,
)
from tnm_fit import (
    make_model, compute_r_squared, run_stage2,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    'os50': '#2166AC', 'os70': '#4393C3',
    'tsA': '#B2182B', 'tsB': '#D6604D',
    'kvA': '#4DAF4A', 'kvB': '#984EA3',
}


def residual_analysis(model, targets, sim_data_kJ):
    """Plot residuals (sim - exp) vs log hold time for all protocols."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'grpA'),
        'tsB': ('ts', 'grpB'),
        'kvA': ('kovacs', 'grpA'),
        'kvB': ('kovacs', 'grpB'),
    }
    titles = {
        'os50': 'One-step 50C', 'os70': 'One-step 70C',
        'tsA': 'Two-step Grp A', 'tsB': 'Two-step Grp B',
        'kvA': 'Kovacs Grp A', 'kvB': 'Kovacs Grp B',
    }

    print(f"    sim_data_kJ keys: {list(sim_data_kJ.keys())}")
    all_residuals = []
    for idx, (sim_key, (exp_key, grp_key)) in enumerate(group_map.items()):
        print(f"    Checking {sim_key}: exp={exp_key}, grp={grp_key}, "
              f"mask_sum={targets[exp_key]['groups'] == grp_key if exp_key in targets else 'N/A'}")
        ax = axes.flat[idx]
        exp = targets[exp_key]
        mask = exp['groups'] == grp_key
        if mask.sum() == 0 or sim_key not in sim_data_kJ:
            continue
        dh_exp = exp['dH'][mask]
        t_exp = exp['t'][mask]
        dh_sim = sim_data_kJ[sim_key][:len(dh_exp)]

        residuals = dh_sim - dh_exp
        all_residuals.extend(residuals)

        ax.plot(t_exp / 60.0, residuals, 'o', color=COLORS[sim_key],
                markersize=7, markerfacecolor='white', markeredgewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Hold time (min)')
        ax.set_ylabel('Residual (kJ/mol)')
        ax.set_title(f'{titles[sim_key]}')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis: TNM Model - Experiment', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_residual_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved residual analysis: {out_path}")

    all_residuals = np.array(all_residuals)
    print(f"\n  Residual summary:")
    print(f"    Mean:  {np.mean(all_residuals):.2f} kJ/mol")
    print(f"    Std:   {np.std(all_residuals):.2f} kJ/mol")
    print(f"    RMSE:  {np.sqrt(np.mean(all_residuals**2)):.2f} kJ/mol")
    print(f"    Max:   {np.max(np.abs(all_residuals)):.2f} kJ/mol")


def bootstrap_uncertainty(targets, stage1_params, n_bootstrap=200, seed=42):
    """Bootstrap parameter uncertainty by resampling hold times."""
    print(f"\n  Bootstrap ({n_bootstrap} resamples)...")

    rng = np.random.RandomState(seed)
    all_params = []

    for b in range(n_bootstrap):
        bs_targets = {}
        for exp_key in ['os50', 'os70', 'ts', 'kovacs']:
            orig = targets[exp_key]
            n = len(orig['t'])
            idx = rng.randint(0, n, n)
            bs_targets[exp_key] = {
                't': orig['t'][idx],
                'dH': orig['dH'][idx],
                'groups': orig['groups'][idx],
            }

        try:
            params, _ = run_stage2(bs_targets, stage1_params, seed=seed + b)
            all_params.append([params['logA'], params['H_star'], params['x'],
                               params['beta'], params['T0']])
        except Exception:
            continue

        if (b + 1) % 50 == 0:
            print(f"    Bootstrap {b+1}/{n_bootstrap}")

    all_params = np.array(all_params)
    param_names = ['logA', 'H_star', 'x', 'beta', 'T0']

    print(f"\n  Parameter uncertainty (95% CI):")
    print(f"  {'Param':<8} {'Mean':<12} {'2.5%':<12} {'97.5%':<12} {'Std':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i, name in enumerate(param_names):
        vals = all_params[:, i]
        mean = np.mean(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        std = np.std(vals)
        print(f"  {name:<8} {mean:<12.4f} {lo:<12.4f} {hi:<12.4f} {std:<12.4f}")

    bs_df = pd.DataFrame(all_params, columns=param_names)
    bs_csv = os.path.join(RESULTS_DIR, 'tnm_bootstrap_params.csv')
    bs_df.to_csv(bs_csv, index=False, float_format='%.6f')
    print(f"\n  Saved bootstrap results: {bs_csv}")

    return all_params, param_names


def parameter_correlation(all_params, param_names):
    """Plot parameter correlation matrix."""
    corr = np.corrcoef(all_params.T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, color='black' if abs(corr[i, j]) < 0.7 else 'white')
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Parameter Correlation Matrix (Bootstrap)')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_param_correlation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved correlation matrix: {out_path}")


def sensitivity_analysis(params):
    """Sensitivity of Tf to cooling rate and T0 variations."""
    print("\n  Sensitivity analysis...")
    logA, H_star, x, beta, T0 = (params['logA'], params['H_star'],
                                  params['x'], params['beta'], params['T0'])

    rates = [0.2, 0.5, 1.0, 2.0, 5.0]
    print(f"\n  Cooling rate sensitivity (base rate = 1.0 K/s):")
    print(f"  {'Rate':<10} {'os50 Tf':<12} {'os70 Tf':<12}")
    for rate in rates:
        m = TNMModel(A=10**logA, H_star=H_star, x=x, beta=beta, T0=T0)
        tf50 = m.simulate_one_step(T_50C, 100.0, T_INITIAL, rate)[0]
        tf70 = m.simulate_one_step(T_70C, 100.0, T_INITIAL, rate)[0]
        print(f"  {rate:<10.1f} {tf50:<12.2f} {tf70:<12.2f}")

    T0_vals = np.linspace(T0 - 10, T0 + 10, 5)
    print(f"\n  T0 sensitivity (base T0 = {T0:.1f} K):")
    print(f"  {'T0 (K)':<10} {'os50 dH_norm':<14} {'os70 dH_norm':<14}")
    for t0 in T0_vals:
        m = TNMModel(A=10**logA, H_star=H_star, x=x, beta=beta, T0=t0)
        dh50 = m.delta_H_normalized(T_50C, 100.0, T_INITIAL, COOLING_RATE)
        dh70 = m.delta_H_normalized(T_70C, 100.0, T_INITIAL, COOLING_RATE)
        print(f"  {t0:<10.1f} {dh50:<14.6f} {dh70:<14.6f}")


def main():
    print("=" * 60)
    print("  TNM Model — Error Analysis")
    print("=" * 60)

    csv_path = os.path.join(RESULTS_DIR, 'tnm_fit_results.csv')
    if not os.path.exists(csv_path):
        print("ERROR: Run tnm_main.py first to generate fit results.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    s2_row = df[df['stage'] == 'Stage2_Absolute'].iloc[0]
    params = {
        'logA': s2_row['log10_A'],
        'H_star': s2_row['H_star_kJmol'] * 1000.0,
        'x': s2_row['x'],
        'beta': s2_row['beta'],
        'T0': s2_row['T0_K'],
    }

    print(f"\n  Loaded fit parameters:")
    print(f"    logA = {params['logA']:.3f}")
    print(f"    H*   = {params['H_star']/1000:.1f} kJ/mol")
    print(f"    x    = {params['x']:.4f}")
    print(f"    beta = {params['beta']:.4f}")
    print(f"    T0   = {params['T0']:.1f} K")

    model = make_model(**params)
    targets = build_target_vectors()

    from tnm_main import compute_sim_kJ_data
    sim_data_kJ, _ = compute_sim_kJ_data(model, targets)

    print("\n[1] Residual analysis...")
    residual_analysis(model, targets, sim_data_kJ)

    print("\n[2] Bootstrap parameter uncertainty...")
    s1_params = {k: params[k] for k in params}
    all_params, param_names = bootstrap_uncertainty(targets, s1_params, n_bootstrap=200)

    if len(all_params) > 0:
        print("\n[3] Parameter correlation...")
        parameter_correlation(all_params, param_names)

    print("\n[4] Sensitivity analysis...")
    sensitivity_analysis(params)

    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
