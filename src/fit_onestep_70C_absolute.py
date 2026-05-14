#!/usr/bin/env python3
"""Fit TNM model to absolute ΔH for 70°C one-step annealing.

Fits run1 and run2 separately (6 params: logA, H*, x, beta, T0, scale).
Extends prediction to 10^5s to verify plateau convergence.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnm_model import TNMModel
from tnm_conditions import HOLD_TIMES_SEC, T_70C, build_target_vectors

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_model(logA, H_star, x, beta, T0, scale):
    A = 10.0 ** logA
    model = TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)
    return model, scale


def delta_H_kJ(model, scale, T_anneal, t_hold):
    """Absolute ΔH in kJ/mol: scale * (T0 - Tf)."""
    Tf_end, _ = model.simulate_one_step(T_anneal, t_hold)
    return scale * (model.T0 - Tf_end)


def cost_function(packed, t_exp, dH_exp, T_anneal):
    """Sum of squared residuals in kJ/mol."""
    logA, H_star, x, beta, T0, scale = packed
    if not (0.001 < x < 1 and 0.01 < beta < 1 and H_star > 30000 and 360 < T0 < 430 and scale > 0):
        return 1e10
    try:
        model, s = make_model(logA, H_star, x, beta, T0, scale)
    except Exception:
        return 1e10

    dH_sim = np.array([delta_H_kJ(model, s, T_anneal, t) for t in t_exp])
    if np.std(dH_sim) < 1e-8:
        return 1e8
    return np.sum((dH_sim - dH_exp) ** 2)


def fit_run(t_exp, dH_exp, T_anneal, run_label, seed=42):
    """Fit 6 TNM parameters to one run."""
    rng = np.random.RandomState(seed)
    bounds = [
        (-25, -12), (50000, 300000), (0.005, 0.9), (0.01, 0.9), (370, 430), (0.1, 5000),
    ]

    best_res = None
    best_cost = np.inf
    n_starts = 25

    print(f"  Fitting {run_label} ({n_starts} starts)...")
    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(cost_function, x0, args=(t_exp, dH_exp, T_anneal),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.fun < best_cost:
            best_cost = res.fun
            best_res = res
        if (k + 1) % 10 == 0:
            print(f"    Start {k+1}/{n_starts}, best cost = {best_cost:.1f}")

    p = best_res.x
    rmse = np.sqrt(best_cost / len(t_exp))
    print(f"    Done: logA={p[0]:.3f}, H*={p[1]/1000:.1f} kJ/mol, x={p[2]:.4f}, "
          f"beta={p[3]:.4f}, T0={p[4]:.1f}K, scale={p[5]:.2f}, RMSE={rmse:.1f} kJ/mol")
    return p, rmse


def main():
    print("=" * 60)
    print("  Phase 1: One-step 70°C — Absolute ΔH Fitting")
    print("=" * 60)

    targets = build_target_vectors()
    exp_70 = targets['os70']
    T_anneal = T_70C

    mask_r1 = exp_70['groups'] == 'run1'
    mask_r2 = exp_70['groups'] == 'run2'
    t_r1 = exp_70['t'][mask_r1]
    dh_r1 = exp_70['dH'][mask_r1]
    t_r2 = exp_70['t'][mask_r2]
    dh_r2 = exp_70['dH'][mask_r2]

    print(f"\n  Run1: {len(t_r1)} pts, ΔH = [{dh_r1.min():.0f}, {dh_r1.max():.0f}] kJ/mol")
    print(f"  Run2: {len(t_r2)} pts, ΔH = [{dh_r2.min():.0f}, {dh_r2.max():.0f}] kJ/mol")

    print("\n[1] Fitting run1...")
    p1, rmse1 = fit_run(t_r1, dh_r1, T_anneal, "Run1")
    print(f"\n[2] Fitting run2...")
    p2, rmse2 = fit_run(t_r2, dh_r2, T_anneal, "Run2")

    m1, s1 = make_model(*p1)
    m2, s2 = make_model(*p2)

    # Extended prediction
    t_ext = np.logspace(-2, 5, 300)
    dh_ext_r1 = np.array([delta_H_kJ(m1, s1, T_anneal, t) for t in t_ext])
    dh_ext_r2 = np.array([delta_H_kJ(m2, s2, T_anneal, t) for t in t_ext])

    # Plateau values
    plateau_r1 = s1 * (p1[4] - T_anneal)
    plateau_r2 = s2 * (p2[4] - T_anneal)
    plateau_diff = abs(plateau_r1 - plateau_r2)

    print(f"\n[3] Plateau convergence at 10^5s:")
    print(f"    Run1 plateau: {plateau_r1:.1f} kJ/mol")
    print(f"    Run2 plateau: {plateau_r2:.1f} kJ/mol")
    print(f"    Difference:   {plateau_diff:.1f} kJ/mol "
          f"({plateau_diff / max(plateau_r1, plateau_r2) * 100:.2f}%)")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.semilogx(t_ext / 60, dh_ext_r1, 'b-', linewidth=2, label='Run1')
    ax.plot(t_r1 / 60, dh_r1, 'bo', markersize=9, markerfacecolor='white',
            markeredgewidth=2, label='Exp')
    ax.axhline(y=plateau_r1, color='gray', linestyle=':', alpha=0.7,
               label=f'Plateau = {plateau_r1:.0f} kJ/mol')
    ax.set_xlabel('Hold time at 70C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title(f'Run1: logA={p1[0]:.2f}, H*={p1[1]/1000:.0f} kJ/mol, '
                 f'x={p1[2]:.3f}, beta={p1[3]:.3f}, RMSE={rmse1:.0f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[0, 1]
    ax.semilogx(t_ext / 60, dh_ext_r2, 'r-', linewidth=2, label='Run2')
    ax.plot(t_r2 / 60, dh_r2, 'ro', markersize=9, markerfacecolor='white',
            markeredgewidth=2, label='Exp')
    ax.axhline(y=plateau_r2, color='gray', linestyle=':', alpha=0.7,
               label=f'Plateau = {plateau_r2:.0f} kJ/mol')
    ax.set_xlabel('Hold time at 70C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title(f'Run2: logA={p2[0]:.2f}, H*={p2[1]/1000:.0f} kJ/mol, '
                 f'x={p2[2]:.3f}, beta={p2[3]:.3f}, RMSE={rmse2:.0f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1, 0]
    ax.semilogx(t_ext / 60, dh_ext_r1, 'b-', linewidth=2, alpha=0.6, label='Run1')
    ax.semilogx(t_ext / 60, dh_ext_r2, 'r-', linewidth=2, alpha=0.6, label='Run2')
    ax.plot(t_r1 / 60, dh_r1, 'bo', markersize=9, markerfacecolor='white',
            markeredgewidth=2)
    ax.plot(t_r2 / 60, dh_r2, 'ro', markersize=9, markerfacecolor='white',
            markeredgewidth=2)
    ax.axhline(y=plateau_r1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=plateau_r2, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.5, 0.05,
            f'Plateau gap: {plateau_diff:.0f} kJ/mol ({plateau_diff/max(plateau_r1,plateau_r2)*100:.1f}%)',
            transform=ax.transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Hold time at 70C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title('One-step 70C: Run1 vs Run2 — Plateau Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1, 1]
    Tf_ext_r1 = np.array([m1.simulate_one_step(T_anneal, t)[0] for t in t_ext])
    Tf_ext_r2 = np.array([m2.simulate_one_step(T_anneal, t)[0] for t in t_ext])
    ax.semilogx(t_ext / 60, Tf_ext_r1 - 273.15, 'b-', linewidth=2, label='Run1')
    ax.semilogx(t_ext / 60, Tf_ext_r2 - 273.15, 'r-', linewidth=2, label='Run2')
    ax.axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='T=70C (equilibrium)')
    ax.set_xlabel('Hold time at 70C (min)')
    ax.set_ylabel('T_f (C)')
    ax.set_title('Fictive Temperature Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('TNM One-Step 70C — Absolute Enthalpy Fitting', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_onestep_70C_absolute.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")

    import pandas as pd
    rows = [{'run': 'run1', 'logA': p1[0], 'A_s': 10**p1[0], 'H_star_kJmol': p1[1]/1000,
             'x': p1[2], 'beta': p1[3], 'T0_K': p1[4], 'scale': p1[5],
             'RMSE_kJmol': rmse1, 'plateau_kJmol': plateau_r1},
            {'run': 'run2', 'logA': p2[0], 'A_s': 10**p2[0], 'H_star_kJmol': p2[1]/1000,
             'x': p2[2], 'beta': p2[3], 'T0_K': p2[4], 'scale': p2[5],
             'RMSE_kJmol': rmse2, 'plateau_kJmol': plateau_r2}]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'tnm_onestep_70C_params.csv'),
                               index=False, float_format='%.6f')
    print("  Saved parameters CSV")
    print("=" * 60)


if __name__ == '__main__':
    main()
