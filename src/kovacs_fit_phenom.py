#!/usr/bin/env python3
"""
Phenomenological fitting of Kovacs up-jump curves using time-shifted KWW.

Model:
    dH(t; g) = H_max - H_max * exp(-((t + c_g) / tau_g)^beta_g)

where c_g = tau_g * (-ln(1 - dH0_g / H_max))^(1/beta_g)

Parameters:
    H_max     : shared asymptote (J/g) — equilibrium dH at T2=90°C
    dH0_50s   : initial dH at t=0 for 50s group (J/g)
    tau_50s   : characteristic relaxation time for 50s group (s)
    beta_50s  : stretching exponent for 50s group
    dH0_500s  : initial dH at t=0 for 500s group (J/g)
    tau_500s  : characteristic relaxation time for 500s group (s)
    beta_500s : stretching exponent for 500s group

Output:
    - Printed fit report with parameters, R², RMSE
    - Extrapolation table
    - Plot saved to results/kovacs_phenom_fit.png
    - Dense sampled curve saved to results/kovacs_phenom_curve.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')


def load_kovacs_data():
    """Load Kovacs experimental data. Returns (t, dH) for each group."""
    csv_path = os.path.join(RESULTS_DIR, 'enthalpy', 'enthalpy_kovacs.csv')
    df = pd.read_csv(csv_path)
    grp_50s = df[df['T1_group'] == '50s']
    grp_500s = df[df['T1_group'] == '500s']
    return (
        grp_50s['T2_hold_s'].values, grp_50s['delta_H_J_per_g'].values,
        grp_500s['T2_hold_s'].values, grp_500s['delta_H_J_per_g'].values,
    )


def kww_time_shifted(t, H_max, dH0, tau, beta):
    """Time-shifted KWW: dH(t) = H_max - H_max * exp(-((t+c)/tau)^beta)."""
    ratio = np.clip(1.0 - dH0 / H_max, 1e-12, 1.0 - 1e-12)
    c = tau * (-np.log(ratio)) ** (1.0 / beta)
    arg = np.clip((t + c) / tau, 1e-12, None)
    return H_max - H_max * np.exp(-(arg ** beta))


def compute_cost(params, t_50, dh_50, t_500, dh_500):
    """Sum of squared residuals for differential_evolution."""
    H_max, dH0_50, tau_50, beta_50, dH0_500, tau_500, beta_500 = params
    # Parameter bounds checks
    if not (4.0 < H_max < 12.0):
        return 1e10
    if not (-2.0 < dH0_50 < H_max * 0.8):
        return 1e10
    if not (1.0 < dH0_500 < H_max * 0.95):
        return 1e10
    if not (0.05 < beta_50 < 0.95):
        return 1e10
    if not (0.05 < beta_500 < 0.95):
        return 1e10
    if not (20 < tau_50 < 5000):
        return 1e10
    if not (20 < tau_500 < 5000):
        return 1e10

    pred_50 = kww_time_shifted(t_50, H_max, dH0_50, tau_50, beta_50)
    pred_500 = kww_time_shifted(t_500, H_max, dH0_500, tau_500, beta_500)
    return np.sum((pred_50 - dh_50) ** 2) + np.sum((pred_500 - dh_500) ** 2)


def compute_effective_c(H_max, dH0, tau, beta):
    """Compute the effective time offset c from dH0."""
    ratio = np.clip(1.0 - dH0 / H_max, 1e-12, 1.0 - 1e-12)
    return tau * (-np.log(ratio)) ** (1.0 / beta)


def fit():
    t_50, dh_50, t_500, dh_500 = load_kovacs_data()
    n_pts = len(t_50) + len(t_500)

    # Parameter bounds: [H_max, dH0_50, tau_50, beta_50, dH0_500, tau_500, beta_500]
    bounds = [
        (5.0, 10.0),       # H_max
        (-1.0, 1.5),       # dH0_50
        (30, 3000),        # tau_50
        (0.1, 0.9),        # beta_50
        (1.5, 5.0),        # dH0_500
        (30, 3000),        # tau_500
        (0.1, 0.9),        # beta_500
    ]

    print("Stage 1: Global search (differential_evolution)...")
    result_de = differential_evolution(
        lambda p: compute_cost(p, t_50, dh_50, t_500, dh_500),
        bounds, seed=42, maxiter=5000, tol=1e-10, polish=False
    )
    print(f"  DE cost = {result_de.fun:.6f}, success={result_de.success}")

    print("Stage 2: Local refinement (least_squares, 'trf')...")
    def residuals(p):
        H_max, dH0_50, tau_50, beta_50, dH0_500, tau_500, beta_500 = p
        pred_50 = kww_time_shifted(t_50, H_max, dH0_50, tau_50, beta_50)
        pred_500 = kww_time_shifted(t_500, H_max, dH0_500, tau_500, beta_500)
        return np.concatenate([pred_50 - dh_50, pred_500 - dh_500])

    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    result_ls = least_squares(
        residuals, result_de.x, bounds=(lb, ub), method='trf',
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000
    )
    p = result_ls.x
    H_max, dH0_50, tau_50, beta_50, dH0_500, tau_500, beta_500 = p

    # Final predictions
    pred_50 = kww_time_shifted(t_50, H_max, dH0_50, tau_50, beta_50)
    pred_500 = kww_time_shifted(t_500, H_max, dH0_500, tau_500, beta_500)
    residuals_all = np.concatenate([pred_50 - dh_50, pred_500 - dh_500])
    rmse = np.sqrt(np.mean(residuals_all ** 2))

    # R-squared
    ss_res_50 = np.sum((pred_50 - dh_50) ** 2)
    ss_tot_50 = np.sum((dh_50 - np.mean(dh_50)) ** 2)
    r2_50 = 1 - ss_res_50 / ss_tot_50
    ss_res_500 = np.sum((pred_500 - dh_500) ** 2)
    ss_tot_500 = np.sum((dh_500 - np.mean(dh_500)) ** 2)
    r2_500 = 1 - ss_res_500 / ss_tot_500

    # Effective c values
    c_50 = compute_effective_c(H_max, dH0_50, tau_50, beta_50)
    c_500 = compute_effective_c(H_max, dH0_500, tau_500, beta_500)

    # ---- Report ----
    print("\n" + "=" * 60)
    print("KOVACS PHENOMENOLOGICAL FIT REPORT")
    print("=" * 60)
    print(f"  H_max     = {H_max:.4f} J/g  (shared asymptote)")
    print(f"  RMSE      = {rmse:.4f} J/g  (n={n_pts})")
    print(f"  R² (50s)  = {r2_50:.4f}")
    print(f"  R² (500s) = {r2_500:.4f}")
    print()
    print(f"  Group 50s  (T1_hold=50s):")
    print(f"    dH0    = {dH0_50:.4f} J/g")
    print(f"    tau    = {tau_50:.1f} s")
    print(f"    beta   = {beta_50:.4f}")
    print(f"    c_eff  = {c_50:.2f} s")
    print(f"  Group 500s (T1_hold=500s):")
    print(f"    dH0    = {dH0_500:.4f} J/g")
    print(f"    tau    = {tau_500:.1f} s")
    print(f"    beta   = {beta_500:.4f}")
    print(f"    c_eff  = {c_500:.2f} s")

    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'Fit50s':>8s}  {'Res':>8s}  |  {'Exp500s':>8s}  {'Fit500s':>8s}  {'Res':>8s}")
    print("  " + "-" * 78)
    for i in range(10):
        print(f"  {t_50[i]:8.3f}  {dh_50[i]:8.4f}  {pred_50[i]:8.4f}  {pred_50[i]-dh_50[i]:8.4f}  |  {dh_500[i]:8.4f}  {pred_500[i]:8.4f}  {pred_500[i]-dh_500[i]:8.4f}")

    # Extrapolation
    print(f"\n  --- Extrapolation ---")
    print(f"  {'t(s)':>10s}  {'50s':>9s}  {'500s':>9s}  {'diff':>8s}  {'gap%':>7s}")
    print("  " + "-" * 50)
    for t_ext in [1000, 2000, 5000, 10000, 50000, 100000, 1000000]:
        v50 = kww_time_shifted(t_ext, H_max, dH0_50, tau_50, beta_50)
        v500 = kww_time_shifted(t_ext, H_max, dH0_500, tau_500, beta_500)
        print(f"  {t_ext:10.0f}  {v50:9.4f}  {v500:9.4f}  {v500-v50:8.4f}  {(v500-v50)/H_max*100:6.2f}%")

    # ---- Save dense curve for TNM reverse-engineering ----
    t_dense = np.logspace(-1, 8, 150)
    dh_dense_50 = kww_time_shifted(t_dense, H_max, dH0_50, tau_50, beta_50)
    dh_dense_500 = kww_time_shifted(t_dense, H_max, dH0_500, tau_500, beta_500)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    curve_df = pd.DataFrame({
        't_s': t_dense,
        'dh_50s_Jg': dh_dense_50,
        'dh_500s_Jg': dh_dense_500,
    })
    curve_df.to_csv(os.path.join(RESULTS_DIR, 'kovacs_phenom_curve.csv'), index=False)
    print(f"\n  Dense curve saved to results/kovacs_phenom_curve.csv ({len(t_dense)} points)")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 6.5))

    t_plot = np.logspace(-1, 6, 600)
    dh_plot_50 = kww_time_shifted(t_plot, H_max, dH0_50, tau_50, beta_50)
    dh_plot_500 = kww_time_shifted(t_plot, H_max, dH0_500, tau_500, beta_500)

    ax.semilogx(t_plot, dh_plot_50, 'C0-', linewidth=2, alpha=0.85, label='Fit 50s (T1_hold=50s)')
    ax.semilogx(t_plot, dh_plot_500, 'C1-', linewidth=2, alpha=0.85, label='Fit 500s (T1_hold=500s)')
    ax.scatter(t_50, dh_50, c='C0', marker='o', s=55, zorder=5, edgecolors='k', linewidth=0.5, label='Exp 50s')
    ax.scatter(t_500, dh_500, c='C1', marker='s', s=55, zorder=5, edgecolors='k', linewidth=0.5, label='Exp 500s')
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5, label='max experimental time')
    ax.axhline(y=H_max, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label=f'$\\Delta H_{{\\rm max}}$ = {H_max:.2f} J/g')

    # Annotate convergence region
    ax.annotate('Convergence\nregion', xy=(30000, H_max * 0.92), fontsize=9,
                ha='center', bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    ax.set_xlabel('ln($t_2$) (s)', fontsize=12)
    ax.set_ylabel('$\\Delta H$ (J/g)', fontsize=12)
    ax.set_title(f'Kovacs Up-Jump (80°C → 90°C) — Phenomenological KWW Fit\n'
                 f'$H_{{\\rm max}}$={H_max:.2f} J/g,  '
                 f'$R^2_{{50s}}$={r2_50:.3f},  $R^2_{{500s}}$={r2_500:.3f},  '
                 f'RMSE={rmse:.3f} J/g', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(5e-2, 1e6)
    ax.set_ylim(bottom=-0.3, top=H_max * 1.08)
    ax.grid(True, alpha=0.25)

    # Inset: zoom on short-time region
    inset = ax.inset_axes([0.18, 0.18, 0.35, 0.35])
    t_zoom = np.logspace(-1, 1.3, 100)
    dh_zoom_50 = kww_time_shifted(t_zoom, H_max, dH0_50, tau_50, beta_50)
    dh_zoom_500 = kww_time_shifted(t_zoom, H_max, dH0_500, tau_500, beta_500)
    inset.semilogx(t_zoom, dh_zoom_50, 'C0-', alpha=0.85)
    inset.semilogx(t_zoom, dh_zoom_500, 'C1-', alpha=0.85)
    inset.scatter(t_50[t_50 < 20], dh_50[t_50 < 20], c='C0', marker='o', s=35, zorder=5, edgecolors='k', linewidth=0.3)
    inset.scatter(t_500[t_500 < 20], dh_500[t_500 < 20], c='C1', marker='s', s=35, zorder=5, edgecolors='k', linewidth=0.3)
    inset.set_xlim(6e-2, 20)
    inset.set_title('Short-time detail', fontsize=8)
    inset.grid(True, alpha=0.2)
    inset.tick_params(labelsize=7)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'kovacs_phenom_fit.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to results/kovacs_phenom_fit.png")

    return {
        'H_max': H_max,
        'rmse': rmse, 'r2_50': r2_50, 'r2_500': r2_500,
        '50s': {'dH0': dH0_50, 'tau': tau_50, 'beta': beta_50, 'c_eff': c_50},
        '500s': {'dH0': dH0_500, 'tau': tau_500, 'beta': beta_500, 'c_eff': c_500},
    }


if __name__ == '__main__':
    fit()
