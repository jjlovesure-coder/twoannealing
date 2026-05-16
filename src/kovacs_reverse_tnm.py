#!/usr/bin/env python3
"""
Reverse-engineer TNM model parameters from the phenomenological Kovacs fit.

Strategy:
  1. Load the phenomenological KWW curve (kovacs_phenom_curve.csv)
  2. Optimize TNM parameters (logA, H_star, x, beta, T0, scale) to match
  3. Compare TNM simulation vs phenom curve vs experimental data

Uses instantaneous-quench TNM — standard in literature for isothermal
annealing / Kovacs fitting (see D'Amore 2006, Grassi 2018).
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')

R_GAS = 8.314

# Kovacs protocol temperatures (Kelvin)
T1 = 80.0 + 273.15   # 353.15 K
T2 = 90.0 + 273.15   # 363.15 K

# T1 hold times
T1_50S = 49.98
T1_500S = 499.98


class TNMModel:
    """TNM model — instantaneous quench + isothermal KWW relaxation.

    Standard approach: freeze-in during cooling is absorbed into T0.
    Literature: D'Amore (2006) for PS gives x≈0.9, β≈0.27.
    """

    def __init__(self, A, H_star, x, beta, T0):
        self.A = A
        self.H_star = H_star
        self.x = x
        self.beta = beta
        self.T0 = T0

    def tau(self, T, Tf):
        exponent = (self.x * self.H_star) / (R_GAS * T) + \
                   ((1 - self.x) * self.H_star) / (R_GAS * Tf)
        exponent = np.clip(exponent, -50, 80)
        return self.A * np.exp(exponent)

    def _isothermal_hold(self, T_hold, t_hold, T_f_start):
        if t_hold <= 0:
            return T_f_start
        n_steps = 12
        t_steps = np.logspace(np.log10(t_hold / 100),
                              np.log10(t_hold), n_steps)
        t_steps = np.unique(np.round(t_steps, 10))

        S = 0.0
        Tf = T_f_start
        t_prev = 0.0
        for t_i in t_steps:
            tau_val = self.tau(T_hold, Tf)
            dt = t_i - t_prev
            t_prev = t_i
            S += dt / tau_val
            Tf = T_hold + (T_f_start - T_hold) * np.exp(-(S ** self.beta))
        return Tf

    def simulate_kovacs(self, t1_hold, t2_sec):
        """Kovacs up-jump: quench T0→T1, hold t1, quench T1→T2, hold t2."""
        Tf_after_t1 = self._isothermal_hold(T1, t1_hold, self.T0)
        Tf_end = self._isothermal_hold(T2, t2_sec, Tf_after_t1)
        dH_norm = (self.T0 - Tf_end) / max(self.T0 - T2, 1.0)
        return np.clip(dH_norm, -1.0, 2.0)


def load_targets():
    curve_path = os.path.join(RESULTS_DIR, 'kovacs_phenom_curve.csv')
    df_curve = pd.read_csv(curve_path)
    t_target = df_curve['t_s'].values
    dh_target_50 = df_curve['dh_50s_Jg'].values
    dh_target_500 = df_curve['dh_500s_Jg'].values

    exp_path = os.path.join(RESULTS_DIR, 'enthalpy', 'enthalpy_kovacs.csv')
    df_exp = pd.read_csv(exp_path)
    grp_50 = df_exp[df_exp['T1_group'] == '50s']
    grp_500 = df_exp[df_exp['T1_group'] == '500s']

    return (t_target, dh_target_50, dh_target_500,
            grp_50['T2_hold_s'].values, grp_50['delta_H_J_per_g'].values,
            grp_500['T2_hold_s'].values, grp_500['delta_H_J_per_g'].values)


def simulate_tnm_vector(params, t2_values, t1_hold):
    logA, H_star, x, beta, T0, scale = params
    A = 10.0 ** logA
    model = TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)
    dh_norm = np.array([model.simulate_kovacs(t1_hold, t2) for t2 in t2_values])
    if np.any(np.isnan(dh_norm)):
        return np.full(len(t2_values), np.nan)
    return scale * dh_norm


def subsample_target(t_full, dh50_full, dh500_full, n_pts=40):
    idx = np.logspace(0, np.log10(len(t_full) - 1), n_pts).astype(int)
    idx = np.unique(np.clip(idx, 0, len(t_full) - 1))
    return t_full[idx], dh50_full[idx], dh500_full[idx]


def compute_cost(params, t_target, dh_target_50, dh_target_500):
    logA, H_star, x, beta, T0, scale = params
    if not (-36 < logA < -8): return 1e10
    if not (50000 < H_star < 600000): return 1e10
    if not (0.05 < x < 0.99): return 1e10
    if not (0.05 < beta < 0.8): return 1e10
    if not (370 < T0 < 430): return 1e10
    if not (0.5 < scale < 20): return 1e10

    try:
        pred_50 = simulate_tnm_vector(params, t_target, T1_50S)
        pred_500 = simulate_tnm_vector(params, t_target, T1_500S)
    except Exception:
        return 1e10

    if np.any(np.isnan(pred_50)) or np.any(np.isnan(pred_500)):
        return 1e10

    return np.sum((pred_50 - dh_target_50) ** 2) + \
           np.sum((pred_500 - dh_target_500) ** 2)


def fit():
    t_full, dh_full_50, dh_full_500, t_exp_50, dh_exp_50, t_exp_500, dh_exp_500 = load_targets()

    # Subsample for speed
    t_target, dh_target_50, dh_target_500 = subsample_target(
        t_full, dh_full_50, dh_full_500, n_pts=30)

    # [logA, H_star, x, beta, T0, scale]
    # Physically-motivated bounds for PS (see D'Amore 2006, Tropin 2015)
    bounds = [
        (-35, -10),         # logA
        (80000, 350000),    # H_star (J/mol)
        (0.1, 0.95),        # x
        (0.1, 0.7),         # beta (stretched: 0.2-0.6 typical for polymers)
        (375, 420),         # T0 (Tg_PS ≈ 373K, T0 slightly above)
        (1.0, 15.0),        # scale
    ]

    def cost(p):
        return compute_cost(p, t_target, dh_target_50, dh_target_500)

    n_starts = 40
    print(f"Multi-start L-BFGS-B ({n_starts} starts, {len(t_target)} pts/group)...")
    rng = np.random.RandomState(42)
    best_result = None
    best_cost = np.inf

    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res
        if (k + 1) % 10 == 0:
            print(f"  Start {k+1}/{n_starts}, best cost = {best_cost:.4f}")

    print(f"  Final best cost = {best_cost:.4f}")
    p = best_result.x
    logA, H_star, x, beta, T0, scale = p

    # Final predictions
    pred_50 = simulate_tnm_vector(p, t_full, T1_50S)
    pred_500 = simulate_tnm_vector(p, t_full, T1_500S)
    pred_exp_50 = simulate_tnm_vector(p, t_exp_50, T1_50S)
    pred_exp_500 = simulate_tnm_vector(p, t_exp_500, T1_500S)

    residual_all = np.concatenate([pred_50 - dh_full_50, pred_500 - dh_full_500])
    rmse = np.sqrt(np.mean(residual_all ** 2))

    # R² vs phenom
    ssr50 = np.sum((pred_50 - dh_full_50)**2)
    sst50 = np.sum((dh_full_50 - np.mean(dh_full_50))**2)
    r2_50p = 1 - ssr50 / sst50
    ssr500 = np.sum((pred_500 - dh_full_500)**2)
    sst500 = np.sum((dh_full_500 - np.mean(dh_full_500))**2)
    r2_500p = 1 - ssr500 / sst500

    # R² vs experiment
    ssr50e = np.sum((pred_exp_50 - dh_exp_50)**2)
    sst50e = np.sum((dh_exp_50 - np.mean(dh_exp_50))**2)
    r2_50e = 1 - ssr50e / sst50e
    ssr500e = np.sum((pred_exp_500 - dh_exp_500)**2)
    sst500e = np.sum((dh_exp_500 - np.mean(dh_exp_500))**2)
    r2_500e = 1 - ssr500e / sst500e

    # ── Report ──
    print()
    print("=" * 60)
    print("TNM REVERSE-ENGINEERING (instantaneous quench)")
    print("=" * 60)
    print(f"  logA     = {logA:.4f}")
    print(f"  A        = {10**logA:.4e} s")
    print(f"  H*       = {H_star/1000:.2f} kJ/mol")
    print(f"  x        = {x:.4f}")
    print(f"  beta     = {beta:.4f}")
    print(f"  T0       = {T0:.2f} K  ({T0-273.15:.1f} °C)")
    print(f"  scale    = {scale:.4f} J/g")
    print(f"  RMSE vs phenom = {rmse:.4f} J/g")
    print(f"  R² vs phenom   = 50s:{r2_50p:.4f}  500s:{r2_500p:.4f}")
    print(f"  R² vs exp      = 50s:{r2_50e:.4f}  500s:{r2_500e:.4f}")

    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'TNM_50':>8s}  {'Phe_50':>8s}  |  {'Exp500s':>8s}  {'TNM500':>8s}  {'Phe500':>8s}")
    print("  " + "-" * 82)
    ph50 = np.interp(t_exp_50, t_full, dh_full_50)
    ph500 = np.interp(t_exp_500, t_full, dh_full_500)
    for i in range(10):
        print(f"  {t_exp_50[i]:8.3f}  {dh_exp_50[i]:8.4f}  {pred_exp_50[i]:8.4f}  {ph50[i]:8.4f}  |  "
              f"{dh_exp_500[i]:8.4f}  {pred_exp_500[i]:8.4f}  {ph500[i]:8.4f}")

    print(f"\n  {'t(s)':>10s}  {'TNM_50':>9s}  {'TNM_500':>9s}  {'diff':>8s}  {'Phe_diff':>9s}")
    print("  " + "-" * 55)
    for t_ext in [1000, 2000, 5000, 10000, 50000, 100000]:
        v50t = simulate_tnm_vector(p, [t_ext], T1_50S)[0]
        v500t = simulate_tnm_vector(p, [t_ext], T1_500S)[0]
        v50p = np.interp(t_ext, t_full, dh_full_50)
        v500p = np.interp(t_ext, t_full, dh_full_500)
        print(f"  {t_ext:10.0f}  {v50t:9.4f}  {v500t:9.4f}  {v500t-v50t:8.4f}  {v500p-v50p:9.4f}")

    # Save
    os.makedirs(os.path.join(RESULTS_DIR, 'tnm'), exist_ok=True)
    df_params = pd.DataFrame([{
        'logA': logA, 'A_s': 10**logA, 'H_star_Jmol': H_star,
        'H_star_kJmol': H_star/1000, 'x': x, 'beta': beta,
        'T0_K': T0, 'T0_C': T0 - 273.15, 'scale_Jg': scale,
        'rmse_phenom': rmse,
        'r2_50s_phenom': r2_50p, 'r2_500s_phenom': r2_500p,
        'r2_50s_exp': r2_50e, 'r2_500s_exp': r2_500e,
    }])
    df_params.to_csv(os.path.join(RESULTS_DIR, 'tnm', 'tnm_reverse_params.csv'), index=False)

    # ── 3-way comparison plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left: semilog
    ax = axes[0]
    t_plot = np.logspace(-1, 6, 600)
    dh_ph50 = np.interp(t_plot, t_full, dh_full_50)
    dh_ph500 = np.interp(t_plot, t_full, dh_full_500)
    dh_t50 = simulate_tnm_vector(p, t_plot, T1_50S)
    dh_t500 = simulate_tnm_vector(p, t_plot, T1_500S)

    ax.semilogx(t_plot, dh_ph50, 'C0--', lw=1.5, alpha=0.5, label='Phenom 50s')
    ax.semilogx(t_plot, dh_ph500, 'C1--', lw=1.5, alpha=0.5, label='Phenom 500s')
    ax.semilogx(t_plot, dh_t50, 'C0-', lw=2, label='TNM 50s')
    ax.semilogx(t_plot, dh_t500, 'C1-', lw=2, label='TNM 500s')
    ax.scatter(t_exp_50, dh_exp_50, c='C0', marker='o', s=50, zorder=5,
               edgecolors='k', linewidth=0.5, label='Exp 50s')
    ax.scatter(t_exp_500, dh_exp_500, c='C1', marker='s', s=50, zorder=5,
               edgecolors='k', linewidth=0.5, label='Exp 500s')
    ax.axvline(x=1000, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('ln($t_2$) (s)')
    ax.set_ylabel(r'$\Delta H$ (J/g)')
    ax.set_title(f'Kovacs Up-Jump (80°C→90°C): TNM (instant. quench) vs Phenom vs Exp\n'
                 f'$R^2_{{\\rm exp}}$: 50s={r2_50e:.3f}, 500s={r2_500e:.3f}')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(5e-2, 1e6)
    ax.grid(True, alpha=0.25)

    # Right: parity
    ax = axes[1]
    ax.scatter(dh_exp_50, pred_exp_50, c='C0', marker='o', s=45, zorder=5,
               edgecolors='k', linewidth=0.3, label='50s')
    ax.scatter(dh_exp_500, pred_exp_500, c='C1', marker='s', s=45, zorder=5,
               edgecolors='k', linewidth=0.3, label='500s')
    ax.plot([0, 7], [0, 7], 'k-', alpha=0.3)
    ax.set_xlabel('Experimental ΔH (J/g)')
    ax.set_ylabel('TNM ΔH (J/g)')
    ax.set_title('Parity Plot')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Annotate parameters
    textstr = (
        f"logA = {logA:.2f}\n"
        f"H* = {H_star/1000:.1f} kJ/mol\n"
        f"x = {x:.3f}\n"
        f"β = {beta:.3f}\n"
        f"T0 = {T0:.1f} K"
    )
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'tnm', 'kovacs_reverse_tnm.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to results/tnm/kovacs_reverse_tnm.png")

    return p, rmse


if __name__ == '__main__':
    fit()
