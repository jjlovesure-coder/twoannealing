#!/usr/bin/env python3
"""
Reverse-engineer TNM model parameters from the phenomenological Kovacs fit.

Strategy:
  1. Load the phenomenological KWW curve (kovacs_phenom_curve.csv)
  2. Densely sample both groups (50s and 500s T1_hold) as target
  3. Optimize TNM parameters (logA, H_star, x, beta, T0, scale) to match
  4. Compare TNM simulation vs phenom curve vs experimental data
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
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
T1_50S = 49.98    # seconds
T1_500S = 499.98  # seconds

# Known from phenom fit: H_max ≈ 6.62 J/g
# At full equilibrium (Tf=T2): dH_norm=1, so scale = H_max
H_MAX_KNOWN = 6.6183


class TNMModel:
    """TNM model — instantaneous quench + isothermal hold + KWW relaxation."""

    def __init__(self, A, H_star, x, beta, T0):
        self.A = A
        self.H_star = H_star
        self.x = x
        self.beta = beta
        self.T0 = T0

    def tau(self, T, Tf):
        exponent = (self.x * self.H_star) / (R_GAS * T) + \
                   ((1 - self.x) * self.H_star) / (R_GAS * Tf)
        return self.A * np.exp(exponent)

    def _isothermal_hold(self, T_hold, t_hold, T_f_start):
        """Isothermal KWW relaxation. Returns T_f_end."""
        if t_hold <= 0:
            return T_f_start

        if t_hold < 0.1:
            n_steps = 5
            t_steps = np.linspace(0, t_hold, n_steps + 1)[1:]
        else:
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
        """Simulate Kovacs up-jump: T0->T1(hold t1)->T2(hold t2).

        Returns delta_H (J/g) = scale * (T0 - Tf_end) / (T0 - T2).
        """
        Tf_after_t1 = self._isothermal_hold(T1, t1_hold, self.T0)
        Tf_end = self._isothermal_hold(T2, t2_sec, Tf_after_t1)
        dH_norm = (self.T0 - Tf_end) / max(self.T0 - T2, 1.0)
        return dH_norm


def load_targets():
    """Load phenomenological curve and experimental data."""
    # Phenomenological target
    curve_path = os.path.join(RESULTS_DIR, 'kovacs_phenom_curve.csv')
    df_curve = pd.read_csv(curve_path)
    t_target = df_curve['t_s'].values
    dh_target_50 = df_curve['dh_50s_Jg'].values
    dh_target_500 = df_curve['dh_500s_Jg'].values

    # Experimental data (for comparison only)
    exp_path = os.path.join(RESULTS_DIR, 'enthalpy', 'enthalpy_kovacs.csv')
    df_exp = pd.read_csv(exp_path)
    grp_50 = df_exp[df_exp['T1_group'] == '50s']
    grp_500 = df_exp[df_exp['T1_group'] == '500s']

    return (
        t_target, dh_target_50, dh_target_500,
        grp_50['T2_hold_s'].values, grp_50['delta_H_J_per_g'].values,
        grp_500['T2_hold_s'].values, grp_500['delta_H_J_per_g'].values,
    )


def simulate_tnm_vector(params, t2_values, t1_hold):
    """Simulate TNM for a list of t2 values, given t1_hold."""
    logA, H_star, x, beta, T0, scale = params
    A = 10.0 ** logA
    model = TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)
    dh_norm = np.array([model.simulate_kovacs(t1_hold, t2) for t2 in t2_values])
    return scale * dh_norm


def compute_cost(params, t_target, dh_target_50, dh_target_500):
    """Sum of squared residuals against phenomenological target."""
    logA, H_star, x, beta, T0, scale = params

    # Parameter sanity
    if not (-30 < logA < -8):
        return 1e10
    if not (30000 < H_star < 500000):
        return 1e10
    if not (0.005 < x < 0.95):
        return 1e10
    if not (0.01 < beta < 0.95):
        return 1e10
    if not (350 < T0 < 450):
        return 1e10
    if not (0.1 < scale < 50):
        return 1e10

    try:
        pred_50 = simulate_tnm_vector(params, t_target, T1_50S)
        pred_500 = simulate_tnm_vector(params, t_target, T1_500S)
    except Exception:
        return 1e10

    return np.sum((pred_50 - dh_target_50) ** 2) + \
           np.sum((pred_500 - dh_target_500) ** 2)


def subsample_target(t_full, dh50_full, dh500_full, n_pts=40):
    """Subsample the dense target curve for faster fitting."""
    idx = np.logspace(0, np.log10(len(t_full) - 1), n_pts).astype(int)
    idx = np.unique(np.clip(idx, 0, len(t_full) - 1))
    return t_full[idx], dh50_full[idx], dh500_full[idx]


def fit():
    t_full, dh_full_50, dh_full_500, t_exp_50, dh_exp_50, t_exp_500, dh_exp_500 = load_targets()

    # Subsample for speed: 40 points per group
    t_target, dh_target_50, dh_target_500 = subsample_target(
        t_full, dh_full_50, dh_full_500, n_pts=40)

    # Parameter bounds: [logA, H_star, x, beta, T0, scale]
    bounds = [
        (-28, -10),          # logA
        (50000, 400000),     # H_star (J/mol)
        (0.005, 0.9),        # x
        (0.01, 0.9),         # beta (KWW for TNM)
        (370, 440),          # T0 (K)
        (1.0, 15.0),         # scale ≈ H_max ≈ 6.6
    ]

    def cost(p):
        return compute_cost(p, t_target, dh_target_50, dh_target_500)

    print(f"Stage 1: Multi-start L-BFGS-B (20 starts, {len(t_target)} target pts per group)...")
    rng = np.random.RandomState(42)
    best_result = None
    best_cost = np.inf

    for k in range(20):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res
        if (k + 1) % 5 == 0:
            print(f"  Start {k+1}/20, best cost = {best_cost:.4f}")

    print(f"  Best cost = {best_cost:.4f}")

    print("Stage 2: Local refinement on full target (least_squares)...")
    def residuals(params):
        pred_50 = simulate_tnm_vector(params, t_full, T1_50S)
        pred_500 = simulate_tnm_vector(params, t_full, T1_500S)
        return np.concatenate([pred_50 - dh_full_50, pred_500 - dh_full_500])

    lb_ls = [b[0] for b in bounds]
    ub_ls = [b[1] for b in bounds]
    result_ls = least_squares(
        residuals, best_result.x, bounds=(lb_ls, ub_ls), method='trf',
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000
    )
    p = result_ls.x
    logA, H_star, x, beta, T0, scale = p

    # Final predictions (on full target set)
    pred_50 = simulate_tnm_vector(p, t_full, T1_50S)
    pred_500 = simulate_tnm_vector(p, t_full, T1_500S)
    pred_exp_50 = simulate_tnm_vector(p, t_exp_50, T1_50S)
    pred_exp_500 = simulate_tnm_vector(p, t_exp_500, T1_500S)

    residuals_all = np.concatenate([pred_50 - dh_full_50, pred_500 - dh_full_500])
    rmse = np.sqrt(np.mean(residuals_all ** 2))

    # R-squared vs phenom curve
    ss_res_50 = np.sum((pred_50 - dh_full_50) ** 2)
    ss_tot_50 = np.sum((dh_full_50 - np.mean(dh_full_50)) ** 2)
    r2_50_phenom = 1 - ss_res_50 / ss_tot_50
    ss_res_500 = np.sum((pred_500 - dh_full_500) ** 2)
    ss_tot_500 = np.sum((dh_full_500 - np.mean(dh_full_500)) ** 2)
    r2_500_phenom = 1 - ss_res_500 / ss_tot_500

    # R-squared vs experiment
    ss_res_50e = np.sum((pred_exp_50 - dh_exp_50) ** 2)
    ss_tot_50e = np.sum((dh_exp_50 - np.mean(dh_exp_50)) ** 2)
    r2_50_exp = 1 - ss_res_50e / ss_tot_50e
    ss_res_500e = np.sum((pred_exp_500 - dh_exp_500) ** 2)
    ss_tot_500e = np.sum((dh_exp_500 - np.mean(dh_exp_500)) ** 2)
    r2_500_exp = 1 - ss_res_500e / ss_tot_500e

    # Report
    print("\n" + "=" * 65)
    print("TNM REVERSE-ENGINEERING REPORT")
    print("=" * 65)
    print(f"  logA      = {logA:.4f}")
    print(f"  A         = {10**logA:.4e} s")
    print(f"  H*        = {H_star/1000:.2f} kJ/mol")
    print(f"  x         = {x:.4f}")
    print(f"  beta      = {beta:.4f}")
    print(f"  T0        = {T0:.2f} K  ({T0-273.15:.1f} °C)")
    print(f"  scale     = {scale:.4f} J/g")
    print(f"  RMSE vs phenom = {rmse:.4f} J/g")
    print(f"  R² vs phenom (50s)  = {r2_50_phenom:.4f}")
    print(f"  R² vs phenom (500s) = {r2_500_phenom:.4f}")
    print(f"  R² vs exp    (50s)  = {r2_50_exp:.4f}")
    print(f"  R² vs exp    (500s) = {r2_500_exp:.4f}")

    # Comparison table at experimental points
    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'TNM50s':>8s}  {'Phen50s':>8s}  |  {'Exp500s':>8s}  {'TNM500s':>8s}  {'Phen500s':>8s}")
    print("  " + "-" * 88)
    phenom_exp_50 = np.interp(t_exp_50, t_full, dh_full_50)
    phenom_exp_500 = np.interp(t_exp_500, t_full, dh_full_500)
    for i in range(10):
        print(f"  {t_exp_50[i]:8.3f}  {dh_exp_50[i]:8.4f}  {pred_exp_50[i]:8.4f}  {phenom_exp_50[i]:8.4f}  |  "
              f"{dh_exp_500[i]:8.4f}  {pred_exp_500[i]:8.4f}  {phenom_exp_500[i]:8.4f}")

    # Extrapolation
    print(f"\n  --- TNM Extrapolation ---")
    print(f"  {'t(s)':>10s}  {'TNM 50s':>9s}  {'TNM 500s':>9s}  {'diff':>8s}  {'Phen diff':>9s}")
    print("  " + "-" * 55)
    for t_ext in [1000, 2000, 5000, 10000, 50000, 100000]:
        v50_tnm = simulate_tnm_vector(p, [t_ext], T1_50S)[0]
        v500_tnm = simulate_tnm_vector(p, [t_ext], T1_500S)[0]
        v50_phe = np.interp(t_ext, t_full, dh_full_50)
        v500_phe = np.interp(t_ext, t_full, dh_full_500)
        print(f"  {t_ext:10.0f}  {v50_tnm:9.4f}  {v500_tnm:9.4f}  {v500_tnm-v50_tnm:8.4f}  {v500_phe-v50_phe:9.4f}")

    # Save results
    os.makedirs(os.path.join(RESULTS_DIR, 'tnm'), exist_ok=True)
    tnm_params_df = pd.DataFrame([{
        'logA': logA, 'A_s': 10**logA, 'H_star_Jmol': H_star,
        'H_star_kJmol': H_star/1000, 'x': x, 'beta': beta,
        'T0_K': T0, 'T0_C': T0 - 273.15, 'scale_Jg': scale,
        'rmse_phenom': rmse,
        'r2_50s_phenom': r2_50_phenom, 'r2_500s_phenom': r2_500_phenom,
        'r2_50s_exp': r2_50_exp, 'r2_500s_exp': r2_500_exp,
    }])
    tnm_params_df.to_csv(os.path.join(RESULTS_DIR, 'tnm', 'tnm_reverse_params.csv'), index=False)
    print(f"\n  Parameters saved to results/tnm/tnm_reverse_params.csv")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left panel: full comparison
    ax = axes[0]
    t_plot = np.logspace(-1, 6, 600)
    dh_phe_50_plot = np.interp(t_plot, t_full, dh_full_50)
    dh_phe_500_plot = np.interp(t_plot, t_full, dh_full_500)
    dh_tnm_50_plot = simulate_tnm_vector(p, t_plot, T1_50S)
    dh_tnm_500_plot = simulate_tnm_vector(p, t_plot, T1_500S)

    ax.semilogx(t_plot, dh_phe_50_plot, 'C0--', linewidth=1.5, alpha=0.5, label='Phenom 50s')
    ax.semilogx(t_plot, dh_phe_500_plot, 'C1--', linewidth=1.5, alpha=0.5, label='Phenom 500s')
    ax.semilogx(t_plot, dh_tnm_50_plot, 'C0-', linewidth=2, label='TNM 50s')
    ax.semilogx(t_plot, dh_tnm_500_plot, 'C1-', linewidth=2, label='TNM 500s')
    ax.scatter(t_exp_50, dh_exp_50, c='C0', marker='o', s=50, zorder=5, edgecolors='k', linewidth=0.5, label='Exp 50s')
    ax.scatter(t_exp_500, dh_exp_500, c='C1', marker='s', s=50, zorder=5, edgecolors='k', linewidth=0.5, label='Exp 500s')
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('ln($t_2$) (s)')
    ax.set_ylabel('$\\Delta H$ (J/g)')
    ax.set_title(f'TNM vs Phenomenological vs Experiment\n'
                 f'$R^2_{{\\rm exp}}$: 50s={r2_50_exp:.3f}, 500s={r2_500_exp:.3f}')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(5e-2, 1e6)
    ax.grid(True, alpha=0.25)

    # Right panel: parity plot
    ax = axes[1]
    all_tnm = np.concatenate([pred_exp_50, pred_exp_500])
    all_exp = np.concatenate([dh_exp_50, dh_exp_500])
    ax.scatter(all_exp, all_tnm, c='C3', alpha=0.7, s=40, zorder=5)
    ax.plot([0, max(all_exp) * 1.05], [0, max(all_exp) * 1.05], 'k-', alpha=0.3)
    ax.set_xlabel('Experimental $\\Delta H$ (J/g)')
    ax.set_ylabel('TNM $\\Delta H$ (J/g)')
    ax.set_title(f'Parity Plot\n$R^2$ = {np.corrcoef(all_exp, all_tnm)[0,1]**2:.3f}')
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'tnm', 'kovacs_reverse_tnm.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to results/tnm/kovacs_reverse_tnm.png")

    return p, rmse


if __name__ == '__main__':
    fit()
