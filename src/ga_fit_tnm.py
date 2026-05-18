#!/usr/bin/env python3
"""
Two-Step Down-Jump (90°C→80°C) — TNM Fit via Genetic Algorithm

Uses scipy's differential_evolution (evolutionary/genetic algorithm) for global
optimization, followed by L-BFGS-B polish.  Compares against the existing
multi-start L-BFGS-B approach to quantify the benefit of global search.
"""

import os, time, numpy as np, pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
R = 8.314

T1, T2 = 363.15, 353.15
T1_50S, T1_500S = 49.98, 499.98


class TNM:
    """Instantaneous-quench TNM."""
    def __init__(self, A, H_star, x, beta, T0):
        self.A, self.H, self.x, self.b, self.T0 = A, H_star, x, beta, T0

    def tau(self, T, Tf):
        e = self.x * self.H / (R * T) + (1 - self.x) * self.H / (R * Tf)
        return self.A * np.exp(np.clip(e, -50, 80))

    def _hold(self, T, t, Tf0):
        if t <= 0:
            return Tf0
        ts = np.unique(np.round(np.logspace(np.log10(t / 100), np.log10(t), 12), 10))
        S, Tf, tp = 0.0, Tf0, 0.0
        for ti in ts:
            S += (ti - tp) / self.tau(T, Tf)
            tp = ti
            Tf = T + (Tf0 - T) * np.exp(-(S ** self.b))
        return Tf

    def simulate(self, t1, t2):
        Tf1 = self._hold(T1, t1, self.T0)
        Tf2 = self._hold(T2, t2, Tf1)
        return np.clip((self.T0 - Tf2) / max(self.T0 - T2, 1.0), -1, 2)


def load_data():
    df = pd.read_csv(os.path.join(RESULTS, 'enthalpy', 'enthalpy_twosteps.csv'))
    g50 = df[df['T1_group'] == '50s']
    g500 = df[df['T1_group'] == '500s']
    return (g50['T2_hold_s'].values, g50['delta_H_J_per_g'].values,
            g500['T2_hold_s'].values, g500['delta_H_J_per_g'].values)


def simulate_vector(params, t2s, t1h):
    logA, H_star, x, beta, T0, scale = params
    m = TNM(A=10**logA, H_star=H_star, x=x, beta=beta, T0=T0)
    return scale * np.array([m.simulate(t1h, t) for t in t2s])


def cost_vector(params, t50, dh50, t500, dh500):
    """Vectorized cost — returns float, safe for DE."""
    logA, H_star, x, beta, T0, scale = params
    try:
        p50 = simulate_vector(params, t50, T1_50S)
        p500 = simulate_vector(params, t500, T1_500S)
    except Exception:
        return 1e10
    if np.any(~np.isfinite(p50)) or np.any(~np.isfinite(p500)):
        return 1e10
    return np.sum((p50 - dh50)**2) + np.sum((p500 - dh500)**2)


def fit():
    t50, dh50, t500, dh500 = load_data()

    bounds = [
        (-35, -15),         # logA
        (100000, 400000),   # H_star J/mol
        (0.2, 0.99),        # x
        (0.1, 0.99),        # beta
        (380, 460),         # T0 K
        (3, 10),            # scale J/g
    ]

    def cost(p):
        return cost_vector(p, t50, dh50, t500, dh500)

    # ── Stage 1: Global search via differential evolution ──
    print("=" * 60)
    print("TNM FIT — GENETIC ALGORITHM (Differential Evolution)")
    print("=" * 60)
    t0 = time.time()

    de_result = differential_evolution(
        cost, bounds,
        seed=42,
        maxiter=2000,
        tol=1e-10,
        popsize=25,            # 25×6 = 150 population members
        mutation=(0.5, 1.2),   # dithering
        recombination=0.9,
        polish=False,           # polish separately for transparency
        workers=1,
        disp=True,
    )
    t_de = time.time() - t0

    # ── Stage 2: Polish with L-BFGS-B ──
    print(f"\n  GA best cost = {de_result.fun:.6f}  (took {t_de:.1f}s)")
    print(f"  GA best params: logA={de_result.x[0]:.3f}  H*={de_result.x[1]/1000:.1f} kJ/mol  "
          f"x={de_result.x[2]:.4f}  beta={de_result.x[3]:.4f}  T0={de_result.x[4]:.1f} K  scale={de_result.x[5]:.4f}")
    print("  Polishing with L-BFGS-B...", flush=True)

    polish_result = minimize(cost, de_result.x, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14})
    t_polish = time.time() - t0 - t_de

    best_p = polish_result.x
    best_c = polish_result.fun
    logA, H_star, x, beta, T0, scale = best_p

    p50 = simulate_vector(best_p, t50, T1_50S)
    p500 = simulate_vector(best_p, t500, T1_500S)
    rmse = np.sqrt(np.mean(np.concatenate([p50 - dh50, p500 - dh500])**2))
    r2_50 = 1 - np.sum((p50 - dh50)**2) / np.sum((dh50 - np.mean(dh50))**2)
    r2_500 = 1 - np.sum((p500 - dh500)**2) / np.sum((dh500 - np.mean(dh500))**2)

    # ── Comparison: multi-start L-BFGS-B ──
    print(f"\n  Polish cost = {best_c:.6f}  (took {t_polish:.1f}s)")
    print(f"  Running multi-start L-BFGS-B for comparison...", flush=True)

    t1 = time.time()
    rng = np.random.RandomState(42)
    bfgs_best_c, bfgs_best_p = np.inf, None
    for k in range(80):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        r = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 400, 'ftol': 1e-14})
        if r.fun < bfgs_best_c:
            bfgs_best_c, bfgs_best_p = r.fun, r.x
    t_bfgs = time.time() - t1

    p50_bfgs = simulate_vector(bfgs_best_p, t50, T1_50S)
    p500_bfgs = simulate_vector(bfgs_best_p, t500, T1_500S)
    rmse_bfgs = np.sqrt(np.mean(np.concatenate([p50_bfgs - dh50, p500_bfgs - dh500])**2))
    r2_50_bfgs = 1 - np.sum((p50_bfgs - dh50)**2) / np.sum((dh50 - np.mean(dh50))**2)
    r2_500_bfgs = 1 - np.sum((p500_bfgs - dh500)**2) / np.sum((dh500 - np.mean(dh500))**2)

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"GA + POLISH  vs  MULTI-START L-BFGS-B")
    print(f"{'='*60}")
    print(f"\n  {'':<14s} {'GA+Polish':>14s} {'Multi-Start':>14s} {'Winner':>10s}")
    print(f"  {'-'*53}")
    print(f"  {'Cost':<14s} {best_c:>14.6f} {bfgs_best_c:>14.6f} {'GA' if best_c < bfgs_best_c else 'BFGS':>10s}")
    print(f"  {'RMSE':<14s} {rmse:>14.4f} {rmse_bfgs:>14.4f}")
    print(f"  {'R² 50s':<14s} {r2_50:>14.4f} {r2_50_bfgs:>14.4f}")
    print(f"  {'R² 500s':<14s} {r2_500:>14.4f} {r2_500_bfgs:>14.4f}")
    print(f"  {'Time (s)':<14s} {t_de + t_polish:>14.1f} {t_bfgs:>14.1f}")

    print(f"\n  GA + Polish Parameters:")
    print(f"    logA  = {logA:.4f}     (A = {10**logA:.4e} s)")
    print(f"    H*    = {H_star/1000:.2f} kJ/mol")
    print(f"    x     = {x:.4f}")
    print(f"    beta  = {beta:.4f}")
    print(f"    T0    = {T0:.2f} K  ({T0-273.15:.1f} °C)")
    print(f"    scale = {scale:.4f} J/g")

    # ── Pointwise table ──
    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'GA_50s':>8s}  {'BFGS50s':>8s}  |  "
          f"{'Exp500s':>8s}  {'GA_500s':>8s}  {'BFGS500s':>8s}")
    for i in range(10):
        print(f"  {t50[i]:8.3f}  {dh50[i]:8.4f}  {p50[i]:8.4f}  {p50_bfgs[i]:8.4f}  |  "
              f"{dh500[i]:8.4f}  {p500[i]:8.4f}  {p500_bfgs[i]:8.4f}")

    # ── Save ──
    os.makedirs(os.path.join(RESULTS, 'tnm'), exist_ok=True)
    pd.DataFrame([{
        'method': 'GA+Polish',
        'logA': logA, 'A_s': 10**logA, 'H_star_kJmol': H_star/1000,
        'x': x, 'beta': beta, 'T0_K': T0, 'T0_C': T0 - 273.15,
        'scale_Jg': scale, 'cost': best_c, 'rmse': rmse,
        'r2_50s': r2_50, 'r2_500s': r2_500,
        'nfe_de': de_result.nfev, 'nfe_polish': polish_result.nfev,
        't_de_s': t_de, 't_polish_s': t_polish,
    }]).to_csv(os.path.join(RESULTS, 'tnm', 'tnm_ga_params.csv'), index=False)

    pd.DataFrame([{
        'method': 'MultiStart_LBFGSB',
        'logA': bfgs_best_p[0], 'A_s': 10**bfgs_best_p[0],
        'H_star_kJmol': bfgs_best_p[1]/1000,
        'x': bfgs_best_p[2], 'beta': bfgs_best_p[3],
        'T0_K': bfgs_best_p[4], 'T0_C': bfgs_best_p[4] - 273.15,
        'scale_Jg': bfgs_best_p[5], 'cost': bfgs_best_c, 'rmse': rmse_bfgs,
        'r2_50s': r2_50_bfgs, 'r2_500s': r2_500_bfgs,
        't_s': t_bfgs,
    }]).to_csv(os.path.join(RESULTS, 'tnm', 'tnm_bfgs_comparison_params.csv'), index=False)

    # ── Convergence trace (re-run DE with store_history) ──
    print("\n  Re-running DE with history tracking for convergence plot...", flush=True)
    from scipy.optimize import OptimizeResult
    history = []
    def callback(xk, convergence=0):
        history.append(cost(xk))

    _ = differential_evolution(
        cost, bounds, seed=42, maxiter=2000, tol=1e-10,
        popsize=25, mutation=(0.5, 1.2), recombination=0.9,
        polish=False, workers=1, disp=False,
        callback=callback,
    )
    # Compute rolling best
    rolling_best = np.minimum.accumulate(history)

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TNM Fit via Genetic Algorithm (Differential Evolution)', fontsize=14, fontweight='bold')

    # Main fit plot
    ax = axes[0, 0]
    t_plt = np.logspace(-1, 6, 300)
    ga50p = simulate_vector(best_p, t_plt, T1_50S)
    ga500p = simulate_vector(best_p, t_plt, T1_500S)
    bg50p = simulate_vector(bfgs_best_p, t_plt, T1_50S)
    bg500p = simulate_vector(bfgs_best_p, t_plt, T1_500S)

    ax.scatter(t50, dh50, c='#1f77b4', marker='o', s=60, zorder=10,
               edgecolors='k', lw=0.6, label='Exp 50s')
    ax.scatter(t500, dh500, c='#ff7f0e', marker='s', s=60, zorder=10,
               edgecolors='k', lw=0.6, label='Exp 500s')
    ax.semilogx(t_plt, ga50p, '#1f77b4', ls='-', lw=2.2, label='GA 50s')
    ax.semilogx(t_plt, ga500p, '#ff7f0e', ls='-', lw=2.2, label='GA 500s')
    ax.semilogx(t_plt, bg50p, '#1f77b4', ls='--', lw=1.2, alpha=0.6, label='BFGS 50s')
    ax.semilogx(t_plt, bg500p, '#ff7f0e', ls='--', lw=1.2, alpha=0.6, label='BFGS 500s')
    ax.axhline(y=scale, color='red', ls=':', lw=1.2, alpha=0.5)
    ax.axvline(x=1000, color='gray', ls='--', alpha=0.25)
    ax.set_xlabel('$t_2$ (s)', fontsize=12)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=12)
    ax.set_title('Two-Step (90°C→80°C): TNM Fit Comparison', fontsize=12, fontweight='bold')
    ax.set_xlim(5e-2, 1e6); ax.set_ylim(-0.3, scale * 1.15)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc='lower right')

    # Convergence plot
    ax = axes[0, 1]
    ax.semilogy(rolling_best, '#2ca02c', lw=1.5, label='DE best cost')
    ax.axhline(y=best_c, color='green', ls=':', lw=1.0, alpha=0.7, label=f'GA final: {best_c:.4f}')
    ax.axhline(y=bfgs_best_c, color='#d62728', ls=':', lw=1.0, alpha=0.7, label=f'BFGS best: {bfgs_best_c:.4f}')
    ax.set_xlabel('Function evaluations', fontsize=12)
    ax.set_ylabel('Cost (SSE)', fontsize=12)
    ax.set_title('DE Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Parity plot
    ax = axes[1, 0]
    ax.scatter(dh50, p50, c='#1f77b4', marker='o', s=50, zorder=5, edgecolors='k', lw=0.3, label='GA 50s')
    ax.scatter(dh500, p500, c='#ff7f0e', marker='s', s=50, zorder=5, edgecolors='k', lw=0.3, label='GA 500s')
    ax.scatter(dh50, p50_bfgs, c='#1f77b4', marker='o', s=25, alpha=0.3, label='BFGS 50s')
    ax.scatter(dh500, p500_bfgs, c='#ff7f0e', marker='s', s=25, alpha=0.3, label='BFGS 500s')
    lims = [-0.5, scale * 1.05]
    ax.plot(lims, lims, 'k-', alpha=0.3, lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Experimental ΔH (J/g)', fontsize=12)
    ax.set_ylabel('Predicted ΔH (J/g)', fontsize=12)
    ax.set_title('Parity Plot', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Residuals
    ax = axes[1, 1]
    res_ga_50 = p50 - dh50
    res_ga_500 = p500 - dh500
    res_bfgs_50 = p50_bfgs - dh50
    res_bfgs_500 = p500_bfgs - dh500

    ax.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax.semilogx(t50, res_ga_50, 'o', c='#1f77b4', ms=7, label='GA 50s')
    ax.semilogx(t500, res_ga_500, 's', c='#ff7f0e', ms=7, label='GA 500s')
    ax.semilogx(t50, res_bfgs_50, 'o', c='#1f77b4', ms=4, alpha=0.4, label='BFGS 50s')
    ax.semilogx(t500, res_bfgs_500, 's', c='#ff7f0e', ms=4, alpha=0.4, label='BFGS 500s')
    ax.set_xlabel('$t_2$ (s)', fontsize=12)
    ax.set_ylabel('Residual (J/g)', fontsize=12)
    ax.set_title(f'Residuals (GA RMSE={rmse:.4f}, BFGS RMSE={rmse_bfgs:.4f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate with params
    axes[0, 0].text(0.03, 0.96,
                    f"GA + Polish\n  log A = {logA:.2f}\n"
                    f"  H* = {H_star/1000:.1f} kJ/mol\n  x = {x:.4f}\n  β = {beta:.4f}\n"
                    f"  T₀ = {T0:.1f} K\n  scale = {scale:.2f} J/g\n"
                    f"R² 50s = {r2_50:.4f}  R² 500s = {r2_500:.4f}\nRMSE = {rmse:.4f}",
                    transform=axes[0, 0].transAxes, fontsize=8.5, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, ec='gray'))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RESULTS, 'tnm', 'twosteps_tnm_ga_fit.png'), dpi=200)
    plt.close()
    print(f"  Plot saved to results/tnm/twosteps_tnm_ga_fit.png")

    return best_p, rmse, (r2_50, r2_500)


if __name__ == '__main__':
    fit()
