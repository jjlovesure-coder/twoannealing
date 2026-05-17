#!/usr/bin/env python3
"""
Two-Step Down-Jump (90°C→80°C) — TNM Direct Fit to Experiment

CRITICAL: Fits TNM directly to experimental data, NOT to KWW phenom curve.
KWW phenom gives tau_50 > tau_500, which contradicts TNM physics
(higher Tf → smaller tau). Direct fit avoids this contradiction.

Uses instantaneous-quench TNM with multi-start L-BFGS-B.
"""

import os, numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
R = 8.314

T1 = 363.15; T2 = 353.15
T1_50S = 49.98; T1_500S = 499.98


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


def cost(params, t50, dh50, t500, dh500):
    logA, H_star, x, beta, T0, scale = params
    if not (-40 < logA < -5):     return 1e10
    if not (30000 < H_star < 800000): return 1e10
    if not (0.01 < x < 0.999):    return 1e10
    if not (0.05 < beta < 0.999): return 1e10
    if not (360 < T0 < 500):      return 1e10
    if not (1 < scale < 15):      return 1e10
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

    def c(p):
        return cost(p, t50, dh50, t500, dh500)

    n_starts = 80
    print(f"Instantaneous-quench TNM — {n_starts} starts...", flush=True)
    rng = np.random.RandomState(42)
    best_c, best_p = np.inf, None
    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        r = minimize(c, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 400, 'ftol': 1e-14})
        if r.fun < best_c:
            best_c, best_p = r.fun, r.x
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{n_starts}, best cost = {best_c:.6f}", flush=True)

    print(f"  Final best cost = {best_c:.6f}", flush=True)
    logA, H_star, x, beta, T0, scale = best_p

    p50 = simulate_vector(best_p, t50, T1_50S)
    p500 = simulate_vector(best_p, t500, T1_500S)
    rmse = np.sqrt(np.mean(np.concatenate([p50 - dh50, p500 - dh500])**2))
    r2_50 = 1 - np.sum((p50 - dh50)**2) / np.sum((dh50 - np.mean(dh50))**2)
    r2_500 = 1 - np.sum((p500 - dh500)**2) / np.sum((dh500 - np.mean(dh500))**2)

    print(f"\n{'='*60}")
    print(f"TWO-STEP TNM — DIRECT FIT TO EXPERIMENT")
    print(f"{'='*60}")
    print(f"  logA   = {logA:.4f}    (A = {10**logA:.4e} s)")
    print(f"  H*     = {H_star/1000:.2f} kJ/mol")
    print(f"  x      = {x:.4f}")
    print(f"  β      = {beta:.4f}")
    print(f"  T0     = {T0:.2f} K  ({T0-273.15:.1f} °C)")
    print(f"  scale  = {scale:.4f} J/g")
    print(f"  RMSE   = {rmse:.4f} J/g")
    print(f"  R²     = 50s: {r2_50:.4f}   500s: {r2_500:.4f}")

    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'TNM50s':>8s}  {'res':>8s}  |  "
          f"{'Exp500s':>8s}  {'TNM500s':>8s}  {'res':>8s}")
    for i in range(10):
        print(f"  {t50[i]:8.3f}  {dh50[i]:8.4f}  {p50[i]:8.4f}  {p50[i]-dh50[i]:8.4f}  |  "
              f"{dh500[i]:8.4f}  {p500[i]:8.4f}  {p500[i]-dh500[i]:8.4f}")

    print(f"\n  {'t(s)':>10s}  {'TNM 50s':>9s}  {'TNM 500s':>9s}  {'diff':>8s}")
    for te in [1000, 2000, 5000, 10000, 50000, 100000]:
        vt50 = simulate_vector(best_p, [te], T1_50S)[0]
        vt500 = simulate_vector(best_p, [te], T1_500S)[0]
        print(f"  {te:10.0f}  {vt50:9.4f}  {vt500:9.4f}  {vt500-vt50:8.4f}")

    os.makedirs(os.path.join(RESULTS, 'tnm'), exist_ok=True)
    pd.DataFrame([{
        'logA': logA, 'A_s': 10**logA, 'H_star_kJmol': H_star/1000,
        'x': x, 'beta': beta, 'T0_K': T0, 'T0_C': T0 - 273.15,
        'scale_Jg': scale, 'rmse': rmse,
        'r2_50s': r2_50, 'r2_500s': r2_500,
    }]).to_csv(os.path.join(RESULTS, 'tnm', 'tnm_twosteps_params.csv'), index=False)

    # Plot
    t_plt = np.logspace(-1, 6, 300)
    tnm50p = simulate_vector(best_p, t_plt, T1_50S)
    tnm500p = simulate_vector(best_p, t_plt, T1_500S)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(t50, dh50, c='#1f77b4', marker='o', s=70, zorder=10,
               edgecolors='k', lw=0.6, label='Exp 50s')
    ax.scatter(t500, dh500, c='#ff7f0e', marker='s', s=70, zorder=10,
               edgecolors='k', lw=0.6, label='Exp 500s')
    ax.semilogx(t_plt, tnm50p, '#1f77b4', ls='-', lw=2.2, label='TNM 50s')
    ax.semilogx(t_plt, tnm500p, '#ff7f0e', ls='-', lw=2.2, label='TNM 500s')
    ax.axhline(y=scale, color='red', ls=':', lw=1.2, alpha=0.5)
    ax.axvline(x=1000, color='gray', ls='--', alpha=0.25)
    ax.set_xlabel('$t_2$ (s)', fontsize=13)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=13)
    ax.set_title('Two-Step (90°C→80°C): TNM Direct Fit', fontsize=13, fontweight='bold')
    ax.set_xlim(5e-2, 1e6); ax.set_ylim(-0.3, scale * 1.15)
    ax.grid(True, alpha=0.2)

    ins = ax.inset_axes([0.55, 0.10, 0.42, 0.35])
    tz = np.logspace(-1, 1.3, 100)
    ins.semilogx(tz, simulate_vector(best_p, tz, T1_50S), '#1f77b4', ls='-', lw=1.5)
    ins.semilogx(tz, simulate_vector(best_p, tz, T1_500S), '#ff7f0e', ls='-', lw=1.5)
    ins.scatter(t50[t50 < 20], dh50[t50 < 20], c='#1f77b4', marker='o', s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.scatter(t500[t500 < 20], dh500[t500 < 20], c='#ff7f0e', marker='s', s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.set_xlim(0.06, 20); ins.set_ylim(-0.05, scale * 0.75)
    ins.set_title('Short-time (0.06–20 s)', fontsize=8.5, pad=3)
    ins.grid(True, alpha=0.2); ins.tick_params(labelsize=7)
    zx0, zx1, zy0, zy1 = 0.06, 20, -0.1, scale * 0.75
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, lw=1.5, ec='green', fc='green', alpha=0.07, zorder=2))
    fig.add_artist(ConnectionPatch(xyA=(zx1, zy1), coordsA=ax.transData, xyB=(1, 1),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))
    fig.add_artist(ConnectionPatch(xyA=(zx0, zy0), coordsA=ax.transData, xyB=(0, 0),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))

    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9,
                  markeredgecolor='k', markeredgewidth=0.5, label='Exp 50s'),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markersize=9,
                  markeredgecolor='k', markeredgewidth=0.5, label='Exp 500s'),
           Line2D([0], [0], color='#1f77b4', lw=2.2, label='TNM model')]
    ax.legend(handles=leg, fontsize=10, loc='lower right')
    ax.text(0.03, 0.97,
            f"TNM (instantaneous quench)\n  log A = {logA:.2f}\n"
            f"  H* = {H_star/1000:.1f} kJ/mol\n  x = {x:.4f}\n  β = {beta:.4f}\n"
            f"  T₀ = {T0:.1f} K ({T0-273.15:.0f}°C)\n  ΔH_max = {scale:.2f} J/g\n"
            f"R² 50s = {r2_50:.4f}  R² 500s = {r2_500:.4f}  RMSE = {rmse:.4f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, ec='gray'))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'tnm', 'twosteps_tnm_fit.png'), dpi=200)
    plt.close()
    print(f"\n  Plot saved to results/tnm/twosteps_tnm_fit.png")

    return best_p, rmse, (r2_50, r2_500)


if __name__ == '__main__':
    fit()
