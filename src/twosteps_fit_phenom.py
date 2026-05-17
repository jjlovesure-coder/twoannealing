#!/usr/bin/env python3
"""
Two-Step Down-Jump — KWW Phenomenological Fit

Model: dH(t; g) = H_max - H_max * exp(-((t + c_g) / tau_g)^beta_g)
       where c_g = tau_g * (-ln(1 - dH0_g / H_max))^(1/beta_g)

Output: results/twosteps_phenom_curve.csv (dense sampled curve)
        results/twosteps_phenom_fit.png
"""

import os, numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')


def load_data():
    df = pd.read_csv(os.path.join(RESULTS, 'enthalpy', 'enthalpy_twosteps.csv'))
    g50 = df[df['T1_group'] == '50s']
    g500 = df[df['T1_group'] == '500s']
    return (g50['T2_hold_s'].values, g50['delta_H_J_per_g'].values,
            g500['T2_hold_s'].values, g500['delta_H_J_per_g'].values)


def kww_ts(t, H_max, dH0, tau, beta):
    ratio = np.clip(1.0 - dH0 / H_max, 1e-12, 1.0 - 1e-12)
    c = tau * (-np.log(ratio)) ** (1.0 / beta)
    return H_max - H_max * np.exp(-((t + c) / tau) ** beta)


def fit():
    t50, dh50, t500, dh500 = load_data()

    def cost(p):
        H_max, dH0_50, tau_50, beta_50, dH0_500, tau_500, beta_500 = p
        if not (3 < H_max < 8): return 1e10
        if not (-1 < dH0_50 < H_max * 0.9): return 1e10
        if not (2 < dH0_500 < H_max * 0.98): return 1e10
        if not (0.05 < beta_50 < 0.95): return 1e10
        if not (0.05 < beta_500 < 0.999): return 1e10
        if not (20 < tau_50 < 5000): return 1e10
        if not (10 < tau_500 < 5000): return 1e10
        p50 = kww_ts(t50, H_max, dH0_50, tau_50, beta_50)
        p500 = kww_ts(t500, H_max, dH0_500, tau_500, beta_500)
        return np.sum((p50 - dh50) ** 2) + np.sum((p500 - dh500) ** 2)

    bounds = [(3.0, 7.0), (-1, 2), (20, 5000), (0.1, 0.9),
              (2.5, 5.5), (10, 3000), (0.1, 0.999)]

    print("KWW fit (DE + polish)...", flush=True)
    res = differential_evolution(cost, bounds, seed=42, maxiter=3000,
                                 tol=1e-10, polish=True)
    H_max, dH0_50, tau_50, beta_50, dH0_500, tau_500, beta_500 = res.x

    p50 = kww_ts(t50, H_max, dH0_50, tau_50, beta_50)
    p500 = kww_ts(t500, H_max, dH0_500, tau_500, beta_500)
    rmse = np.sqrt(np.mean(np.concatenate([p50 - dh50, p500 - dh500]) ** 2))
    r2_50 = 1 - np.sum((p50 - dh50) ** 2) / np.sum((dh50 - np.mean(dh50)) ** 2)
    r2_500 = 1 - np.sum((p500 - dh500) ** 2) / np.sum((dh500 - np.mean(dh500)) ** 2)

    print(f"\n{'='*55}")
    print(f"TWO-STEP KWW PHENOM. FIT")
    print(f"{'='*55}")
    print(f"  H_max  = {H_max:.4f} J/g")
    print(f"  50s:   dH0={dH0_50:.4f}  tau={tau_50:.0f}s  beta={beta_50:.4f}")
    print(f"  500s:  dH0={dH0_500:.4f}  tau={tau_500:.0f}s  beta={beta_500:.4f}")
    print(f"  RMSE   = {rmse:.4f} J/g")
    print(f"  R²     = 50s:{r2_50:.4f}  500s:{r2_500:.4f}")
    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'Fit50s':>8s}  {'Res':>8s}  |  "
          f"{'Exp500s':>8s}  {'Fit500s':>8s}  {'Res':>8s}")
    for i in range(10):
        print(f"  {t50[i]:8.3f}  {dh50[i]:8.4f}  {p50[i]:8.4f}  {p50[i]-dh50[i]:8.4f}  |  "
              f"{dh500[i]:8.4f}  {p500[i]:8.4f}  {p500[i]-dh500[i]:8.4f}")

    # Save dense curve
    t_dense = np.logspace(-1, 8, 150)
    dh_d50 = kww_ts(t_dense, H_max, dH0_50, tau_50, beta_50)
    dh_d500 = kww_ts(t_dense, H_max, dH0_500, tau_500, beta_500)
    pd.DataFrame({'t_s': t_dense, 'dh_50s_Jg': dh_d50, 'dh_500s_Jg': dh_d500}).to_csv(
        os.path.join(RESULTS, 'twosteps_phenom_curve.csv'), index=False)

    # Plot
    t_plt = np.logspace(-1, 6, 600)
    k50p = kww_ts(t_plt, H_max, dH0_50, tau_50, beta_50)
    k500p = kww_ts(t_plt, H_max, dH0_500, tau_500, beta_500)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(t50, dh50, c='#1f77b4', marker='o', s=70, zorder=10, edgecolors='k', lw=0.6)
    ax.scatter(t500, dh500, c='#ff7f0e', marker='s', s=70, zorder=10, edgecolors='k', lw=0.6)
    ax.semilogx(t_plt, k50p, '#1f77b4', ls='-', lw=2.2)
    ax.semilogx(t_plt, k500p, '#ff7f0e', ls='-', lw=2.2)
    ax.axhline(y=H_max, color='red', ls=':', lw=1.2, alpha=0.5)
    ax.axvline(x=1000, color='gray', ls='--', alpha=0.25)
    ax.set_xlabel('$t_2$ (s)', fontsize=13)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=13)
    ax.set_title('Two-Step (90°C→80°C) — KWW Phenomenological Fit', fontsize=13, fontweight='bold')
    ax.set_xlim(5e-2, 1e6); ax.set_ylim(-0.3, H_max * 1.15)
    ax.grid(True, alpha=0.2)

    ins = ax.inset_axes([0.55, 0.10, 0.42, 0.35])
    tz = np.logspace(-1, 1.3, 150)
    ins.semilogx(tz, kww_ts(tz, H_max, dH0_50, tau_50, beta_50), '#1f77b4', ls='-', lw=1.5)
    ins.semilogx(tz, kww_ts(tz, H_max, dH0_500, tau_500, beta_500), '#ff7f0e', ls='-', lw=1.5)
    ins.scatter(t50[t50 < 20], dh50[t50 < 20], c='#1f77b4', marker='o', s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.scatter(t500[t500 < 20], dh500[t500 < 20], c='#ff7f0e', marker='s', s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.set_xlim(0.06, 20); ins.set_ylim(-0.05, H_max * 0.78)
    ins.set_title('Short-time (0.06–20 s)', fontsize=8.5, pad=3)
    ins.grid(True, alpha=0.2); ins.tick_params(labelsize=7)

    zx0, zx1, zy0, zy1 = 0.06, 20, -0.1, H_max * 0.78
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, lw=1.5, ec='green', fc='green', alpha=0.07, zorder=2))
    fig.add_artist(ConnectionPatch(xyA=(zx1, zy1), coordsA=ax.transData, xyB=(1, 1),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))
    fig.add_artist(ConnectionPatch(xyA=(zx0, zy0), coordsA=ax.transData, xyB=(0, 0),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))

    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Exp 50s'),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Exp 500s'),
           Line2D([0], [0], color='#1f77b4', lw=2.2, label='KWW fit')]
    ax.legend(handles=leg, fontsize=10, loc='lower right')

    ax.text(0.03, 0.97,
            f"KWW fit\n  H_max = {H_max:.3f} J/g\n"
            f"  50s:  dH₀={dH0_50:.2f}  τ={tau_50:.0f}s  β={beta_50:.4f}\n"
            f"  500s: dH₀={dH0_500:.2f}  τ={tau_500:.0f}s  β={beta_500:.4f}\n"
            f"R² 50s={r2_50:.4f}  R² 500s={r2_500:.4f}  RMSE={rmse:.4f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, ec='gray'))

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'twosteps_phenom_fit.png'), dpi=200)
    plt.close()
    print(f"\n  Saved: results/twosteps_phenom_curve.csv")
    print(f"  Saved: results/twosteps_phenom_fit.png")

    return {'H_max': H_max, 'rmse': rmse, 'r2_50': r2_50, 'r2_500': r2_500}


if __name__ == '__main__':
    fit()
