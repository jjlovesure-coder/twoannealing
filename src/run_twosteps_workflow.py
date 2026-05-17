#!/usr/bin/env python3
"""
Two-Step Down-Jump (90°C→80°C) — Full Analysis Workflow

Stages:
  1. KWW phenomenological fit
  2. TNM direct fit to experimental data
  3. Final comparison plot (Exp + KWW + TNM)
"""

import subprocess, sys, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
SCRIPTS = {
    'phenom': os.path.join(ROOT, 'src', 'twosteps_fit_phenom.py'),
    'tnm':    os.path.join(ROOT, 'src', 'twosteps_fit_tnm.py'),
}


def run_script(name, path):
    print(f"\n{'='*70}")
    print(f"  STAGE: {name}")
    print(f"{'='*70}\n")
    t0 = time.time()
    env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
    r = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
    if r.returncode != 0:
        print(f"  ERROR in {name}"); return False
    print(f"\n  OK ({time.time()-t0:.0f}s)")
    return True


def main():
    print("=" * 70)
    print("  TWO-STEP ANNEALING — FULL WORKFLOW")
    print("=" * 70)

    if not run_script("KWW Phenom. Fit", SCRIPTS['phenom']):
        sys.exit(1)
    if not run_script("TNM Direct Fit", SCRIPTS['tnm']):
        sys.exit(1)

    # ── Stage 3: Final comparison plot ──
    print(f"\n{'='*70}")
    print(f"  STAGE: Final Comparison Plot")
    print(f"{'='*70}\n")
    t0 = time.time()

    R = 8.314
    T1, T2 = 363.15, 353.15
    T1_50S, T1_500S = 49.98, 499.98

    df = pd.read_csv(os.path.join(RESULTS, 'enthalpy', 'enthalpy_twosteps.csv'))
    g50 = df[df['T1_group'] == '50s']; g500 = df[df['T1_group'] == '500s']
    t50e, d50e = g50['T2_hold_s'].values, g50['delta_H_J_per_g'].values
    t500e, d500e = g500['T2_hold_s'].values, g500['delta_H_J_per_g'].values

    df_phe = pd.read_csv(os.path.join(RESULTS, 'twosteps_phenom_curve.csv'))
    t_phe, d50_phe, d500_phe = df_phe['t_s'].values, df_phe['dh_50s_Jg'].values, df_phe['dh_500s_Jg'].values

    # Load TNM params
    df_tnm = pd.read_csv(os.path.join(RESULTS, 'tnm', 'tnm_twosteps_params.csv'))
    logA = df_tnm['logA'].values[0]
    Hs = df_tnm['H_star_kJmol'].values[0] * 1000
    x = df_tnm['x'].values[0]
    b = df_tnm['beta'].values[0]
    T0 = df_tnm['T0_K'].values[0]
    sc = df_tnm['scale_Jg'].values[0]
    r2_50 = df_tnm['r2_50s'].values[0]
    r2_500 = df_tnm['r2_500s'].values[0]
    rmse = df_tnm['rmse'].values[0] if 'rmse' in df_tnm.columns else 0

    class TNM:
        def __init__(s, A, H, x, b, T0): s.A, s.H, s.x, s.b, s.T0 = A, H, x, b, T0
        def tau(s, T, Tf): return s.A * np.exp(np.clip(s.x*s.H/(R*T) + (1-s.x)*s.H/(R*Tf), -50, 80))
        def hold(s, T, t, Tf0):
            if t <= 0: return Tf0
            ts = np.unique(np.round(np.logspace(np.log10(t/100), np.log10(t), 12), 10))
            S, Tf, tp = 0.0, Tf0, 0.0
            for ti in ts: S += (ti-tp)/s.tau(T, Tf); tp = ti; Tf = T + (Tf0-T)*np.exp(-(S**s.b))
            return Tf
        def sim(s, t1, t2):
            return np.clip((s.T0 - s.hold(T2, t2, s.hold(T1, t1, s.T0))) / max(s.T0 - T2, 1.0), -1, 2)

    tnm = TNM(A=10**logA, H=Hs, x=x, b=b, T0=T0)

    def sv(t2s, t1h):
        return sc * np.array([tnm.sim(t1h, t) for t in t2s])

    # Plot
    t_plt = np.logspace(-1, 6, 400)
    tnm50 = sv(t_plt, T1_50S); tnm500 = sv(t_plt, T1_500S)
    kww50 = np.interp(t_plt, t_phe, d50_phe); kww500 = np.interp(t_plt, t_phe, d500_phe)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(t50e, d50e, c='#1f77b4', marker='o', s=70, zorder=10, edgecolors='k', lw=0.6, label='Exp 50s')
    ax.scatter(t500e, d500e, c='#ff7f0e', marker='s', s=70, zorder=10, edgecolors='k', lw=0.6, label='Exp 500s')
    ax.semilogx(t_plt, kww50, '#1f77b4', ls='--', lw=1.5, alpha=0.45, label='KWW 50s')
    ax.semilogx(t_plt, kww500, '#ff7f0e', ls='--', lw=1.5, alpha=0.45, label='KWW 500s')
    ax.semilogx(t_plt, tnm50, '#1f77b4', ls='-', lw=2.2, label='TNM 50s')
    ax.semilogx(t_plt, tnm500, '#ff7f0e', ls='-', lw=2.2, label='TNM 500s')
    ax.axhline(y=sc, color='red', ls=':', lw=1.2, alpha=0.5)
    ax.axvline(x=1000, color='gray', ls='--', alpha=0.25)
    ax.set_xlabel('$t_2$ (s)', fontsize=13)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=13)
    ax.set_title('Two-Step Down-Jump (90°C→80°C): TNM vs KWW vs Experiment',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(5e-2, 1e6); ax.set_ylim(-0.3, sc * 1.15)
    ax.grid(True, alpha=0.2)

    ins = ax.inset_axes([0.55, 0.10, 0.42, 0.35])
    tz = np.logspace(-1, 1.3, 100)
    ins.semilogx(tz, sv(tz, T1_50S), '#1f77b4', ls='-', lw=1.5)
    ins.semilogx(tz, sv(tz, T1_500S), '#ff7f0e', ls='-', lw=1.5)
    ins.scatter(t50e[t50e < 20], d50e[t50e < 20], c='#1f77b4', marker='o', s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.scatter(t500e[t500e < 20], d500e[t500e < 20], c='#ff7f0e', marker='s', s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.set_xlim(0.06, 20); ins.set_ylim(-0.05, sc * 0.75)
    ins.set_title('Short-time (0.06–20 s)', fontsize=8.5, pad=3)
    ins.grid(True, alpha=0.2); ins.tick_params(labelsize=7)

    zx0, zx1, zy0, zy1 = 0.06, 20, -0.1, sc * 0.75
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, lw=1.5, ec='green', fc='green', alpha=0.07, zorder=2))
    fig.add_artist(ConnectionPatch(xyA=(zx1, zy1), coordsA=ax.transData, xyB=(1, 1),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))
    fig.add_artist(ConnectionPatch(xyA=(zx0, zy0), coordsA=ax.transData, xyB=(0, 0),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))

    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Exp 50s'),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Exp 500s'),
           Line2D([0], [0], color='#1f77b4', lw=2.2, label='TNM model'),
           Line2D([0], [0], color='#1f77b4', lw=1.5, ls='--', alpha=0.45, label='KWW fit')]
    ax.legend(handles=leg, fontsize=10, loc='lower right')

    ax.text(0.03, 0.97,
            f"TNM parameters\n  log A = {logA:.2f}\n  H* = {Hs/1000:.1f} kJ/mol\n"
            f"  x = {x:.3f}\n  β = {b:.3f}\n  T₀ = {T0:.1f} K ({T0-273.15:.0f}°C)\n"
            f"  ΔH_max = {sc:.2f} J/g\n"
            f"R² 50s = {r2_50:.4f}  R² 500s = {r2_500:.4f}  RMSE = {rmse:.4f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, ec='gray'))

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'twosteps_full_comparison.png'), dpi=200)
    plt.close()

    k50p = np.interp(t50e, t_phe, d50_phe)
    k500p = np.interp(t500e, t_phe, d500_phe)
    tnm50e = sv(t50e, T1_50S); tnm500e = sv(t500e, T1_500S)
    df_comp = pd.DataFrame({
        't_s': np.concatenate([t50e, t500e]),
        'group': ['50s'] * 10 + ['500s'] * 10,
        'exp_Jg': np.concatenate([d50e, d500e]),
        'kww_Jg': np.concatenate([k50p, k500p]),
        'tnm_Jg': np.concatenate([tnm50e, tnm500e]),
    })
    df_comp.to_csv(os.path.join(RESULTS, 'twosteps_comparison_table.csv'), index=False)

    print(f"  OK ({time.time() - t0:.0f}s)")
    print(f"\n{'='*70}")
    print(f"  WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"  Outputs:")
    print(f"    results/twosteps_phenom_curve.csv")
    print(f"    results/twosteps_phenom_fit.png")
    print(f"    results/tnm/tnm_twosteps_params.csv")
    print(f"    results/twosteps_full_comparison.png")
    print(f"    results/twosteps_comparison_table.csv")
    print(f"\n  TNM: logA={logA:.2f} H*={Hs/1000:.1f}kJ x={x:.3f} β={b:.3f} "
          f"T0={T0:.1f}K ΔHmax={sc:.2f}")
    print(f"  R²: 50s={r2_50:.4f}  500s={r2_500:.4f}  RMSE={rmse:.4f}")


if __name__ == '__main__':
    main()
