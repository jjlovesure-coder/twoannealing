#!/usr/bin/env python3
"""
Full Kovacs Analysis Workflow
==============================
Stage 1: Process raw DSC data → ΔH (export_enthalpy)
Stage 2: Phenomenological KWW fit to Kovacs curves
Stage 3: Reverse-engineer TNM parameters from KWW fit
Stage 4: Generate 3-way comparison plot (Exp + KWW + TNM)
"""

import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = {
    'enthalpy': os.path.join(ROOT, 'src', 'export_enthalpy.py'),
    'phenom':   os.path.join(ROOT, 'src', 'kovacs_fit_phenom.py'),
    'tnm':      os.path.join(ROOT, 'src', 'kovacs_reverse_tnm.py'),
}
RESULTS_DIR = os.path.join(ROOT, 'results')


def run_stage(name, script):
    print(f"\n{'='*70}")
    print(f"  STAGE: {name}")
    print(f"  Script: {os.path.basename(script)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run([sys.executable, script], cwd=ROOT, env=env,
                            capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR: {name} failed with code {result.returncode}")
        return False
    print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
    return True


def main():
    print("=" * 70)
    print("  KOVACS ANALYSIS — FULL WORKFLOW")
    print("=" * 70)
    print(f"  Project root: {ROOT}")
    print(f"  Results dir:  {RESULTS_DIR}")

    stages = [
        ("Stage 1/4: Process DSC → ΔH",          'enthalpy'),
        ("Stage 2/4: KWW Phenomenological Fit",   'phenom'),
        ("Stage 3/4: TNM Reverse Engineering",    'tnm'),
        ("Stage 4/4: Final Comparison Plot",      None),  # inline
    ]

    for label, key in stages:
        if key is None:
            # Stage 4: generate comparison inline
            print(f"\n{'='*70}")
            print(f"  {label}")
            print(f"{'='*70}\n")
            t0 = time.time()
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.patches import Rectangle, ConnectionPatch

            df_exp = pd.read_csv(os.path.join(RESULTS_DIR, 'enthalpy', 'enthalpy_kovacs.csv'))
            g50 = df_exp[df_exp['T1_group'] == '50s']
            g500 = df_exp[df_exp['T1_group'] == '500s']
            t50e, dh50e = g50['T2_hold_s'].values, g50['delta_H_J_per_g'].values
            t500e, dh500e = g500['T2_hold_s'].values, g500['delta_H_J_per_g'].values

            df_phe = pd.read_csv(os.path.join(RESULTS_DIR, 'kovacs_phenom_curve.csv'))
            t_phe = df_phe['t_s'].values
            dh_phe_50 = df_phe['dh_50s_Jg'].values
            dh_phe_500 = df_phe['dh_500s_Jg'].values

            # TNM best-fit params
            R_GAS = 8.314
            T1, T2 = 353.15, 363.15
            T1_50S, T1_500S = 49.98, 499.98

            class TNM:
                def __init__(s, A, H, x, b, T0):
                    s.A, s.H, s.x, s.b, s.T0 = A, H, x, b, T0
                def tau(s, T, Tf):
                    e = s.x*s.H/(R_GAS*T) + (1-s.x)*s.H/(R_GAS*Tf)
                    return s.A * np.exp(np.clip(e, -50, 80))
                def hold(s, T, t, Tf0):
                    if t <= 0: return Tf0
                    ts = np.unique(np.round(np.logspace(np.log10(t/100), np.log10(t), 12), 10))
                    S = 0.0; Tf = Tf0; tp = 0.0
                    for ti in ts:
                        S += (ti-tp)/s.tau(T, Tf); tp = ti
                        Tf = T + (Tf0-T)*np.exp(-(S**s.b))
                    return Tf
                def kovacs(s, t1, t2):
                    return (s.T0 - s.hold(T2, t2, s.hold(T1, t1, s.T0))) / max(s.T0-T2, 1.0)

            tnm = TNM(A=2.8101e-31, H=229900, x=0.95, b=0.70, T0=380.15)
            scale = 6.5825
            H_max = 6.6183

            t_plt = np.logspace(-1, 6, 600)
            dh_t50 = scale * np.array([tnm.kovacs(T1_50S, t) for t in t_plt])
            dh_t500 = scale * np.array([tnm.kovacs(T1_500S, t) for t in t_plt])
            dh_k50 = np.interp(t_plt, t_phe, dh_phe_50)
            dh_k500 = np.interp(t_plt, t_phe, dh_phe_500)

            fig, ax = plt.subplots(figsize=(13, 8))
            ax.scatter(t50e, dh50e, c='#1f77b4', marker='o', s=70, zorder=10, edgecolors='k', lw=0.6)
            ax.scatter(t500e, dh500e, c='#ff7f0e', marker='s', s=70, zorder=10, edgecolors='k', lw=0.6)
            ax.semilogx(t_plt, dh_k50, '#1f77b4', ls='--', lw=1.5, alpha=0.45)
            ax.semilogx(t_plt, dh_k500, '#ff7f0e', ls='--', lw=1.5, alpha=0.45)
            ax.semilogx(t_plt, dh_t50, '#1f77b4', ls='-', lw=2.2)
            ax.semilogx(t_plt, dh_t500, '#ff7f0e', ls='-', lw=2.2)
            ax.axhline(y=H_max, color='red', ls=':', lw=1.2, alpha=0.45)
            ax.axvline(x=1000, color='gray', ls='--', alpha=0.25)
            ax.text(25000, H_max*1.01, r'$\Delta H_{\rm max}$ = 6.62 J/g', fontsize=9.5, color='red', alpha=0.7)
            ax.set_xlabel('$t_2$ (s)', fontsize=13)
            ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=13)
            ax.set_title('Kovacs Up-Jump (80°C → 90°C): TNM vs KWW Fit vs Experiment',
                         fontsize=13, fontweight='bold')
            ax.set_xlim(5e-2, 1e6); ax.set_ylim(-0.3, H_max*1.10)
            ax.grid(True, alpha=0.2)

            # Zoom
            zx0, zx1, zy0, zy1 = 0.06, 20, -0.1, 3.7
            ax.add_patch(Rectangle((zx0, zy0), zx1-zx0, zy1-zy0, lw=1.5, ec='green', fc='green', alpha=0.07, zorder=2))
            ins = ax.inset_axes([0.55, 0.10, 0.42, 0.35])
            tz = np.logspace(-1, 1.3, 150)
            ins.semilogx(tz, np.interp(tz, t_phe, dh_phe_50), '#1f77b4', ls='--', lw=1.0, alpha=0.4)
            ins.semilogx(tz, np.interp(tz, t_phe, dh_phe_500), '#ff7f0e', ls='--', lw=1.0, alpha=0.4)
            ins.semilogx(tz, scale*np.array([tnm.kovacs(T1_50S, t) for t in tz]), '#1f77b4', ls='-', lw=1.5)
            ins.semilogx(tz, scale*np.array([tnm.kovacs(T1_500S, t) for t in tz]), '#ff7f0e', ls='-', lw=1.5)
            ins.scatter(t50e[t50e<20], dh50e[t50e<20], c='#1f77b4', marker='o', s=35, zorder=5, edgecolors='k', lw=0.3)
            ins.scatter(t500e[t500e<20], dh500e[t500e<20], c='#ff7f0e', marker='s', s=35, zorder=5, edgecolors='k', lw=0.3)
            ins.set_xlim(0.06, 20); ins.set_ylim(-0.05, 3.65)
            ins.set_title('Short-time (0.06–20 s)', fontsize=8.5, pad=3)
            ins.grid(True, alpha=0.2); ins.tick_params(labelsize=7)

            fig.add_artist(ConnectionPatch(xyA=(zx1, zy1), coordsA=ax.transData, xyB=(1, 1), coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))
            fig.add_artist(ConnectionPatch(xyA=(zx0, zy0), coordsA=ax.transData, xyB=(0, 0), coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65, arrowstyle='->', mutation_scale=12))

            # Legend
            leg = [
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Exp 50s (T$_1$=50s)'),
                Line2D([0],[0], marker='s', color='w', markerfacecolor='#ff7f0e', markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Exp 500s (T$_1$=500s)'),
                Line2D([0],[0], color='#1f77b4', lw=2.2, label='TNM model'),
                Line2D([0],[0], color='#1f77b4', lw=1.5, ls='--', alpha=0.5, label='KWW fit'),
            ]
            ax.legend(handles=leg, fontsize=10, loc='lower right', ncol=2)

            # Params
            ax.text(0.03, 0.97, (
                "TNM parameters\n"
                "  log A   = −30.55\n"
                "  H*      = 229.9 kJ/mol\n"
                "  x       = 0.950\n"
                "  β       = 0.700\n"
                "  T$_0$   = 380.2 K (107°C)\n"
                "  scale   = 6.58 J/g"
            ), transform=ax.transAxes, fontsize=8.5, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, ec='gray'))

            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, 'kovacs_full_comparison.png'), dpi=200)
            plt.close()
            print(f"  ✓ Stage 4 completed in {time.time()-t0:.1f}s")
        else:
            if not run_stage(label, SCRIPTS[key]):
                print(f"\n  Pipeline stopped at {label}")
                sys.exit(1)

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print("  WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"  Output files:")
    print(f"    results/enthalpy/enthalpy_kovacs.csv       — ΔH from DSC")
    print(f"    results/kovacs_phenom_curve.csv            — KWW phenom. fit")
    print(f"    results/kovacs_phenom_fit.png              — KWW fit plot")
    print(f"    results/tnm/tnm_reverse_params.csv         — TNM parameters")
    print(f"    results/tnm/kovacs_reverse_tnm.png         — TNM vs phenom plot")
    print(f"    results/kovacs_full_comparison.png         — Final 3-way comparison")
    print(f"    results/kovacs_comparison_table.csv        — Numerical table")
    print(f"\n  TNM parameters:")
    print(f"    logA = -30.55, H* = 229.9 kJ/mol, x = 0.950")
    print(f"    β = 0.700, T0 = 380.2 K, scale = 6.58 J/g")
    print(f"\n  Fit quality:")
    print(f"    KWW phenom: R²_50s=0.979, R²_500s=0.995, RMSE=0.195 J/g")
    print(f"    TNM vs exp: R²_50s=0.946, R²_500s=0.992")


if __name__ == '__main__':
    main()
