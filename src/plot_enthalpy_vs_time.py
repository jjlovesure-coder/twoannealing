#!/usr/bin/env python3
"""
ΔH vs annealing time t₂: corrected integration bounds vs old (40°C) vs production (70°C).

Each experiment shows two groups (50s / 500s) on semilog ΔH(t₂) axes.
Uses actual experimental t₂ values from the DSC program headers.
"""

import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results', 'enthalpy')

sys.path.insert(0, os.path.join(ROOT, 'src'))
from recompute_enthalpy import (
    load_empty, load_dsc_experiment, detect_heating_ramps,
    process_experiment, CONV_JG
)

interp_empty = load_empty()

def integrals_to_dH(integrals, ref_integral):
    return (np.array(integrals) - ref_integral) * CONV_JG


def get_t2_values(exp_name):
    """Load t2 hold times from existing enthalpy CSV (40°C version has the correct times)."""
    if 'kovacs_80-90' in exp_name.lower() or 'new' in exp_name.lower():
        df = pd.read_csv(os.path.join(RESULTS, 'enthalpy_kovacs_80-90C_40C.csv'))
    elif 'kovacs' in exp_name.lower():
        df = pd.read_csv(os.path.join(RESULTS, 'enthalpy_kovacs_40C.csv'))
    elif 'two' in exp_name.lower():
        df = pd.read_csv(os.path.join(RESULTS, 'enthalpy_twosteps_40C.csv'))
    elif '50' in exp_name:
        df = pd.read_csv(os.path.join(RESULTS, 'enthalpy_onestep_50C_40C.csv'))
    elif '70' in exp_name:
        df = pd.read_csv(os.path.join(RESULTS, 'enthalpy_onestep_70C_40C.csv'))
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")
    return df['T2_hold_s'].values


def compute_dH_curves(data_file, sheet, t_low, exp_name, ref_file=None, ref_sheet=None):
    """Compute dH at given integration bound. Returns t2, dH arrays."""
    res, _, _ = process_experiment(data_file, sheet, interp_empty, t_low)
    integrals = np.array([r['integral'] for r in res])

    if ref_file:
        ref_res, _, _ = process_experiment(ref_file, ref_sheet, interp_empty, t_low)
        ref_int = ref_res[0]['integral']
    else:
        ref_int = integrals[0]

    dH = integrals_to_dH(integrals, ref_int)
    t2 = get_t2_values(exp_name)

    if len(t2) != len(dH):
        t2 = t2[:len(dH)]

    return t2, dH


# ═══════════════════════════════════════════════════════════════════════
# Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

experiments = [
    {
        'name': 'Kovacs (old, 80→90°C)',
        'file': 'pskovacs.xlsx', 'sheet': 'PS-kovacs-01',
        't_correct': 30, 't_old': 40, 't_prod': 70,
        'T1': 80, 'T2': 90,
    },
    {
        'name': 'Two-Step (90→80°C)',
        'file': 'twosteps.xlsx', 'sheet': 'PS-02',
        't_correct': 30, 't_old': 40, 't_prod': 70,
        'T1': 90, 'T2': 80,
    },
    {
        'name': 'Kovacs New (80→90°C)',
        'file': 'ps-02.xlsx', 'sheet': 'ps-02',
        't_correct': 10, 't_old': 40, 't_prod': 40,
        'ref_file': 'ps-ref-02.xlsx', 'ref_sheet': 'ps-ref-02',
        'T1': 80, 'T2': 90,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Plot: one figure per experiment, ΔH vs t₂
# ═══════════════════════════════════════════════════════════════════════

for exp in experiments:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    name = exp['name']
    f = os.path.join(DATA, exp['file'])
    s = exp['sheet']
    rf = os.path.join(DATA, exp['ref_file']) if 'ref_file' in exp else None
    rs = exp['ref_sheet'] if 'ref_file' in exp else None

    t2_c, dH_c = compute_dH_curves(f, s, exp['t_correct'], exp['name'], rf, rs)
    t2_o, dH_o = compute_dH_curves(f, s, exp['t_old'], exp['name'], rf, rs)
    t2_p, dH_p = compute_dH_curves(f, s, exp['t_prod'], exp['name'], rf, rs)

    # Split into 50s and 500s groups (first half = 50s, second half = 500s)
    n = len(dH_c)
    n_half = n // 2

    # 50s group
    ax.semilogx(t2_c[:n_half], dH_c[:n_half], 'o-', c='#2ca02c', ms=10, lw=2.2,
                label=f'{exp["t_correct"]}°C (correct) 50s')
    ax.semilogx(t2_o[:n_half], dH_o[:n_half], 's--', c='#ff7f0e', ms=8, lw=1.5,
                label=f'{exp["t_old"]}°C (old) 50s')
    ax.semilogx(t2_p[:n_half], dH_p[:n_half], '^:', c='#d62728', ms=7, lw=1.2, alpha=0.7,
                label=f'{exp["t_prod"]}°C (prod) 50s')

    # 500s group
    ax.semilogx(t2_c[n_half:], dH_c[n_half:], 'o-', c='#2ca02c', ms=12, lw=2.5,
                markerfacecolor='none', markeredgewidth=2,
                label=f'{exp["t_correct"]}°C (correct) 500s')
    ax.semilogx(t2_o[n_half:], dH_o[n_half:], 's--', c='#ff7f0e', ms=9, lw=1.8,
                markerfacecolor='none', markeredgewidth=1.5,
                label=f'{exp["t_old"]}°C (old) 500s')
    ax.semilogx(t2_p[n_half:], dH_p[n_half:], '^:', c='#d62728', ms=8, lw=1.4, alpha=0.7,
                markerfacecolor='none', markeredgewidth=1,
                label=f'{exp["t_prod"]}°C (prod) 500s')

    # Mark T1 group separation
    yl = ax.get_ylim()
    ax.axvline(x=500, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(50, yl[1] * 0.92, '50s', ha='center', fontsize=10, fontweight='bold', color='gray')
    ax.text(5000, yl[1] * 0.92, '500s', ha='center', fontsize=10, fontweight='bold', color='gray')

    ax.set_xlabel(r'Annealing time $t_2$ (s)', fontsize=13)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=13)
    ax.set_title(f'{name}\n$T_1$={exp["T1"]}°C, $T_2$={exp["T2"]}°C  |  '
                 f'Integration: {exp["t_correct"]}°C(correct) vs {exp["t_old"]}°C(old) vs {exp["t_prod"]}°C(prod)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.05, 2e3)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, ncol=2, loc='lower right')

    # Annotation box
    max_diff = np.max(np.abs(dH_c - dH_o))
    max_diff_p = np.max(np.abs(dH_c - dH_p))
    mean_dH = np.mean(np.abs(dH_c[1:]))
    ax.text(0.02, 0.97,
            f'Correct ({exp["t_correct"]}°C) vs Old ({exp["t_old"]}°C):\n'
            f'  max|Δ| = {max_diff:.4f} J/g  ({max_diff/mean_dH*100:.1f}%)\n'
            f'Correct ({exp["t_correct"]}°C) vs Prod ({exp["t_prod"]}°C):\n'
            f'  max|Δ| = {max_diff_p:.4f} J/g  ({max_diff_p/mean_dH*100:.1f}%)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, ec='gray'))

    fig.tight_layout()
    safe = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg')
    out = os.path.join(RESULTS, f'enthalpy_correction_{safe}.png')
    fig.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Combined summary figure
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
fig.suptitle('Enthalpy vs Annealing Time: Correct vs Old Integration Bounds', fontsize=13, fontweight='bold')

for idx, exp in enumerate(experiments):
    ax = axes[idx]
    f = os.path.join(DATA, exp['file'])
    s = exp['sheet']
    rf = os.path.join(DATA, exp['ref_file']) if 'ref_file' in exp else None
    rs = exp['ref_sheet'] if 'ref_file' in exp else None

    t2_c, dH_c = compute_dH_curves(f, s, exp['t_correct'], exp['name'], rf, rs)
    t2_o, dH_o = compute_dH_curves(f, s, exp['t_old'], exp['name'], rf, rs)

    n = len(dH_c)
    n_half = n // 2

    # Note: dH[n_half:] needs t2[n_half:] as corresponding times
    if n_half > 0:
        # 500s group data starts at index n_half
        ax.semilogx(t2_c[n_half:], dH_c[n_half:], 'o-', c='#2ca02c', ms=7, lw=2,
                    markerfacecolor='none', markeredgewidth=1.5, label=f'{exp["t_correct"]}°C 500s')
        ax.semilogx(t2_o[n_half:], dH_o[n_half:], 's--', c='#ff7f0e', ms=6, lw=1.5,
                    markerfacecolor='none', markeredgewidth=1, label=f'{exp["t_old"]}°C 500s')

    ax.semilogx(t2_c[:n_half], dH_c[:n_half], 'o-', c='#2ca02c', ms=8, lw=2.2,
                label=f'{exp["t_correct"]}°C 50s')
    ax.semilogx(t2_o[:n_half], dH_o[:n_half], 's--', c='#ff7f0e', ms=7, lw=1.5,
                label=f'{exp["t_old"]}°C 50s')

    ax.set_xlabel(r'$t_2$ (s)', fontsize=11)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=11)
    ax.set_title(exp['name'], fontsize=10, fontweight='bold')
    ax.set_xlim(0.05, 2e3)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc='lower right')

    max_diff = np.max(np.abs(dH_c - dH_o))
    mean_dH = np.mean(np.abs(dH_c[1:]))
    ax.text(0.02, 0.97,
            f'max|Δ| = {max_diff:.3f} J/g ({max_diff/mean_dH*100:.1f}%)',
            transform=ax.transAxes, fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, ec='gray'))

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_sum = os.path.join(RESULTS, 'enthalpy_correction_all_experiments.png')
fig.savefig(out_sum, dpi=200)
plt.close()
print(f"Saved: {out_sum}")

print("\nDone. All ΔH vs t₂ comparison plots saved.")
