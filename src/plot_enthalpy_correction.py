#!/usr/bin/env python3
"""
Plot before/after comparison of enthalpy correction (integration bounds fix).

Generates per-experiment and summary figures comparing:
  - Corrected bound (10°C or 30°C, matching actual ramp start)
  - Old bound (40°C, for fair same-experiment comparison)
  - Production bound (70°C, the currently-used symlinked data)

Core insight: empty-crucible baseline cancels in difference → integration lower
bound should match the actual temperature ramp start for each experiment.
"""

import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'enthalpy')

# ── Load data ──

def load_csv(name):
    p = os.path.join(RESULTS, name)
    if os.path.islink(p):
        p = os.path.join(RESULTS, os.readlink(p))
    return pd.read_csv(p)

# Per-experiment data (all computed by recompute_enthalpy.py)
csvs = {
    'Kovacs 80→90°C (new)': {
        'correct': ('10°C', pd.read_csv(os.path.join(RESULTS, 'enthalpy_kovacs_80-90C_10C.csv'))),
        'old40':   ('40°C', pd.read_csv(os.path.join(RESULTS, 'enthalpy_kovacs_80-90C_40C.csv'))),
    },
}

# For other experiments, load from recompute output (30°C vs 40°C) plus 70°C prod
for name, f30, f40, f70 in [
    ('Kovacs (old)',      'enthalpy_kovacs_40C.csv',         'enthalpy_kovacs_40C.csv',         'enthalpy_kovacs_70C.csv'),
    ('Two-Step',          'enthalpy_twosteps_40C.csv',       'enthalpy_twosteps_40C.csv',       'enthalpy_twosteps_70C.csv'),
    ('One-Step 50°C',     'enthalpy_onestep_50C_40C.csv',    'enthalpy_onestep_50C_40C.csv',    'enthalpy_onestep_50C_70C.csv'),
    ('One-Step 70°C',     'enthalpy_onestep_70C_40C.csv',    'enthalpy_onestep_70C_40C.csv',    'enthalpy_onestep_70C_70C.csv'),
]:
    # We need to recompute 30°C vs 40°C from the recompute run
    pass


# We'll compute from scratch for reliable comparison
sys.path.insert(0, os.path.join(ROOT, 'src'))
from recompute_enthalpy import (
    load_empty, load_dsc_experiment, detect_heating_ramps,
    find_liquid_onset, compute_integral, process_experiment,
    M_SAMPLE, HEATING_RATE, BETA, CONV_JG
)

DATA = os.path.join(ROOT, 'data')

interp_empty = load_empty()

def integrals_to_dH(integrals, ref_integral):
    return (np.array(integrals) - ref_integral) * CONV_JG

def process_and_compare(data_file, sheet, t_int_low_correct, t_int_low_old, t_int_low_prod):
    """Process one experiment at 3 integration bounds."""
    results_c, _, _ = process_experiment(data_file, sheet, interp_empty, t_int_low_correct)
    int_c = [r['integral'] for r in results_c]

    results_o, _, _ = process_experiment(data_file, sheet, interp_empty, t_int_low_old)
    int_o = [r['integral'] for r in results_o]

    results_p, _, _ = process_experiment(data_file, sheet, interp_empty, t_int_low_prod)
    int_p = [r['integral'] for r in results_p]

    dH_c = integrals_to_dH(int_c, int_c[0])
    dH_o = integrals_to_dH(int_o, int_o[0])
    dH_p = integrals_to_dH(int_p, int_p[0])

    T_on = np.array([r['T_onset_C'] for r in results_c])

    return dH_c, dH_o, dH_p, T_on


# ═══════════════════════════════════════════════════════════════════════
# ALL EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════

experiments = [
    {
        'name': 'Kovacs 80→90°C (new, ps-02)',
        'file': 'ps-02.xlsx', 'sheet': 'ps-02',
        't_correct': 10, 't_old': 40, 't_prod': 40,
        'ref_file': 'ps-ref-02.xlsx', 'ref_sheet': 'ps-ref-02',
    },
    {
        'name': 'Kovacs (old, pskovacs)',
        'file': 'pskovacs.xlsx', 'sheet': 'PS-kovacs-01',
        't_correct': 30, 't_old': 40, 't_prod': 70,
    },
    {
        'name': 'Two-Step (twosteps)',
        'file': 'twosteps.xlsx', 'sheet': 'PS-02',
        't_correct': 30, 't_old': 40, 't_prod': 70,
    },
    {
        'name': 'One-Step 50°C',
        'file': 'PS-onestep-01.xlsx', 'sheet': 'PS-onestep-01',
        't_correct': 30, 't_old': 40, 't_prod': 70,
    },
    {
        'name': 'One-Step 70°C',
        'file': 'PS-onestep-02.xlsx', 'sheet': 'PS-onestep-02',
        't_correct': 30, 't_old': 40, 't_prod': 70,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Per-experiment dH curves (corrected vs old)
# ═══════════════════════════════════════════════════════════════════════

n_exp = len(experiments)
fig, axes = plt.subplots(2, 3, figsize=(21, 13))
axes_flat = axes.flatten()
fig.suptitle('Enthalpy Correction: Correct Integration Bounds vs Old', fontsize=14, fontweight='bold')

for idx, exp in enumerate(experiments):
    ax = axes_flat[idx]
    name = exp['name']
    f = os.path.join(DATA, exp['file'])
    s = exp['sheet']

    if 'ref_file' in exp and exp['ref_file']:
        # External reference
        ref_res_c, _, _ = process_experiment(
            os.path.join(DATA, exp['ref_file']), exp['ref_sheet'],
            interp_empty, exp['t_correct'])
        ref_res_o, _, _ = process_experiment(
            os.path.join(DATA, exp['ref_file']), exp['ref_sheet'],
            interp_empty, exp['t_old'])

        res_c, _, _ = process_experiment(f, s, interp_empty, exp['t_correct'])
        res_o, _, _ = process_experiment(f, s, interp_empty, exp['t_old'])
        res_p, _, _ = process_experiment(f, s, interp_empty, exp['t_prod'])

        int_c = [r['integral'] for r in res_c]
        int_o = [r['integral'] for r in res_o]
        int_p = [r['integral'] for r in res_p]

        dH_c = integrals_to_dH(int_c, ref_res_c[0]['integral'])
        dH_o = integrals_to_dH(int_o, ref_res_o[0]['integral'])
        dH_p = integrals_to_dH(int_p, ref_res_o[0]['integral'])
    else:
        dH_c, dH_o, dH_p, _ = process_and_compare(
            f, s, exp['t_correct'], exp['t_old'], exp['t_prod'])

    x = np.arange(1, len(dH_c) + 1)
    w = 0.25

    ax.bar(x - w, dH_c, w, color='#2ca02c', alpha=0.85, label=f'{exp["t_correct"]}°C (correct)')
    ax.bar(x,     dH_o, w, color='#ff7f0e', alpha=0.7, label=f'{exp["t_old"]}°C (old)')
    ax.bar(x + w, dH_p, w, color='#d62728', alpha=0.5, label=f'{exp["t_prod"]}°C (prod)')

    diff_co = dH_c - dH_o
    max_diff = np.max(np.abs(diff_co))
    mean_dH = np.mean(np.abs(dH_c[1:])) if len(dH_c) > 1 else 1

    ax.set_title(f'{name}\nmax|Δ|={max_diff:.3f} J/g ({max_diff/mean_dH*100:.1f}% rel)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Ramp #', fontsize=9)
    ax.set_ylabel('ΔH (J/g)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.tick_params(labelsize=8)

# Remove extra subplot
if n_exp < 6:
    axes_flat[5].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out1 = os.path.join(RESULTS, 'enthalpy_correction_comparison.png')
fig.savefig(out1, dpi=200)
plt.close()
print(f"Saved: {out1}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Detailed – Kovacs old (correct vs old) with dH difference
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Kovacs (old): Integration Bound Correction — 30°C vs 40°C vs 70°C', fontsize=13, fontweight='bold')

dH_30, dH_40, dH_70, T_on = process_and_compare(
    os.path.join(DATA, 'pskovacs.xlsx'), 'PS-kovacs-01', 30, 40, 70)

n_ramps = len(dH_30)
x = np.arange(n_ramps)

# Panel 1: Direct comparison
ax = axes[0]
w = 0.25
ax.bar(x - w, dH_30, w, color='#2ca02c', alpha=0.9, label='30°C (correct)')
ax.bar(x,     dH_40, w, color='#ff7f0e', alpha=0.7, label='40°C (old)')
ax.bar(x + w, dH_70, w, color='#d62728', alpha=0.5, label='70°C (prod)')

# Annotate group separation (first 10 = 50s, next 10 = 500s)
ax.axvline(x=9.5, color='gray', ls='--', lw=1.5, alpha=0.5)
ax.text(4.5, ax.get_ylim()[1] * 0.92, '50s group', ha='center', fontsize=10, fontweight='bold', color='gray')
ax.text(14.5, ax.get_ylim()[1] * 0.92, '500s group', ha='center', fontsize=10, fontweight='bold', color='gray')

ax.set_xlabel('Ramp #', fontsize=11)
ax.set_ylabel('ΔH (J/g)', fontsize=11)
ax.set_title('Direct Comparison', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# Panel 2: Difference from corrected
ax = axes[1]
diff_40 = dH_30 - dH_40
diff_70 = dH_30 - dH_70
ax.bar(x - 0.15, diff_40, 0.3, color='#ff7f0e', alpha=0.8, label='30°C − 40°C')
ax.bar(x + 0.15, diff_70, 0.3, color='#d62728', alpha=0.6, label='30°C − 70°C')
ax.axhline(y=0, color='gray', ls='-', lw=0.8)
ax.axvline(x=9.5, color='gray', ls='--', lw=1.5, alpha=0.5)
ax.text(4.5, ax.get_ylim()[1] * 0.85, '50s', ha='center', fontsize=10, fontweight='bold', color='gray')
ax.text(14.5, ax.get_ylim()[1] * 0.85, '500s', ha='center', fontsize=10, fontweight='bold', color='gray')

ax.set_xlabel('Ramp #', fontsize=11)
ax.set_ylabel('ΔH difference (J/g)', fontsize=11)
ax.set_title(f'Error from wrong bound\nmax|Δ30-40|={np.max(np.abs(diff_40)):.3f}  max|Δ30-70|={np.max(np.abs(diff_70)):.3f} J/g',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

fig.tight_layout(rect=[0, 0, 1, 0.94])
out2 = os.path.join(RESULTS, 'enthalpy_correction_kovacs_detail.png')
fig.savefig(out2, dpi=200)
plt.close()
print(f"Saved: {out2}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: DSC raw curves with integration ranges highlighted
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('DSC Raw Curves: Integration Range Comparison', fontsize=13, fontweight='bold')

# Panel A: New Kovacs ramp #1 (ps-02, first heating) — 10°C vs 40°C
ax = axes[0, 0]
exp_data, _ = load_dsc_experiment(os.path.join(DATA, 'ps-02.xlsx'), 'ps-02')
T_all = exp_data['Temp'].values
DSC_all = exp_data['DSC'].values
t_all = exp_data['Time'].values
ramps = detect_heating_ramps(T_all, t_all)
s, e = ramps[0]
T_seg = T_all[s:e+1]
DSC_seg = DSC_all[s:e+1]
T_onset = find_liquid_onset(T_seg, DSC_seg)

ax.plot(T_seg, DSC_seg, '#1f77b4', lw=1.2, label='DSC (sample − ref)')
ax.axvline(x=10, color='#2ca02c', ls='--', lw=2, alpha=0.7, label=f'10°C (correct start)')
ax.axvline(x=40, color='#ff7f0e', ls='--', lw=2, alpha=0.7, label=f'40°C (old start)')
ax.axvline(x=70, color='#d62728', ls='--', lw=2, alpha=0.5, label=f'70°C (prod start)')
ax.axvline(x=T_onset, color='gray', ls='-', lw=1.5, alpha=0.6, label=f'T_onset={T_onset:.1f}°C')
ax.axvspan(10, 40, alpha=0.08, color='#2ca02c')
ax.axvspan(40, 70, alpha=0.08, color='#ff7f0e')

ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('DSC (µW)', fontsize=11)
ax.set_title(f'Kovacs New (ps-02): 10→180°C ramp\n'
             f'Shaded area = signal lost with higher lower bound',
             fontsize=10, fontweight='bold')
ax.set_xlim(5, T_onset + 5)
yl = ax.get_ylim()
ax.set_ylim(yl[0], yl[1] * 1.1)
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.2)

# Panel B: DSC raw (pskovacs) first heating — 30°C vs 40°C
ax = axes[0, 1]
exp_data, _ = load_dsc_experiment(os.path.join(DATA, 'pskovacs.xlsx'), 'PS-kovacs-01')
T_all = exp_data['Temp'].values
DSC_all = exp_data['DSC'].values
t_all = exp_data['Time'].values
ramps = detect_heating_ramps(T_all, t_all)
s, e = ramps[0]
T_seg = T_all[s:e+1]
DSC_seg = DSC_all[s:e+1]
T_onset = find_liquid_onset(T_seg, DSC_seg)
empty_vals = interp_empty(T_seg)

ax.plot(T_seg, DSC_seg, '#1f77b4', lw=1.2, label='DSC (sample − ref)')
ax.plot(T_seg, empty_vals, '#7f7f7f', lw=1.0, alpha=0.5, label='Empty crucible baseline')
ax.axvline(x=30, color='#2ca02c', ls='--', lw=2, alpha=0.7, label=f'30°C (correct start)')
ax.axvline(x=40, color='#ff7f0e', ls='--', lw=2, alpha=0.7, label=f'40°C (old start)')
ax.axvline(x=70, color='#d62728', ls='--', lw=2, alpha=0.5, label=f'70°C (prod start)')
ax.axvline(x=T_onset, color='gray', ls='-', lw=1.5, alpha=0.6, label=f'T_onset={T_onset:.1f}°C')
ax.axvspan(30, 40, alpha=0.08, color='#2ca02c')
ax.axvspan(40, 70, alpha=0.08, color='#ff7f0e')

ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('DSC (µW)', fontsize=11)
ax.set_title(f'Kovacs Old (pskovacs): 30→200°C ramp\n'
             f'30-40°C: DSC ≈ empty baseline → near-zero integrand\n'
             f'40-70°C: DSC > empty → significant lost signal',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.2)

# Panel C: Two-Step — DSC raw with integrand
ax = axes[1, 0]
exp_data, _ = load_dsc_experiment(os.path.join(DATA, 'twosteps.xlsx'), 'PS-02')
T_all = exp_data['Temp'].values
DSC_all = exp_data['DSC'].values
t_all = exp_data['Time'].values
ramps = detect_heating_ramps(T_all, t_all)
s, e = ramps[0]
T_seg = T_all[s:e+1]
DSC_seg = DSC_all[s:e+1]
T_onset = find_liquid_onset(T_seg, DSC_seg)

ax.plot(T_seg, DSC_seg, '#1f77b4', lw=1.2, label='DSC (sample − ref)')
ax.axvline(x=30, color='#2ca02c', ls='--', lw=2, alpha=0.7, label=f'30°C (correct)')
ax.axvline(x=40, color='#ff7f0e', ls='--', lw=2, alpha=0.7, label=f'40°C (old)')
ax.axvline(x=70, color='#d62728', ls='--', lw=2, alpha=0.5, label=f'70°C (prod)')
ax.axvline(x=T_onset, color='gray', ls='-', lw=1.5, alpha=0.6, label=f'T_onset={T_onset:.1f}°C')

ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('DSC (µW)', fontsize=11)
ax.set_title(f'Two-Step (twosteps): 30→200°C ramp', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.2)

# Panel D: Error vs ramp index (all experiments)
ax = axes[1, 1]
all_exp_names = []
all_correct = []
all_old = []
all_prod = []

for exp in experiments:
    name = exp['name']
    f = os.path.join(DATA, exp['file'])
    s = exp['sheet']

    if 'ref_file' in exp and exp['ref_file']:
        ref_c, _, _ = process_experiment(
            os.path.join(DATA, exp['ref_file']), exp['ref_sheet'],
            interp_empty, exp['t_correct'])
        ref_o, _, _ = process_experiment(
            os.path.join(DATA, exp['ref_file']), exp['ref_sheet'],
            interp_empty, exp['t_old'])

        res_c, _, _ = process_experiment(f, s, interp_empty, exp['t_correct'])
        res_o, _, _ = process_experiment(f, s, interp_empty, exp['t_old'])
        res_p, _, _ = process_experiment(f, s, interp_empty, exp['t_prod'])

        int_c = [r['integral'] for r in res_c]
        int_o = [r['integral'] for r in res_o]
        int_p = [r['integral'] for r in res_p]

        dH_c = integrals_to_dH(int_c, ref_c[0]['integral'])
        dH_o = integrals_to_dH(int_o, ref_o[0]['integral'])
        dH_p = integrals_to_dH(int_p, ref_o[0]['integral'])
    else:
        dH_c, dH_o, dH_p, _ = process_and_compare(
            f, s, exp['t_correct'], exp['t_old'], exp['t_prod'])

    diff_o = dH_c - dH_o
    diff_p = dH_c - dH_p
    for i in range(len(dH_c)):
        all_exp_names.append(f'{name[:12]}')
        all_correct.append(dH_c[i])
        all_old.append(diff_o[i])
        all_prod.append(diff_p[i])

colors_map = {'Kovacs 80→90°C': '#1f77b4', 'Kovacs (old,': '#ff7f0e',
              'Two-Step (tw': '#2ca02c', 'One-Step 50°': '#9467bd', 'One-Step 70°': '#d62728'}

x_full = np.arange(len(all_correct))
for i in range(len(x_full)):
    key = all_exp_names[i]
    for prefix, c in colors_map.items():
        if key.startswith(prefix):
            col = c
            break
    else:
        col = 'gray'
    ax.scatter(all_correct[i], all_old[i], c=col, marker='o', s=45, alpha=0.7, zorder=5,
               edgecolors='k', lw=0.3)
    ax.scatter(all_correct[i], all_prod[i], c=col, marker='s', s=30, alpha=0.4, zorder=4,
               edgecolors='k', lw=0.2)

ax.axhline(y=0, color='gray', ls='-', lw=0.8)
ax.set_xlabel('Corrected ΔH (J/g)', fontsize=11)
ax.set_ylabel('ΔH error = Corrected − Wrong (J/g)', fontsize=11)
ax.set_title('Error vs Corrected ΔH (all experiments)\n○ = old bound (40°C)   □ = prod (70°C)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.2)

# Legend
from matplotlib.lines import Line2D
leg_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=k)
           for k, c in colors_map.items()]
leg_els.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='vs old 40°C'))
leg_els.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=5, alpha=0.5, label='vs prod 70°C'))
ax.legend(handles=leg_els, fontsize=6, loc='upper left', ncol=2)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out3 = os.path.join(RESULTS, 'enthalpy_correction_dsc_integration.png')
fig.savefig(out3, dpi=200)
plt.close()
print(f"Saved: {out3}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Summary — max|Δ| per experiment
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))

# Collect summary stats
summary = []
for exp in experiments:
    f = os.path.join(DATA, exp['file'])
    s = exp['sheet']
    if 'ref_file' in exp and exp['ref_file']:
        ref_c, _, _ = process_experiment(
            os.path.join(DATA, exp['ref_file']), exp['ref_sheet'],
            interp_empty, exp['t_correct'])
        ref_o, _, _ = process_experiment(
            os.path.join(DATA, exp['ref_file']), exp['ref_sheet'],
            interp_empty, exp['t_old'])

        res_c, _, _ = process_experiment(f, s, interp_empty, exp['t_correct'])
        res_o, _, _ = process_experiment(f, s, interp_empty, exp['t_old'])
        res_p, _, _ = process_experiment(f, s, interp_empty, exp['t_prod'])

        int_c = [r['integral'] for r in res_c]
        int_o = [r['integral'] for r in res_o]
        int_p = [r['integral'] for r in res_p]

        dH_c = integrals_to_dH(int_c, ref_c[0]['integral'])
        dH_o = integrals_to_dH(int_o, ref_o[0]['integral'])
        dH_p = integrals_to_dH(int_p, ref_o[0]['integral'])
    else:
        dH_c, dH_o, dH_p, _ = process_and_compare(
            f, s, exp['t_correct'], exp['t_old'], exp['t_prod'])

    diff_o = np.abs(dH_c - dH_o)
    diff_p = np.abs(dH_c - dH_p)
    mean_dH = np.mean(np.abs(dH_c[1:]))
    summary.append({
        'name': exp['name'],
        'max_o': np.max(diff_o), 'max_p': np.max(diff_p),
        'mean_o': np.mean(diff_o), 'mean_p': np.mean(diff_p),
        'rel_o': np.max(diff_o) / mean_dH * 100 if mean_dH > 0 else 0,
        'rel_p': np.max(diff_p) / mean_dH * 100 if mean_dH > 0 else 0,
        't_c': exp['t_correct'], 't_o': exp['t_old'], 't_p': exp['t_prod'],
    })

names = [s['name'] for s in summary]
y = np.arange(len(names))
h = 0.35

ax.barh(y + h/2, [s['rel_o'] for s in summary], h, color='#ff7f0e', alpha=0.8,
        label=f'Error: {summary[0]["t_c"]}°C→{summary[0]["t_o"]}°C shift')
ax.barh(y - h/2, [s['rel_p'] for s in summary], h, color='#d62728', alpha=0.6,
        label=f'Error: {summary[0]["t_c"]}°C→{summary[0]["t_p"]}°C shift')

# Add text annotations
for i, s in enumerate(summary):
    ax.text(s['rel_o'] + 0.5, i + h/2, f'{s["max_o"]:.2f} J/g', va='center', fontsize=8, fontweight='bold', color='#ff7f0e')
    ax.text(s['rel_p'] + 0.5, i - h/2, f'{s["max_p"]:.2f} J/g', va='center', fontsize=8, fontweight='bold', color='#d62728')

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Relative error (%) = max|Δ| / mean(ΔH) × 100', fontsize=12)
ax.set_title('Integration Bound Correction — Error Magnitude', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='x')

# Annotation explaining
ax.text(0.98, 0.02,
        f'Correct bound = actual ramp start temperature\n'
        f'Old bound used 40°C arbitrarily\n'
        f'Production data used 70°C (worst)\n'
        f'All differences systematic (ΔH underestimated by higher lower bound)',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

fig.tight_layout()
out4 = os.path.join(RESULTS, 'enthalpy_correction_summary.png')
fig.savefig(out4, dpi=200)
plt.close()
print(f"Saved: {out4}")

print(f"\nAll 4 figures saved to results/enthalpy/")
