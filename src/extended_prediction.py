#!/usr/bin/env python3
"""
Extended-time prediction: simulate TNM model far beyond experimental times
to verify plateau convergence across protocols.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C,
    T1_HOLD_SHORT, T1_HOLD_LONG, build_target_vectors,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load fitted params
csv_path = os.path.join(RESULTS_DIR, 'tnm_fit_results.csv')
df = pd.read_csv(csv_path)
s2 = df[df['stage'] == 'Stage2_Absolute'].iloc[0]
params = {
    'logA': s2['log10_A'], 'H_star': s2['H_star_kJmol'] * 1000,
    'x': s2['x'], 'beta': s2['beta'], 'T0': s2['T0_K'],
}

model = TNMModel(A=10**params['logA'], H_star=params['H_star'],
                 x=params['x'], beta=params['beta'], T0=params['T0'])

print(f"Parameters: logA={params['logA']:.3f}, H*={params['H_star']/1000:.1f} kJ/mol, "
      f"x={params['x']:.4f}, beta={params['beta']:.4f}, T0={params['T0']:.1f}K")

# Extended time range: 0.1s to 10^7s
t_ext = np.logspace(-1, 7, 200)
targets = build_target_vectors()

# ---- Plot 1: One-step ----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel 1: One-step 50C + 70C
ax = axes[0, 0]
for T_anneal, T_label, color in [(T_50C, '50C', '#2166AC'), (T_70C, '70C', '#4393C3')]:
    dh = np.array([model.delta_H_normalized(T_anneal, t) for t in t_ext])
    ax.semilogx(t_ext / 60, dh, '-', color=color, linewidth=2, label=f'{T_label} (model)')
    # Experimental points
    exp_data = targets['os50' if T_label == '50C' else 'os70']
    mask = exp_data['groups'] == 'run1'
    dh_exp = exp_data['dH'][mask]
    # Normalize exp to [0,1] for visual comparison
    dh_exp_norm = (dh_exp - dh_exp.min()) / (dh_exp.max() - dh_exp.min() + 1e-15)
    ax.plot(exp_data['t'][mask] / 60, dh_exp_norm, 'o', color=color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Equilibrium')
ax.set_xlabel('Hold time (min)')
ax.set_ylabel('Normalized ΔH')
ax.set_title('One-step: 50C & 70C (extended)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')

# Panel 2: Two-step hi->lo (90C->80C)
ax = axes[0, 1]
for t1, grp, color in [(T1_HOLD_SHORT, 'Grp A (0.83min@90C)', '#B2182B'),
                         (T1_HOLD_LONG, 'Grp B (8.33min@90C)', '#D6604D')]:
    dh = np.array([model.delta_H_two_step(T_90C, t1, T_80C, t) for t in t_ext])
    ax.semilogx(t_ext / 60, dh, '-', color=color, linewidth=2, label=grp)
# Experimental points
exp_ts = targets['ts']
for grp_key, color in [('ts_grpA', '#B2182B'), ('ts_grpB', '#D6604D')]:
    mask = exp_ts['groups'] == grp_key
    dh_exp = exp_ts['dH'][mask]
    dh_exp_norm = (dh_exp - dh_exp.min()) / (dh_exp.max() - dh_exp.min() + 1e-15)
    ax.plot(exp_ts['t'][mask] / 60, dh_exp_norm, 'o', color=color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('T2 hold time at 80C (min)')
ax.set_ylabel('Normalized ΔH')
ax.set_title('Two-step hi→lo: 90C→80C (extended)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')

# Panel 3: Kovacs up-jump (80C->90C)
ax = axes[0, 2]
for t1, grp, color in [(T1_HOLD_SHORT, 'Grp A (0.83min@80C)', '#4DAF4A'),
                         (T1_HOLD_LONG, 'Grp B (8.33min@80C)', '#984EA3')]:
    dh = np.array([model.delta_H_two_step(T_80C, t1, T_90C, t) for t in t_ext])
    ax.semilogx(t_ext / 60, dh, '-', color=color, linewidth=2, label=grp)
exp_kv = targets['kovacs']
for grp_key, color in [('kov_grpA', '#4DAF4A'), ('kov_grpB', '#984EA3')]:
    mask = exp_kv['groups'] == grp_key
    dh_exp = exp_kv['dH'][mask]
    dh_exp_norm = (dh_exp - dh_exp.min()) / (dh_exp.max() - dh_exp.min() + 1e-15)
    ax.plot(exp_kv['t'][mask] / 60, dh_exp_norm, 'o', color=color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('T2 hold time at 90C (min)')
ax.set_ylabel('Normalized ΔH')
ax.set_title('Kovacs up-jump: 80C→90C (extended)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')

# Panel 4: Absolute dH (kJ/mol) — One-step
ax = axes[1, 0]
for T_anneal, T_label, color in [(T_50C, '50C', '#2166AC'), (T_70C, '70C', '#4393C3')]:
    dh_norm = np.array([model.delta_H_normalized(T_anneal, t) for t in t_ext])
    # Scale to kJ using experimental range
    exp_data = targets['os50' if T_label == '50C' else 'os70']
    mask = exp_data['groups'] == 'run1'
    dh_exp = exp_data['dH'][mask]
    a, b = np.polyfit(dh_norm[:10], [model.delta_H_normalized(T_anneal, th) for th in HOLD_TIMES_SEC], 1)
    # Actually, use direct fit to exp kJ
    _, dh_sim_raw = np.zeros(10), np.array([model.delta_H_normalized(T_anneal, th) for th in HOLD_TIMES_SEC])
    # Fit at experimental points
    dh_sim_at_exp = np.interp(exp_data['t'][mask], HOLD_TIMES_SEC,
                              np.array([model.delta_H_normalized(T_anneal, th) for th in HOLD_TIMES_SEC]))
    a, b = np.polyfit(dh_sim_at_exp, dh_exp, 1)
    dh_kJ = a * dh_norm + b
    dh_exp_kJ_scaled = a * dh_sim_at_exp + b
    ax.semilogx(t_ext / 60, dh_kJ, '-', color=color, linewidth=2, label=f'{T_label}')
    ax.plot(exp_data['t'][mask] / 60, dh_exp, 'o', color=color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
ax.set_xlabel('Hold time (min)')
ax.set_ylabel('ΔH (kJ/mol)')
ax.set_title('One-step: Absolute enthalpy')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')

# Panel 5: Absolute dH (kJ/mol) — Two-step
ax = axes[1, 1]
for t1, grp, grp_key, color in [(T1_HOLD_SHORT, 'Grp A', 'ts_grpA', '#B2182B'),
                                   (T1_HOLD_LONG, 'Grp B', 'ts_grpB', '#D6604D')]:
    dh_norm = np.array([model.delta_H_two_step(T_90C, t1, T_80C, t) for t in t_ext])
    mask = exp_ts['groups'] == grp_key
    dh_exp = exp_ts['dH'][mask]
    t_exp_ts = exp_ts['t'][mask]
    dh_sim_at_exp = np.interp(t_exp_ts, HOLD_TIMES_SEC,
                              np.array([model.delta_H_two_step(T_90C, t1, T_80C, th) for th in HOLD_TIMES_SEC]))
    a, b = np.polyfit(dh_sim_at_exp, dh_exp, 1)
    dh_kJ = a * dh_norm + b
    ax.semilogx(t_ext / 60, dh_kJ, '-', color=color, linewidth=2, label=grp)
    ax.plot(t_exp_ts / 60, dh_exp, 'o', color=color,
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
ax.set_xlabel('T2 hold time at 80C (min)')
ax.set_ylabel('ΔH (kJ/mol)')
ax.set_title('Two-step: Absolute enthalpy')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')

# Panel 6: Tf evolution across all temperatures
ax = axes[1, 2]
T_range = np.linspace(300, 480, 100)
for t_hold in [0.1, 1, 10, 100, 1000, 1e4, 1e5]:
    Tf_vals = np.array([model.simulate_one_step(T, t_hold)[0] for T in T_range])
    ax.plot(T_range - 273.15, Tf_vals - 273.15, '-', linewidth=1,
            alpha=0.7, label=f't={t_hold:.0f}s' if t_hold < 1e4 else f't={t_hold/3600:.1f}h')
ax.plot([0, 200], [0, 200], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('T (C)')
ax.set_ylabel('T_f (C)')
ax.set_title('T_f vs T at various hold times')
ax.legend(fontsize=6, loc='lower right')
ax.grid(True, alpha=0.3)

logA_str = f'{params["logA"]:.2f}'
H_str = f'{params["H_star"]/1000:.0f}'
x_str = f'{params["x"]:.3f}'
b_str = f'{params["beta"]:.3f}'
T0_str = f'{params["T0"]:.0f}'
plt.suptitle(f'TNM Extended-Time Prediction\n'
             f'logA={logA_str}, H*={H_str} kJ/mol, '
             f'x={x_str}, beta={b_str}, T0={T0_str}K',
             fontsize=13)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'tnm_extended_prediction.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ---- Convergence analysis ----
print("\n=== Plateau Convergence Analysis ===")
print("Checking if Grp A and Grp B converge at long T2 times...")

for name, T1, T2, t1_s, t1_l in [
    ('Two-step hi→lo', T_90C, T_80C, T1_HOLD_SHORT, T1_HOLD_LONG),
    ('Kovacs up-jump', T_80C, T_90C, T1_HOLD_SHORT, T1_HOLD_LONG),
]:
    dh_A_inf = model.delta_H_two_step(T1, t1_s, T2, 1e7)
    dh_B_inf = model.delta_H_two_step(T1, t1_l, T2, 1e7)
    dh_A_exp = model.delta_H_two_step(T1, t1_s, T2, 1000)
    dh_B_exp = model.delta_H_two_step(T1, t1_l, T2, 1000)
    print(f"\n{name} (T1={T1-273:.0f}C -> T2={T2-273:.0f}C):")
    print(f"  Grp A: dH(t2=1000s)={dh_A_exp:.4f}, dH(t2=1e7s)={dh_A_inf:.4f}")
    print(f"  Grp B: dH(t2=1000s)={dh_B_exp:.4f}, dH(t2=1e7s)={dh_B_inf:.4f}")
    print(f"  Convergence gap at 1e7s: {abs(dh_A_inf - dh_B_inf):.6f}")
    print(f"  Gap at 1000s: {abs(dh_A_exp - dh_B_exp):.4f}")
    print(f"  Gap reduction: {abs(dh_A_exp - dh_B_exp) - abs(dh_A_inf - dh_B_inf):.4f}")
