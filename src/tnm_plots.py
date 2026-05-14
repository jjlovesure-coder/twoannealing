"""
Visualization for TNM model fitting results.

Generates:
  1. One-step fit (50C + 70C)
  2. Two-step fit (Group A + B)
  3. Kovacs prediction
  4. Parity plot
  5. T_f evolution curves
  6. Residual analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tnm_model import TNMModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')

# Color scheme
C_OS50 = '#2166AC'
C_OS70 = '#4393C3'
C_TS_A = '#B2182B'
C_TS_B = '#D6604D'
C_KV_A = '#4DAF4A'
C_KV_B = '#984EA3'


def plot_one_step_fit(exp_data, sim_data, params, output_path):
    """One-step fit: delta_H vs log(hold_time) for 50C and 70C."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (exp_df_key, T_C, color) in enumerate([
        ('os50', 50, C_OS50), ('os70', 70, C_OS70),
    ]):
        ax = axes[ax_idx]
        exp = exp_data[exp_df_key]

        # Split into run1 and run2
        n = len(exp['dH'])
        mid = n // 2
        t_hold_min = exp['t'] / 60.0
        t_r1 = t_hold_min[:mid]
        dh_r1 = exp['dH'][:mid]
        t_r2 = t_hold_min[mid:]
        dh_r2 = exp['dH'][mid:]

        ax.plot(t_r1, dh_r1, 'o', color=color, markersize=9,
                markerfacecolor='white', markeredgewidth=1.5, label='Run 1')
        ax.plot(t_r2, dh_r2, 's', color=color, markersize=9,
                markerfacecolor='white', markeredgewidth=1.5, label='Run 2')

        # Model curve (use unique hold times from run 1)
        dh_s = sim_data[exp_df_key]
        if len(dh_s) == len(t_r1):
            ax.plot(t_r1, dh_s, '-', color=color, linewidth=2.0, alpha=0.7,
                    label='TNM fit')
        else:
            # Interpolate sim data onto run1 hold times
            from tnm_conditions import HOLD_TIMES_SEC
            ax.plot(HOLD_TIMES_SEC / 60.0, dh_s[:10], '-', color=color,
                    linewidth=2.0, alpha=0.7, label='TNM fit')

        ax.set_xlabel(f'Hold time at {T_C}C (min)')
        ax.set_ylabel('deltaH (kJ/mol)')
        ax.set_title(f'One-step @ {T_C}C')
        ax.set_xscale('log')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_two_step_fit(exp_data, sim_data, output_path):
    """Two-step fit: delta_H vs log(T2_hold) for Group A and B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    exp = exp_data['ts']
    t1_vals = exp['groups']
    t_exp = exp['t'] / 60.0
    dH_exp = exp['dH']

    for ax_idx, (grp_label, grp_key, color) in enumerate([
        ('T1=0.833 min (Grp A)', 'ts_grpA', C_TS_A),
        ('T1=8.333 min (Grp B)', 'ts_grpB', C_TS_B),
    ]):
        ax = axes[ax_idx]
        mask = t1_vals == grp_key
        t_grp = t_exp[mask]
        dh_grp = dH_exp[mask]

        ax.plot(t_grp, dh_grp, 'o', color=color, markersize=9,
                markerfacecolor='white', markeredgewidth=1.5, label='Exp')

        # Model
        ax.plot(t_grp, sim_data[grp_key], '-', color=color, linewidth=2.0,
                alpha=0.7, label='TNM fit')

        ax.set_xlabel('T2 hold time at 80C (min)')
        ax.set_ylabel('deltaH (kJ/mol)')
        ax.set_title(f'Two-step: {grp_label}')
        ax.set_xscale('log')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_kovacs_prediction(exp_data, sim_data, output_path):
    """Kovacs prediction: delta_H vs log(T2_hold), with expected hump."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    exp = exp_data['kovacs']
    t1_vals = exp['groups']
    t_exp = exp['t'] / 60.0
    dH_exp = exp['dH']

    for ax_idx, (grp_label, grp_key, color) in enumerate([
        ('T1=0.833 min @ 80C (Grp A)', 'kov_grpA', C_KV_A),
        ('T1=8.333 min @ 80C (Grp B)', 'kov_grpB', C_KV_B),
    ]):
        ax = axes[ax_idx]
        mask = t1_vals == grp_key
        t_grp = t_exp[mask]
        dh_grp = dH_exp[mask]

        ax.plot(t_grp, dh_grp, 'o', color=color, markersize=9,
                markerfacecolor='white', markeredgewidth=1.5, label='Exp')

        ax.plot(t_grp, sim_data[grp_key], '-', color=color, linewidth=2.0,
                alpha=0.7, label='TNM prediction')

        ax.set_xlabel('T2 hold time at 90C (min)')
        ax.set_ylabel('deltaH (kJ/mol)')
        ax.set_title(f'Kovacs up-jump: {grp_label}')
        ax.set_xscale('log')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_parity(sim_data, exp_data, output_path):
    """Parity plot: delta_H_model vs delta_H_exp for all conditions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    all_sim = []
    all_exp = []
    colors_list = []
    labels_list = []

    datasets = [
        ('os50', 'os50', C_OS50, 'One-step 50C'),
        ('os70', 'os70', C_OS70, 'One-step 70C'),
        ('ts_grpA', 'ts', C_TS_A, 'Two-step Grp A'),
        ('ts_grpB', 'ts', C_TS_B, 'Two-step Grp B'),
        ('kov_grpA', 'kovacs', C_KV_A, 'Kovacs Grp A'),
        ('kov_grpB', 'kovacs', C_KV_B, 'Kovacs Grp B'),
    ]

    for sim_key, exp_key, color, label in datasets:
        exp = exp_data[exp_key]
        if 'groups' in exp:
            if exp_key == 'ts':
                mask = exp['groups'] == sim_key
            elif exp_key == 'kovacs':
                mask = exp['groups'] == sim_key
            else:
                continue
            dH_e = exp['dH'][mask]
            dH_s = sim_data[sim_key]
            # Match by order
            if len(dH_e) <= len(dH_s):
                dH_s = dH_s[:len(dH_e)]
        else:
            dH_e = exp['dH']
            dH_s = sim_data[sim_key]
            dH_s = dH_s[:len(dH_e)]

        ax.scatter(dH_e, dH_s, c=color, label=label, alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=50)
        all_sim.extend(dH_s)
        all_exp.extend(dH_e)

    all_sim = np.array(all_sim)
    all_exp = np.array(all_exp)

    # Diagonal
    lims = [min(all_exp.min(), all_sim.min()) * 0.9,
            max(all_exp.max(), all_sim.max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    # R^2
    ss_res = np.sum((all_sim - all_exp) ** 2)
    ss_tot = np.sum((all_exp - np.mean(all_exp)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
    rmse = np.sqrt(np.mean((all_sim - all_exp) ** 2))

    ax.text(0.05, 0.95, f'R^2 = {r2:.3f}\nRMSE = {rmse:.1f} kJ/mol',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Experimental deltaH (kJ/mol)')
    ax.set_ylabel('TNM model deltaH (kJ/mol)')
    ax.set_title('Parity Plot: TNM Model vs Experiment')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_tf_evolution(model, output_path):
    """T_f evolution for representative conditions — simplified visualization."""
    from tnm_conditions import T_50C, T_70C, T_80C, T_90C, HOLD_TIMES_SEC

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: One-step normalized delta_H vs hold time
    ax = axes[0, 0]
    dh_50 = np.array([model.delta_H_normalized(T_50C, th) for th in HOLD_TIMES_SEC])
    dh_70 = np.array([model.delta_H_normalized(T_70C, th) for th in HOLD_TIMES_SEC])
    ax.plot(HOLD_TIMES_SEC / 60.0, dh_50, 'o-', color=C_OS50, linewidth=1.5,
            markersize=6, label='50C')
    ax.plot(HOLD_TIMES_SEC / 60.0, dh_70, 's-', color=C_OS70, linewidth=1.5,
            markersize=6, label='70C')
    ax.set_xlabel('Hold time (min)')
    ax.set_ylabel('Normalized delta_H')
    ax.set_title('One-step: Normalized enthalpy')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Two-step normalized delta_H vs T2 hold
    ax = axes[0, 1]
    t1_A, t1_B = 0.833 * 60.0, 8.333 * 60.0
    dh_tsA = np.array([model.delta_H_two_step(T_90C, t1_A, T_80C, th)
                        for th in HOLD_TIMES_SEC])
    dh_tsB = np.array([model.delta_H_two_step(T_90C, t1_B, T_80C, th)
                        for th in HOLD_TIMES_SEC])
    ax.plot(HOLD_TIMES_SEC / 60.0, dh_tsA, 'o-', color=C_TS_A, linewidth=1.5,
            markersize=6, label='Grp A (t1=0.833min)')
    ax.plot(HOLD_TIMES_SEC / 60.0, dh_tsB, 's-', color=C_TS_B, linewidth=1.5,
            markersize=6, label='Grp B (t1=8.333min)')
    ax.set_xlabel('T2 hold time at 80C (min)')
    ax.set_ylabel('Normalized delta_H')
    ax.set_title('Two-step hi->lo: Normalized enthalpy')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: Kovacs normalized delta_H vs T2 hold
    ax = axes[0, 2]
    dh_kvA = np.array([model.delta_H_two_step(T_80C, t1_A, T_90C, th)
                        for th in HOLD_TIMES_SEC])
    dh_kvB = np.array([model.delta_H_two_step(T_80C, t1_B, T_90C, th)
                        for th in HOLD_TIMES_SEC])
    ax.plot(HOLD_TIMES_SEC / 60.0, dh_kvA, 'o-', color=C_KV_A, linewidth=1.5,
            markersize=6, label='Grp A (t1=0.833min)')
    ax.plot(HOLD_TIMES_SEC / 60.0, dh_kvB, 's-', color=C_KV_B, linewidth=1.5,
            markersize=6, label='Grp B (t1=8.333min)')
    ax.set_xlabel('T2 hold time at 90C (min)')
    ax.set_ylabel('Normalized delta_H')
    ax.set_title('Kovacs up-jump: Normalized enthalpy')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 4: tau(T, Tf) map
    ax = axes[1, 0]
    T_range = np.linspace(300, 480, 50)
    Tf_range = np.linspace(320, 480, 50)
    TT, TfTf = np.meshgrid(T_range, Tf_range)
    tau_grid = model.tau(TT, TfTf)
    c = ax.contourf(TT - 273.15, TfTf - 273.15, np.log10(tau_grid),
                     levels=20, cmap='viridis')
    plt.colorbar(c, ax=ax, label='log10(tau / s)')
    ax.plot([T_range[0]-273.15, T_range[-1]-273.15],
            [T_range[0]-273.15, T_range[-1]-273.15], 'w--', alpha=0.5)
    ax.set_xlabel('T (C)')
    ax.set_ylabel('T_f (C)')
    ax.set_title('Relaxation time map: log10(tau)')

    # Panel 5: Effect of x (nonlinearity) on shape
    ax = axes[1, 1]
    for x_val in [0.1, 0.3, 0.5, 0.7]:
        m = TNMModel(A=10**model.logA if hasattr(model, 'logA') else model.A,
                      H_star=model.H_star, x=x_val, beta=model.beta, T0=model.T0)
        dh = np.array([m.delta_H_normalized(T_50C, th) for th in HOLD_TIMES_SEC])
        dh_norm = (dh - dh.min()) / (dh.max() - dh.min() + 1e-10)
        ax.plot(HOLD_TIMES_SEC / 60.0, dh_norm, '-', linewidth=1.5,
                label=f'x={x_val:.1f}')
    ax.set_xlabel('Hold time (min)')
    ax.set_ylabel('Normalized shape')
    ax.set_title('Effect of nonlinearity x on curve shape')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 6: Effect of beta on shape
    ax = axes[1, 2]
    for beta_val in [0.2, 0.4, 0.6, 0.8]:
        m = TNMModel(A=model.A, H_star=model.H_star, x=model.x,
                      beta=beta_val, T0=model.T0)
        dh = np.array([m.delta_H_normalized(T_50C, th) for th in HOLD_TIMES_SEC])
        dh_norm = (dh - dh.min()) / (dh.max() - dh.min() + 1e-10)
        ax.plot(HOLD_TIMES_SEC / 60.0, dh_norm, '-', linewidth=1.5,
                label=f'beta={beta_val:.1f}')
    ax.set_xlabel('Hold time (min)')
    ax.set_ylabel('Normalized shape')
    ax.set_title('Effect of beta on curve shape')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_residuals(exp_data, sim_data, output_path):
    """Residual analysis: delta_H_residual vs hold time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # One-step residuals (use run1 only)
    ax = axes[0]
    for protocol, T_C, color in [
        ('os50', 50, C_OS50), ('os70', 70, C_OS70)
    ]:
        exp = exp_data[protocol]
        mask = exp['groups'] == 'run1'
        t_exp = exp['t'][mask] / 60.0
        dH_e = exp['dH'][mask]
        dH_s = sim_data[protocol]
        dH_s = dH_s[:len(dH_e)]

        residuals = dH_s - dH_e
        ax.plot(t_exp, residuals, 'o', color=color, markersize=7,
                markerfacecolor='white', markeredgewidth=1.2,
                label=f'{T_C}C')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Hold time (min)')
    ax.set_ylabel('Residual (sim - exp) (kJ/mol)')
    ax.set_title('One-step residuals')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Two-step + Kovacs residuals
    ax = axes[1]
    for grp_key, label, color in [
        ('ts_grpA', 'TS Grp A', C_TS_A),
        ('ts_grpB', 'TS Grp B', C_TS_B),
        ('kov_grpA', 'Kov Grp A', C_KV_A),
        ('kov_grpB', 'Kov Grp B', C_KV_B),
    ]:
        exp_key = 'ts' if grp_key.startswith('ts') else 'kovacs'
        exp = exp_data[exp_key]
        mask = exp['groups'] == grp_key
        dH_e = exp['dH'][mask]
        dH_s = sim_data[grp_key]
        dH_s = dH_s[:len(dH_e)]
        t_exp = exp['t'][mask] / 60.0

        residuals = dH_s - dH_e
        ax.plot(t_exp, residuals, 'o', color=color, markersize=7,
                markerfacecolor='white', markeredgewidth=1.2,
                label=label)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('T2 hold time (min)')
    ax.set_ylabel('Residual (sim - exp) (kJ/mol)')
    ax.set_title('Two-step & Kovacs residuals')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")
