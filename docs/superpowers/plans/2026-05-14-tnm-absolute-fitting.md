# TNM Absolute Enthalpy Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit TNM model directly to absolute ΔH (kJ/mol) for one-step 70°C and two-step 90→80°C data, then extend predictions to 10^5s to verify plateau convergence between paired experimental conditions.

**Architecture:** Two self-contained scripts sharing `tnm_model.py` (instantaneous-quench TNM) and `tnm_conditions.py` (data loading). Each script: loads data → fits 6 TNM parameters via L-BFGS-B → simulates to 10^5s → generates 4-panel figure + convergence report.

**Tech Stack:** Python 3, numpy, scipy (L-BFGS-B), matplotlib, pandas

---

## File Structure

| File | Role |
|------|------|
| `src/tnm_model.py` | Instantaneous-quench TNM (already exists, reuse) |
| `src/tnm_conditions.py` | Data loading + protocol temps (already exists, reuse) |
| `src/fit_onestep_70C_absolute.py` | **Create** — Phase 1: fit 70°C run1/run2 separately |
| `src/fit_twostep_absolute.py` | **Create** — Phase 2: fit two-step Grp A/B jointly |

---

### Task 1: Phase 1 — One-step 70°C absolute fitting + extended prediction

**Files:**
- Create: `src/fit_onestep_70C_absolute.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Fit TNM model to absolute ΔH for 70°C one-step annealing.

Fits run1 and run2 separately (6 params: logA, H*, x, beta, T0, scale).
Extends prediction to 10^5s to verify plateau convergence.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnm_model import TNMModel
from tnm_conditions import HOLD_TIMES_SEC, T_70C, build_target_vectors

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)

R_GAS = 8.314


def make_model(logA, H_star, x, beta, T0, scale):
    A = 10.0 ** logA
    model = TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)
    return model, scale


def delta_H_kJ(model, scale, T_anneal, t_hold):
    """Absolute ΔH in kJ/mol: scale * (T0 - Tf)."""
    Tf_end, _ = model.simulate_one_step(T_anneal, t_hold)
    return scale * (model.T0 - Tf_end)


def cost_function(packed, t_exp, dH_exp, T_anneal):
    """Sum of squared residuals in kJ/mol."""
    logA, H_star, x, beta, T0, scale = packed
    if not (0.001 < x < 1 and 0.01 < beta < 1 and H_star > 30000 and 360 < T0 < 430 and scale > 0):
        return 1e10
    try:
        model, s = make_model(logA, H_star, x, beta, T0, scale)
    except Exception:
        return 1e10

    dH_sim = np.array([delta_H_kJ(model, s, T_anneal, t) for t in t_exp])
    if np.std(dH_sim) < 1e-8:
        return 1e8
    return np.sum((dH_sim - dH_exp) ** 2)


def fit_run(t_exp, dH_exp, T_anneal, run_label, seed=42):
    """Fit 6 TNM parameters to one run."""
    rng = np.random.RandomState(seed)
    bounds = [
        (-25, -12),         # logA
        (50000, 300000),    # H_star (J/mol)
        (0.005, 0.9),       # x
        (0.01, 0.9),        # beta
        (370, 430),         # T0 (K)
        (0.1, 5000),        # scale (kJ/mol/K)
    ]

    best_res = None
    best_cost = np.inf
    n_starts = 30

    print(f"  Fitting {run_label} ({n_starts} starts)...")
    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(cost_function, x0, args=(t_exp, dH_exp, T_anneal),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.fun < best_cost:
            best_cost = res.fun
            best_res = res
        if (k + 1) % 10 == 0:
            print(f"    Start {k+1}/{n_starts}, best cost = {best_cost:.1f}")

    p = best_res.x
    rmse = np.sqrt(best_cost / len(t_exp))
    print(f"    Done: logA={p[0]:.3f}, H*={p[1]/1000:.1f} kJ/mol, x={p[2]:.4f}, "
          f"beta={p[3]:.4f}, T0={p[4]:.1f}K, scale={p[5]:.2f}, RMSE={rmse:.1f} kJ/mol")
    return p, rmse


def main():
    print("=" * 60)
    print("  Phase 1: One-step 70°C — Absolute ΔH Fitting")
    print("=" * 60)

    targets = build_target_vectors()
    exp_70 = targets['os70']
    T_anneal = T_70C  # 343.15 K

    # Split run1 and run2
    mask_r1 = exp_70['groups'] == 'run1'
    mask_r2 = exp_70['groups'] == 'run2'
    t_r1 = exp_70['t'][mask_r1]
    dh_r1 = exp_70['dH'][mask_r1]
    t_r2 = exp_70['t'][mask_r2]
    dh_r2 = exp_70['dH'][mask_r2]

    print(f"\n  Run1: {len(t_r1)} pts, ΔH = [{dh_r1.min():.0f}, {dh_r1.max():.0f}] kJ/mol")
    print(f"  Run2: {len(t_r2)} pts, ΔH = [{dh_r2.min():.0f}, {dh_r2.max():.0f}] kJ/mol")

    # Fit each run
    print("\n[1] Fitting run1...")
    p1, rmse1 = fit_run(t_r1, dh_r1, T_anneal, "Run1")
    print(f"\n[2] Fitting run2...")
    p2, rmse2 = fit_run(t_r2, dh_r2, T_anneal, "Run2")

    # Build models
    m1, s1 = make_model(*p1)
    m2, s2 = make_model(*p2)

    # Extended prediction to 10^5s
    t_ext = np.logspace(-2, 5, 300)
    dh_ext_r1 = np.array([delta_H_kJ(m1, s1, T_anneal, t) for t in t_ext])
    dh_ext_r2 = np.array([delta_H_kJ(m2, s2, T_anneal, t) for t in t_ext])

    # Plateau values at t -> infinity (Tf -> T_anneal)
    plateau_r1 = s1 * (p1[4] - T_anneal)
    plateau_r2 = s2 * (p2[4] - T_anneal)
    plateau_diff = abs(plateau_r1 - plateau_r2)

    print(f"\n[3] Plateau convergence at 10^5s:")
    print(f"    Run1 plateau: {plateau_r1:.1f} kJ/mol")
    print(f"    Run2 plateau: {plateau_r2:.1f} kJ/mol")
    print(f"    Difference:   {plateau_diff:.1f} kJ/mol "
          f"({plateau_diff / max(plateau_r1, plateau_r2) * 100:.2f}%)")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Run1
    ax = axes[0, 0]
    ax.semilogx(t_ext / 60, dh_ext_r1, 'b-', linewidth=2, label=f'Run1')
    ax.plot(t_r1 / 60, dh_r1, 'bo', markersize=9, markerfacecolor='white',
            markeredgewidth=2, label='Exp')
    ax.axhline(y=plateau_r1, color='gray', linestyle=':', alpha=0.7,
               label=f'Plateau = {plateau_r1:.0f} kJ/mol')
    ax.set_xlabel('Hold time at 70°C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title(f'Run1: logA={p1[0]:.2f}, H*={p1[1]/1000:.0f} kJ/mol, '
                 f'x={p1[2]:.3f}, β={p1[3]:.3f}, RMSE={rmse1:.0f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Run2
    ax = axes[0, 1]
    ax.semilogx(t_ext / 60, dh_ext_r2, 'r-', linewidth=2, label=f'Run2')
    ax.plot(t_r2 / 60, dh_r2, 'ro', markersize=9, markerfacecolor='white',
            markeredgewidth=2, label='Exp')
    ax.axhline(y=plateau_r2, color='gray', linestyle=':', alpha=0.7,
               label=f'Plateau = {plateau_r2:.0f} kJ/mol')
    ax.set_xlabel('Hold time at 70°C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title(f'Run2: logA={p2[0]:.2f}, H*={p2[1]/1000:.0f} kJ/mol, '
                 f'x={p2[2]:.3f}, β={p2[3]:.3f}, RMSE={rmse2:.0f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: Overlay comparison
    ax = axes[1, 0]
    ax.semilogx(t_ext / 60, dh_ext_r1, 'b-', linewidth=2, alpha=0.6, label='Run1')
    ax.semilogx(t_ext / 60, dh_ext_r2, 'r-', linewidth=2, alpha=0.6, label='Run2')
    ax.plot(t_r1 / 60, dh_r1, 'bo', markersize=9, markerfacecolor='white',
            markeredgewidth=2)
    ax.plot(t_r2 / 60, dh_r2, 'ro', markersize=9, markerfacecolor='white',
            markeredgewidth=2)
    ax.axhline(y=plateau_r1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=plateau_r2, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.5, 0.05,
            f'Plateau gap: {plateau_diff:.0f} kJ/mol ({plateau_diff/max(plateau_r1,plateau_r2)*100:.1f}%)',
            transform=ax.transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Hold time at 70°C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title('One-step 70°C: Run1 vs Run2 — Plateau Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 4: Tf evolution
    ax = axes[1, 1]
    Tf_ext_r1 = np.array([m1.simulate_one_step(T_anneal, t)[0] for t in t_ext])
    Tf_ext_r2 = np.array([m2.simulate_one_step(T_anneal, t)[0] for t in t_ext])
    ax.semilogx(t_ext / 60, Tf_ext_r1 - 273.15, 'b-', linewidth=2, label='Run1')
    ax.semilogx(t_ext / 60, Tf_ext_r2 - 273.15, 'r-', linewidth=2, label='Run2')
    ax.axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='T=70°C (equilibrium)')
    ax.set_xlabel('Hold time at 70°C (min)')
    ax.set_ylabel('T_f (°C)')
    ax.set_title('Fictive Temperature Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('TNM One-Step 70°C — Absolute Enthalpy Fitting', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_onestep_70C_absolute.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")

    # Save parameters
    import pandas as pd
    rows = []
    for label, p, rmse in [('run1', p1, rmse1), ('run2', p2, rmse2)]:
        rows.append({
            'run': label, 'logA': p[0], 'A_s': 10**p[0],
            'H_star_kJmol': p[1]/1000, 'x': p[2], 'beta': p[3],
            'T0_K': p[4], 'scale': p[5], 'RMSE_kJmol': rmse,
            'plateau_kJmol': p[5] * (p[4] - T_anneal),
        })
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'tnm_onestep_70C_params.csv'),
                               index=False, float_format='%.6f')
    print("  Saved parameters CSV")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the script**

```bash
cd /root/twoannealing && python src/fit_onestep_70C_absolute.py
```

Expected: Two fits complete, plateau values reported, gap < 15%, 4-panel figure saved.

- [ ] **Step 3: Verify output**

```bash
ls -la results/tnm/tnm_onestep_70C_absolute.png results/tnm/tnm_onestep_70C_params.csv
```

Expected: Both files exist.

- [ ] **Step 4: Commit**

```bash
git add src/fit_onestep_70C_absolute.py
git commit -m "feat: add one-step 70C absolute enthalpy TNM fitting + extended prediction"
```

---

### Task 2: Phase 2 — Two-step 90→80°C absolute fitting + extended prediction

**Files:**
- Create: `src/fit_twostep_absolute.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Fit TNM model to absolute ΔH for two-step 90°C→80°C annealing.

Fits Grp A (T1=50s) and Grp B (T1=500s) jointly (6 params).
Extends T2 prediction to 10^5s to verify plateau convergence.
The S-shaped two-step curve (slow→fast→plateau) is captured by the
double-exponential two-step TNM formula.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_80C, T_90C, T1_HOLD_SHORT, T1_HOLD_LONG,
    build_target_vectors,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)

R_GAS = 8.314


def make_model(logA, H_star, x, beta, T0, scale):
    A = 10.0 ** logA
    model = TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)
    return model, scale


def delta_H_two_step_kJ(model, scale, T1, t1, T2, t2):
    """Absolute ΔH (kJ/mol) for two-step protocol."""
    Tf_end, _ = model.simulate_two_step(T1, t1, T2, t2)
    return scale * (model.T0 - Tf_end)


def cost_function(packed, tA, dhA, tB, dhB):
    """Joint cost: sum of squared residuals for both groups."""
    logA, H_star, x, beta, T0, scale = packed
    if not (0.001 < x < 1 and 0.01 < beta < 1 and H_star > 30000 and 360 < T0 < 430 and scale > 0):
        return 1e10
    try:
        model, s = make_model(logA, H_star, x, beta, T0, scale)
    except Exception:
        return 1e10

    chi2 = 0.0
    for t_exp, dh_exp, t1_val in [(tA, dhA, T1_HOLD_SHORT), (tB, dhB, T1_HOLD_LONG)]:
        dH_sim = np.array([delta_H_two_step_kJ(model, s, T_90C, t1_val, T_80C, t)
                           for t in t_exp])
        if np.std(dH_sim) < 1e-8:
            return 1e8
        chi2 += np.sum((dH_sim - dh_exp) ** 2)
    return chi2


def main():
    print("=" * 60)
    print("  Phase 2: Two-step 90°C->80°C — Absolute ΔH Fitting")
    print("=" * 60)

    targets = build_target_vectors()
    exp_ts = targets['ts']
    T1, T2 = T_90C, T_80C

    mask_A = exp_ts['groups'] == 'ts_grpA'
    mask_B = exp_ts['groups'] == 'ts_grpB'
    t_A = exp_ts['t'][mask_A]
    dh_A = exp_ts['dH'][mask_A]
    t_B = exp_ts['t'][mask_B]
    dh_B = exp_ts['dH'][mask_B]

    print(f"\n  Grp A (T1={T1_HOLD_SHORT}s @ 90°C): {len(t_A)} pts, "
          f"ΔH = [{dh_A.min():.0f}, {dh_A.max():.0f}] kJ/mol")
    print(f"  Grp B (T1={T1_HOLD_LONG}s @ 90°C): {len(t_B)} pts, "
          f"ΔH = [{dh_B.min():.0f}, {dh_B.max():.0f}] kJ/mol")

    # Joint fit
    rng = np.random.RandomState(42)
    bounds = [
        (-25, -12), (50000, 300000), (0.005, 0.9), (0.01, 0.9), (370, 430), (0.1, 5000),
    ]
    best_res, best_cost = None, np.inf
    n_starts = 30

    print(f"\n  Joint fitting Grp A + Grp B ({n_starts} starts)...")
    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(cost_function, x0, args=(t_A, dh_A, t_B, dh_B),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.fun < best_cost:
            best_cost = res.fun
            best_res = res
        if (k + 1) % 10 == 0:
            print(f"    Start {k+1}/{n_starts}, best cost = {best_cost:.1f}")

    p = best_res.x
    n_total = len(t_A) + len(t_B)
    rmse = np.sqrt(best_cost / n_total)
    print(f"\n  Best: logA={p[0]:.3f}, H*={p[1]/1000:.1f} kJ/mol, x={p[2]:.4f}, "
          f"beta={p[3]:.4f}, T0={p[4]:.1f}K, scale={p[5]:.2f}, RMSE={rmse:.1f} kJ/mol")

    model, scale = make_model(*p)

    # Extended T2 prediction to 10^5s
    t_ext = np.logspace(-2, 5, 300)
    dh_ext_A = np.array([delta_H_two_step_kJ(model, scale, T1, T1_HOLD_SHORT, T2, t)
                         for t in t_ext])
    dh_ext_B = np.array([delta_H_two_step_kJ(model, scale, T1, T1_HOLD_LONG, T2, t)
                         for t in t_ext])

    # Plateau at t2 -> infinity
    plateau = scale * (p[4] - T2)
    dh_A_1e5 = delta_H_two_step_kJ(model, scale, T1, T1_HOLD_SHORT, T2, 1e5)
    dh_B_1e5 = delta_H_two_step_kJ(model, scale, T1, T1_HOLD_LONG, T2, 1e5)
    gap_1e5 = abs(dh_A_1e5 - dh_B_1e5)

    print(f"\n  Plateau convergence:")
    print(f"    Theoretical plateau: {plateau:.1f} kJ/mol")
    print(f"    Grp A at 10^5s:      {dh_A_1e5:.1f} kJ/mol")
    print(f"    Grp B at 10^5s:      {dh_B_1e5:.1f} kJ/mol")
    print(f"    Gap at 10^5s:        {gap_1e5:.1f} kJ/mol "
          f"({gap_1e5/max(dh_A_1e5,dh_B_1e5)*100:.2f}%)")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'A': '#B2182B', 'B': '#D6604D'}

    # Panel 1: Grp A
    ax = axes[0, 0]
    ax.semilogx(t_ext / 60, dh_ext_A, '-', color=colors['A'], linewidth=2, label='Grp A')
    ax.plot(t_A / 60, dh_A, 'o', color=colors['A'], markersize=9,
            markerfacecolor='white', markeredgewidth=2, label='Exp')
    ax.axhline(y=plateau, color='gray', linestyle=':', alpha=0.7,
               label=f'Plateau = {plateau:.0f} kJ/mol')
    ax.set_xlabel('T2 hold time at 80°C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title(f'Grp A (T1=0.83min @ 90°C): S-shaped → plateau')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Grp B
    ax = axes[0, 1]
    ax.semilogx(t_ext / 60, dh_ext_B, '-', color=colors['B'], linewidth=2, label='Grp B')
    ax.plot(t_B / 60, dh_B, 'o', color=colors['B'], markersize=9,
            markerfacecolor='white', markeredgewidth=2, label='Exp')
    ax.axhline(y=plateau, color='gray', linestyle=':', alpha=0.7,
               label=f'Plateau = {plateau:.0f} kJ/mol')
    ax.set_xlabel('T2 hold time at 80°C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title(f'Grp B (T1=8.33min @ 90°C): S-shaped → plateau')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: Overlay comparison
    ax = axes[1, 0]
    ax.semilogx(t_ext / 60, dh_ext_A, '-', color=colors['A'], linewidth=2, alpha=0.6,
                label='Grp A')
    ax.semilogx(t_ext / 60, dh_ext_B, '-', color=colors['B'], linewidth=2, alpha=0.6,
                label='Grp B')
    ax.plot(t_A / 60, dh_A, 'o', color=colors['A'], markersize=9,
            markerfacecolor='white', markeredgewidth=2)
    ax.plot(t_B / 60, dh_B, 'o', color=colors['B'], markersize=9,
            markerfacecolor='white', markeredgewidth=2)
    ax.axhline(y=plateau, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.5, 0.05,
            f'Plateau gap at 10^5s: {gap_1e5:.0f} kJ/mol ({gap_1e5/plateau*100:.1f}%)\n'
            f'logA={p[0]:.2f}, H*={p[1]/1000:.0f} kJ/mol, x={p[2]:.3f}, β={p[3]:.3f}',
            transform=ax.transAxes, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('T2 hold time at 80°C (min)')
    ax.set_ylabel('ΔH (kJ/mol)')
    ax.set_title('Two-step 90→80°C: Grp A vs Grp B — Plateau Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 4: Tf evolution
    ax = axes[1, 1]
    Tf_ext_A = np.array([model.simulate_two_step(T1, T1_HOLD_SHORT, T2, t)[0]
                         for t in t_ext])
    Tf_ext_B = np.array([model.simulate_two_step(T1, T1_HOLD_LONG, T2, t)[0]
                         for t in t_ext])
    ax.semilogx(t_ext / 60, Tf_ext_A - 273.15, '-', color=colors['A'], linewidth=2,
                label='Grp A')
    ax.semilogx(t_ext / 60, Tf_ext_B - 273.15, '-', color=colors['B'], linewidth=2,
                label='Grp B')
    ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5, label='T=80°C (equilibrium)')
    ax.set_xlabel('T2 hold time at 80°C (min)')
    ax.set_ylabel('T_f (°C)')
    ax.set_title('Fictive Temperature Evolution (Two-step)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('TNM Two-Step 90°C→80°C — S-shaped Relaxation + Plateau Convergence',
                 fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_twostep_absolute.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")

    # Save parameters
    import pandas as pd
    pd.DataFrame([{
        'logA': p[0], 'A_s': 10**p[0], 'H_star_kJmol': p[1]/1000,
        'x': p[2], 'beta': p[3], 'T0_K': p[4], 'T0_C': p[4]-273.15,
        'scale': p[5], 'RMSE_kJmol': rmse, 'plateau_kJmol': plateau,
    }]).to_csv(os.path.join(RESULTS_DIR, 'tnm_twostep_params.csv'),
               index=False, float_format='%.6f')
    print("  Saved parameters CSV")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the script**

```bash
cd /root/twoannealing && python src/fit_twostep_absolute.py
```

Expected: Joint fit complete, plateau reported, gap at 10^5s < 10%, S-shaped prediction curve visible.

- [ ] **Step 3: Verify output**

```bash
ls -la results/tnm/tnm_twostep_absolute.png results/tnm/tnm_twostep_params.csv
```

Expected: Both files exist.

- [ ] **Step 4: Commit**

```bash
git add src/fit_twostep_absolute.py
git commit -m "feat: add two-step 90C->80C absolute enthalpy TNM fitting + extended prediction"
```

---

## Verification Checklist

1. [ ] `python src/fit_onestep_70C_absolute.py` — Run1/Run2 fits converge, plateau gap < 15%
2. [ ] `python src/fit_twostep_absolute.py` — Joint fit converges, plateau gap < 10%, S-shaped curve visible
3. [ ] Both PNG outputs in `results/tnm/` show experimental points + extended prediction lines
4. [ ] Both CSVs saved with fitted parameters
