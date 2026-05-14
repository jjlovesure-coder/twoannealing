# TNM Model Verification & PS Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix TNM model physics bugs, verify against MATLAB reference, fit to PS annealing data, and analyze errors.

**Architecture:** Pure isothermal TNM (Mode A) for MATLAB verification in new files; corrected finite-rate cooling TNM (Mode B) by fixing Boltzmann superposition in existing files. Data loading fixed for v2 CSV format. New multi-stage fitting with shape + absolute-scale optimization.

**Tech Stack:** Python 3, numpy, scipy (L-BFGS-B optimization), matplotlib, pandas

**Reference:** MATLAB code at `/root/twoannealing/results/tnm/gemini-code-1778722682691.txt`

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `src/tnm_reference.py` | Pure isothermal TNM (Mode A) | **Create** |
| `src/verify_matlab_reference.py` | MATLAB comparison script | **Create** |
| `src/tnm_model.py` | Corrected TNM with finite-rate cooling (Mode B) | **Modify** |
| `src/tnm_conditions.py` | Protocol definitions, data loading | **Modify** |
| `src/tnm_fit.py` | Fitting cost functions, optimization | **Modify** |
| `src/tnm_main.py` | Driver script | **Modify** |
| `src/tnm_plots.py` | Visualization | **Modify** (minor) |
| `src/tnm_error_analysis.py` | Residuals, bootstrap, diagnostics | **Create** |

---

## Phase 1: MATLAB Reference Verification

### Task 1.1: Write pure isothermal TNM reference module

**Files:**
- Create: `src/tnm_reference.py`

- [ ] **Step 1: Write the module**

```python
"""
Pure isothermal TNM model — exactly reproduces MATLAB reference formulas.
No cooling ramps. Instantaneous temperature jumps only.

TNM equations (Song et al. 2020):
  tau = exp(log(A) + x*H*/(R*T) + (1-x)*H*/(R*Tf))
  Tf(t) = Ta + (T0 - Ta) * exp(-S(t)^beta)   [one-step]
  Tf(t2) = T2 + (T0-T1)*exp(-(S1+S2)^beta) + (T1-T2)*exp(-S2^beta)  [two-step]
"""
import numpy as np

R_GAS = 8.314


def make_tau_func(A, H_star, x):
    """Return tau(T, Tf) function with given parameters."""
    logA = np.log(A)
    def tau(T, Tf):
        exponent = (x * H_star) / (R_GAS * T) + \
                   ((1 - x) * H_star) / (R_GAS * Tf)
        return np.exp(logA + exponent)
    return tau


def one_step(T0, Ta, t_array, A, H_star, x, beta):
    """One-step isothermal annealing.

    Starts at Tf(0) = T0, instant quench to Ta, isothermal hold.
    Returns Tf(t) array.
    """
    tau_func = make_tau_func(A, H_star, x)
    Tf = np.zeros(len(t_array))
    Tf[0] = T0
    S = 0.0

    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i - 1]
        tau_val = tau_func(Ta, Tf[i - 1])
        S += dt / tau_val
        Tf[i] = Ta + (T0 - Ta) * np.exp(-(S ** beta))

    return Tf


def two_step_hi_to_lo(T0, T1, t1_total, T2, t2_array, A, H_star, x, beta):
    """Two-step hi-to-lo: T0 -> T1 (hold t1) -> T2 (hold t2).

    Returns Tf(t2) array for the second hold.
    """
    tau_func = make_tau_func(A, H_star, x)

    # Step 1: hold at T1
    n1 = 500
    t1_sim = np.linspace(0, t1_total, n1)
    Tf_step1 = T0
    S1 = 0.0
    for i in range(1, n1):
        dt = t1_sim[i] - t1_sim[i - 1]
        tau_val = tau_func(T1, Tf_step1)
        S1 += dt / tau_val
        Tf_step1 = T1 + (T0 - T1) * np.exp(-(S1 ** beta))

    # Step 2: hold at T2
    Tf = np.zeros(len(t2_array))
    Tf[0] = Tf_step1
    S2 = 0.0
    for j in range(1, len(t2_array)):
        dt = t2_array[j] - t2_array[j - 1]
        tau_val = tau_func(T2, Tf[j - 1])
        S2 += dt / tau_val
        Tf[j] = T2 + (T0 - T1) * np.exp(-((S1 + S2) ** beta)) + \
                (T1 - T2) * np.exp(-(S2 ** beta))

    return Tf


def two_step_lo_to_hi(T0, T1, t1_total, T2, t2_array, A, H_star, x, beta):
    """Two-step lo-to-hi (Kovacs up-jump): T0 -> T1 (hold t1) -> T2 (hold t2).

    T1 < T2 (up-jump). Same formula as hi-to-lo — it's general.
    """
    return two_step_hi_to_lo(T0, T1, t1_total, T2, t2_array, A, H_star, x, beta)
```

- [ ] **Step 2: Verify module imports**

Run: `cd /root/twoannealing && python -c "from src.tnm_reference import one_step, two_step_hi_to_lo, two_step_lo_to_hi; print('OK')"`
Expected: `OK`

---

### Task 1.2: Write verification script against MATLAB

**Files:**
- Create: `src/verify_matlab_reference.py`

- [ ] **Step 1: Write the verification script**

```python
#!/usr/bin/env python3
"""
Verify TNM reference implementation against MATLAB results.
Generates comparison plot showing Tf evolution for all 3 protocols.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnm_reference import one_step, two_step_hi_to_lo, two_step_lo_to_hi

# MATLAB reference parameters (Au-based metallic glass)
H_STAR = 164e3       # J/mol
A = 6e-22            # seconds
X = 0.6
BETA = 0.43
T0 = 430.0           # K, initial fictive temperature
R = 8.314

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_matlab_values():
    """Pre-compute expected values using MATLAB formulas directly."""
    # The formulas in the MATLAB code are:
    # Figure 1a: one-step at Ta=383K, t = logspace(-1, 4, 1000)
    # Figure 1b: two-step hi-to-lo, T1=383K, t1=50s, T2=373K, t2=logspace(-1, 5, 1000)
    # Figure 1c: memory, T1=373K, t1=50s, T2=383K, t2=logspace(-1, 4, 1000)

    # Protocol (a): One-step 383K
    Ta_1a = 383.0
    t_1a = np.logspace(-1, 4, 1000)
    Tf_1a = one_step(T0, Ta_1a, t_1a, A, H_STAR, X, BETA)

    # Protocol (b): Two-step hi-to-lo (383K -> 373K)
    T1_1b = 383.0
    t1_1b = 50.0
    T2_1b = 373.0
    t_1b = np.logspace(-1, 5, 1000)
    Tf_1b = two_step_hi_to_lo(T0, T1_1b, t1_1b, T2_1b, t_1b, A, H_STAR, X, BETA)

    # Protocol (c): Memory / Kovacs up-jump (373K -> 383K)
    T1_1c = 373.0
    t1_1c = 50.0
    T2_1c = 383.0
    t_1c = np.logspace(-1, 4, 1000)
    Tf_1c = two_step_lo_to_hi(T0, T1_1c, t1_1c, T2_1c, t_1c, A, H_STAR, X, BETA)

    return {
        'a': (t_1a, Tf_1a, Ta_1a),
        'b': (t_1b, Tf_1b, T1_1b, T2_1b),
        'c': (t_1c, Tf_1c, T1_1c, T2_1c),
    }


def verify_consistency(results):
    """Check physical consistency of Tf evolution."""
    checks = []

    # Protocol (a): Tf must be between Ta and T0, monotonic decrease
    t_a, Tf_a, Ta = results['a']
    checks.append(('a: Tf within range', np.all(Tf_a <= T0) and np.all(Tf_a >= Ta)))
    checks.append(('a: Tf monotonic', np.all(np.diff(Tf_a) <= 0)))
    checks.append(('a: Tf(0) = T0', abs(Tf_a[0] - T0) < 1e-10))

    # Protocol (b): Tf at end of step1 < T1, further decreases at T2
    t_b, Tf_b, T1_b, T2_b = results['b']
    checks.append(('b: Tf within range', np.all(Tf_b <= T0) and np.all(Tf_b >= T2_b)))
    checks.append(('b: Tf(0) < T1', Tf_b[0] < T1_b))
    checks.append(('b: Tf monotonic', np.all(np.diff(Tf_b) <= 0)))

    # Protocol (c): Kovacs up-jump — Tf should show "memory hump"
    t_c, Tf_c, T1_c, T2_c = results['c']
    checks.append(('c: Tf(0) < T2', Tf_c[0] < T2_c))
    checks.append(('c: Tf returns toward T2', Tf_c[-1] > Tf_c[0]))

    all_pass = True
    for name, passed in checks:
        status = 'PASS' if passed else 'FAIL'
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    return all_pass


def plot_results(results):
    """Generate 3-panel comparison figure matching MATLAB layout."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): One-step monotonic at 383K
    t_a, Tf_a, Ta = results['a']
    axes[0].semilogx(t_a, Tf_a, 'b-', linewidth=2)
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Fictive Temperature T_f (K)')
    axes[0].set_title(f'(a) Monotonic (383 K)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=Ta, color='gray', linestyle='--', alpha=0.5)

    # Panel (b): Two-step hi-to-lo (383K -> 373K)
    t_b, Tf_b, T1_b, T2_b = results['b']
    axes[1].semilogx(t_b, Tf_b, 'b-', linewidth=2)
    axes[1].set_xlabel('t_2 (s)')
    axes[1].set_ylabel('Fictive Temperature T_f (K)')
    axes[1].set_title(f'(b) Two-step (383 K -> 373 K)')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=T2_b, color='gray', linestyle='--', alpha=0.5)

    # Panel (c): Memory / Kovacs up-jump (373K -> 383K)
    t_c, Tf_c, T1_c, T2_c = results['c']
    axes[2].semilogx(t_c, Tf_c, 'r-', linewidth=2)
    axes[2].set_xlabel('t_2 (s)')
    axes[2].set_ylabel('Fictive Temperature T_f (K)')
    axes[2].set_title(f'(c) Memory (373 K -> 383 K)')
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=T2_c, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'ref_verification.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved verification plot: {out_path}")


def main():
    print("=" * 60)
    print("  TNM Reference Verification (vs MATLAB)")
    print("=" * 60)
    print(f"\n  Parameters:")
    print(f"    H* = {H_STAR/1000:.0f} kJ/mol")
    print(f"    A  = {A:.1e} s")
    print(f"    x  = {X}")
    print(f"    beta = {BETA}")
    print(f"    T0 = {T0} K")

    print("\n[1] Computing Tf for all 3 protocols...")
    results = compute_matlab_values()

    print("\n[2] Physical consistency checks:")
    all_pass = verify_consistency(results)
    print(f"\n  All checks passed: {all_pass}")

    print("\n[3] Generating comparison plots...")
    plot_results(results)

    # Save numerical reference for future use
    ref_csv = os.path.join(RESULTS_DIR, 'ref_data', '')
    os.makedirs(ref_csv, exist_ok=True)
    for key, (t, Tf, *temps) in results.items():
        fname = os.path.join(ref_csv, f'ref_protocol_{key}.csv')
        cols_data = {'t_s': t, 'Tf_K': Tf}
        import pandas as pd
        pd.DataFrame(cols_data).to_csv(fname, index=False, float_format='%.8f')
    print(f"\n  Saved reference data to {ref_csv}")

    print("\n" + "=" * 60)
    print("  VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run verification script**

Run: `cd /root/twoannealing && python src/verify_matlab_reference.py`
Expected: All consistency checks PASS, plot saved to `results/tnm/ref_verification.png`

- [ ] **Step 3: Commit**

```bash
git add src/tnm_reference.py src/verify_matlab_reference.py
git commit -m "feat: add pure isothermal TNM reference model and MATLAB verification"
```

---

## Phase 2: Fix Model Physics (tnm_model.py)

### Task 2.1: Add Boltzmann superposition helper and fix _simulate_hold

**Files:**
- Modify: `src/tnm_model.py`

- [ ] **Step 1: Add `_compute_Tf_from_history` static method**

Replace the entire content of `src/tnm_model.py` with the corrected version. The key change is in `_simulate_hold` — instead of the single-step KWW formula, it now uses full Boltzmann superposition over the complete temperature history.

```python
"""
TNM (Tool-Narayanaswamy-Moynihan) model for structural relaxation in glasses.

Based on Song et al. (2020) "Activation Entropy as a Key Factor Controlling
the Memory Effect in Glasses" and Moynihan et al. (1976).

Implements finite-rate cooling/heating with correct Boltzmann superposition
throughout the full thermal history.
"""

import numpy as np

R_GAS = 8.314  # J/(mol·K)


class TNMModel:
    """TNM model for glass structural relaxation.

    Parameters
    ----------
    A : float
        Pre-exponential factor (seconds).
    H_star : float
        Activation enthalpy (J/mol).
    x : float
        Nonlinearity parameter, 0 < x <= 1.
    beta : float
        KWW stretch exponent, 0 < beta <= 1.
    T0 : float
        Equilibrium fictive temperature reference (K), typically near Tg.
    """

    def __init__(self, A, H_star, x, beta, T0):
        self.A = A
        self.H_star = H_star
        self.x = x
        self.beta = beta
        self.T0 = T0

    def tau(self, T, Tf):
        """Relaxation time at temperature T with fictive temperature Tf.

        Song et al. Eq. 3 / Suppl. Eq. 12:
        tau = A * exp[ x*H*/(R*T) + (1-x)*H*/(R*Tf) ]
        """
        exponent = (self.x * self.H_star) / (R_GAS * T) + \
                   ((1 - self.x) * self.H_star) / (R_GAS * Tf)
        return self.A * np.exp(exponent)

    @staticmethod
    def _kww(xi, beta):
        """KWW stretched exponential: phi(xi) = exp(-xi^beta)."""
        return np.exp(-(xi ** beta))

    @staticmethod
    def _compute_Tf(T_hist, xi_hist, beta):
        """Boltzmann superposition: compute Tf from full temperature + reduced time history.

        Tf(T_n, xi_n) = T_n - SUM_j (T_{j+1} - T_j) * exp(-(xi_n - xi_j)^beta)

        Parameters
        ----------
        T_hist : array of shape (n_steps+1,)
            Temperature history, including initial temperature.
        xi_hist : array of shape (n_steps+1,)
            Reduced time at each step (xi_hist[0] = 0 at start).
        beta : float
            KWW stretch exponent.

        Returns
        -------
        Tf : float
            Fictive temperature at the final step.
        """
        n = len(xi_hist) - 1
        xi_n = xi_hist[n]
        Tf = T_hist[n]
        for j in range(n - 1, -1, -1):
            dxi = xi_n - xi_hist[j]
            kww_val = TNMModel._kww(dxi, beta)
            if kww_val < 1e-10:
                break
            dT = T_hist[j + 1] - T_hist[j]
            Tf -= dT * kww_val
        return Tf

    def _simulate_ramp(self, T_start, T_end, rate, T_f_init, xi_init=0.0):
        """Simulate a temperature ramp with Boltzmann superposition.

        Returns (T_f_end, xi_end, T_hist, xi_hist).
        """
        dT_total = abs(T_end - T_start)
        if dT_total < 0.1 or rate <= 0:
            T_hist = np.array([T_start])
            xi_hist = np.array([xi_init])
            return T_f_init, xi_init, T_hist, xi_hist

        duration = dT_total / rate
        n_steps = max(10, int(np.ceil(dT_total / 2.0)))
        T_vals = np.linspace(T_start, T_end, n_steps + 1)[1:]

        T_hist = np.zeros(n_steps + 1)
        xi_hist = np.zeros(n_steps + 1)
        T_hist[0] = T_start
        xi_hist[0] = xi_init

        for i in range(n_steps):
            T_i = T_vals[i]
            T_f_prev = self._compute_Tf(T_hist[:i + 1], xi_hist[:i + 1], self.beta)
            tau_i = self.tau(T_i, T_f_prev)
            dt = duration / n_steps
            xi_hist[i + 1] = xi_hist[i] + dt / tau_i
            T_hist[i + 1] = T_i

        T_f_end = self._compute_Tf(T_hist, xi_hist, self.beta)
        return T_f_end, xi_hist[-1], T_hist, xi_hist

    def _simulate_hold(self, T_hold, t_hold, T_f_init, xi_init=0.0,
                       T_hist_prev=None, xi_hist_prev=None):
        """Isothermal hold with correct Boltzmann superposition.

        If T_hist_prev/xi_hist_prev are provided, the hold continues from
        the preceding thermal history (e.g., cooling ramp).

        Returns (T_f_end, xi_end, T_f_start, T_hist_full, xi_hist_full).
        """
        if T_hist_prev is not None and xi_hist_prev is not None:
            T_hist = np.array(list(T_hist_prev), dtype=float)
            xi_hist = np.array(list(xi_hist_prev), dtype=float)
        else:
            T_hist = np.array([T_f_init], dtype=float)
            xi_hist = np.array([xi_init], dtype=float)

        T_f_start = self._compute_Tf(T_hist, xi_hist, self.beta)

        if t_hold <= 0:
            return T_f_start, xi_init, T_f_start, T_hist, xi_hist

        # Generate hold time steps
        if t_hold < 1.0:
            n_steps = max(3, min(30, int(t_hold / 0.005) + 3))
            t_steps = np.linspace(0, t_hold, n_steps + 1)[1:]
        else:
            n_steps = max(8, min(50, int(np.log10(t_hold) * 20 + 10)))
            t_start_log = max(0.02, t_hold / 200)
            t_steps = np.logspace(np.log10(t_start_log),
                                  np.log10(t_hold), n_steps)
            t_steps = np.unique(np.round(t_steps, 8))

        n_existing = len(T_hist)
        T_hist = np.concatenate([T_hist, np.full(len(t_steps), T_hold)])
        xi_hist = np.concatenate([xi_hist, np.zeros(len(t_steps))])

        t_prev = 0.0
        for k, t_k in enumerate(t_steps):
            idx = n_existing + k
            Tf_prev = self._compute_Tf(T_hist[:idx], xi_hist[:idx], self.beta)
            tau_i = self.tau(T_hold, Tf_prev)
            dt = t_k - t_prev
            t_prev = t_k
            xi_hist[idx] = xi_hist[idx - 1] + dt / tau_i

        T_f_end = self._compute_Tf(T_hist, xi_hist, self.beta)
        return T_f_end, xi_hist[-1], T_f_start, T_hist, xi_hist

    def simulate_one_step(self, T_anneal, t_hold, T_initial=473.15,
                          cooling_rate=1.0, heating_rate=None):
        """One-step annealing: cool -> hold.

        Returns (T_f_end, T_f_hold_start).
        heating_rate is accepted but not used (reserved for future).
        """
        T_f_after_cool, xi_after_cool, T_hist_cool, xi_hist_cool = \
            self._simulate_ramp(T_initial, T_anneal, cooling_rate, T_initial, 0.0)
        T_f_end, xi_end, T_f_hold_start, _, _ = self._simulate_hold(
            T_anneal, t_hold, T_f_after_cool, xi_after_cool,
            T_hist_prev=T_hist_cool, xi_hist_prev=xi_hist_cool)
        return T_f_end, T_f_hold_start

    def simulate_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=473.15,
                          cooling_rate=1.0, heating_rate=None):
        """Two-step annealing: cool->hold1->cool->hold2.

        Returns (T_f_end_at_T2, T_f_start_of_hold2).
        heating_rate is accepted but not used (reserved for future).
        """
        # Cool to T1
        T_f_1, xi_1, T_hist_1, xi_hist_1 = self._simulate_ramp(
            T_initial, T1, cooling_rate, T_initial, 0.0)
        # Hold at T1
        T_f_1e, xi_1e, _, T_hist_1e, xi_hist_1e = self._simulate_hold(
            T1, t1_hold, T_f_1, xi_1,
            T_hist_prev=T_hist_1, xi_hist_prev=xi_hist_1)
        # Cool to T2 (continue from end of hold1)
        T_f_2, xi_2, T_hist_2, xi_hist_2 = self._simulate_ramp(
            T1, T2, cooling_rate, T_f_1e, xi_1e)
        # Prepend hold1 history to ramp2 history for continuity
        T_hist_full = np.concatenate([T_hist_1e, T_hist_2[1:]])
        xi_hist_full = np.concatenate([xi_hist_1e, xi_hist_2[1:]])
        # Hold at T2
        T_f_2e, xi_2e, T_f_hold_start, _, _ = self._simulate_hold(
            T2, t2_hold, T_f_2, xi_2,
            T_hist_prev=T_hist_full, xi_hist_prev=xi_hist_full)
        return T_f_2e, T_f_hold_start

    def delta_H_normalized(self, T_anneal, t_hold, T_initial=473.15,
                           cooling_rate=1.0):
        """One-step delta_H normalized to [0, 1] using T0 reference.

        delta_H = (Tf_end - T_anneal) / (T0 - T_anneal)
        Range: 0 (fully relaxed, Tf=T_anneal) to ~1 (unrelaxed, Tf=T0).
        """
        T_f_end, T_f_start = self.simulate_one_step(
            T_anneal, t_hold, T_initial, cooling_rate)
        return (T_f_end - T_anneal) / max(self.T0 - T_anneal, 1.0)

    def delta_H_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=473.15,
                         cooling_rate=1.0):
        """Two-step delta_H normalized using T0 reference.

        delta_H = (Tf_end - T2) / (T0 - T2)
        """
        T_f_end, T_f_start = self.simulate_two_step(
            T1, t1_hold, T2, t2_hold, T_initial, cooling_rate)
        return (T_f_end - T2) / max(self.T0 - T2, 1.0)
```

- [ ] **Step 2: Quick smoke test — verify import and basic simulation runs**

Run:
```bash
cd /root/twoannealing && python -c "
import numpy as np
from src.tnm_model import TNMModel
# Test with Au MBMG params from MATLAB reference
model = TNMModel(A=6e-22, H_star=164e3, x=0.6, beta=0.43, T0=430.0)
Tf_end, Tf_start = model.simulate_one_step(T_anneal=383.0, t_hold=100.0)
print(f'One-step: Tf_end={Tf_end:.3f} K')
# Test two-step
Tf_end2, Tf_start2 = model.simulate_two_step(T1=383.0, t1_hold=50.0, T2=373.0, t2_hold=100.0)
print(f'Two-step: Tf_end={Tf_end2:.3f} K')
# Physical check: Tf should be between T_anneal and T_initial
assert 383.0 <= Tf_end <= 473.15, 'Tf_end out of range!'
assert 373.0 <= Tf_end2 <= 473.15, 'Tf_end2 out of range!'
print('All checks passed!')
"
```
Expected: `All checks passed!`

- [ ] **Step 3: Compare old vs new for instantaneous quench (should match)**

Run:
```bash
cd /root/twoannealing && python -c "
import numpy as np
from src.tnm_model import TNMModel

# Model with finite cooling
A, H, x, beta = 6e-22, 164e3, 0.6, 0.43
m = TNMModel(A, H, x, beta, T0=430)

# Instantaneous quench: cooling_rate = VERY high
# Old behavior: hold formula should be close when xi_ramp is negligible
Tf_slow, _ = m.simulate_one_step(T_anneal=383.0, t_hold=1000.0, cooling_rate=1.0)
Tf_fast, _ = m.simulate_one_step(T_anneal=383.0, t_hold=1000.0, cooling_rate=1e6)
print(f'Slow cool (1 K/s): Tf_end = {Tf_slow:.4f} K')
print(f'Fast cool (1e6 K/s): Tf_end = {Tf_fast:.4f} K')
print(f'Difference: {abs(Tf_slow - Tf_fast):.6f} K')
# Fast cooling should give LOWER Tf (less relaxation during cool = more room to relax during hold)
assert Tf_fast >= Tf_slow, 'Fast cool should give higher Tf (less pre-relaxation)'
print('Physical consistency: OK')
"
```
Expected: `Physical consistency: OK`

- [ ] **Step 4: Commit**

```bash
git add src/tnm_model.py
git commit -m "fix: correct Boltzmann superposition in TNM hold simulation

Replace single-step KWW formula in _simulate_hold with full Boltzmann
superposition over complete temperature history via _compute_Tf_from_history.
Add heating_rate parameter to simulate_one_step/simulate_two_step signatures.
Refactor ramp to return history arrays for continuity into hold."
```

---

## Phase 3: Fix Data Loading (tnm_conditions.py)

### Task 3.1: Fix all data loading bugs

**Files:**
- Modify: `src/tnm_conditions.py`

- [ ] **Step 1: Write the corrected file**

Replace the content of `src/tnm_conditions.py` with:

```python
"""
Protocol definitions and data loading for TNM model fitting.

Protocols:
- One-step 50C / 70C (two runs each, different cooling rates)
- Two-step hi->lo 90C->80C (Grp A: T1=0.833min, Grp B: T1=8.333min)
- Kovacs up-jump 80C->90C (Grp A: T1=0.833min, Grp B: T1=8.333min)
"""

import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dsc')

# Hold times in seconds (from experiment design)
HOLD_TIMES_SEC = np.array([
    0.1002, 0.4998, 0.7980, 1.0002, 4.9980,
    10.0200, 49.9800, 100.0200, 499.9800, 1000.0200
])

# Cooling/heating rates
COOLING_RATE = 1.0       # 60 C/min = 1.0 K/s
HEATING_RATE = 10.0 / 60.0  # 10 C/min = 0.1667 K/s

# Temperatures in Kelvin
T_INITIAL = 200 + 273.15   # 473.15 K (start above Tg)
T_50C = 50 + 273.15        # 323.15 K
T_70C = 70 + 273.15        # 343.15 K
T_80C = 80 + 273.15        # 353.15 K
T_90C = 90 + 273.15        # 363.15 K

# T1 hold times for two-step/Kovacs (seconds)
T1_HOLD_SHORT = 0.833 * 60.0   # 49.98 s (Group A)
T1_HOLD_LONG = 8.333 * 60.0    # 499.98 s (Group B)


def load_experimental_data():
    """Load all experimental CSV files. Returns dict of DataFrames."""
    data = {}
    for name, csv_file in [
        ('os50', 'onestep_50C_v2_results.csv'),
        ('os70', 'onestep_70C_v2_results.csv'),
        ('ts', 'twosteps_v2_results.csv'),
        ('kovacs', 'kovacs_v2_results.csv'),
    ]:
        path = os.path.join(RESULTS_DIR, csv_file)
        df = pd.read_csv(path)
        # Subtract the +1 kJ/mol processing offset
        if 'delta_H_kJmol' in df.columns:
            df['delta_H_kJmol'] = df['delta_H_kJmol'] - 1.0
        data[name] = df
    return data


def extract_one_step(df):
    """Extract hold times and delta_H from one-step DataFrame.

    Data has columns: ramp, hold_s, T_onset_C, delta_H_kJmol
    20 rows: first 10 = run1 (fast cool), last 10 = run2 (slow cool).

    Returns (t_hold_sec, delta_H, groups).
    """
    t_hold_sec = df['hold_s'].values
    delta_H = df['delta_H_kJmol'].values
    n = len(t_hold_sec)
    # First half = run1, second half = run2
    groups = np.array(['run1'] * (n // 2) + ['run2'] * (n - n // 2))
    return t_hold_sec, delta_H, groups


def extract_two_step(df):
    """Extract hold times and delta_H from two-step/Kovacs DataFrame.

    Data has columns: ramp, T1_hold_s, T2_hold_s, T_onset_C, delta_H_kJmol
    T1_hold_s = 49.98 (Grp A) or 499.98 (Grp B).

    Returns (t2_hold_sec, delta_H, groups).
    """
    t2_hold_sec = df['T2_hold_s'].values
    delta_H = df['delta_H_kJmol'].values
    t1_vals = df['T1_hold_s'].values
    # Threshold: 100s separates short (49.98s) from long (499.98s)
    groups = np.where(t1_vals < 100, 'grpA', 'grpB')
    return t2_hold_sec, delta_H, groups


def build_target_vectors():
    """Build target data structures for fitting.

    Returns dict:
        'os50': {'t': array, 'dH': array, 'groups': array}
        'os70': {'t': array, 'dH': array, 'groups': array}
        'ts': {'t': array, 'dH': array, 'groups': array}
        'kovacs': {'t': array, 'dH': array, 'groups': array}
    """
    exp_data = load_experimental_data()

    result = {}

    # One-step: use extract_one_step helper
    for key in ['os50', 'os70']:
        t, dh, grp = extract_one_step(exp_data[key])
        result[key] = {'t': t, 'dH': dh, 'groups': grp}

    # Two-step and Kovacs: use extract_two_step helper
    for key in ['ts', 'kovacs']:
        t, dh, grp = extract_two_step(exp_data[key])
        result[key] = {'t': t, 'dH': dh, 'groups': grp}

    return result
```

- [ ] **Step 2: Verify data loading works correctly**

Run:
```bash
cd /root/twoannealing && python -c "
from src.tnm_conditions import build_target_vectors, HOLD_TIMES_SEC

targets = build_target_vectors()
for key, d in targets.items():
    print(f'{key}: {len(d[\"t\"])} points, dH range=[{d[\"dH\"].min():.1f}, {d[\"dH\"].max():.1f}] kJ/mol')
    unique_groups = sorted(set(d['groups']))
    print(f'  groups: {unique_groups}')
    for g in unique_groups:
        mask = d['groups'] == g
        print(f'  {g}: {mask.sum()} points')
print()
print(f'HOLD_TIMES_SEC: {HOLD_TIMES_SEC}')
print(f'Match data[0:10]: {list(HOLD_TIMES_SEC)}')
"
```
Expected: 20 points per protocol, groups shown, dH range starts near 0 (after offset subtraction).

- [ ] **Step 3: Commit**

```bash
git add src/tnm_conditions.py
git commit -m "fix: correct data loading for v2 CSV format

Fix CSV filenames (add _v2 infix), column names (hold_s/T1_hold_s/T2_hold_s
instead of hold_min/T1_hold_min/T2_hold_min), remove spurious *60 unit
conversion, and fix group classification threshold for second-based values.
Also subtract +1 kJ/mol processing offset from delta_H values."
```

---

## Phase 4: Fitting Strategy (tnm_fit.py, tnm_main.py)

### Task 4.1: Rewrite cost functions with wider bounds

**Files:**
- Modify: `src/tnm_fit.py`

- [ ] **Step 1: Write new fitting module**

```python
"""
Staged parameter optimization for TNM model fitting to DSC enthalpy data.

Strategy:
  Stage 1: Shape-fitting one-step (normalized [0,1]) — fits x, beta.
  Stage 2: Absolute-scale fitting with all protocols (kJ/mol) — refines all params.
  Stage 3: Kovacs prediction (no fitting, hold-out validation).
"""

import numpy as np
from scipy.optimize import minimize
from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C, T_INITIAL,
    COOLING_RATE, build_target_vectors, T1_HOLD_SHORT, T1_HOLD_LONG,
)


def make_model(logA, H_star, x, beta, T0):
    A = 10.0 ** logA
    return TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)


def simulate_protocol_raw(model, protocol_key):
    """Simulate raw delta_H_norm for a protocol at the 10 canonical hold times.

    Returns (t_sec, dh_sim_norm) arrays.
    """
    if protocol_key in ('os50', 'os70'):
        T_a = T_50C if protocol_key == 'os50' else T_70C
        dh = np.array([model.delta_H_normalized(T_a, th, T_INITIAL, COOLING_RATE)
                       for th in HOLD_TIMES_SEC])
        return HOLD_TIMES_SEC.copy(), dh

    elif protocol_key == 'tsA':
        dh = np.array([model.delta_H_two_step(T_90C, T1_HOLD_SHORT, T_80C, th,
                                               T_INITIAL, COOLING_RATE)
                       for th in HOLD_TIMES_SEC])
        return HOLD_TIMES_SEC.copy(), dh

    elif protocol_key == 'tsB':
        dh = np.array([model.delta_H_two_step(T_90C, T1_HOLD_LONG, T_80C, th,
                                               T_INITIAL, COOLING_RATE)
                       for th in HOLD_TIMES_SEC])
        return HOLD_TIMES_SEC.copy(), dh

    elif protocol_key == 'kvA':
        dh = np.array([model.delta_H_two_step(T_80C, T1_HOLD_SHORT, T_90C, th,
                                               T_INITIAL, COOLING_RATE)
                       for th in HOLD_TIMES_SEC])
        return HOLD_TIMES_SEC.copy(), dh

    elif protocol_key == 'kvB':
        dh = np.array([model.delta_H_two_step(T_80C, T1_HOLD_LONG, T_90C, th,
                                               T_INITIAL, COOLING_RATE)
                       for th in HOLD_TIMES_SEC])
        return HOLD_TIMES_SEC.copy(), dh

    else:
        raise ValueError(f"Unknown protocol: {protocol_key}")


def simulate_protocol_match_exp(model, protocol_key, targets):
    """Simulate delta_H_norm, interpolated to match experimental hold times.

    Returns (dh_sim_norm, dh_exp) arrays matched at experimental time points.
    """
    # Get sim at canonical hold times
    t_sim, dh_sim = simulate_protocol_raw(model, protocol_key)

    # Get experimental data for this protocol
    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'grpA'),
        'tsB': ('ts', 'grpB'),
        'kvA': ('kovacs', 'grpA'),
        'kvB': ('kovacs', 'grpB'),
    }
    exp_key, grp_key = group_map[protocol_key]
    exp = targets[exp_key]
    mask = exp['groups'] == grp_key
    dh_exp = exp['dH'][mask]
    t_exp = exp['t'][mask]

    # Normalize both sim and exp to [0, 1] for shape comparison
    s_min, s_max = dh_sim.min(), dh_sim.max()
    if s_max - s_min < 1e-10:
        dh_sim_norm = np.zeros_like(dh_sim)
    else:
        dh_sim_norm = (dh_sim - s_min) / (s_max - s_min)

    e_min, e_max = dh_exp.min(), dh_exp.max()
    if e_max - e_min < 1e-10:
        dh_exp_norm = np.zeros_like(dh_exp)
    else:
        dh_exp_norm = (dh_exp - e_min) / (e_max - e_min)

    # Interpolate sim to experimental time points
    dh_sim_interp = np.interp(t_exp, t_sim, dh_sim_norm)

    return dh_sim_interp, dh_exp_norm, dh_exp, t_exp, dh_sim, t_sim


def stage1_cost_shape(packed, targets):
    """Stage 1: Shape-only cost on one-step 50C + 70C (normalized [0,1])."""
    logA, H_star, x, beta, T0 = packed[0], packed[1], packed[2], packed[3], packed[4]
    if not (0.01 < x <= 0.99 and 0.05 < beta <= 0.99 and H_star > 50000 and T0 > 350):
        return 1e12

    try:
        model = make_model(logA, H_star, x, beta, T0)
    except Exception:
        return 1e12

    chi2 = 0.0
    n_pts = 0
    for key in ['os50', 'os70']:
        dh_sim_norm, dh_exp_norm, _, _, _, _ = \
            simulate_protocol_match_exp(model, key, targets)
        chi2 += np.sum((dh_sim_norm - dh_exp_norm) ** 2)
        n_pts += len(dh_sim_norm)

    return np.sqrt(chi2 / max(n_pts, 1))


def stage2_cost_absolute(packed, targets):
    """Stage 2: Absolute-scale cost using linear mapping sim_norm -> kJ/mol.

    Fits: dh_exp ≈ a * dh_sim_norm + b  for each protocol group independently,
    then computes weighted RMS across all fitted protocols.
    """
    logA, H_star, x, beta, T0 = packed[0], packed[1], packed[2], packed[3], packed[4]
    if not (0.01 < x <= 0.99 and 0.05 < beta <= 0.99 and H_star > 50000 and T0 > 350):
        return 1e12

    try:
        model = make_model(logA, H_star, x, beta, T0)
    except Exception:
        return 1e12

    chi2 = 0.0
    n_pts = 0
    for key in ['os50', 'os70', 'tsA', 'tsB']:
        _, dh_exp_norm_vals, dh_exp_kJ, _, dh_sim_norm_raw, _ = \
            simulate_protocol_match_exp(model, key, targets)

        # Scale sim norm to match exp kJ via linear fit (per protocol group)
        a, b = np.polyfit(dh_exp_norm_vals, dh_exp_kJ, 1)
        dh_sim_kJ = a * dh_exp_norm_vals + b
        chi2 += np.sum((dh_sim_kJ - dh_exp_kJ) ** 2)
        n_pts += len(dh_exp_kJ)

    return np.sqrt(chi2 / max(n_pts, 1))


def run_stage1(targets, seed=None):
    """Multi-start L-BFGS-B for one-step shape fitting."""
    rng = np.random.RandomState(seed)
    bounds = [
        (-40, -8),          # logA
        (100000, 500000),   # H_star (J/mol)
        (0.05, 0.9),        # x
        (0.1, 0.9),         # beta
        (365, 400),         # T0 (K)
    ]

    best_result = None
    best_cost = np.inf
    n_starts = 30

    print(f"  Stage 1: Multi-start L-BFGS-B ({n_starts} starts, 5 params)...")
    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(
            stage1_cost_shape, x0, args=(targets,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8},
        )
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res
        if (k + 1) % 5 == 0:
            print(f"    Start {k+1}/{n_starts}, best cost = {best_cost:.6f}")

    print(f"    Final best cost: {best_cost:.6f}")
    best = best_result.x
    params = {'logA': best[0], 'H_star': best[1], 'x': best[2],
              'beta': best[3], 'T0': best[4]}
    return params, best_result


def run_stage2(targets, stage1_params, seed=None):
    """Refine with absolute-scale cost using all protocols."""
    x0 = [stage1_params['logA'], stage1_params['H_star'],
          stage1_params['x'], stage1_params['beta'], stage1_params['T0']]
    bounds = [
        (-40, -8), (100000, 500000), (0.05, 0.9), (0.1, 0.9), (365, 400),
    ]

    print("  Stage 2: L-BFGS-B with absolute-scale cost...")
    result = minimize(
        stage2_cost_absolute, x0, args=(targets,),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-10},
    )
    print(f"    Stage 2 best cost: {result.fun:.6f}")

    best = result.x
    params = {'logA': best[0], 'H_star': best[1], 'x': best[2],
              'beta': best[3], 'T0': best[4]}
    return params, result


def compute_r_squared(dH_sim, dH_exp):
    ss_res = np.sum((dH_sim - dH_exp) ** 2)
    ss_tot = np.sum((dH_exp - np.mean(dH_exp)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

- [ ] **Step 2: Verify import**

Run: `cd /root/twoannealing && python -c "from src.tnm_fit import run_stage1, run_stage2, compute_r_squared; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/tnm_fit.py
git commit -m "feat: redesign fitting with wider bounds, 5-param optimization

Replace shape-only cost with 2-stage strategy: Stage 1 shape-fits one-step
curves (x, beta dominant), Stage 2 uses absolute kJ/mol cost with all
protocols. T0 now treated as fitting parameter (365-400K). Bounds widened
significantly for PS (logA: -40 to -8, H*: 100-500 kJ/mol, x: 0.05-0.9,
beta: 0.1-0.9)."
```

---

### Task 4.2: Update driver script

**Files:**
- Modify: `src/tnm_main.py`

- [ ] **Step 1: Write updated driver**

```python
#!/usr/bin/env python3
"""
TNM Model Driver — Corrected model + 2-stage fitting + Kovacs prediction.

Fit strategy:
  Stage 1: Shape-fit one-step (normalized [0,1]) → rough x, beta.
  Stage 2: Absolute-scale fit (kJ/mol) with one-step + two-step → refine all.
  Stage 3: Kovacs prediction (no fitting, hold-out validation).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C, T_INITIAL,
    COOLING_RATE, build_target_vectors, T1_HOLD_SHORT, T1_HOLD_LONG,
)
from tnm_fit import (
    make_model, run_stage1, run_stage2, compute_r_squared,
    simulate_protocol_match_exp, simulate_protocol_raw,
)
from tnm_plots import (
    plot_one_step_fit, plot_two_step_fit, plot_kovacs_prediction,
    plot_parity, plot_tf_evolution, plot_residuals,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_sim_kJ_data(model, targets):
    """Compute simulated delta_H in kJ/mol for all 6 protocol groups.

    Uses per-group linear scaling: sim_norm -> exp_kJ.
    """
    sim_data_kJ = {}
    scaling = {}

    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'grpA'),
        'tsB': ('ts', 'grpB'),
        'kvA': ('kovacs', 'grpA'),
        'kvB': ('kovacs', 'grpB'),
    }

    for sim_key, (exp_key, grp_key) in group_map.items():
        # Get sim at canonical times
        _, dh_sim_raw = simulate_protocol_raw(model, sim_key)

        # Get exp
        exp = targets[exp_key]
        mask = exp['groups'] == grp_key
        dh_exp = exp['dH'][mask]
        t_exp = exp['t'][mask]

        # Normalize sim to [0,1]
        s_min, s_max = dh_sim_raw.min(), dh_sim_raw.max()
        if s_max - s_min < 1e-10:
            dh_sim_norm = np.zeros_like(dh_sim_raw)
        else:
            dh_sim_norm = (dh_sim_raw - s_min) / (s_max - s_min)

        # Normalize exp to [0,1]
        e_min, e_max = dh_exp.min(), dh_exp.max()
        if e_max - e_min < 1e-10:
            dh_exp_norm = np.zeros_like(dh_exp)
        else:
            dh_exp_norm = (dh_exp - e_min) / (e_max - e_min)

        # Interpolate sim to exp times
        dh_sim_norm_interp = np.interp(t_exp, HOLD_TIMES_SEC, dh_sim_norm)

        # Linear scale to kJ
        a, b = np.polyfit(dh_sim_norm_interp, dh_exp, 1)
        scaling[sim_key] = (a, b)
        sim_data_kJ[sim_key] = a * dh_sim_norm_interp + b

    return sim_data_kJ, scaling


def compute_all_r2(sim_data_kJ, targets):
    """Compute R^2 for each group using kJ/mol values."""
    r2_values = {}
    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'grpA'),
        'tsB': ('ts', 'grpB'),
        'kvA': ('kovacs', 'grpA'),
        'kvB': ('kovacs', 'grpB'),
    }
    for sim_key, (exp_key, grp_key) in group_map.items():
        exp = targets[exp_key]
        mask = exp['groups'] == grp_key
        dh_exp = exp['dH'][mask]
        if sim_key in sim_data_kJ:
            dh_sim = sim_data_kJ[sim_key]
            r2_values[sim_key] = compute_r_squared(dh_sim, dh_exp)
        else:
            r2_values[sim_key] = np.nan
    return r2_values


def main():
    print("=" * 70)
    print("  TNM Model — Corrected Fitting to PS Annealing Data")
    print("=" * 70)

    # Load targets
    print("\n[1] Loading experimental data...")
    targets = build_target_vectors()
    for key, d in targets.items():
        print(f"  {key}: {len(d['t'])} pts, "
              f"dH range=[{d['dH'].min():.1f}, {d['dH'].max():.1f}] kJ/mol")

    # Stage 1: Shape-fit one-step
    print("\n[2] Stage 1: Shape-fitting one-step (50C + 70C)...")
    s1_params, s1_result = run_stage1(targets, seed=42)
    model_s1 = make_model(**s1_params)

    sim_kJ_s1, scaling_s1 = compute_sim_kJ_data(model_s1, targets)
    r2_s1 = compute_all_r2(sim_kJ_s1, targets)

    print(f"\n  Stage 1 results:")
    print(f"    log10(A) = {s1_params['logA']:.3f}  "
          f"(A = {10**s1_params['logA']:.2e} s)")
    print(f"    H*       = {s1_params['H_star']/1000:.1f} kJ/mol")
    print(f"    x        = {s1_params['x']:.4f}")
    print(f"    beta     = {s1_params['beta']:.4f}")
    print(f"    T0       = {s1_params['T0']:.1f} K ({s1_params['T0']-273.15:.1f} C)")
    print(f"    Cost     = {s1_result.fun:.6f}")
    for k in ['os50', 'os70', 'tsA', 'tsB', 'kvA', 'kvB']:
        v = r2_s1.get(k, np.nan)
        if not np.isnan(v):
            print(f"    R^2 {k:6s} = {v:.4f}")

    # Stage 2: Absolute-scale fit
    print("\n[3] Stage 2: Absolute-scale refinement (all protocols)...")
    s2_params, s2_result = run_stage2(targets, s1_params, seed=42)
    model_s2 = make_model(**s2_params)

    sim_kJ_s2, scaling_s2 = compute_sim_kJ_data(model_s2, targets)
    r2_s2 = compute_all_r2(sim_kJ_s2, targets)

    print(f"\n  Stage 2 results:")
    print(f"    log10(A) = {s2_params['logA']:.3f}  "
          f"(A = {10**s2_params['logA']:.2e} s)")
    print(f"    H*       = {s2_params['H_star']/1000:.1f} kJ/mol")
    print(f"    x        = {s2_params['x']:.4f}")
    print(f"    beta     = {s2_params['beta']:.4f}")
    print(f"    T0       = {s2_params['T0']:.1f} K ({s2_params['T0']-273.15:.1f} C)")
    print(f"    Cost     = {s2_result.fun:.6f}")
    for k in ['os50', 'os70', 'tsA', 'tsB', 'kvA', 'kvB']:
        v = r2_s2.get(k, np.nan)
        if not np.isnan(v):
            print(f"    R^2 {k:6s} = {v:.4f}")

    # Stage 3: Kovacs prediction
    print("\n[4] Stage 3: Kovacs prediction (hold-out validation)...")
    for k in ['kvA', 'kvB']:
        v = r2_s2.get(k, np.nan)
        print(f"    R^2 {k:6s} = {v:.4f}")

    # Save results
    print("\n[5] Saving fit results...")
    results_rows = []
    for stage_name, params, r2_dict in [
        ('Stage1_Shape', s1_params, r2_s1),
        ('Stage2_Absolute', s2_params, r2_s2),
    ]:
        row = {
            'stage': stage_name,
            'log10_A': params['logA'],
            'A_s': 10.0 ** params['logA'],
            'H_star_kJmol': params['H_star'] / 1000.0,
            'x': params['x'],
            'beta': params['beta'],
            'T0_K': params['T0'],
            'T0_C': params['T0'] - 273.15,
        }
        for k, v in r2_dict.items():
            row[f'R2_{k}'] = v
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(RESULTS_DIR, 'tnm_fit_results.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Display final summary
    print("\n  " + "=" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['stage']}:")
        print(f"    log10(A/s) = {row['log10_A']:.3f}, A = {row['A_s']:.2e} s")
        print(f"    H* = {row['H_star_kJmol']:.1f} kJ/mol, x = {row['x']:.4f}, "
              f"beta = {row['beta']:.4f}, T0 = {row['T0_K']:.1f} K")
        for k, v in row.items():
            if k.startswith('R2_') and not (isinstance(v, float) and np.isnan(v)):
                print(f"    {k}: {v:.4f}")
    print("  " + "=" * 80)

    # Generate plots
    print("\n[6] Generating plots...")
    sim_for_plot = {k: sim_kJ_s2[k] for k in sim_kJ_s2}

    plot_one_step_fit(
        targets, sim_for_plot, s2_params,
        os.path.join(RESULTS_DIR, 'tnm_onestep_fit.png'),
    )
    plot_two_step_fit(
        targets, sim_for_plot,
        os.path.join(RESULTS_DIR, 'tnm_twostep_fit.png'),
    )
    plot_kovacs_prediction(
        targets, sim_for_plot,
        os.path.join(RESULTS_DIR, 'tnm_kovacs_prediction.png'),
    )
    plot_parity(
        sim_for_plot, targets,
        os.path.join(RESULTS_DIR, 'tnm_parity.png'),
    )
    plot_tf_evolution(
        model_s2,
        os.path.join(RESULTS_DIR, 'tnm_tf_evolution.png'),
    )
    plot_residuals(
        targets, sim_for_plot,
        os.path.join(RESULTS_DIR, 'tnm_residuals.png'),
    )

    # Summary
    print("\n" + "=" * 70)
    print("  TNM FITTING COMPLETE")
    print("=" * 70)
    print(f"  Final: logA={s2_params['logA']:.3f}, "
          f"H*={s2_params['H_star']/1000:.1f} kJ/mol, "
          f"x={s2_params['x']:.4f}, beta={s2_params['beta']:.4f}, "
          f"T0={s2_params['T0']:.1f} K")
    print(f"  Outputs in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify driver imports**

Run: `cd /root/twoannealing && python -c "import src.tnm_main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/tnm_main.py
git commit -m "feat: update driver for 5-param fitting with corrected model"
```

---

### Task 4.3: Update plot functions for new data format

**Files:**
- Modify: `src/tnm_plots.py`

- [ ] **Step 1: Fix plot_one_step_fit to use new data structure**

The current `plot_one_step_fit` splits into run1/run2 by sequential index. With the corrected `tnm_conditions.py`, the one-step data still splits correctly into `run1`/`run2` groups. However, we need to verify the plot code handles the group-based access correctly. The minimal fix is to update the hold time annotation.

No major changes needed in plots.py — the `groups` field in targets already works. The hold time values used in plots were already in seconds (using `exp['t'] / 60.0` to convert to minutes for display), which is fine since `t` is now correctly in seconds.

- [ ] **Step 2: Verify plot imports**

Run: `cd /root/twoannealing && python -c "from src.tnm_plots import plot_one_step_fit, plot_parity; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit (if changes made)**

```bash
git add src/tnm_plots.py
git commit -m "fix: adapt plots for corrected data loading"
```
(If no changes needed, skip this commit.)

---

## Phase 5: Error Analysis

### Task 5.1: Write comprehensive error analysis module

**Files:**
- Create: `src/tnm_error_analysis.py`

- [ ] **Step 1: Write error analysis script**

```python
#!/usr/bin/env python3
"""
Comprehensive error analysis for TNM model fitting.

Generates:
  1. Per-protocol residual analysis (residuals vs log hold time)
  2. Bootstrap parameter uncertainty (200 resamples)
  3. Parameter correlation analysis
  4. Kovacs memory effect amplitude comparison
  5. Sensitivity to cooling rate and T0
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C, T_INITIAL,
    COOLING_RATE, build_target_vectors, T1_HOLD_SHORT, T1_HOLD_LONG,
)
from tnm_fit import (
    make_model, simulate_protocol_raw, simulate_protocol_match_exp,
    compute_r_squared, stage2_cost_absolute, run_stage2,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'os50': '#2166AC', 'os70': '#4393C3',
    'tsA': '#B2182B', 'tsB': '#D6604D',
    'kvA': '#4DAF4A', 'kvB': '#984EA3',
}


def residual_analysis(model, targets, sim_data_kJ):
    """Plot residuals (sim - exp) vs log hold time for all protocols."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'grpA'),
        'tsB': ('ts', 'grpB'),
        'kvA': ('kovacs', 'grpA'),
        'kvB': ('kovacs', 'grpB'),
    }

    titles = {
        'os50': 'One-step 50C', 'os70': 'One-step 70C',
        'tsA': 'Two-step Grp A', 'tsB': 'Two-step Grp B',
        'kvA': 'Kovacs Grp A', 'kvB': 'Kovacs Grp B',
    }

    all_residuals = []

    for idx, (sim_key, (exp_key, grp_key)) in enumerate(group_map.items()):
        ax = axes.flat[idx]
        exp = targets[exp_key]
        mask = exp['groups'] == grp_key
        dh_exp = exp['dH'][mask]
        t_exp = exp['t'][mask]
        dh_sim = sim_data_kJ[sim_key]

        residuals = dh_sim - dh_exp
        all_residuals.extend(residuals)

        ax.plot(t_exp / 60.0, residuals, 'o', color=COLORS[sim_key],
                markersize=7, markerfacecolor='white', markeredgewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Hold time (min)')
        ax.set_ylabel('Residual (kJ/mol)')
        ax.set_title(f'{titles[sim_key]}')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis: TNM Model - Experiment', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_residual_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved residual analysis: {out_path}")

    # Summary statistics
    all_residuals = np.array(all_residuals)
    print(f"\n  Residual summary:")
    print(f"    Mean:  {np.mean(all_residuals):.2f} kJ/mol")
    print(f"    Std:   {np.std(all_residuals):.2f} kJ/mol")
    print(f"    RMSE:  {np.sqrt(np.mean(all_residuals**2)):.2f} kJ/mol")
    print(f"    Max:   {np.max(np.abs(all_residuals)):.2f} kJ/mol")


def bootstrap_uncertainty(targets, stage1_params, n_bootstrap=200, seed=42):
    """Bootstrap parameter uncertainty by resampling hold times."""
    print(f"\n  Bootstrap ({n_bootstrap} resamples)...")

    rng = np.random.RandomState(seed)
    all_params = []

    group_keys = ['os50', 'os70', 'tsA', 'tsB']

    for b in range(n_bootstrap):
        # Build bootstrapped targets by resampling within each protocol
        bs_targets = {}
        for exp_key in ['os50', 'os70', 'ts', 'kovacs']:
            orig = targets[exp_key]
            n = len(orig['t'])
            idx = rng.randint(0, n, n)
            bs_targets[exp_key] = {
                't': orig['t'][idx],
                'dH': orig['dH'][idx],
                'groups': orig['groups'][idx],
            }

        try:
            params, result = run_stage2(bs_targets, stage1_params, seed=seed + b)
            all_params.append([params['logA'], params['H_star'], params['x'],
                               params['beta'], params['T0']])
        except Exception:
            continue

        if (b + 1) % 50 == 0:
            print(f"    Bootstrap {b+1}/{n_bootstrap}")

    all_params = np.array(all_params)
    param_names = ['logA', 'H_star', 'x', 'beta', 'T0']

    print(f"\n  Parameter uncertainty (95% CI):")
    print(f"  {'Param':<8} {'Mean':<12} {'2.5%':<12} {'97.5%':<12} {'Std':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i, name in enumerate(param_names):
        vals = all_params[:, i]
        mean = np.mean(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        std = np.std(vals)
        print(f"  {name:<8} {mean:<12.4f} {lo:<12.4f} {hi:<12.4f} {std:<12.4f}")

    # Save bootstrap results
    bs_df = pd.DataFrame(all_params, columns=param_names)
    bs_csv = os.path.join(RESULTS_DIR, 'tnm_bootstrap_params.csv')
    bs_df.to_csv(bs_csv, index=False, float_format='%.6f')
    print(f"\n  Saved bootstrap results: {bs_csv}")

    return all_params, param_names


def parameter_correlation(all_params, param_names):
    """Plot parameter correlation matrix."""
    corr = np.corrcoef(all_params.T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, color='black' if abs(corr[i, j]) < 0.7 else 'white')
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Parameter Correlation Matrix (Bootstrap)')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'tnm_param_correlation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved correlation matrix: {out_path}")


def sensitivity_analysis(params):
    """Sensitivity of R^2 to cooling rate and T0 variations."""
    print("\n  Sensitivity analysis...")

    logA, H_star, x, beta, T0 = (params['logA'], params['H_star'],
                                  params['x'], params['beta'], params['T0'])
    model_base = make_model(logA, H_star, x, beta, T0)

    # Cooling rate sensitivity
    rates = [0.2, 0.5, 1.0, 2.0, 5.0]
    print(f"\n  Cooling rate sensitivity (base rate = 1.0 K/s):")
    print(f"  {'Rate':<10} {'os50 Tf':<12} {'os70 Tf':<12}")
    for rate in rates:
        m = TNMModel(A=10**logA, H_star=H_star, x=x, beta=beta, T0=T0)
        tf50 = m.simulate_one_step(T_50C, 100.0, T_INITIAL, rate)[0]
        tf70 = m.simulate_one_step(T_70C, 100.0, T_INITIAL, rate)[0]
        print(f"  {rate:<10.1f} {tf50:<12.2f} {tf70:<12.2f}")

    # T0 sensitivity
    T0_vals = np.linspace(T0 - 10, T0 + 10, 5)
    print(f"\n  T0 sensitivity (base T0 = {T0:.1f} K):")
    print(f"  {'T0 (K)':<10} {'os50 dH_norm':<14} {'os70 dH_norm':<14}")
    for t0 in T0_vals:
        m = TNMModel(A=10**logA, H_star=H_star, x=x, beta=beta, T0=t0)
        dh50 = m.delta_H_normalized(T_50C, 100.0, T_INITIAL, COOLING_RATE)
        dh70 = m.delta_H_normalized(T_70C, 100.0, T_INITIAL, COOLING_RATE)
        print(f"  {t0:<10.1f} {dh50:<14.6f} {dh70:<14.6f}")


def main():
    print("=" * 60)
    print("  TNM Model — Error Analysis")
    print("=" * 60)

    # Load fit results
    csv_path = os.path.join(RESULTS_DIR, 'tnm_fit_results.csv')
    if not os.path.exists(csv_path):
        print("ERROR: Run tnm_main.py first to generate fit results.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    s2_row = df[df['stage'] == 'Stage2_Absolute'].iloc[0]
    params = {
        'logA': s2_row['log10_A'],
        'H_star': s2_row['H_star_kJmol'] * 1000.0,
        'x': s2_row['x'],
        'beta': s2_row['beta'],
        'T0': s2_row['T0_K'],
    }

    print(f"\n  Loaded fit parameters:")
    print(f"    logA = {params['logA']:.3f}")
    print(f"    H*   = {params['H_star']/1000:.1f} kJ/mol")
    print(f"    x    = {params['x']:.4f}")
    print(f"    beta = {params['beta']:.4f}")
    print(f"    T0   = {params['T0']:.1f} K")

    # Build model and sim data
    model = make_model(**params)
    targets = build_target_vectors()

    # Compute sim data in kJ/mol (same as tnm_main.py)
    from tnm_main import compute_sim_kJ_data
    sim_data_kJ, _ = compute_sim_kJ_data(model, targets)

    # 1. Residual analysis
    print("\n[1] Residual analysis...")
    residual_analysis(model, targets, sim_data_kJ)

    # 2. Bootstrap uncertainty
    print("\n[2] Bootstrap parameter uncertainty...")
    s1_params = {k: params[k] for k in params}
    all_params, param_names = bootstrap_uncertainty(targets, s1_params, n_bootstrap=200)

    if len(all_params) > 0:
        # 3. Parameter correlation
        print("\n[3] Parameter correlation...")
        parameter_correlation(all_params, param_names)

    # 4. Sensitivity analysis
    print("\n[4] Sensitivity analysis...")
    sensitivity_analysis(params)

    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Quick check imports**

Run: `cd /root/twoannealing && python -c "import src.tnm_error_analysis; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/tnm_error_analysis.py
git commit -m "feat: add comprehensive TNM error analysis module

Includes residual analysis, bootstrap uncertainty (200 resamples),
parameter correlation matrix, and sensitivity to cooling rate / T0."
```

---

## Phase 6: Final Integration & Run

### Task 6.1: Full pipeline run

- [ ] **Step 1: Run MATLAB verification**

```bash
cd /root/twoannealing && python src/verify_matlab_reference.py
```
Expected: All checks PASS.

- [ ] **Step 2: Run TNM fitting**

```bash
cd /root/twoannealing && python src/tnm_main.py
```
Expected: Parameters NOT at bounds, meaningful R² values, plots saved.

- [ ] **Step 3: Run error analysis**

```bash
cd /root/twoannealing && python src/tnm_error_analysis.py
```
Expected: Bootstrap CI computed, residual summary printed.

- [ ] **Step 4: Verify all outputs exist**

```bash
ls -la /root/twoannealing/results/tnm/tnm_*.png /root/twoannealing/results/tnm/tnm_*.csv
```
Expected: All plot and data files present.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete TNM model verification, fitting, and error analysis

- Pure isothermal TNM verified against MATLAB reference (Au MBMG)
- Fixed Boltzmann superposition in hold simulation
- Corrected v2 data loading (filenames, columns, units, offset)
- 5-parameter fitting (logA, H*, x, beta, T0) with wider bounds
- 2-stage strategy: shape-fit then absolute-scale refinement
- Comprehensive error analysis: residuals, bootstrap, correlation, sensitivity"
```

---

## Verification Checklist

After all tasks complete, verify:

1. [ ] `python src/verify_matlab_reference.py` — All 6 physical consistency checks PASS
2. [ ] `python src/tnm_main.py` — Parameters not at bounds; R² > 0.5 for one-step; R² > 0.7 for two-step
3. [ ] `python src/tnm_error_analysis.py` — Bootstrap CI computed; residual mean close to 0
4. [ ] All plots in `results/tnm/` look physically reasonable
5. [ ] `git status` — clean working tree
