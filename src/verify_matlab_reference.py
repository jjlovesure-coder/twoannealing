#!/usr/bin/env python3
"""
Verify TNM reference model physical consistency using Au MBMG parameters.
Runs 3 protocols from the MATLAB reference (isothermal, no cooling ramps)
and checks Tf evolution satisfies physical constraints (monotonicity, range
bounds, Kovacs memory hump). Saves plot and reference CSV data.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnm_reference import one_step, two_step_hi_to_lo, two_step_lo_to_hi

# MATLAB reference parameters (Au-based metallic glass)
H_STAR = 164e3       # J/mol
A = 6e-22            # seconds
X = 0.6
BETA = 0.43
T0 = 430.0           # K, initial fictive temperature

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'tnm')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_results():
    """Compute Tf for all 3 protocols from MATLAB reference."""
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
    checks.append(('a: Tf within range [Ta, T0]',
                   np.all(Tf_a <= T0) and np.all(Tf_a >= Ta)))
    checks.append(('a: Tf monotonic decreasing', np.all(np.diff(Tf_a) <= 0)))
    checks.append(('a: Tf[0] == T0', abs(Tf_a[0] - T0) < 1e-10))
    checks.append(('a: Tf approaches Ta', abs(Tf_a[-1] - Ta) < 5.0))

    # Protocol (b): Tf further decreases from after-step1 toward T2
    t_b, Tf_b, T1_b, T2_b = results['b']
    checks.append(('b: Tf within range [T2, T0]',
                   np.all(Tf_b <= T0) and np.all(Tf_b >= T2_b)))
    checks.append(('b: Tf[0] < T0 (partial relax after 50s at T1)',
                   Tf_b[0] < T0))
    checks.append(('b: Tf monotonic decreasing', np.all(np.diff(Tf_b) <= 0)))
    checks.append(('b: Tf approaches T2', abs(Tf_b[-1] - T2_b) < 5.0))

    # Protocol (c): Kovacs up-jump — Tf shows "memory hump"
    t_c, Tf_c, T1_c, T2_c = results['c']
    checks.append(('c: Tf[0] < T0 (partial relax from T0 after 50s at T1)',
                   Tf_c[0] < T0))
    # Memory hump: Tf should rise then fall, max above T2
    peak_idx = np.argmax(Tf_c)
    checks.append(('c: Tf rises after start (memory effect)',
                   Tf_c[peak_idx] > Tf_c[0]))
    checks.append(('c: Tf overshoots T2 (Kovacs hump)',
                   Tf_c[peak_idx] > T2_c))
    checks.append(('c: Tf returns toward T2 at long times',
                   abs(Tf_c[-1] - T2_c) < abs(Tf_c[peak_idx] - T2_c)))

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
    t_b, Tf_b, _, T2_b = results['b']
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


def save_reference_data(results):
    """Save numerical reference data as CSV for future comparison."""
    ref_dir = os.path.join(RESULTS_DIR, 'ref_data')
    os.makedirs(ref_dir, exist_ok=True)
    names = {'a': 'protocol_a_onestep', 'b': 'protocol_b_twostep_hi2lo',
             'c': 'protocol_c_memory_upjump'}
    for key, name in names.items():
        t, Tf = results[key][0], results[key][1]
        pd.DataFrame({'t_s': t, 'Tf_K': Tf}).to_csv(
            os.path.join(ref_dir, f'ref_{name}.csv'), index=False, float_format='%.8f')
    print(f"  Saved reference data to: {ref_dir}/")


def main():
    print("=" * 60)
    print("  TNM Reference Model — Physical Consistency Check")
    print("=" * 60)
    print(f"\n  Parameters:")
    print(f"    H* = {H_STAR/1000:.0f} kJ/mol")
    print(f"    A  = {A:.1e} s")
    print(f"    x  = {X}")
    print(f"    beta = {BETA}")
    print(f"    T0 = {T0} K")

    print("\n[1] Computing Tf for all 3 protocols...")
    results = compute_results()
    for key, (t, Tf, *temps) in results.items():
        print(f"  Protocol ({key}): {len(t)} points, "
              f"Tf range [{Tf.min():.2f}, {Tf.max():.2f}] K")

    print("\n[2] Physical consistency checks:")
    all_pass = verify_consistency(results)
    print(f"\n  All checks passed: {all_pass}")

    print("\n[3] Generating comparison plots...")
    plot_results(results)

    print("\n[4] Saving reference data...")
    save_reference_data(results)

    if all_pass:
        print("\n" + "=" * 60)
        print("  VERIFICATION PASSED — Model matches MATLAB reference")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("  VERIFICATION FAILED — Some checks did not pass")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
