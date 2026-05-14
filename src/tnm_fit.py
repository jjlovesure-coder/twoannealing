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

    Returns (dh_sim_norm, dh_exp_norm, dh_exp_kJ, t_exp, dh_sim_raw, t_sim).
    """
    t_sim, dh_sim = simulate_protocol_raw(model, protocol_key)

    group_map = {
        'os50': ('os50', 'run1'),
        'os70': ('os70', 'run1'),
        'tsA': ('ts', 'ts_grpA'),
        'tsB': ('ts', 'ts_grpB'),
        'kvA': ('kovacs', 'kov_grpA'),
        'kvB': ('kovacs', 'kov_grpB'),
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

    Fits dh_exp ≈ a * dh_sim_norm + b per protocol group,
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
        dh_sim_norm_vals, dh_exp_norm, dh_exp_kJ, _, _, _ = \
            simulate_protocol_match_exp(model, key, targets)

        a, b = np.polyfit(dh_sim_norm_vals, dh_exp_kJ, 1)
        dh_sim_kJ = a * dh_sim_norm_vals + b
        chi2 += np.sum((dh_sim_kJ - dh_exp_kJ) ** 2)
        n_pts += len(dh_exp_kJ)

    return np.sqrt(chi2 / max(n_pts, 1))


def run_stage1(targets, seed=None):
    """Multi-start L-BFGS-B for one-step shape fitting."""
    rng = np.random.RandomState(seed)
    bounds = [
        (-30, -12),         # logA (A ~ 1e-30 to 1e-12 s)
        (40000, 500000),    # H_star (J/mol)
        (0.01, 0.9),        # x
        (0.02, 0.9),         # beta
        (360, 400),         # T0 (K, near PS Tg ~373K)
    ]

    best_result = None
    best_cost = np.inf
    n_starts = 60

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
        (-25, -12), (60000, 400000), (0.05, 0.8), (0.1, 0.7), (370, 400),
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
