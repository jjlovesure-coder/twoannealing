"""
Staged parameter optimization for TNM model fitting to DSC enthalpy data.

Uses shape-based fitting: both simulated and experimental delta_H are
normalized to [0, 1] before comparison.
"""

import numpy as np
from scipy.optimize import minimize
from tnm_model import TNMModel
from tnm_conditions import (
    HOLD_TIMES_SEC, T_50C, T_70C, T_80C, T_90C, T_INITIAL,
    COOLING_RATE, build_target_vectors,
)


def make_model(logA, H_star, x, beta, T0):
    A = 10.0 ** logA
    return TNMModel(A=A, H_star=H_star, x=x, beta=beta, T0=T0)


def simulate_protocol_dh(model, protocol_key, targets):
    """Simulate delta_H for a protocol, normalized to [0, 1].

    Returns (dh_sim_norm, dh_exp_norm).
    """
    if protocol_key in ('os50', 'os70'):
        T_a = T_50C if protocol_key == 'os50' else T_70C
        dh_raw = np.array([model.delta_H_normalized(
            T_a, th, T_INITIAL, COOLING_RATE) for th in HOLD_TIMES_SEC])
    elif protocol_key == 'tsA':
        t1 = 0.833 * 60.0
        dh_raw = np.array([model.delta_H_two_step(
            T_90C, t1, T_80C, th, T_INITIAL, COOLING_RATE)
            for th in HOLD_TIMES_SEC])
    elif protocol_key == 'tsB':
        t1 = 8.333 * 60.0
        dh_raw = np.array([model.delta_H_two_step(
            T_90C, t1, T_80C, th, T_INITIAL, COOLING_RATE)
            for th in HOLD_TIMES_SEC])
    elif protocol_key == 'kvA':
        t1 = 0.833 * 60.0
        dh_raw = np.array([model.delta_H_two_step(
            T_80C, t1, T_90C, th, T_INITIAL, COOLING_RATE)
            for th in HOLD_TIMES_SEC])
    elif protocol_key == 'kvB':
        t1 = 8.333 * 60.0
        dh_raw = np.array([model.delta_H_two_step(
            T_80C, t1, T_90C, th, T_INITIAL, COOLING_RATE)
            for th in HOLD_TIMES_SEC])
    else:
        return None, None

    # Normalize sim to [0, 1]
    dmin, dmax = dh_raw.min(), dh_raw.max()
    if dmax - dmin < 1e-10:
        dh_sim_norm = np.zeros_like(dh_raw)
    else:
        dh_sim_norm = (dh_raw - dmin) / (dmax - dmin)

    # Map protocol key to experimental data group
    group_map = {
        'os50': ('os50', None),   # use all data, no group filter
        'os70': ('os70', None),
        'tsA': ('ts', 'ts_grpA'),
        'tsB': ('ts', 'ts_grpB'),
        'kvA': ('kovacs', 'kov_grpA'),
        'kvB': ('kovacs', 'kov_grpB'),
    }
    exp_key, grp_key = group_map[protocol_key]
    exp = targets[exp_key]

    if grp_key is not None and 'groups' in exp:
        mask = exp['groups'] == grp_key
        dh_exp_raw = exp['dH'][mask]
        t_exp = exp['t'][mask]
    else:
        dh_exp_raw = exp['dH']
        t_exp = exp['t']

    emin, emax = dh_exp_raw.min(), dh_exp_raw.max()
    if emax - emin < 1e-10:
        dh_exp_norm = np.zeros_like(dh_exp_raw)
    else:
        dh_exp_norm = (dh_exp_raw - emin) / (emax - emin)

    # Interpolate sim to exp hold times (t_exp already set above)
    dh_sim_interp = np.interp(t_exp, HOLD_TIMES_SEC, dh_sim_norm)
    dh_exp_interp = dh_exp_norm

    return dh_sim_interp, dh_exp_interp


def stage1_cost_shape(packed, targets, T0):
    """Shape-based cost: normalized one-step 50C + 70C."""
    logA, H_star, x, beta = packed[0], packed[1], packed[2], packed[3]
    if not (0 < x <= 1 and 0 < beta <= 1 and H_star > 0):
        return 1e12
    model = make_model(logA, H_star, x, beta, T0)

    chi2 = 0.0
    for key in ['os50', 'os70']:
        dh_sim, dh_exp = simulate_protocol_dh(model, key, targets)
        if dh_sim is None:
            return 1e12
        chi2 += np.sum((dh_sim - dh_exp) ** 2)

    return np.sqrt(chi2 / (len(dh_sim) + len(dh_sim)))


def stage2_cost_shape(packed, targets, T0):
    """Shape-based cost: one-step + two-step."""
    logA, H_star, x, beta = packed[0], packed[1], packed[2], packed[3]
    if not (0 < x <= 1 and 0 < beta <= 1 and H_star > 0):
        return 1e12
    model = make_model(logA, H_star, x, beta, T0)

    chi2 = 0.0
    n = 0
    for key in ['os50', 'os70', 'tsA', 'tsB']:
        dh_sim, dh_exp = simulate_protocol_dh(model, key, targets)
        if dh_sim is None:
            return 1e12
        chi2 += np.sum((dh_sim - dh_exp) ** 2)
        n += len(dh_sim)

    return np.sqrt(chi2 / max(n, 1))


def run_stage1(targets, T0, seed=None):
    """Multi-start L-BFGS-B for one-step shape fitting."""
    rng = np.random.RandomState(seed)
    bounds = [
        (-20, -8),        # logA
        (60000, 200000),  # H_star (J/mol)
        (0.1, 0.8),       # x
        (0.2, 0.7),       # beta
    ]

    best_result = None
    best_cost = np.inf
    n_starts = 20

    print(f"  Stage 1: Multi-start L-BFGS-B ({n_starts} starts)...")
    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(
            stage1_cost_shape, x0, args=(targets, T0),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6},
        )
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res
        if (k + 1) % 5 == 0:
            print(f"    Start {k+1}/{n_starts}, best cost = {best_cost:.6f}")

    print(f"    Final best cost: {best_cost:.6f}")
    best = best_result.x
    params = {'logA': best[0], 'H_star': best[1], 'x': best[2], 'beta': best[3]}
    return params, best_result


def run_stage2(targets, T0, stage1_params, seed=None):
    """Refine with two-step data added."""
    x0 = [stage1_params['logA'], stage1_params['H_star'],
          stage1_params['x'], stage1_params['beta']]
    bounds = [
        (-20, -8), (60000, 200000), (0.1, 0.8), (0.2, 0.7),
    ]

    print("  Stage 2: L-BFGS-B with two-step data...")
    result = minimize(
        stage2_cost_shape, x0, args=(targets, T0),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8},
    )
    print(f"    Stage 2 best cost: {result.fun:.6f}")

    best = result.x
    params = {'logA': best[0], 'H_star': best[1], 'x': best[2], 'beta': best[3]}
    return params, result


def compute_r_squared(dH_sim, dH_exp):
    ss_res = np.sum((dH_sim - dH_exp) ** 2)
    ss_tot = np.sum((dH_exp - np.mean(dH_exp)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot
