"""
Protocol definitions and batch simulation for TNM model fitting.

Defines all 6 experimental protocols:
- One-step 50C / 70C
- Two-step hi->lo (Grp A: T1=0.833min, Grp B: T1=8.333min)
- Kovacs up-jump (Grp A: T1=0.833min, Grp B: T1=8.333min)
"""

import os
import numpy as np
import pandas as pd
from tnm_model import TNMModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dsc')

# Hold times in seconds (converted from minutes)
HOLD_TIMES_MIN = np.array([
    0.001670, 0.008330, 0.013300, 0.016670, 0.083300,
    0.167000, 0.833000, 1.667000, 8.333000, 16.667000
])
HOLD_TIMES_SEC = HOLD_TIMES_MIN * 60.0

# Cooling/heating rates from DSC program metadata
COOLING_RATE = 60.0 / 60.0   # 60 C/min = 1.0 K/s
HEATING_RATE = 10.0 / 60.0   # 10 C/min = 0.1667 K/s

# Temperatures in Kelvin
T_INITIAL = 200 + 273.15   # 473.15 K
T_50C = 50 + 273.15        # 323.15 K
T_70C = 70 + 273.15        # 343.15 K
T_80C = 80 + 273.15        # 353.15 K
T_90C = 90 + 273.15        # 363.15 K


def load_experimental_data():
    """Load all experimental CSV files. Returns dict of DataFrames."""
    data = {}
    for name, csv_file in [
        ('os50', 'onestep_50C_results.csv'),
        ('os70', 'onestep_70C_results.csv'),
        ('ts', 'twosteps_enthalpy_results.csv'),
        ('kovacs', 'kovacs_enthalpy_results.csv'),
    ]:
        path = os.path.join(RESULTS_DIR, csv_file)
        df = pd.read_csv(path)
        data[name] = df
    return data


def extract_hold_times_and_dH(df, protocol_type):
    """Extract arrays of hold times (s) and delta_H values from a DataFrame.

    For one-step: uses hold_min column
    For two-step/Kovacs: uses T2_hold_min column

    Returns (t_hold_sec, delta_H, group_labels) where group_labels is
    'run1', 'run2' for one-step or 'grpA', 'grpB' for two-step/Kovacs.
    """
    t_hold_sec = df['hold_min' if 'hold_min' in df.columns
                     else 'T2_hold_min'].values * 60.0
    delta_H = df['delta_H_kJmol'].values

    if 'T1_hold_min' in df.columns:
        t1_vals = df['T1_hold_min'].values
        groups = np.where(t1_vals < 1.0, 'grpA', 'grpB')
    else:
        n = len(t_hold_sec)
        groups = np.array(['run1'] * (n // 2) + ['run2'] * (n - n // 2))

    return t_hold_sec, delta_H, groups


def run_one_step_simulation(model, T_anneal):
    """Run one-step simulation for all 10 hold times.

    Returns (t_hold_sec, delta_H_sim) arrays.
    """
    delta_H_sim = np.zeros(len(HOLD_TIMES_SEC))
    for i, t_hold in enumerate(HOLD_TIMES_SEC):
        delta_H_sim[i] = model.simulate_one_step(
            T_anneal=T_anneal,
            t_hold=t_hold,
            T_initial=T_INITIAL,
            cooling_rate=COOLING_RATE,
            heating_rate=HEATING_RATE,
        )
    return HOLD_TIMES_SEC.copy(), delta_H_sim


def run_two_step_simulation(model, T1, t1_hold, T2):
    """Run two-step simulation for all 10 T2 hold times.

    Returns (t2_hold_sec, delta_H_sim) arrays.
    """
    delta_H_sim = np.zeros(len(HOLD_TIMES_SEC))
    for i, t2_hold in enumerate(HOLD_TIMES_SEC):
        delta_H_sim[i] = model.simulate_two_step(
            T1=T1, t1_hold=t1_hold, T2=T2, t2_hold=t2_hold,
            T_initial=T_INITIAL,
            cooling_rate=COOLING_RATE,
            heating_rate=HEATING_RATE,
        )
    return HOLD_TIMES_SEC.copy(), delta_H_sim


def simulate_all_protocols(model):
    """Run all 6 protocol simulations.

    Returns dict mapping protocol name -> (t_hold, delta_H_sim, exp_data_dict).
    """
    results = {}

    # One-step 50C
    t, dh = run_one_step_simulation(model, T_50C)
    results['os50'] = {'t_hold_sec': t, 'delta_H_sim': dh}

    # One-step 70C
    t, dh = run_one_step_simulation(model, T_70C)
    results['os70'] = {'t_hold_sec': t, 'delta_H_sim': dh}

    # Two-step Grp A: T1=0.833 min @ 90C, T2=80C
    t1_A = 0.833 * 60.0
    t, dh = run_two_step_simulation(model, T_90C, t1_A, T_80C)
    results['ts_grpA'] = {'t_hold_sec': t, 'delta_H_sim': dh}

    # Two-step Grp B: T1=8.333 min @ 90C, T2=80C
    t1_B = 8.333 * 60.0
    t, dh = run_two_step_simulation(model, T_90C, t1_B, T_80C)
    results['ts_grpB'] = {'t_hold_sec': t, 'delta_H_sim': dh}

    # Kovacs Grp A: T1=0.833 min @ 80C, T2=90C (up-jump)
    t, dh = run_two_step_simulation(model, T_80C, t1_A, T_90C)
    results['kov_grpA'] = {'t_hold_sec': t, 'delta_H_sim': dh}

    # Kovacs Grp B: T1=8.333 min @ 80C, T2=90C (up-jump)
    t, dh = run_two_step_simulation(model, T_80C, t1_B, T_90C)
    results['kov_grpB'] = {'t_hold_sec': t, 'delta_H_sim': dh}

    return results


def build_target_vectors():
    """Build concatenated target vectors for fitting.

    Returns:
        t_all: concatenated hold times
        dH_all: concatenated experimental delta_H
        sim_func: function(model) -> concat simulated delta_H
        groups: list of protocol names for each data point
    """
    exp_data = load_experimental_data()

    # One-step 50C
    t_os50, dH_os50, grp_os50 = extract_hold_times_and_dH(exp_data['os50'], 'os')
    # One-step 70C
    t_os70, dH_os70, grp_os70 = extract_hold_times_and_dH(exp_data['os70'], 'os')

    # Two-step
    df_ts = exp_data['ts']
    t_ts = df_ts['T2_hold_min'].values * 60.0
    dH_ts = df_ts['delta_H_kJmol'].values
    t1_vals_ts = df_ts['T1_hold_min'].values
    grp_ts = np.where(t1_vals_ts < 1.0, 'ts_grpA', 'ts_grpB')

    # Kovacs
    df_kv = exp_data['kovacs']
    t_kv = df_kv['T2_hold_min'].values * 60.0
    dH_kv = df_kv['delta_H_kJmol'].values
    t1_vals_kv = df_kv['T1_hold_min'].values
    grp_kv = np.where(t1_vals_kv < 1.0, 'kov_grpA', 'kov_grpB')

    return {
        'os50': {'t': t_os50, 'dH': dH_os50, 'groups': grp_os50},
        'os70': {'t': t_os70, 'dH': dH_os70, 'groups': grp_os70},
        'ts': {'t': t_ts, 'dH': dH_ts, 'groups': grp_ts},
        'kovacs': {'t': t_kv, 'dH': dH_kv, 'groups': grp_kv},
    }
