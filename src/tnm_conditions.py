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
COOLING_RATE = 1.0           # 60 C/min = 1.0 K/s
HEATING_RATE = 10.0 / 60.0   # 10 C/min = 0.1667 K/s

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

    Data columns: ramp, hold_s, T_onset_C, delta_H_kJmol
    20 rows: first 10 = run1, last 10 = run2.

    Returns (t_hold_sec, delta_H, groups).
    """
    t_hold_sec = df['hold_s'].values
    delta_H = df['delta_H_kJmol'].values
    n = len(t_hold_sec)
    groups = np.array(['run1'] * (n // 2) + ['run2'] * (n - n // 2))
    return t_hold_sec, delta_H, groups


def extract_two_step(df):
    """Extract hold times and delta_H from two-step/Kovacs DataFrame.

    Data columns: ramp, T1_hold_s, T2_hold_s, T_onset_C, delta_H_kJmol
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

    Returns dict with keys os50, os70, ts, kovacs.
    Each value: {'t': array, 'dH': array, 'groups': array}.
    """
    exp_data = load_experimental_data()

    result = {}
    for key in ['os50', 'os70']:
        t, dh, grp = extract_one_step(exp_data[key])
        result[key] = {'t': t, 'dH': dh, 'groups': grp}

    for key in ['ts', 'kovacs']:
        t, dh, grp = extract_two_step(exp_data[key])
        # Add prefix to distinguish ts vs kovacs groups
        prefix = 'ts' if key == 'ts' else 'kov'
        prefixed = np.array([f'{prefix}_{g}' for g in grp])
        result[key] = {'t': t, 'dH': dh, 'groups': prefixed}

    return result
