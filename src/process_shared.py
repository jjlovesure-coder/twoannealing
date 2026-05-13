"""
Shared utilities for DSC data processing with Tg peak integration.

Consolidates data loading, heating ramp detection, and the sigmoid-blended
baseline method for extracting physical aging enthalpy from DSC curves.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

# ── Instrument & sample constants ─────────────────────────────────────────
M_SAMPLE = 4.7        # mg
MW = 280000            # g/mol
HEATING_RATE = 10.0    # °C/min
COOLING_RATE = 60.0    # °C/min
BETA = HEATING_RATE / 60.0   # °C/s
DT_DT = 1.0 / BETA           # 6 s/°C
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)  # uW·°C → J/g
CONV_KJMOL = CONV_FACTOR * MW / 1000     # uW·°C → kJ/mol

# Sigmoid baseline parameters for PS at 10°C/min
SIGMOID_T_MID = 100.0   # Tg midpoint (°C)
SIGMOID_WIDTH = 5.0     # transition width (°C)
PRE_TG = (55, 85)       # pre-Tg linear fit range (°C)
POST_TG = (120, 145)    # post-Tg linear fit range (°C)
T_INT_LOW = 50          # integration range start (°C)
T_INT_HIGH = 150        # integration range end (°C)

# Program step mapping
HEATING_T_START = 30.0   # heating ramp starts from this T
HEATING_T_END = 200.0    # heating ramp ends at this T
MIN_HEATING_RATE = 2.0   # °C/min threshold for detecting heating


# ── Data loading ──────────────────────────────────────────────────────────

def load_dsc_simple(filename, sheet=None):
    """Load a simple DSC run (empty crucible or reference) — single ramp."""
    if sheet is None:
        df = pd.read_excel(filename)
    else:
        df = pd.read_excel(filename, sheet_name=sheet)

    header_row = None
    for i in range(len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i
            break

    if header_row is None:
        for i in range(len(df)):
            try:
                a = float(str(df.iloc[i, 0]).strip())
                b = float(str(df.iloc[i, 1]).strip())
                if np.isfinite(a) and np.isfinite(b):
                    header_row = i
                    break
            except (ValueError, TypeError):
                continue

    v0 = df.iloc[header_row, 0]
    if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
        data_start = header_row + 1
        v0_next = df.iloc[data_start, 0]
        if v0_next is not None and isinstance(v0_next, str) and 'min' in str(v0_next).lower():
            data_start += 1
        data = df.iloc[data_start:].copy()
    else:
        data = df.iloc[header_row:].copy()

    data = data.dropna(axis=1, how='all').reset_index(drop=True)
    ncols = data.shape[1]
    col_names = ['Time', 'Temp', 'DSC', 'DDSC'] + \
                [f'Extra_{i}' for i in range(max(0, ncols - 4))]
    data.columns = col_names[:ncols] if ncols <= len(col_names) else \
        col_names + [f'Extra_{i}' for i in range(len(col_names), ncols)]
    data = data[['Time', 'Temp', 'DSC', 'DDSC']].astype(float).reset_index(drop=True)
    return data


# ── Empty crucible baseline ──────────────────────────────────────────────


def load_empty_crucible(csv_path):
    """Load empty crucible baseline CSV (single heating ramp 30->~168C).

    Returns T (°C) and DSC (uW) arrays for the heating portion.
    """
    df = pd.read_csv(csv_path)
    T = df['Temp/Cel'].values.astype(float)
    DSC = df['DSC/uW'].values.astype(float)
    mask = T >= 30.0
    return T[mask], DSC[mask]


def subtract_empty_crucible(T_sample, DSC_sample, T_empty, DSC_empty):
    """Interpolate empty crucible DSC to sample T grid and subtract.

    Returns DSC_corrected = DSC_sample - DSC_empty_interp.
    """
    f_empty = interp1d(T_empty, DSC_empty, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    DSC_empty_interp = f_empty(T_sample)
    return DSC_sample - DSC_empty_interp


def detect_t_onset(T, DSC_corrected, post_Tg_range=(120, 150)):
    """Detect supercooled-liquid onset temperature from corrected DSC.

    Fits a linear baseline to the post-Tg region, then scans downward
    from the high-T end to find where the corrected signal deviates
    beyond 3x the post-Tg residual noise level.

    Returns T_onset in °C, or 100.0 if detection fails.
    """
    post_mask = (T >= post_Tg_range[0]) & (T <= post_Tg_range[1])
    if np.sum(post_mask) < 5:
        return 100.0

    post_T = T[post_mask]
    post_DSC = DSC_corrected[post_mask]
    coeffs = np.polyfit(post_T, post_DSC, 1)
    baseline = np.polyval(coeffs, post_T)
    residuals = post_DSC - baseline
    noise_std = np.std(residuals)
    threshold = 3.0 * max(noise_std, 0.01)

    scan_mask = (T >= 80) & (T <= 150)
    scan_idx = np.where(scan_mask)[0]
    if len(scan_idx) == 0:
        return 100.0

    full_baseline = np.polyval(coeffs, T)
    excess = np.abs(DSC_corrected - full_baseline)

    for i in range(len(scan_idx) - 1, -1, -1):
        idx = scan_idx[i]
        if excess[idx] > threshold:
            T_candidate = T[idx]
            return float(np.clip(T_candidate, 95, 140))

    return 100.0


def load_dsc_experiment(filename, sheet):
    """Load multi-ramp experiment with temperature program table.

    Returns (data_df, program_list) where:
        data_df: DataFrame with columns Time, Temp, DSC, DDSC
        program_list: list of dicts with step, T_start, T_end, rate, time_min
    """
    df = pd.read_excel(filename, sheet_name=sheet)

    # Extract temperature program table (rows 7 to NaN header)
    program = []
    for i in range(7, len(df)):
        v1 = df.iloc[i, 1]
        if v1 is not None and isinstance(v1, str) and \
           ('温度' in str(v1) or '冷却' in str(v1) or '温度程序' in str(v1)):
            break
        try:
            step_str = str(df.iloc[i, 1]).strip()
            step_num = int(float(step_str))
            t_start = float(str(df.iloc[i, 2]).strip())
            t_end = float(str(df.iloc[i, 3]).strip())
            rate = float(str(df.iloc[i, 4]).strip())
            time_val = float(str(df.iloc[i, 5]).strip())
            program.append({
                'step': step_num,
                'T_start': t_start,
                'T_end': t_end,
                'rate': rate,
                'time_min': time_val,
            })
        except (ValueError, TypeError, IndexError):
            continue

    # Find data header and parse data columns
    header_row = None
    for i in range(70, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i
            break

    data_start = header_row + 2
    raw = df.iloc[data_start:].copy()
    data = pd.DataFrame({
        'Time': pd.to_numeric(raw.iloc[:, 0], errors='coerce'),
        'Temp': pd.to_numeric(raw.iloc[:, 1], errors='coerce'),
        'DSC': pd.to_numeric(raw.iloc[:, 2], errors='coerce'),
        'DDSC': pd.to_numeric(raw.iloc[:, 3], errors='coerce'),
    }).dropna().reset_index(drop=True)
    data = data.astype(float)
    return data, program


# ── Ramp detection ────────────────────────────────────────────────────────

def detect_heating_ramps(T, t, min_rate=MIN_HEATING_RATE,
                         min_duration_pts=500):
    """Detect heating ramps (30→200°C at ~10°C/min) using temp gradient.

    Returns list of (start_idx, end_idx) tuples for ALL detected heating ramps.
    """
    n = len(T)
    window = 30
    dTdt = np.zeros(n)
    for i in range(window, n - window):
        dTdt[i] = (T[i + window] - T[i - window]) / \
                  max(t[i + window] - t[i - window], 1e-12)

    is_heating = dTdt > min_rate
    segments = []
    in_seg = False
    start = 0
    for i in range(n):
        if is_heating[i] and not in_seg:
            in_seg = True
            start = i
        elif not is_heating[i] and in_seg:
            if i - start > min_duration_pts:
                seg_T = T[start:i]
                if np.min(seg_T) < HEATING_T_START + 10 and \
                   np.max(seg_T) > HEATING_T_END - 20:
                    segments.append((start, i - 1))
            in_seg = False
    if in_seg and n - start > min_duration_pts:
        seg_T = T[start:n]
        if np.min(seg_T) < HEATING_T_START + 10 and \
           np.max(seg_T) > HEATING_T_END - 20:
            segments.append((start, n - 1))
    return segments


# ── Sigmoid baseline for Tg peak integration ──────────────────────────────

def _sigmoid(T, T_mid, width):
    """Logistic sigmoid: s(T) = 1 / (1 + exp(-(T - T_mid) / width))."""
    return 1.0 / (1.0 + np.exp(-(T - T_mid) / width))


def estimate_sigmoid_params(reference_curves):
    """Estimate sigmoid midpoint and width from reference (shortest-hold) ramps.

    Parameters
    ----------
    reference_curves : list of (T_array, DSC_array) tuples
        The shortest-hold ramps from each group, with minimal recovery peak.

    Returns
    -------
    T_mid, width : float
        Sigmoid parameters for the Cp step at Tg.
    """
    if len(reference_curves) == 0:
        return SIGMOID_T_MID, SIGMOID_WIDTH

    # Interpolate all reference curves to a common fine grid
    T_common = np.arange(50, 151, 0.2)
    dsc_interp = []
    for T_ramp, DSC_ramp in reference_curves:
        f = interp1d(T_ramp, DSC_ramp, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        dsc_interp.append(f(T_common))
    dsc_mean = np.mean(dsc_interp, axis=0)

    # Subtract straight line connecting pre-Tg and post-Tg means
    pre_mask = (T_common >= PRE_TG[0]) & (T_common <= PRE_TG[1])
    post_mask = (T_common >= POST_TG[0]) & (T_common <= POST_TG[1])
    T_pre_mid = np.mean(T_common[pre_mask])
    T_post_mid = np.mean(T_common[post_mask])
    dsc_pre = np.mean(dsc_mean[pre_mask])
    dsc_post = np.mean(dsc_mean[post_mask])
    slope = (dsc_post - dsc_pre) / (T_post_mid - T_pre_mid)
    intercept = dsc_pre - slope * T_pre_mid
    straight_line = slope * T_common + intercept

    # Residual = Cp step shape
    residual = dsc_mean - straight_line

    # Fit sigmoid to residual: a * s(T) + b
    def sigmoid_model(T, a, T_mid, width, b):
        return a * _sigmoid(T, T_mid, width) + b

    try:
        popt, _ = curve_fit(
            sigmoid_model, T_common, residual,
            p0=[residual.max() - residual.min(),
                SIGMOID_T_MID, SIGMOID_WIDTH, residual.min()],
            bounds=([0, 85, 2, -np.inf], [np.inf, 115, 15, np.inf]),
            maxfev=5000,
        )
        _, T_mid, width, _ = popt
        return float(np.clip(T_mid, 90, 110)), float(np.clip(width, 3, 10))
    except Exception:
        return SIGMOID_T_MID, SIGMOID_WIDTH


def fit_tg_baseline(T_seg, DSC_seg, sigmoid_params=None,
                    pre_Tg=PRE_TG, post_Tg=POST_TG,
                    T_int_low=T_INT_LOW, T_int_high=T_INT_HIGH):
    """Fit sigmoid-blended baseline through Tg and compute excess area.

    baseline(T) = L_pre(T) * (1 - s(T)) + L_post(T) * s(T)
    where s(T) = 1 / (1 + exp(-(T - T_mid) / width))

    L_pre is fit to DSC in pre_Tg range (e.g., 55-85°C)
    L_post is fit to DSC in post_Tg range (e.g., 120-145°C)

    Parameters
    ----------
    T_seg : array
        Temperature values for one heating ramp (°C).
    DSC_seg : array
        DSC signal values (uW).
    sigmoid_params : (T_mid, width) or None
        Sigmoid parameters. If None, use defaults for PS.
    pre_Tg, post_Tg : (low, high) tuples
        Temperature ranges for linear baseline fits.
    T_int_low, T_int_high : float
        Integration range for excess enthalpy.

    Returns
    -------
    dict with:
        excess_area : float, integrated excess DSC (uW·°C)
        pre_fit : (slope, intercept) of pre-Tg line
        post_fit : (slope, intercept) of post-Tg line
        T_mid, width : sigmoid parameters used
        baseline_values : array of baseline on T_seg
        excess_values : array of excess DSC on T_seg
    """
    if sigmoid_params is None:
        T_mid = SIGMOID_T_MID
        width = SIGMOID_WIDTH
    else:
        T_mid, width = sigmoid_params

    # Filter to integration range
    int_mask = (T_seg >= T_int_low) & (T_seg <= T_int_high)
    T_range = T_seg[int_mask]
    DSC_range = DSC_seg[int_mask]

    # Fit pre-Tg line
    pre_mask = (T_range >= pre_Tg[0]) & (T_range <= pre_Tg[1])
    if np.sum(pre_mask) < 5:
        # Fallback: use first 20% of range
        n_fallback = max(5, len(T_range) // 5)
        pre_mask = np.zeros(len(T_range), dtype=bool)
        pre_mask[:n_fallback] = True
    pre_coeffs = np.polyfit(T_range[pre_mask], DSC_range[pre_mask], 1)

    # Fit post-Tg line
    post_mask = (T_range >= post_Tg[0]) & (T_range <= post_Tg[1])
    if np.sum(post_mask) < 5:
        # Fallback: use last 20% of range
        n_fallback = max(5, len(T_range) // 5)
        post_mask = np.zeros(len(T_range), dtype=bool)
        post_mask[-n_fallback:] = True
    post_coeffs = np.polyfit(T_range[post_mask], DSC_range[post_mask], 1)

    # Build sigmoid-blended baseline
    s = _sigmoid(T_range, T_mid, width)
    L_pre = np.polyval(pre_coeffs, T_range)
    L_post = np.polyval(post_coeffs, T_range)
    baseline = L_pre * (1 - s) + L_post * s

    # Excess = DSC - baseline
    excess = DSC_range - baseline

    # Integrate excess area
    excess_area = trapezoid(excess, T_range)

    return {
        'excess_area': excess_area,
        'pre_fit': (pre_coeffs[0], pre_coeffs[1]),
        'post_fit': (post_coeffs[0], post_coeffs[1]),
        'T_mid': T_mid,
        'width': width,
        'baseline_values': baseline,
        'excess_values': excess,
        'T_range': T_range,
    }


def compute_enthalpy(excess_area_uWC):
    """Convert integrated excess DSC (uW·°C) to enthalpy (kJ/mol)."""
    return excess_area_uWC * CONV_KJMOL
