"""
Export enthalpy change data for one-step, two-step, and Kovacs annealing
with enriched column information (annealing temperatures, cooling rates, etc.).
Saves to the data/ folder.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'enthalpy')
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

M_SAMPLE = 4.7
HEATING_RATE = 10.0
BETA = HEATING_RATE / 60.0
DT_DT = 1.0 / BETA
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)
CONV_JG = CONV_FACTOR  # µW·°C → J/g
T_INT_LOW_DEFAULT = 40


def load_empty():
    """加载空坩埚基线，返回插值函数"""
    df = pd.read_excel(os.path.join(DATA_DIR, 'ps-empty-01.xlsx'))
    for i in range(len(df)):
        try:
            a = float(str(df.iloc[i, 0]).strip())
            b = float(str(df.iloc[i, 1]).strip())
            if np.isfinite(a) and np.isfinite(b):
                hr = i
                break
        except Exception:
            continue
    raw = df.iloc[hr:]
    Te = pd.to_numeric(raw.iloc[:, 1], errors='coerce').dropna().values
    De = pd.to_numeric(raw.iloc[:, 2], errors='coerce').dropna().values
    return interp1d(Te, De, kind='linear', bounds_error=False, fill_value='extrapolate')


def load_dsc_experiment(filename, sheet):
    df = pd.read_excel(filename, sheet_name=sheet)
    program = []
    for i in range(7, len(df)):
        v1 = df.iloc[i, 1]
        if v1 is not None and isinstance(v1, str) and ('温度' in str(v1) or '冷却' in str(v1)):
            break
        try:
            step_num = int(float(str(df.iloc[i, 1]).strip()))
            t_start = float(str(df.iloc[i, 2]).strip())
            t_end = float(str(df.iloc[i, 3]).strip())
            rate = float(str(df.iloc[i, 4]).strip())
            time_val = float(str(df.iloc[i, 5]).strip())
            program.append({
                'step': step_num, 'T_start_C': t_start, 'T_end_C': t_end,
                'rate_K_per_min': rate, 'hold_min': time_val,
            })
        except Exception:
            continue

    header_row = None
    for i in range(5, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i
            break
    raw = df.iloc[header_row + 2:].copy()
    data = pd.DataFrame({
        'Time': pd.to_numeric(raw.iloc[:, 0], errors='coerce'),
        'Temp': pd.to_numeric(raw.iloc[:, 1], errors='coerce'),
        'DSC': pd.to_numeric(raw.iloc[:, 2], errors='coerce'),
        'DDSC': pd.to_numeric(raw.iloc[:, 3], errors='coerce'),
    }).dropna().reset_index(drop=True).astype(float)
    return data, program


def detect_heating_ramps(T, t, min_rate=4.0, min_duration_pts=500,
                         T_range_min=40, T_range_max=175):
    n = len(T)
    window = 30
    dTdt = np.zeros(n)
    for i in range(window, n - window):
        dTdt[i] = (T[i + window] - T[i - window]) / max(t[i + window] - t[i - window], 1e-12)
    is_h = dTdt > min_rate
    segs = []
    in_seg = False
    start = 0
    for i in range(n):
        if is_h[i] and not in_seg:
            in_seg = True
            start = i
        elif not is_h[i] and in_seg:
            if i - start > min_duration_pts:
                seg_T = T[start:i]
                if np.min(seg_T) < T_range_min and np.max(seg_T) >= T_range_max:
                    segs.append((start, i - 1))
            in_seg = False
    if in_seg and n - start > min_duration_pts:
        seg_T = T[start:n]
        if np.min(seg_T) < T_range_min and np.max(seg_T) >= T_range_max:
            segs.append((start, n - 1))
    return segs


def find_liquid_onset(T_seg, DSC_seg, T_liq_fit_lo=110, T_liq_fit_hi=160, T_tg_lo=70, T_tg_hi=115):
    liq_mask = (T_seg >= T_liq_fit_lo) & (T_seg <= T_liq_fit_hi)
    if liq_mask.sum() < 10:
        return np.nan
    coeffs = np.polyfit(T_seg[liq_mask], DSC_seg[liq_mask], 1)
    liquid_line = np.polyval(coeffs, T_seg)
    tg_mask = (T_seg >= T_tg_lo) & (T_seg <= T_tg_hi)
    if tg_mask.sum() < 10:
        return np.nan
    peak_offset = np.argmax(DSC_seg[tg_mask])
    peak_idx = np.arange(len(T_seg))[tg_mask][peak_offset]
    after_peak = np.arange(peak_idx, len(T_seg))
    diff = DSC_seg[after_peak] - liquid_line[after_peak]
    for j in range(1, len(diff)):
        if diff[j - 1] >= 0 and diff[j] < 0:
            return T_seg[after_peak[j]]
    return np.nan


def process_onestep(data_file, sheet, T_anneal, label, interp_empty, t_int_low=T_INT_LOW_DEFAULT):
    """Process one-step annealing and return enriched DataFrame."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program
                     if p['T_start_C'] < p['T_end_C']
                     and (p['T_end_C'] - p['T_start_C']) >= 150]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            # Find the preceding steps
            step_to_Tanneal = None  # step_num-2: cooling to T_anneal
            step_to_Tlow = None     # step_num-1: T_anneal → T_low (cooling to glass)
            for p in program:
                if p['step'] == step_num - 2:
                    step_to_Tanneal = p
                elif p['step'] == step_num - 1:
                    step_to_Tlow = p
            t_hold = step_to_Tanneal['hold_min'] if step_to_Tanneal else None
            cool_rate = step_to_Tlow['rate_K_per_min'] if step_to_Tlow else None
        else:
            t_hold = None
            cool_rate = None
        conditions.append({
            'ramp_idx': idx + 1, 'hold_min': t_hold,
            'cooling_rate_K_per_min': cool_rate,
            'start_idx': s, 'end_idx': e,
        })

    for c in conditions:
        c['hold_s'] = c['hold_min'] * 60 if c['hold_min'] else None

    T_onsets = []
    all_integrals = []

    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e + 1]
        DSC_seg = DSC_exp[s:e + 1]
        T_onset = find_liquid_onset(T_seg, DSC_seg)
        T_onsets.append(T_onset)

        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear', bounds_error=False, fill_value='extrapolate')
        if not np.isnan(T_onset):
            T_grid = np.arange(t_int_low, T_onset + 0.01, 0.1)
            delta_DSC = interp_dsc(T_grid) - interp_empty(T_grid)
            all_integrals.append(trapezoid(delta_DSC, T_grid))
        else:
            all_integrals.append(np.nan)

    integrals = np.array(all_integrals)
    ref = integrals[0]
    dH = (integrals - ref) * CONV_JG

    # Determine cooling group (50s vs 500s) from program pattern
    n = len(conditions)
    midpoint = n // 2
    cooling_groups = ['50s'] * midpoint + ['500s'] * (n - midpoint)

    results = []
    for i, cond in enumerate(conditions):
        results.append({
            'ramp': i + 1,
            'experiment': label,
            'T_anneal_C': T_anneal,
            'cooling_rate_to_glass_K_per_min': cond['cooling_rate_K_per_min'],
            'heating_rate_K_per_min': HEATING_RATE,
            'hold_s': cond['hold_s'],
            'cooling_group': cooling_groups[i],
            'T_onset_C': T_onsets[i],
            'delta_H_J_per_g': dH[i],
        })

    return pd.DataFrame(results)


def process_twosteps(data_file, sheet, label, interp_empty, t_int_low=T_INT_LOW_DEFAULT):
    """Process two-step annealing and return enriched DataFrame."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program
                     if p['T_start_C'] < p['T_end_C']
                     and (p['T_end_C'] - p['T_start_C']) >= 150]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            step_to_T1 = None    # step_num-3: cool to T1 (pre-annealing)
            step_to_T2 = None    # step_num-2: T1 → T2 (annealing)
            step_to_Tlow = None  # step_num-1: T2 → T_low (cooling to glass)
            for p in program:
                if p['step'] == step_num - 3:
                    step_to_T1 = p
                elif p['step'] == step_num - 2:
                    step_to_T2 = p
                elif p['step'] == step_num - 1:
                    step_to_Tlow = p

            T1_C = step_to_T1['T_end_C'] if step_to_T1 else None
            T2_C = step_to_T2['T_end_C'] if step_to_T2 else None
            T1_hold_min = step_to_T1['hold_min'] if step_to_T1 else None
            T2_hold_min = step_to_T2['hold_min'] if step_to_T2 else None
            cool_rate = step_to_Tlow['rate_K_per_min'] if step_to_Tlow else None
            T1_cool_rate = step_to_T1['rate_K_per_min'] if step_to_T1 else None
        else:
            T1_C = T2_C = T1_hold_min = T2_hold_min = cool_rate = T1_cool_rate = None

        conditions.append({
            'ramp_idx': idx + 1,
            'T1_C': T1_C, 'T2_C': T2_C,
            'T1_hold_min': T1_hold_min, 'T2_hold_min': T2_hold_min,
            'T1_cool_rate_K_per_min': T1_cool_rate,
            'cooling_rate_to_glass_K_per_min': cool_rate,
            'start_idx': s, 'end_idx': e,
        })

    for c in conditions:
        c['T1_hold_s'] = c['T1_hold_min'] * 60 if c['T1_hold_min'] else None
        c['T2_hold_s'] = c['T2_hold_min'] * 60 if c['T2_hold_min'] else None

    T_onsets = []
    all_integrals = []

    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e + 1]
        DSC_seg = DSC_exp[s:e + 1]
        T_onset = find_liquid_onset(T_seg, DSC_seg)
        T_onsets.append(T_onset)

        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear', bounds_error=False, fill_value='extrapolate')
        if not np.isnan(T_onset):
            T_grid = np.arange(t_int_low, T_onset + 0.01, 0.1)
            delta_DSC = interp_dsc(T_grid) - interp_empty(T_grid)
            all_integrals.append(trapezoid(delta_DSC, T_grid))
        else:
            all_integrals.append(np.nan)

    integrals = np.array(all_integrals)
    ref = integrals[0]
    dH = (integrals - ref) * CONV_JG

    # Build T1 group labels from actual hold times
    T1_groups = []
    for c in conditions:
        if c['T1_hold_s'] is not None:
            T1_groups.append(f"{int(round(c['T1_hold_s']))}s")
        else:
            T1_groups.append('?')

    results = []
    for i, cond in enumerate(conditions):
        results.append({
            'ramp': i + 1,
            'experiment': label,
            'T1_C': cond['T1_C'],
            'T2_C': cond['T2_C'],
            'T1_hold_s': cond['T1_hold_s'],
            'T2_hold_s': cond['T2_hold_s'],
            'T1_cool_rate_K_per_min': cond['T1_cool_rate_K_per_min'],
            'cooling_rate_to_glass_K_per_min': cond['cooling_rate_to_glass_K_per_min'],
            'heating_rate_K_per_min': HEATING_RATE,
            'T1_group': T1_groups[i],
            'T_onset_C': T_onsets[i],
            'delta_H_J_per_g': dH[i],
        })

    return pd.DataFrame(results)


def process_kovacs(data_file, sheet, label, interp_empty, t_int_low=T_INT_LOW_DEFAULT,
                   ref_integral=None):
    """Process Kovacs up-jump annealing and return enriched DataFrame.
    If ref_integral is provided, use it as the reference (instead of first ramp)."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program
                     if p['T_start_C'] < p['T_end_C']
                     and (p['T_end_C'] - p['T_start_C']) >= 150]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            step_to_T1 = None    # step_num-3: cool to T1 (pre-annealing)
            step_to_T2 = None    # step_num-2: T1 → T2 (up-jump annealing)
            step_to_Tlow = None  # step_num-1: T2 → T_low (cooling to glass)
            for p in program:
                if p['step'] == step_num - 3:
                    step_to_T1 = p
                elif p['step'] == step_num - 2:
                    step_to_T2 = p
                elif p['step'] == step_num - 1:
                    step_to_Tlow = p

            T1_C = step_to_T1['T_end_C'] if step_to_T1 else None
            T2_C = step_to_T2['T_end_C'] if step_to_T2 else None
            T1_hold_min = step_to_T1['hold_min'] if step_to_T1 else None
            T2_hold_min = step_to_T2['hold_min'] if step_to_T2 else None
            cool_rate = step_to_Tlow['rate_K_per_min'] if step_to_Tlow else None
            T1_cool_rate = step_to_T1['rate_K_per_min'] if step_to_T1 else None
            T2_rate = step_to_T2['rate_K_per_min'] if step_to_T2 else None
        else:
            T1_C = T2_C = T1_hold_min = T2_hold_min = cool_rate = T1_cool_rate = T2_rate = None

        conditions.append({
            'ramp_idx': idx + 1,
            'T1_C': T1_C, 'T2_C': T2_C,
            'T1_hold_min': T1_hold_min, 'T2_hold_min': T2_hold_min,
            'T1_cool_rate_K_per_min': T1_cool_rate,
            'T2_rate_K_per_min': T2_rate,
            'cooling_rate_to_glass_K_per_min': cool_rate,
            'start_idx': s, 'end_idx': e,
        })

    for c in conditions:
        c['T1_hold_s'] = c['T1_hold_min'] * 60 if c['T1_hold_min'] else None
        c['T2_hold_s'] = c['T2_hold_min'] * 60 if c['T2_hold_min'] else None

    T_onsets = []
    all_integrals = []
    overshoot_peaks = []

    for cond in conditions:
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e + 1]
        DSC_seg = DSC_exp[s:e + 1]
        T_onset = find_liquid_onset(T_seg, DSC_seg)
        T_onsets.append(T_onset)

        # Overshoot peak
        tg_mask = (T_seg >= 75) & (T_seg <= 110)
        ov_peak = np.max(DSC_seg[tg_mask]) - np.interp(100, T_seg, DSC_seg)
        overshoot_peaks.append(ov_peak)

        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear', bounds_error=False, fill_value='extrapolate')
        if not np.isnan(T_onset):
            T_grid = np.arange(t_int_low, T_onset + 0.01, 0.1)
            delta_DSC = interp_dsc(T_grid) - interp_empty(T_grid)
            all_integrals.append(trapezoid(delta_DSC, T_grid))
        else:
            all_integrals.append(np.nan)

    integrals = np.array(all_integrals)
    ref = ref_integral if ref_integral is not None else integrals[0]
    dH = (integrals - ref) * CONV_JG

    # Build T1 group labels from actual hold times
    T1_groups = []
    for c in conditions:
        if c['T1_hold_s'] is not None:
            T1_groups.append(f"{int(round(c['T1_hold_s']))}s")
        else:
            T1_groups.append('?')

    results = []
    for i, cond in enumerate(conditions):
        results.append({
            'ramp': i + 1,
            'experiment': label,
            'T1_C': cond['T1_C'],
            'T2_C': cond['T2_C'],
            'T1_hold_s': cond['T1_hold_s'],
            'T2_hold_s': cond['T2_hold_s'],
            'T1_cool_rate_K_per_min': cond['T1_cool_rate_K_per_min'],
            'up_jump_rate_K_per_min': cond['T2_rate_K_per_min'],
            'cooling_rate_to_glass_K_per_min': cond['cooling_rate_to_glass_K_per_min'],
            'heating_rate_K_per_min': HEATING_RATE,
            'T1_group': T1_groups[i],
            'T_onset_C': T_onsets[i],
            'delta_H_J_per_g': dH[i],
            'overshoot_peak_uW': overshoot_peaks[i],
        })

    return pd.DataFrame(results)


def process_reference_scan(data_file, sheet, interp_empty, t_int_low=T_INT_LOW_DEFAULT):
    """Process a reference (unaged) DSC scan and return its integral value.
    Used as the reference baseline for enthalpy calculations."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    if len(ramps) == 0:
        raise ValueError(f"No heating ramps found in reference file: {data_file}")

    s, e = ramps[0]
    T_seg = T_exp[s:e + 1]
    DSC_seg = DSC_exp[s:e + 1]
    T_onset = find_liquid_onset(T_seg, DSC_seg)

    interp_dsc = interp1d(T_seg, DSC_seg, kind='linear', bounds_error=False, fill_value='extrapolate')
    T_grid = np.arange(t_int_low, T_onset + 0.01, 0.1)
    delta_DSC = interp_dsc(T_grid) - interp_empty(T_grid)
    ref_integral = trapezoid(delta_DSC, T_grid)
    print(f"  Reference scan: T_onset={T_onset:.2f}°C, integral={ref_integral:.2f} µW·°C²")
    return ref_integral, T_onset


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export enthalpy data from DSC experiments.')
    parser.add_argument('--t-int-low', type=float, default=T_INT_LOW_DEFAULT,
                        help=f'Lower integration bound in °C (default: {T_INT_LOW_DEFAULT})')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output CSV filenames (e.g. "_70C")')
    args = parser.parse_args()
    t_int_low = args.t_int_low
    suffix = args.suffix or f'_{int(t_int_low)}C'

    print("=" * 70)
    print(f"  Exporting enthalpy data (T_int_low = {t_int_low}°C)")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load empty baseline
    interp_empty = load_empty()

    range_label = f'{int(t_int_low)}-Tg'

    # ── One-step 50°C ──
    print(f"\n[1/5] One-step annealing 50°C (int: {range_label}) ...")
    df_os_50 = process_onestep(
        os.path.join(DATA_DIR, 'PS-onestep-01.xlsx'),
        sheet='PS-onestep-01', T_anneal=50, label='One-step 50C',
        interp_empty=interp_empty, t_int_low=t_int_low,
    )
    out = os.path.join(RESULTS_DIR, f'enthalpy_onestep_50C{suffix}.csv')
    df_os_50.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_os_50)} rows)")
    print(df_os_50.to_string(index=False))

    # ── One-step 70°C ──
    print(f"\n[2/5] One-step annealing 70°C (int: {range_label}) ...")
    df_os_70 = process_onestep(
        os.path.join(DATA_DIR, 'PS-onestep-02.xlsx'),
        sheet='PS-onestep-02', T_anneal=70, label='One-step 70C',
        interp_empty=interp_empty, t_int_low=t_int_low,
    )
    out = os.path.join(RESULTS_DIR, f'enthalpy_onestep_70C{suffix}.csv')
    df_os_70.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_os_70)} rows)")
    print(df_os_70.to_string(index=False))

    # ── Two-step ──
    print(f"\n[3/5] Two-step annealing (int: {range_label}) ...")
    df_ts = process_twosteps(
        os.path.join(DATA_DIR, 'twosteps.xlsx'),
        sheet='PS-02', label='Two-step',
        interp_empty=interp_empty, t_int_low=t_int_low,
    )
    out = os.path.join(RESULTS_DIR, f'enthalpy_twosteps{suffix}.csv')
    df_ts.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_ts)} rows)")
    print(df_ts.to_string(index=False))

    # ── Kovacs (original) ──
    print(f"\n[4/5] Kovacs up-jump annealing (int: {range_label}) ...")
    df_kov = process_kovacs(
        os.path.join(DATA_DIR, 'pskovacs.xlsx'),
        sheet='PS-kovacs-01', label='Kovacs',
        interp_empty=interp_empty, t_int_low=t_int_low,
    )
    out = os.path.join(RESULTS_DIR, f'enthalpy_kovacs{suffix}.csv')
    df_kov.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_kov)} rows)")
    print(df_kov.to_string(index=False))

    # ── Kovacs 80→90°C (new, 10→180 ramp, with external reference) ──
    print(f"\n[5/5] Kovacs 80→90°C (int: {range_label}, ext ref) ...")
    ref_integral, ref_T_onset = process_reference_scan(
        os.path.join(DATA_DIR, 'ps-ref-02.xlsx'),
        sheet='ps-ref-02',
        interp_empty=interp_empty, t_int_low=t_int_low,
    )
    df_kov_new = process_kovacs(
        os.path.join(DATA_DIR, 'ps-02.xlsx'),
        sheet='ps-02', label='Kovacs 80-90C',
        interp_empty=interp_empty, t_int_low=t_int_low,
        ref_integral=ref_integral,
    )
    out = os.path.join(RESULTS_DIR, f'enthalpy_kovacs_80-90C{suffix}.csv')
    df_kov_new.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_kov_new)} rows)")
    print(df_kov_new.to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"  All enthalpy data (T_int_low={t_int_low}°C) exported to {RESULTS_DIR}/")
    print(f"{'=' * 70}")

    # ── Generate plots ──
    print(f"\n[6/7] Generating combined result plots...")
    plot_all_results(df_os_50, df_os_70, df_ts, df_kov, t_int_low,
                     df_kov_new=df_kov_new)

    print(f"\n[7/7] Generating new Kovacs 80→90°C plot...")
    plot_kovacs_new(df_kov_new, ref_T_onset, t_int_low)


def plot_all_results(df_os_50, df_os_70, df_ts, df_kov, t_int_low=None,
                     df_kov_new=None):
    """Generate 2x2 comparison plot for all experiments.
    If df_kov_new is provided, it is overlaid on the Kovacs panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    int_label = f'(Integral: {int(t_int_low)}°C–Tg)' if t_int_low else ''

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Enthalpy Recovery — All Experiments {int_label}', fontsize=14, fontweight='bold')
    color_cycle = ['#2166AC', '#B2182B', '#4DAF4A', '#FF7F00', '#984EA3']
    marker_cycle = ['o', 's', '^', 'D', 'v']

    def _get_groups(df):
        grp_col = 'cooling_group' if 'cooling_group' in df.columns else 'T1_group'
        return sorted(df[grp_col].unique())

    def _get_group_colors_markers(all_dfs):
        all_groups = set()
        for df in all_dfs:
            if df is not None:
                all_groups.update(_get_groups(df))
        groups_sorted = sorted(all_groups, key=lambda x: int(x.replace('s', '')) if x.replace('s', '').isdigit() else 0)
        return ({g: color_cycle[i % len(color_cycle)] for i, g in enumerate(groups_sorted)},
                {g: marker_cycle[i % len(marker_cycle)] for i, g in enumerate(groups_sorted)})

    colors, markers = _get_group_colors_markers([df_os_50, df_os_70, df_ts, df_kov, df_kov_new])

    def plot_one(ax, df, title, x_col, xlabel):
        grp_col = 'cooling_group' if 'cooling_group' in df.columns else 'T1_group'
        for grp in sorted(df[grp_col].unique()):
            g = df[df[grp_col] == grp]
            if len(g) > 0:
                ax.plot(g[x_col], g['delta_H_J_per_g'], marker=markers.get(grp, 'o'),
                        color=colors.get(grp, 'gray'), linewidth=1.5, markersize=7,
                        markerfacecolor='white', markeredgewidth=1.5, label=grp)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('ΔH (J/g)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    # Panel 1: One-step 50°C
    plot_one(axes[0, 0], df_os_50, 'One-step annealing @50°C', 'hold_s',
             'Hold time at 50°C (s)')

    # Panel 2: One-step 70°C
    plot_one(axes[0, 1], df_os_70, 'One-step annealing @70°C', 'hold_s',
             'Hold time at 70°C (s)')

    # Panel 3: Two-step
    plot_one(axes[1, 0], df_ts, 'Two-step annealing (90→80°C)', 'T2_hold_s',
             'T2 hold time at 80°C (s)')

    # Panel 4: Kovacs (old + new overlaid)
    ax4 = axes[1, 1]
    grp_col = 'T1_group'
    # Old Kovacs data — solid lines
    for grp in sorted(df_kov[grp_col].unique()):
        g = df_kov[df_kov[grp_col] == grp]
        if len(g) > 0:
            ax4.plot(g['T2_hold_s'], g['delta_H_J_per_g'],
                     marker=markers.get(grp, 'o'), color=colors.get(grp, 'gray'),
                     linewidth=1.5, markersize=7, markerfacecolor='white',
                     markeredgewidth=1.5, linestyle='-', label=f'{grp} (old)')
    # New Kovacs data — dashed lines with larger markers
    if df_kov_new is not None and len(df_kov_new) > 0:
        for grp in sorted(df_kov_new[grp_col].unique()):
            g = df_kov_new[df_kov_new[grp_col] == grp]
            if len(g) > 0:
                ax4.plot(g['T2_hold_s'], g['delta_H_J_per_g'],
                         marker=markers.get(grp, 'o'), color=colors.get(grp, 'gray'),
                         linewidth=2.0, markersize=9, markerfacecolor=colors.get(grp, 'gray'),
                         markeredgewidth=1.5, linestyle='--', label=f'{grp} (new)')
    ax4.set_xlabel('T2 hold time at 90°C (s)')
    ax4.set_ylabel('ΔH (J/g)')
    ax4.set_title('Kovacs up-jump (80→90°C)')
    ax4.set_xscale('log')
    ax4.invert_yaxis()
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = f'_{int(t_int_low)}C' if t_int_low else ''
    png_path = os.path.join(RESULTS_DIR, f'enthalpy_all_results{suffix}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {png_path}")
    plt.close()


def plot_kovacs_new(df_kov_new, ref_T_onset, t_int_low=None):
    """Generate a dedicated plot for the new Kovacs 80→90°C experiment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    int_label = f'(Integral: {int(t_int_low)}°C–Tg)' if t_int_low else ''

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Kovacs 80→90°C — New Experiment {int_label}',
                 fontsize=14, fontweight='bold')

    grp_col = 'T1_group'
    color_cycle = ['#2166AC', '#B2182B', '#4DAF4A', '#FF7F00']
    marker_cycle = ['o', 's', '^', 'D']

    for i, grp in enumerate(sorted(df_kov_new[grp_col].unique())):
        g = df_kov_new[df_kov_new[grp_col] == grp]
        color = color_cycle[i % len(color_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        ax.plot(g['T2_hold_s'], g['delta_H_J_per_g'],
                marker=marker, color=color, linewidth=1.5, markersize=8,
                markerfacecolor='white', markeredgewidth=1.5, label=f'T1 hold = {grp}')
        for _, row in g.iterrows():
            ax.annotate(f"{row['T2_hold_s']:.1f}s",
                        (row['T2_hold_s'], row['delta_H_J_per_g']),
                        textcoords="offset points", xytext=(8, -5),
                        fontsize=8, color=color)

    ax.set_xlabel('T2 hold time at 90°C (s)')
    ax.set_ylabel('ΔH (J/g)')
    ax.set_title(f'Kovacs up-jump 80→90°C | Ref T_onset = {ref_T_onset:.1f}°C')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = f'_{int(t_int_low)}C' if t_int_low else ''
    png_path = os.path.join(RESULTS_DIR, f'enthalpy_kovacs_80-90C{suffix}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {png_path}")
    plt.close()


if __name__ == '__main__':
    main()
