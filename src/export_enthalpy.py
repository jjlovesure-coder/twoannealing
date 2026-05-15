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
T_INT_LOW = 30


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
    for i in range(70, len(df)):
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


def detect_heating_ramps(T, t, min_rate=4.0, min_duration_pts=500):
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
                if np.min(seg_T) < 40 and np.max(seg_T) > 180:
                    segs.append((start, i - 1))
            in_seg = False
    if in_seg and n - start > min_duration_pts:
        seg_T = T[start:n]
        if np.min(seg_T) < 40 and np.max(seg_T) > 180:
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


def process_onestep(data_file, sheet, T_anneal, label):
    """Process one-step annealing and return enriched DataFrame."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program if p['T_start_C'] == 30 and p['T_end_C'] == 200]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            # Find the preceding steps
            step_to_Tanneal = None  # step_num-2: 200 → T_anneal (annealing step)
            step_to_30 = None       # step_num-1: T_anneal → 30 (cooling to glass)
            for p in program:
                if p['step'] == step_num - 2:
                    step_to_Tanneal = p
                elif p['step'] == step_num - 1:
                    step_to_30 = p
            t_hold = step_to_Tanneal['hold_min'] if step_to_Tanneal else None
            cool_rate = step_to_30['rate_K_per_min'] if step_to_30 else None
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
            T_grid = np.arange(T_INT_LOW, T_onset + 0.01, 0.1)
            all_integrals.append(trapezoid(interp_dsc(T_grid), T_grid))
        else:
            all_integrals.append(np.nan)

    integrals = np.array(all_integrals)
    # ΔH_released = integral_ramp − integral_ref (unannealed baseline)
    # Positive = exothermic (energy released during annealing)
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


def process_twosteps(data_file, sheet, label):
    """Process two-step annealing and return enriched DataFrame."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program if p['T_start_C'] == 30 and p['T_end_C'] == 200]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            step_to_T1 = None    # step_num-3: 200 → T1 (pre-annealing)
            step_to_T2 = None    # step_num-2: T1 → T2 (annealing)
            step_to_30 = None    # step_num-1: T2 → 30 (cooling to glass)
            for p in program:
                if p['step'] == step_num - 3:
                    step_to_T1 = p
                elif p['step'] == step_num - 2:
                    step_to_T2 = p
                elif p['step'] == step_num - 1:
                    step_to_30 = p

            T1_C = step_to_T1['T_end_C'] if step_to_T1 else None
            T2_C = step_to_T2['T_end_C'] if step_to_T2 else None
            T1_hold_min = step_to_T1['hold_min'] if step_to_T1 else None
            T2_hold_min = step_to_T2['hold_min'] if step_to_T2 else None
            cool_rate = step_to_30['rate_K_per_min'] if step_to_30 else None
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
            T_grid = np.arange(T_INT_LOW, T_onset + 0.01, 0.1)
            all_integrals.append(trapezoid(interp_dsc(T_grid), T_grid))
        else:
            all_integrals.append(np.nan)

    integrals = np.array(all_integrals)
    # ΔH_released = integral_ramp − integral_ref (unannealed baseline)
    # Positive = exothermic (energy released during annealing)
    ref = integrals[0]
    dH = (integrals - ref) * CONV_JG

    # Determine T1 group (50s vs 500s)
    n = len(conditions)
    midpoint = n // 2
    T1_groups = ['50s'] * midpoint + ['500s'] * (n - midpoint)

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


def process_kovacs(data_file, sheet, label):
    """Process Kovacs up-jump annealing and return enriched DataFrame."""
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program if p['T_start_C'] == 30 and p['T_end_C'] == 200]

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            step_to_T1 = None    # step_num-3: 200 → T1 (pre-annealing)
            step_to_T2 = None    # step_num-2: T1 → T2 (up-jump annealing)
            step_to_30 = None    # step_num-1: T2 → 30 (cooling to glass)
            for p in program:
                if p['step'] == step_num - 3:
                    step_to_T1 = p
                elif p['step'] == step_num - 2:
                    step_to_T2 = p
                elif p['step'] == step_num - 1:
                    step_to_30 = p

            T1_C = step_to_T1['T_end_C'] if step_to_T1 else None
            T2_C = step_to_T2['T_end_C'] if step_to_T2 else None
            T1_hold_min = step_to_T1['hold_min'] if step_to_T1 else None
            T2_hold_min = step_to_T2['hold_min'] if step_to_T2 else None
            cool_rate = step_to_30['rate_K_per_min'] if step_to_30 else None
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
            T_grid = np.arange(T_INT_LOW, T_onset + 0.01, 0.1)
            all_integrals.append(trapezoid(interp_dsc(T_grid), T_grid))
        else:
            all_integrals.append(np.nan)

    integrals = np.array(all_integrals)
    # ΔH_released = integral_ramp − integral_ref (unannealed baseline)
    # Positive = exothermic (energy released during annealing)
    ref = integrals[0]
    dH = (integrals - ref) * CONV_JG

    n = len(conditions)
    midpoint = n // 2
    T1_groups = ['50s'] * midpoint + ['500s'] * (n - midpoint)

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


def main():
    print("=" * 70)
    print("  Exporting enthalpy data to data/ folder")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── One-step 50°C ──
    print("\n[1/4] One-step annealing 50°C ...")
    df_os_50 = process_onestep(
        os.path.join(DATA_DIR, 'PS-onestep-01.xlsx'),
        sheet='PS-onestep-01', T_anneal=50, label='One-step 50C'
    )
    out = os.path.join(RESULTS_DIR, 'enthalpy_onestep_50C.csv')
    df_os_50.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_os_50)} rows)")
    print(df_os_50.to_string(index=False))

    # ── One-step 70°C ──
    print("\n[2/4] One-step annealing 70°C ...")
    df_os_70 = process_onestep(
        os.path.join(DATA_DIR, 'PS-onestep-02.xlsx'),
        sheet='PS-onestep-02', T_anneal=70, label='One-step 70C'
    )
    out = os.path.join(RESULTS_DIR, 'enthalpy_onestep_70C.csv')
    df_os_70.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_os_70)} rows)")
    print(df_os_70.to_string(index=False))

    # ── Two-step ──
    print("\n[3/4] Two-step annealing ...")
    df_ts = process_twosteps(
        os.path.join(DATA_DIR, 'twosteps.xlsx'),
        sheet='PS-02', label='Two-step'
    )
    out = os.path.join(RESULTS_DIR, 'enthalpy_twosteps.csv')
    df_ts.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_ts)} rows)")
    print(df_ts.to_string(index=False))

    # ── Kovacs ──
    print("\n[4/4] Kovacs up-jump annealing ...")
    df_kov = process_kovacs(
        os.path.join(DATA_DIR, 'pskovacs.xlsx'),
        sheet='PS-kovacs-01', label='Kovacs'
    )
    out = os.path.join(RESULTS_DIR, 'enthalpy_kovacs.csv')
    df_kov.to_csv(out, index=False, float_format='%.4f')
    print(f"  Saved {out}  ({len(df_kov)} rows)")
    print(df_kov.to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"  All enthalpy data exported to {RESULTS_DIR}/")
    print(f"{'=' * 70}")

    # ── Generate plots ──
    print(f"\n[5/5] Generating result plots...")
    plot_all_results(df_os_50, df_os_70, df_ts, df_kov)


def plot_all_results(df_os_50, df_os_70, df_ts, df_kov):
    """Generate 2x2 comparison plot for all experiments."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {'50s': '#2166AC', '500s': '#B2182B'}
    markers = {'50s': 'o', '500s': 's'}

    def plot_one(ax, df, title, x_col, xlabel):
        for grp in ['50s', '500s']:
            g = df[df['cooling_group'] == grp] if 'cooling_group' in df.columns else df[df['T1_group'] == grp]
            if len(g) == 0:
                g1 = df[df['T1_group'] == grp] if 'T1_group' in df.columns else pd.DataFrame()
                g2 = df[df['cooling_group'] == grp] if 'cooling_group' in df.columns else pd.DataFrame()
                g = g1 if len(g1) > 0 else g2
            if len(g) > 0:
                ax.plot(g[x_col], g['delta_H_J_per_g'], marker=markers.get(grp, 'o'),
                        color=colors.get(grp, 'gray'), linewidth=1.5, markersize=7,
                        markerfacecolor='white', markeredgewidth=1.5, label=grp)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('ΔH (J/g)')
        ax.set_title(title)
        ax.set_xscale('log')
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

    # Panel 4: Kovacs
    ax4 = axes[1, 1]
    for grp in ['50s', '500s']:
        g = df_kov[df_kov['T1_group'] == grp]
        if len(g) > 0:
            ax4.plot(g['T2_hold_s'], g['delta_H_J_per_g'],
                     marker=markers.get(grp, 'o'), color=colors.get(grp, 'gray'),
                     linewidth=1.5, markersize=7, markerfacecolor='white',
                     markeredgewidth=1.5, label=grp)
    ax4.set_xlabel('T2 hold time at 90°C (s)')
    ax4.set_ylabel('ΔH (J/g)')
    ax4.set_title('Kovacs up-jump (80→90°C)')
    ax4.set_xscale('log')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'enthalpy_all_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {png_path}")
    plt.close()


if __name__ == '__main__':
    main()
