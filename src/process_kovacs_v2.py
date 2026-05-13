"""
Two-step annealing DSC — v2: condition-dependent liquid onset integration.
Compares fixed integration ranges vs. per-ramp liquid onset temperature.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dsc')

M_SAMPLE = 4.7
HEATING_RATE = 10.0
BETA = HEATING_RATE / 60.0
DT_DT = 1.0 / BETA
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)
MW = 280000
CONV_KJMOL = CONV_FACTOR * MW / 1000

T_INT_LOW = 30  # fixed lower bound

# ── Data loading (same as v1) ─────────────────────────────────────────────
def load_dsc_simple(filename, sheet=None):
    if sheet is None: df = pd.read_excel(filename)
    else: df = pd.read_excel(filename, sheet_name=sheet)
    header_row = None
    for i in range(len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i; break
    if header_row is None:
        for i in range(len(df)):
            try:
                a=float(str(df.iloc[i,0]).strip()); b=float(str(df.iloc[i,1]).strip())
                if np.isfinite(a) and np.isfinite(b): header_row = i; break
            except: continue
    v0 = df.iloc[header_row, 0]
    if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
        data_start = header_row + 1
        v0_next = df.iloc[data_start, 0]
        if v0_next is not None and isinstance(v0_next, str) and 'min' in str(v0_next).lower():
            data_start += 1
        data = df.iloc[data_start:].copy()
    else: data = df.iloc[header_row:].copy()
    data = data.dropna(axis=1, how='all').reset_index(drop=True)
    ncols = data.shape[1]
    col_names = ['Time', 'Temp', 'DSC', 'DDSC'] + [f'Extra_{i}' for i in range(max(0,ncols-4))]
    data.columns = col_names[:ncols]
    return data[['Time','Temp','DSC','DDSC']].astype(float).reset_index(drop=True)


def load_dsc_experiment(filename, sheet):
    df = pd.read_excel(filename, sheet_name=sheet)
    program = []
    for i in range(7, len(df)):
        v1 = df.iloc[i, 1]
        if v1 is not None and isinstance(v1, str) and ('温度' in str(v1) or '冷却' in str(v1)): break
        try:
            step_num = int(float(str(df.iloc[i,1]).strip()))
            t_start = float(str(df.iloc[i,2]).strip())
            t_end = float(str(df.iloc[i,3]).strip())
            rate = float(str(df.iloc[i,4]).strip())
            time_val = float(str(df.iloc[i,5]).strip())
            program.append({'step':step_num,'T_start':t_start,'T_end':t_end,
                           'rate':rate,'time_min':time_val})
        except: continue
    header_row = None
    for i in range(90, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i; break
    raw = df.iloc[header_row+2:].copy()
    data = pd.DataFrame({
        'Time': pd.to_numeric(raw.iloc[:,0], errors='coerce'),
        'Temp': pd.to_numeric(raw.iloc[:,1], errors='coerce'),
        'DSC':  pd.to_numeric(raw.iloc[:,2], errors='coerce'),
        'DDSC': pd.to_numeric(raw.iloc[:,3], errors='coerce'),
    }).dropna().reset_index(drop=True).astype(float)
    return data, program


def detect_heating_ramps(T, t, min_rate=4.0, min_duration_pts=500):
    n = len(T); window = 30
    dTdt = np.zeros(n)
    for i in range(window, n-window):
        dTdt[i] = (T[i+window]-T[i-window]) / max(t[i+window]-t[i-window], 1e-12)
    is_h = dTdt > min_rate
    segs = []; in_seg = False; start = 0
    for i in range(n):
        if is_h[i] and not in_seg: in_seg = True; start = i
        elif not is_h[i] and in_seg:
            if i-start > min_duration_pts:
                seg_T = T[start:i]
                if np.min(seg_T)<40 and np.max(seg_T)>180: segs.append((start,i-1))
            in_seg = False
    if in_seg and n-start > min_duration_pts:
        seg_T = T[start:n]
        if np.min(seg_T)<40 and np.max(seg_T)>180: segs.append((start,n-1))
    return segs


# ── NEW: Liquid onset detection ───────────────────────────────────────────
def find_liquid_onset(T_seg, DSC_seg, T_liq_fit_lo=110, T_liq_fit_hi=160,
                       T_tg_lo=70, T_tg_hi=115):
    """Find temperature where DSC returns to liquid baseline after Tg overshoot.

    Returns T_onset (°C) or np.nan if not found.
    """
    # 1. Fit liquid baseline
    liq_mask = (T_seg >= T_liq_fit_lo) & (T_seg <= T_liq_fit_hi)
    if liq_mask.sum() < 10:
        return np.nan
    coeffs = np.polyfit(T_seg[liq_mask], DSC_seg[liq_mask], 1)
    liquid_line = np.polyval(coeffs, T_seg)

    # 2. Find overshoot peak in Tg region
    tg_mask = (T_seg >= T_tg_lo) & (T_seg <= T_tg_hi)
    if tg_mask.sum() < 10:
        return np.nan
    peak_offset = np.argmax(DSC_seg[tg_mask])
    peak_idx = np.arange(len(T_seg))[tg_mask][peak_offset]

    # 3. Scan after peak to find crossover (DSC crosses below liquid line)
    after_peak = np.arange(peak_idx, len(T_seg))
    diff = DSC_seg[after_peak] - liquid_line[after_peak]

    for j in range(1, len(diff)):
        if diff[j-1] >= 0 and diff[j] < 0:
            return T_seg[after_peak[j]]

    return np.nan


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Two-step annealing v2 — Liquid-onset-based integration comparison")
    print("=" * 70)

    # Load
    exp_data, program = load_dsc_experiment(os.path.join(DATA_DIR, 'twosteps.xlsx'), sheet='PS-02')
    T_exp = exp_data['Temp'].values
    t_exp = exp_data['Time'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    print(f"  Found {len(ramps)} heating ramps")

    # Map conditions
    heating_steps = [p for p in program if p['T_start']==30 and p['T_end']==200]
    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]; step_num = h_step['step']
            cool_step = anneal_step_1 = None
            for p in program:
                if p['step'] == step_num-3: cool_step = p
                elif p['step'] == step_num-2: anneal_step_1 = p
            t1_hold = cool_step['time_min'] if cool_step else None
            t2_hold = anneal_step_1['time_min'] if anneal_step_1 else None
        else: t1_hold = t2_hold = None
        conditions.append({'ramp_idx':idx+1, 'T1_hold_min':t1_hold, 'T2_hold_min':t2_hold,
                          'start_idx':s, 'end_idx':e})

    # Convert to seconds, filter
    MIN_HOLD = 0.1 * 60
    for c in conditions:
        c['T1_hold_s'] = c['T1_hold_min']*60 if c['T1_hold_min'] else None
        c['T2_hold_s'] = c['T2_hold_min']*60 if c['T2_hold_min'] else None
    conditions = [c for c in conditions if c['T2_hold_s'] and c['T2_hold_s'] >= MIN_HOLD]

    # ── Analyze each ramp: find T_onset, compute integrals for multiple ranges ──
    fixed_highs = [90, 95, 100, 105, 110]
    range_labels = [f'30–{h}°C' for h in fixed_highs] + ['30–T_onset']

    print(f"\n  Per-ramp analysis (liquid onset + multi-range integrals):")
    print(f"  {'Ramp':<6} {'T_onset':>8}  " + "  ".join(f"{lbl:>14}" for lbl in range_labels))

    all_results = {lbl: [] for lbl in range_labels}
    T_onsets = []

    for ramp_idx, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]; DSC_seg = DSC_exp[s:e+1]

        # Find liquid onset
        T_onset = find_liquid_onset(T_seg, DSC_seg)
        T_onsets.append(T_onset)

        # Compute integrals for each range
        integrals = {}
        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear',
                              bounds_error=False, fill_value='extrapolate')

        for T_high in fixed_highs:
            T_grid = np.arange(T_INT_LOW, T_high + 0.01, 0.1)
            DSC_grid = interp_dsc(T_grid)
            integrals[f'30–{T_high}°C'] = trapezoid(DSC_grid, T_grid)

        # Standard range: 30°C to T_onset
        if not np.isnan(T_onset):
            T_grid = np.arange(T_INT_LOW, T_onset + 0.01, 0.1)
            DSC_grid = interp_dsc(T_grid)
            integrals['30–T_onset'] = trapezoid(DSC_grid, T_grid)
        else:
            integrals['30–T_onset'] = np.nan

        row = f"  {ramp_idx+1:<4d}  {T_onset:8.1f}"
        for lbl in range_labels:
            row += f"  {integrals[lbl]:>14.1f}"
            all_results[lbl].append(integrals[lbl])
        print(row)

    # ── Compute ΔH (reference-based) for each range ──
    print(f"\n  ── ΔH comparison (kJ/mol, ref = shortest T2) ──")

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Panel 1: DSC curves with T_onset markers
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(conditions)))
    for i, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]; DSC_seg = DSC_exp[s:e+1]
        ax.plot(T_seg, DSC_seg, alpha=0.5, linewidth=0.6, color=colors[i])
        if not np.isnan(T_onsets[i]):
            ax.axvline(T_onsets[i], color=colors[i], alpha=0.4, linewidth=0.5, linestyle='--')
    ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('DSC (µW)')
    ax.set_title('DSC curves + liquid onset markers')
    ax.set_xlim(28, 170)

    # Panel 2: T_onset vs annealing time
    ax = axes[0, 1]
    t2_vals = [c['T2_hold_s'] for c in conditions]
    t1_vals = [c['T1_hold_s'] for c in conditions]
    grp_a = [i for i, t in enumerate(t1_vals) if t and t < 60]
    grp_b = [i for i, t in enumerate(t1_vals) if t and t > 60]
    ax.plot([t2_vals[i] for i in grp_a], [T_onsets[i] for i in grp_a],
            'o-', color='#2166AC', label='T1=50s')
    ax.plot([t2_vals[i] for i in grp_b], [T_onsets[i] for i in grp_b],
            's-', color='#B2182B', label='T1=500s')
    ax.set_xlabel('T2 hold time (s)'); ax.set_ylabel('T_liquid_onset (°C)')
    ax.set_title('Liquid onset temperature vs annealing')
    ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

    # Panels 3-6: ΔH vs T2 for each integration range
    plot_ranges = ['30–95°C', '30–100°C', '30–105°C', '30–T_onset']
    for pi, rl in enumerate(plot_ranges):
        ax = axes[(pi+2)//3, (pi+2)%3]

        integrals = np.array(all_results[rl])
        ref = integrals[0]
        delta_H = (integrals - ref) * CONV_KJMOL
        offset = max(0, -delta_H.min()) + 1.0
        delta_H_pos = delta_H + offset

        ga = [delta_H_pos[i] for i in grp_a]
        gb = [delta_H_pos[i] for i in grp_b]
        ta = [t2_vals[i] for i in grp_a]
        tb = [t2_vals[i] for i in grp_b]

        ax.plot(ta, ga, 'o-', color='#2166AC', markersize=8, markerfacecolor='white',
                markeredgewidth=1.5, label='T1=50s')
        ax.plot(tb, gb, 's-', color='#B2182B', markersize=8, markerfacecolor='white',
                markeredgewidth=1.5, label='T1=500s')
        ax.set_xlabel('T2 hold time (s)'); ax.set_ylabel('ΔH (kJ/mol)')
        ax.set_title(f'Range: {rl}')
        ax.set_xscale('log'); ax.invert_yaxis(); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, 'twosteps_v2_comparison.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  Saved {out_png}")

    # ── Save CSV for standard range ──
    rl = '30–T_onset'
    integrals = np.array(all_results[rl])
    ref = integrals[0]
    delta_H = (integrals - ref) * CONV_KJMOL
    offset = max(0, -delta_H.min()) + 1.0
    delta_H_pos = delta_H + offset

    results = []
    for i, cond in enumerate(conditions):
        results.append({
            'ramp': i+1,
            'T1_hold_s': cond['T1_hold_s'],
            'T2_hold_s': cond['T2_hold_s'],
            'T_onset_C': T_onsets[i],
            'delta_H_kJmol': delta_H_pos[i],
        })

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, 'twosteps_v2_results.csv')
    results_df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"  Saved {out_csv}")

    print(f"\n  Standard range (30–T_onset) results:")
    for _, r in results_df.iterrows():
        print(f"    R{r['ramp']:.0f}: T1={r['T1_hold_s']:.0f}s, T2={r['T2_hold_s']:.0f}s, "
              f"T_onset={r['T_onset_C']:.1f}°C, ΔH={r['delta_H_kJmol']:.2f} kJ/mol")

    return results_df


if __name__ == '__main__':
    results = main()
