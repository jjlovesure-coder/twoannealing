"""
One-step annealing DSC v2 — Liquid-onset integration for both 50°C and 70°C.
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
BETA = 10.0 / 60.0
DT_DT = 1.0 / BETA
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)
MW = 280000
CONV_KJMOL = CONV_FACTOR * MW / 1000

T_INT_LOW = 30

# ── Data loading ──────────────────────────────────────────────────────────
def load_dsc_simple(filename):
    df = pd.read_excel(filename)
    header_row = None
    for i in range(len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0): header_row = i; break
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
        if v0_next is not None and isinstance(v0_next, str) and 'min' in str(v0_next).lower(): data_start += 1
        data = df.iloc[data_start:].copy()
    else: data = df.iloc[header_row:].copy()
    data = data.dropna(axis=1, how='all').reset_index(drop=True)
    ncols = data.shape[1]
    col_names = ['Time','Temp','DSC','DDSC'] + [f'Extra_{i}' for i in range(max(0,ncols-4))]
    data.columns = col_names[:ncols]
    return data[['Time','Temp','DSC','DDSC']].astype(float).reset_index(drop=True)


def load_dsc_experiment(filename, sheet=None):
    if sheet: df = pd.read_excel(filename, sheet_name=sheet)
    else: df = pd.read_excel(filename)
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
            program.append({'step':step_num,'T_start':t_start,'T_end':t_end,'rate':rate,'time_min':time_val})
        except: continue
    header_row = None
    for i in range(70, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0): header_row = i; break
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


def find_liquid_onset(T_seg, DSC_seg, T_liq_fit_lo=110, T_liq_fit_hi=160, T_tg_lo=70, T_tg_hi=115):
    liq_mask = (T_seg >= T_liq_fit_lo) & (T_seg <= T_liq_fit_hi)
    if liq_mask.sum() < 10: return np.nan
    coeffs = np.polyfit(T_seg[liq_mask], DSC_seg[liq_mask], 1)
    liquid_line = np.polyval(coeffs, T_seg)
    tg_mask = (T_seg >= T_tg_lo) & (T_seg <= T_tg_hi)
    if tg_mask.sum() < 10: return np.nan
    peak_offset = np.argmax(DSC_seg[tg_mask])
    peak_idx = np.arange(len(T_seg))[tg_mask][peak_offset]
    after_peak = np.arange(peak_idx, len(T_seg))
    diff = DSC_seg[after_peak] - liquid_line[after_peak]
    for j in range(1, len(diff)):
        if diff[j-1] >= 0 and diff[j] < 0: return T_seg[after_peak[j]]
    return np.nan


# ── Process one experiment ────────────────────────────────────────────────
def process_onestep(data_file, sheet, T_anneal, label):
    print(f"\n{'='*60}")
    print(f"  {label} — T_anneal={T_anneal}°C")
    print(f"{'='*60}")

    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values; DSC_exp = exp_data['DSC'].values; t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    print(f"  Found {len(ramps)} heating ramps")

    # Map: step 3n+1=200→T_anneal(hold), 3n+2=T_anneal→30, 3n+3=30→200
    heating_steps = [p for p in program if p['T_start']==30 and p['T_end']==200]
    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]; step_num = h_step['step']
            anneal_step = None
            for p in program:
                if p['step'] == step_num-2: anneal_step = p; break  # 200→T_anneal
            t_hold = anneal_step['time_min'] if anneal_step else None
        else: t_hold = None
        conditions.append({'ramp_idx':idx+1, 'hold_min':t_hold, 'start_idx':s, 'end_idx':e})

    # Convert to seconds
    for c in conditions:
        c['hold_s'] = c['hold_min']*60 if c['hold_min'] else None

    fixed_highs = [90, 95, 100, 105, 110]
    range_labels = [f'30–{h}°C' for h in fixed_highs] + ['30–T_onset']
    all_integrals = {lbl: [] for lbl in range_labels}
    T_onsets = []

    for ramp_idx, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]; DSC_seg = DSC_exp[s:e+1]
        T_onset = find_liquid_onset(T_seg, DSC_seg)
        T_onsets.append(T_onset)
        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear', bounds_error=False, fill_value='extrapolate')
        for T_high in fixed_highs:
            T_grid = np.arange(T_INT_LOW, T_high+0.01, 0.1)
            all_integrals[f'30–{T_high}°C'].append(trapezoid(interp_dsc(T_grid), T_grid))
        if not np.isnan(T_onset):
            T_grid = np.arange(T_INT_LOW, T_onset+0.01, 0.1)
            all_integrals['30–T_onset'].append(trapezoid(interp_dsc(T_grid), T_grid))
        else:
            all_integrals['30–T_onset'].append(np.nan)

    return conditions, T_onsets, all_integrals, T_exp, DSC_exp


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # Process both files
    files = [
        (os.path.join(DATA_DIR, 'PS-onestep-01.xlsx'), 'PS-onestep-01', 50, 'Onestep 50°C'),
        (os.path.join(DATA_DIR, 'PS-onestep-02.xlsx'), 'PS-onestep-02', 70, 'Onestep 70°C'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    all_dfs = []

    for fi, (fname, sheet, T_anneal, label) in enumerate(files):
        conditions, T_onsets, all_integrals, T_exp, DSC_exp = process_onestep(fname, sheet, T_anneal, label)

        n_half = len(conditions)//2
        t_hold = [c['hold_s'] for c in conditions]

        # Standard range results
        rl = '30–T_onset'
        integrals = np.array(all_integrals[rl])
        ref = integrals[0]
        delta_H = (integrals - ref) * CONV_KJMOL
        offset = max(0, -delta_H.min()) + 1.0
        dH = delta_H + offset

        results = [{'ramp':i+1, 'hold_s':c['hold_s'], 'T_onset_C':T_onsets[i],
                    'delta_H_kJmol': dH[i]} for i, c in enumerate(conditions)]
        all_dfs.append(pd.DataFrame(results))

        row_base = fi * 2
        # Panel: DSC + T_onset
        ax = axes[fi, 0]
        for i in [0, 2, 4, n_half, n_half+2, n_half+4]:
            s,e = conditions[i]['start_idx'], conditions[i]['end_idx']
            ax.plot(T_exp[s:e+1], DSC_exp[s:e+1], alpha=0.7, linewidth=0.8)
            if not np.isnan(T_onsets[i]): ax.axvline(T_onsets[i], alpha=0.3, linestyle='--')
        ax.set_xlabel('T (°C)'); ax.set_ylabel('DSC (µW)')
        ax.set_title(f'{label}: DSC + T_onset'); ax.set_xlim(28, 170)

        # Panel: T_onset vs hold time
        ax = axes[fi, 1]
        r1 = range(n_half); r2 = range(n_half, len(conditions))
        ax.plot([t_hold[i] for i in r1], [T_onsets[i] for i in r1], 'o-', color='#2166AC', label='cooling 50s')
        ax.plot([t_hold[i] for i in r2], [T_onsets[i] for i in r2], 's-', color='#B2182B', label='cooling 500s')
        ax.set_xlabel('Hold time (s)'); ax.set_ylabel('T_onset (°C)')
        ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title(f'{label}: T_onset vs annealing')

        # Panel: ΔH (30–T_onset) vs hold time
        ax = axes[fi, 2]
        ax.plot([t_hold[i] for i in r1], [dH[i] for i in r1], 'o-', color='#2166AC',
                markersize=8, markerfacecolor='white', markeredgewidth=1.5, label='cooling 50s')
        ax.plot([t_hold[i] for i in r2], [dH[i] for i in r2], 's-', color='#B2182B',
                markersize=8, markerfacecolor='white', markeredgewidth=1.5, label='cooling 500s')
        ax.set_xlabel('Hold time (s)'); ax.set_ylabel('ΔH (kJ/mol)')
        ax.set_title(f'{label}: ΔH (30–T_onset)'); ax.set_xscale('log')
        ax.invert_yaxis(); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        print(f"\n  {label} standard range results (30–T_onset):")
        for _, r in pd.DataFrame(results).iterrows():
            print(f"    R{r['ramp']:.0f}: hold={r['hold_s']:.0f}s, "
                  f"T_onset={r['T_onset_C']:.1f}°C, ΔH={r['delta_H_kJmol']:.2f} kJ/mol")

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, 'onestep_v2_comparison.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {out_png}")

    for fi, df in enumerate(all_dfs):
        tag = ['50C', '70C'][fi]
        out_csv = os.path.join(RESULTS_DIR, f'onestep_{tag}_v2_results.csv')
        df.to_csv(out_csv, index=False, float_format='%.4f')
        print(f"  Saved {out_csv}")

    return all_dfs


if __name__ == '__main__':
    results = main()
