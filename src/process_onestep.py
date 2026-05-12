"""
One-step annealing DSC data processing for Polystyrene (PS).
Direct heat-flow integration: ΔH = (1/(β·m)) × ∫(DSC_sample − DSC_empty) dT
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
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# ── Constants ──────────────────────────────────────────────────────────────
M_SAMPLE = 4.7
HEATING_RATE = 10.0
BETA = HEATING_RATE / 60.0
DT_DT = 1.0 / BETA
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)
MW = 280000
CONV_KJMOL = CONV_FACTOR * MW / 1000

T_INT_LOW  = 35
T_INT_HIGH = 95

# ── Data loading ──────────────────────────────────────────────────────────
def load_dsc_simple(filename):
    df = pd.read_excel(filename)
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
                    header_row = i; break
            except: continue

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
    col_names = ['Time', 'Temp', 'DSC', 'DDSC'] + [f'Extra_{i}' for i in range(max(0, ncols-4))]
    data.columns = col_names[:ncols]
    data = data[['Time', 'Temp', 'DSC', 'DDSC']].astype(float).reset_index(drop=True)
    return data


def load_dsc_experiment(filename, sheet=None):
    if sheet:
        df = pd.read_excel(filename, sheet_name=sheet)
    else:
        df = pd.read_excel(filename)

    program = []
    for i in range(7, len(df)):
        v1 = df.iloc[i, 1]
        if v1 is not None and isinstance(v1, str) and ('温度' in str(v1) or '冷却' in str(v1)):
            break
        try:
            step_str = str(df.iloc[i, 1]).strip()
            step_num = int(float(step_str))
            t_start  = float(str(df.iloc[i, 2]).strip())
            t_end    = float(str(df.iloc[i, 3]).strip())
            rate     = float(str(df.iloc[i, 4]).strip())
            time_val = float(str(df.iloc[i, 5]).strip())
            program.append({
                'step': step_num, 'T_start': t_start, 'T_end': t_end,
                'rate': rate, 'time_min': time_val,
            })
        except (ValueError, TypeError, IndexError):
            continue

    header_row = None
    for i in range(70, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i; break

    data_start = header_row + 2
    raw = df.iloc[data_start:].copy()
    data = pd.DataFrame({
        'Time': pd.to_numeric(raw.iloc[:, 0], errors='coerce'),
        'Temp': pd.to_numeric(raw.iloc[:, 1], errors='coerce'),
        'DSC':  pd.to_numeric(raw.iloc[:, 2], errors='coerce'),
        'DDSC': pd.to_numeric(raw.iloc[:, 3], errors='coerce'),
    }).dropna().reset_index(drop=True).astype(float)
    return data, program


def detect_heating_ramps(T, t, min_rate=4.0, min_duration_pts=500):
    n = len(T)
    window = 30
    dTdt = np.zeros(n)
    for i in range(window, n - window):
        dTdt[i] = (T[i + window] - T[i - window]) / max(t[i + window] - t[i - window], 1e-12)
    is_heating = dTdt > min_rate
    segments = []
    in_seg = False; start = 0
    for i in range(n):
        if is_heating[i] and not in_seg: in_seg = True; start = i
        elif not is_heating[i] and in_seg:
            if i - start > min_duration_pts:
                seg_T = T[start:i]
                if np.min(seg_T) < 40 and np.max(seg_T) > 180:
                    segments.append((start, i - 1))
            in_seg = False
    if in_seg and n - start > min_duration_pts:
        seg_T = T[start:n]
        if np.min(seg_T) < 40 and np.max(seg_T) > 180:
            segments.append((start, n - 1))
    return segments


# ── Processing function ────────────────────────────────────────────────────
def process_onestep(data_file, sheet, T_anneal, label, out_name):
    print(f"\n{'='*70}")
    print(f"  One-step annealing ({label}) — T_anneal = {T_anneal}°C")
    print(f"  Direct heat-flow integration ({T_INT_LOW}–{T_INT_HIGH}°C)")
    print(f"{'='*70}")

    # Load empty
    empty = load_dsc_simple(os.path.join(DATA_DIR, 'ps-empty-01.xlsx'))
    T_grid = np.arange(np.ceil(30.0), np.floor(168.0) + 0.01, 0.1)
    interp_empty = interp1d(empty['Temp'], empty['DSC'], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    DSC_empty_grid = interp_empty(T_grid)

    # Load experiment
    exp_data, program = load_dsc_experiment(data_file, sheet=sheet)
    T_exp = exp_data['Temp'].values
    t_exp = exp_data['Time'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    print(f"  Found {len(ramps)} heating ramps")

    # Map annealing conditions: step 3n = heating, preceded by 3n-2=200→T_anneal
    heating_steps = [p for p in program if p['T_start'] == 30 and p['T_end'] == 200]
    print(f"  Heating steps in program: {len(heating_steps)}")

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            anneal_step = None
            for p in program:
                if p['step'] == step_num - 2:  # 200→T_anneal
                    anneal_step = p; break
            t_hold = anneal_step['time_min'] if anneal_step else None
        else:
            t_hold = None
        conditions.append({
            'ramp_idx': idx + 1, 'hold_min': t_hold,
            'start_idx': s, 'end_idx': e,
        })

    print(f"  Annealing conditions (hold @ {T_anneal}°C):")
    for c in conditions:
        print(f"    Ramp {c['ramp_idx']:2d}: hold = {c['hold_min']:8.4f} min")

    # Compute raw integrals
    raw_integrals = []
    for ramp_idx, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]; DSC_seg = DSC_exp[s:e+1]
        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        delta_DSC = interp_dsc(T_grid) - DSC_empty_grid
        int_mask = (T_grid >= T_INT_LOW) & (T_grid <= T_INT_HIGH)
        raw_integrals.append(trapezoid(delta_DSC[int_mask], T_grid[int_mask]))

    # Global reference + offset for positive values
    ref_global = raw_integrals[0]
    all_diffs = [(r - ref_global) * CONV_KJMOL for r in raw_integrals]
    offset = max(0, -min(all_diffs)) + 1.0
    results = []
    for ramp_idx, cond in enumerate(conditions):
        integral = raw_integrals[ramp_idx]
        delta_H_raw = (integral - ref_global) * CONV_KJMOL
        delta_H = delta_H_raw + offset
        results.append({
            'ramp': ramp_idx + 1, 'hold_min': cond['hold_min'],
            'delta_H_kJmol': delta_H,
        })
        print(f"    Ramp {ramp_idx+1:2d}: hold = {cond['hold_min']:8.4f} min → ΔH = {delta_H:.2f} kJ/mol")

    results_df = pd.DataFrame(results)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: ΔH vs hold time
    ax1 = axes[0]
    # Split into first 10 and second 10 (replicates)
    r1 = results_df.iloc[:10]
    r2 = results_df.iloc[10:]
    ax1.plot(r1['hold_min'], r1['delta_H_kJmol'], 'o-', color='#2166AC', linewidth=1.8,
             markersize=9, markerfacecolor='white', markeredgewidth=1.5, label='Run 1')
    ax1.plot(r2['hold_min'], r2['delta_H_kJmol'], 's--', color='#B2182B', linewidth=1.8,
             markersize=9, markerfacecolor='white', markeredgewidth=1.5, label='Run 2')
    ax1.set_xlabel(f'Hold time at {T_anneal}°C (min)')
    ax1.set_ylabel(f'ΔH (kJ/mol)')
    ax1.set_title(f'{label}: Released enthalpy vs annealing time')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: ΔH bar chart
    ax2 = axes[1]
    x_pos = np.arange(len(results_df))
    bar_colors = ['#2166AC'] * 10 + ['#B2182B'] * 10
    ax2.bar(x_pos, results_df['delta_H_kJmol'], color=bar_colors, edgecolor='black',
            linewidth=0.5, alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{r['hold_min']:.3f}" for _, r in results_df.iterrows()],
                        fontsize=6, rotation=45)
    ax2.set_ylabel(f'ΔH (kJ/mol)')
    ax2.set_xlabel(f'Hold time at {T_anneal}°C (min)')
    ax2.set_title(f'{label}: ΔH distribution')
    ax2.axvline(9.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f'{out_name}_results.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {png_path}")

    csv_path = os.path.join(RESULTS_DIR, f'{out_name}_results.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved {csv_path}")

    # Summary
    print(f"\n  Results ({label}):")
    print(f"  {'Ramp':<6} {'hold@' + str(T_anneal) + '°C':>12}  {'ΔH (kJ/mol)':>12}")
    print(f"  {'-'*35}")
    for _, r in results_df.iterrows():
        print(f"  {r['ramp']:<4.0f}  {r['hold_min']:>12.4f}  {r['delta_H_kJmol']:>12.2f}")
    run1 = results_df.iloc[:10]
    run2 = results_df.iloc[10:]
    print(f"\n  Run 1: ΔH = {run1['delta_H_kJmol'].min():.2f} – {run1['delta_H_kJmol'].max():.2f} kJ/mol")
    print(f"  Run 2: ΔH = {run2['delta_H_kJmol'].min():.2f} – {run2['delta_H_kJmol'].max():.2f} kJ/mol")

    return results_df


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    r1 = process_onestep(
        os.path.join(DATA_DIR, 'PS-onestep-01.xlsx'),
        sheet='PS-onestep-01',
        T_anneal=50,
        label='One-step @50°C',
        out_name='onestep_50C',
    )
    r2 = process_onestep(
        os.path.join(DATA_DIR, 'PS-onestep-02.xlsx'),
        sheet='PS-onestep-02',
        T_anneal=70,
        label='One-step @70°C',
        out_name='onestep_70C',
    )
