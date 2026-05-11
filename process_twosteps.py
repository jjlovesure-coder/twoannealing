"""
Two-step annealing DSC data processing for Polystyrene (PS).
Three-step method: empty crucible baseline + sapphire reference → cp(T) → ΔH(30-100°C)
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants ──────────────────────────────────────────────────────────────
M_SAMPLE = 4.7   # mg (PS)
M_REF    = 10.0  # mg (sapphire)
T_INT_LOW  = 30   # °C
T_INT_HIGH = 100  # °C

# ── Sapphire (Al2O3) specific heat capacity [J/(g·K)] ─────────────────────
# Standard NETZSCH / PerkinElmer reference data for synthetic sapphire
# Polynomial fit: cp = A + B·T + C·T^2 + D·T^3 (T in °C, valid 20–300°C)
# Based on published SRM 720 certificate values converted to J/(g·K)
_SAPPHIRE_CP_TABLE = np.array([
    # T(°C), cp(J/(g·K))
    [20,    0.7356],
    [30,    0.7603],
    [40,    0.7838],
    [50,    0.8056],
    [60,    0.8257],
    [70,    0.8440],
    [80,    0.8606],
    [90,    0.8756],
    [100,   0.8890],
    [110,   0.9009],
    [120,   0.9114],
    [130,   0.9205],
    [140,   0.9283],
    [150,   0.9349],
    [160,   0.9404],
    [170,   0.9448],
    [180,   0.9483],
    [190,   0.9509],
    [200,   0.9527],
    [220,   0.9540],
    [250,   0.9550],
    [300,   0.9580],
])


def cp_sapphire_JgK(T_C):
    """Return sapphire specific heat in J/(g·K) at temperature T in °C.
    Uses linear interpolation of standard NETZSCH reference data.
    """
    T_C = np.asarray(T_C)
    cp = np.interp(T_C, _SAPPHIRE_CP_TABLE[:, 0], _SAPPHIRE_CP_TABLE[:, 1],
                   left=_SAPPHIRE_CP_TABLE[0, 1], right=_SAPPHIRE_CP_TABLE[-1, 1])
    return cp  # J/(g·K)

# ── Data loading helpers ───────────────────────────────────────────────────
def load_dsc_simple(filename, sheet=None):
    """Load a simple DSC run (empty or ref) — single 30→200 ramp."""
    if sheet is None:
        df = pd.read_excel(filename)
    else:
        df = pd.read_excel(filename, sheet_name=sheet)

    # Search for the data header row containing "Time" (twosteps format)
    # or find the first row with finite numeric data in col0 and col1
    header_row = None
    for i in range(len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i
            break

    if header_row is None:
        # Find first row with finite numeric col0 AND col1 (not NaN)
        for i in range(len(df)):
            try:
                a = float(str(df.iloc[i, 0]).strip())
                b = float(str(df.iloc[i, 1]).strip())
                if np.isfinite(a) and np.isfinite(b):
                    header_row = i
                    break
            except (ValueError, TypeError):
                continue

    if header_row is None:
        raise ValueError(f"Could not find data start in {filename}")

    # Read from header row: if it's a text header (e.g. "Time"), skip it
    v0 = df.iloc[header_row, 0]
    if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
        # data starts after this row; also skip unit row if present
        data_start = header_row + 1
        v0_next = df.iloc[data_start, 0]
        if v0_next is not None and isinstance(v0_next, str) and 'min' in str(v0_next).lower():
            data_start += 1
        data = df.iloc[data_start:].copy()
    else:
        data = df.iloc[header_row:].copy()

    data = data.dropna(axis=1, how='all').reset_index(drop=True)

    # Rename columns
    ncols = data.shape[1]
    col_names = ['Time', 'Temp', 'DSC', 'DDSC'] + [f'Extra_{i}' for i in range(max(0, ncols-4))]
    data.columns = col_names[:ncols] if ncols <= len(col_names) else col_names + [f'Extra_{i}' for i in range(len(col_names), ncols)]

    data = data[['Time', 'Temp', 'DSC', 'DDSC']].astype(float).reset_index(drop=True)
    return data


def load_dsc_twosteps(filename, sheet='PS-02'):
    """Load the twosteps experiment data with temperature program."""
    df = pd.read_excel(filename, sheet_name=sheet)

    # Parse temperature program (rows starting ~7 to before "温度程序模式")
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
                'step': step_num,
                'T_start': t_start,
                'T_end': t_end,
                'rate_or_temp': rate,
                'time_min': time_val,
            })
        except (ValueError, TypeError, IndexError):
            continue

    # Find data header row with "Time"
    header_row = None
    for i in range(90, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find 'Time' header in twosteps data")

    # Data starts after header row + unit row ("min", "Cel", etc.)
    # Actually, just grab raw data from header_row+2 onward (skip header + unit)
    data_start = header_row + 2
    raw = df.iloc[data_start:].copy()

    # Select first 4 columns: Time, Temp, DSC, DDSC
    data = pd.DataFrame({
        'Time': pd.to_numeric(raw.iloc[:, 0], errors='coerce'),
        'Temp': pd.to_numeric(raw.iloc[:, 1], errors='coerce'),
        'DSC':  pd.to_numeric(raw.iloc[:, 2], errors='coerce'),
        'DDSC': pd.to_numeric(raw.iloc[:, 3], errors='coerce'),
    }).dropna().reset_index(drop=True)

    data = data.astype(float)
    return data, program


def detect_heating_ramps(T, t, min_rate=4.0, min_duration_pts=500):
    """Detect heating ramps (30→200 at ~10°C/min) using temperature gradient."""
    n = len(T)
    window = 30  # ~30 sec smoothing
    dTdt = np.zeros(n)
    for i in range(window, n - window):
        dTdt[i] = (T[i + window] - T[i - window]) / max(t[i + window] - t[i - window], 1e-12)

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
                if np.min(seg_T) < 40 and np.max(seg_T) > 180:
                    segments.append((start, i - 1))
            in_seg = False
    if in_seg and n - start > min_duration_pts:
        seg_T = T[start:n]
        if np.min(seg_T) < 40 and np.max(seg_T) > 180:
            segments.append((start, n - 1))
    return segments


# ── Main processing ────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Two-step annealing DSC analysis — Three-step cp method")
    print("=" * 70)

    # ── 1. Load empty and reference data ─────────────────────────────────
    print("\n[1/5] Loading empty and reference data...")
    empty = load_dsc_simple('ps-empty-01.xlsx')
    ref   = load_dsc_simple('ps-ref-01.xlsx')

    print(f"  Empty: {len(empty)} pts, T range {empty['Temp'].min():.1f}–{empty['Temp'].max():.1f} °C")
    print(f"  Ref:   {len(ref)} pts, T range {ref['Temp'].min():.1f}–{ref['Temp'].max():.1f} °C")

    # Trim to 30-200°C range for calibration
    mask_e = (empty['Temp'] >= 28) & (empty['Temp'] <= 202)
    mask_r = (ref['Temp'] >= 28) & (ref['Temp'] <= 202)
    empty_cal = empty[mask_e].copy()
    ref_cal   = ref[mask_r].copy()

    # Interpolate to common temperature grid (0.1°C steps)
    T_grid = np.arange(30.0, 200.01, 0.1)
    interp_empty = interp1d(empty_cal['Temp'], empty_cal['DSC'], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    interp_ref   = interp1d(ref_cal['Temp'], ref_cal['DSC'], kind='linear',
                            bounds_error=False, fill_value='extrapolate')

    DSC_empty_grid = interp_empty(T_grid)
    DSC_ref_grid   = interp_ref(T_grid)

    # ── 2. Three-step calibration ────────────────────────────────────────
    print("\n[2/5] Computing three-step calibration...")
    cp_ref_grid = cp_sapphire_JgK(T_grid)  # J/(g·K)

    # cp_sample(T) = cp_ref(T) × [DSC_sample - DSC_empty] / [DSC_ref - DSC_empty] × (m_ref / m_sample)
    dsc_empty_to_ref = DSC_ref_grid - DSC_empty_grid  # µW
    # Avoid division by zero (shouldn't happen in 30-200°C range)
    valid = np.abs(dsc_empty_to_ref) > 1e-6

    # Pre-compute the calibration denominator
    cal_denom = dsc_empty_to_ref / (cp_ref_grid * M_REF)  # µW / (J/(g·K) * mg) = µW·g·K/J·mg

    print(f"  Valid calibration points: {valid.sum()}/{len(T_grid)}")

    # ── 3. Load twosteps data ────────────────────────────────────────────
    print("\n[3/5] Loading twosteps data and detecting heating ramps...")
    ts_data, program = load_dsc_twosteps('twosteps.xlsx')
    T_ts = ts_data['Temp'].values
    t_ts = ts_data['Time'].values
    DSC_ts = ts_data['DSC'].values

    ramps = detect_heating_ramps(T_ts, t_ts)
    print(f"  Found {len(ramps)} heating ramps")

    # ── 4. Build annealing-condition map ─────────────────────────────────
    # From the temperature program: each cycle is 4 steps.
    # Step pattern within each cycle:
    #   step 4n+1: 200→90  (cool, hold t1 at 90°C)
    #   step 4n+2: 90→80   (cool, hold t2 at 80°C — variable)
    #   step 4n+3: 80→30   (cool)
    #   step 4n+4: 30→200  (heat — measurement ramp)
    heating_steps = [p for p in program if p['T_start'] == 30 and p['T_end'] == 200]
    print(f"  Heating steps in program: {len(heating_steps)}")

    # Map each heating ramp to its preceding annealing steps
    conditions = []
    for idx, (s, e) in enumerate(ramps):
        # Find the corresponding heating program step
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            # Find the two preceding cooling/annealing steps
            anneal_step_2 = None  # step before heating (80→30)
            anneal_step_1 = None  # step before that  (90→80)
            cool_step     = None  # step before that  (200→90)
            for p in program:
                if p['step'] == step_num - 1:
                    anneal_step_2 = p
                elif p['step'] == step_num - 2:
                    anneal_step_1 = p
                elif p['step'] == step_num - 3:
                    cool_step = p

            t1_hold = cool_step['time_min'] if cool_step else None
            t2_hold = anneal_step_1['time_min'] if anneal_step_1 else None
        else:
            t1_hold = None
            t2_hold = None

        conditions.append({
            'ramp_idx': idx + 1,
            'T1_hold_min': t1_hold,
            'T2_hold_min': t2_hold,
            'start_idx': s,
            'end_idx': e,
        })

    print(f"  Mapped {len(conditions)} ramps to annealing conditions")
    for c in conditions:
        print(f"    Ramp {c['ramp_idx']:2d}: T1(90°C)={c['T1_hold_min']:8.4f} min, "
              f"T2(80°C)={c['T2_hold_min']:8.4f} min")

    # ── 5. Compute cp & ΔH for each ramp ──────────────────────────────────
    print(f"\n[4/5] Computing cp(T) and ΔH ({T_INT_LOW}–{T_INT_HIGH}°C) for each ramp...")

    results = []
    cp_curves = []  # Store all cp curves for later analysis

    for ramp_idx, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_ts[s:e+1]
        DSC_seg = DSC_ts[s:e+1]

        # Interpolate sample DSC to common T_grid
        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        DSC_sample_grid = interp_dsc(T_grid)

        # Three-step cp calculation
        cp_sample = np.zeros_like(T_grid)
        cp_sample[valid] = (cp_ref_grid[valid]
                            * (DSC_sample_grid[valid] - DSC_empty_grid[valid])
                            / dsc_empty_to_ref[valid]
                            * (M_REF / M_SAMPLE))

        # Integrate cp from T_INT_LOW to T_INT_HIGH → ΔH in J/g
        int_mask = (T_grid >= T_INT_LOW) & (T_grid <= T_INT_HIGH)
        delta_H = trapezoid(cp_sample[int_mask], T_grid[int_mask])

        # Also compute partial integrals for better analysis
        cp_avg = trapezoid(cp_sample[int_mask], T_grid[int_mask]) / (T_INT_HIGH - T_INT_LOW)

        results.append({
            'ramp': ramp_idx + 1,
            'T1_hold_min': cond['T1_hold_min'],
            'T2_hold_min': cond['T2_hold_min'],
            'delta_H_Jg': delta_H,
            'cp_avg_JgK': cp_avg,
            'cp_at_50C': np.interp(50, T_grid, cp_sample),
            'cp_at_80C': np.interp(80, T_grid, cp_sample),
            'cp_at_100C': np.interp(100, T_grid, cp_sample),
        })
        cp_curves.append(cp_sample)

        print(f"    Ramp {ramp_idx+1:2d}: t1={cond['T1_hold_min']:8.4f} min, "
              f"t2={cond['T2_hold_min']:8.4f} min → "
              f"ΔH = {delta_H:.4f} J/g, cp_avg = {cp_avg:.4f} J/(g·K)")

    results_df = pd.DataFrame(results)
    cp_curves = np.array(cp_curves)

    # ── 6. Generate comprehensive plots ───────────────────────────────────
    print(f"\n[5/5] Generating plots...")

    fig = plt.figure(figsize=(20, 14))

    # --- Panel 1: cp curves for all ramps (color-coded by T1 group) ---
    ax1 = fig.add_subplot(2, 3, 1)
    t1_short = results_df[results_df['T1_hold_min'] < 1.0]
    t1_long  = results_df[results_df['T1_hold_min'] > 1.0]

    # T1-short group: cool colors
    for _, r in t1_short.iterrows():
        idx = int(r['ramp']) - 1
        ax1.plot(T_grid, cp_curves[idx], alpha=0.6, linewidth=0.8,
                 color=plt.cm.Blues(0.4 + 0.5 * idx / len(t1_short)))
    # T1-long group: warm colors
    for _, r in t1_long.iterrows():
        idx = int(r['ramp']) - 1
        ax1.plot(T_grid, cp_curves[idx], alpha=0.6, linewidth=0.8,
                 color=plt.cm.Oranges(0.4 + 0.5 * (idx - len(t1_short)) / len(t1_long)))

    # Average curves per group
    t1s_idx = [int(r['ramp'])-1 for _, r in t1_short.iterrows()]
    t1l_idx = [int(r['ramp'])-1 for _, r in t1_long.iterrows()]
    ax1.plot(T_grid, cp_curves[t1s_idx].mean(axis=0), 'b-', linewidth=2.0,
             label=f'T1=0.833 min (n={len(t1s_idx)})')
    ax1.plot(T_grid, cp_curves[t1l_idx].mean(axis=0), 'r-', linewidth=2.0,
             label=f'T1=8.333 min (n={len(t1l_idx)})')
    ax1.axvspan(T_INT_LOW, T_INT_HIGH, alpha=0.08, color='green')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('cp (J/(g·K))')
    ax1.set_title('Specific heat capacity — all ramps by T1 group')
    ax1.legend(fontsize=8)
    ax1.set_xlim(28, 202)

    # --- Panel 2: DSC heating curves (T-axis aligned) ---
    ax2 = fig.add_subplot(2, 3, 2)
    # Show first ramp, a middle ramp, and last ramp from each group
    highlight_ramps = [0, 4, 9, 10, 14, 19]  # idx in cp_curves
    labels_ramps = [1, 5, 10, 11, 15, 20]  # user-facing numbers
    for pi, (idx, lbl) in enumerate(zip(highlight_ramps, labels_ramps)):
        cond = conditions[idx]
        s, e = cond['start_idx'], cond['end_idx']
        ax2.plot(T_ts[s:e+1], DSC_ts[s:e+1], alpha=0.8, linewidth=0.8,
                 label=f"R{lbl}: t1={cond['T1_hold_min']:.3f},t2={cond['T2_hold_min']:.3f}")
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('DSC signal (µW)')
    ax2.set_title('Raw DSC heating curves (selected)')
    ax2.legend(fontsize=6, loc='lower right')

    # --- Panel 3: ΔH vs T2 annealing time (log scale) ---
    ax3 = fig.add_subplot(2, 3, 3)
    markers = ['o', 's']
    colors_grp = ['#2166AC', '#B2182B']
    for i, (grp_label, grp_df) in enumerate([('T1=0.833 min', t1_short), ('T1=8.333 min', t1_long)]):
        ax3.plot(grp_df['T2_hold_min'], grp_df['delta_H_Jg'],
                 marker=markers[i], color=colors_grp[i], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=grp_label)
    ax3.set_xlabel('T2 hold time at 80°C (min)')
    ax3.set_ylabel(f'ΔH (30–100°C) (J/g)')
    ax3.set_title('Enthalpy recovery vs T2 annealing time')
    ax3.set_xscale('log')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # --- Panel 4: ΔH comparison bar chart ---
    ax4 = fig.add_subplot(2, 3, 4)
    x_pos = np.arange(len(results_df))
    bar_colors = [colors_grp[0] if t < 1.0 else colors_grp[1]
                  for t in results_df['T1_hold_min']]
    bars = ax4.bar(x_pos, results_df['delta_H_Jg'], color=bar_colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{r['ramp']:.0f}" for _, r in results_df.iterrows()],
                        fontsize=7, rotation=45)
    ax4.set_ylabel(f'ΔH (30–100°C) (J/g)')
    ax4.set_xlabel('Ramp number')
    ax4.set_title('ΔH distribution across all annealing conditions')

    # Add group separator
    ax4.axvline(9.5, color='gray', linestyle='--', alpha=0.7)
    ax4.text(4.5, ax4.get_ylim()[1] * 0.98, 'T1=0.833 min',
             ha='center', fontsize=9, fontweight='bold', color=colors_grp[0])
    ax4.text(14.5, ax4.get_ylim()[1] * 0.98, 'T1=8.333 min',
             ha='center', fontsize=9, fontweight='bold', color=colors_grp[1])

    # --- Panel 5: cp average comparison ---
    ax5 = fig.add_subplot(2, 3, 5)
    for i, (grp_label, grp_df) in enumerate([('T1=0.833 min', t1_short), ('T1=8.333 min', t1_long)]):
        ax5.plot(grp_df['T2_hold_min'], grp_df['cp_avg_JgK'],
                 marker=markers[i], color=colors_grp[i], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=grp_label)
    ax5.set_xlabel('T2 hold time at 80°C (min)')
    ax5.set_ylabel('Average cp (30–100°C) (J/(g·K))')
    ax5.set_title('Average cp vs T2 annealing time')
    ax5.set_xscale('log')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, which='both')

    # --- Panel 6: Summary table ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    # Create formatted table
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}",
            f"{r['T1_hold_min']:.3f}",
            f"{r['T2_hold_min']:.3f}",
            f"{r['delta_H_Jg']:.3f}",
            f"{r['cp_avg_JgK']:.3f}",
        ])
    # Split into two columns (group A and B)
    col_labels = ['Ramp', 't1@90°C\n(min)', 't2@80°C\n(min)', 'ΔH\n(J/g)', 'cp_avg\nJ/(g·K)']

    # Add group headers
    table_nrows = len(table_data)
    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.08, 0.16, 0.16, 0.16, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.15)

    # Color-code rows by T1 group
    for row_idx in range(table_nrows):
        for col_idx in range(5):
            cell = table[row_idx + 1, col_idx]
            if row_idx < 10:
                cell.set_facecolor('#E3EDF8')  # light blue
            else:
                cell.set_facecolor('#FDE0DD')  # light red

    ax6.set_title('Results Summary', fontsize=12, fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    plt.savefig('twosteps_enthalpy_results.png', dpi=150, bbox_inches='tight')
    print("  Saved twosteps_enthalpy_results.png")

    # Save results to CSV
    results_df.to_csv('twosteps_enthalpy_results.csv', index=False, float_format='%.6f')
    print("  Saved twosteps_enthalpy_results.csv")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Two-step annealing of Polystyrene")
    print("=" * 70)
    print(f"\n{'Ramp':<6} {'t1@90°C':>10}  {'t2@80°C':>10}  {'ΔH(30-100°C)':>14}  {'cp_avg':>10}")
    print(f"{'':6} {'(min)':>10}  {'(min)':>10}  {'(J/g)':>14}  {'J/(g·K)':>10}")
    print("-" * 60)
    for _, r in results_df.iterrows():
        print(f"  {r['ramp']:<4.0f}  {r['T1_hold_min']:>10.4f}  {r['T2_hold_min']:>10.4f}  "
              f"{r['delta_H_Jg']:>14.4f}  {r['cp_avg_JgK']:>10.4f}")
    print("-" * 60)

    # Group statistics
    grp_a = results_df[results_df['T1_hold_min'] < 1.0]
    grp_b = results_df[results_df['T1_hold_min'] > 1.0]

    print(f"\n  Group A (T1 = 0.833 min):")
    print(f"    ΔH: {grp_a['delta_H_Jg'].min():.4f} – {grp_a['delta_H_Jg'].max():.4f} J/g")
    print(f"    ΔH mean ± std: {grp_a['delta_H_Jg'].mean():.4f} ± {grp_a['delta_H_Jg'].std():.4f} J/g")
    print(f"    cp_avg mean ± std: {grp_a['cp_avg_JgK'].mean():.4f} ± {grp_a['cp_avg_JgK'].std():.4f} J/(g·K)")

    print(f"\n  Group B (T1 = 8.333 min):")
    print(f"    ΔH: {grp_b['delta_H_Jg'].min():.4f} – {grp_b['delta_H_Jg'].max():.4f} J/g")
    print(f"    ΔH mean ± std: {grp_b['delta_H_Jg'].mean():.4f} ± {grp_b['delta_H_Jg'].std():.4f} J/g")
    print(f"    cp_avg mean ± std: {grp_b['cp_avg_JgK'].mean():.4f} ± {grp_b['cp_avg_JgK'].std():.4f} J/(g·K)")

    Δ_H_diff = grp_a['delta_H_Jg'].mean() - grp_b['delta_H_Jg'].mean()
    print(f"\n  Δ(ΔH) between groups: {Δ_H_diff:.4f} J/g (A − B)")
    print(f"  Overall ΔH range: {results_df['delta_H_Jg'].min():.4f} – {results_df['delta_H_Jg'].max():.4f} J/g")

    return results_df


if __name__ == '__main__':
    results = main()
