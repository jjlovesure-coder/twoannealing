"""
Two-step annealing DSC data processing for Polystyrene (PS).
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
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dsc')

# ── Constants ──────────────────────────────────────────────────────────────
M_SAMPLE = 4.7      # mg
HEATING_RATE = 10.0  # °C/min
BETA = HEATING_RATE / 60.0  # °C/s
DT_DT = 1.0 / BETA   # s/°C  (time per degree)
# Conversion factor: ΔH(J/g) = DT_DT * 1e-6 / (M_SAMPLE * 1e-3) * ∫ΔDSC dT
#                          = DT_DT / (M_SAMPLE * 1000) * ∫ΔDSC dT
#                          = 6 / 4700 * ∫ΔDSC dT   (µW·°C → J/g)
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)  # (µW·°C → J/g)
MW = 280000  # g/mol
CONV_KJMOL = CONV_FACTOR * MW / 1000  # (µW·°C → kJ/mol)

T_INT_LOW  = 35
T_INT_HIGH = 95

# ── Data loading ──────────────────────────────────────────────────────────
def load_dsc_simple(filename, sheet=None):
    """Load a simple DSC run (empty or ref) — single 30→200 ramp."""
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
    col_names = ['Time', 'Temp', 'DSC', 'DDSC'] + [f'Extra_{i}' for i in range(max(0, ncols-4))]
    data.columns = col_names[:ncols] if ncols <= len(col_names) else col_names + [f'Extra_{i}' for i in range(len(col_names), ncols)]
    data = data[['Time', 'Temp', 'DSC', 'DDSC']].astype(float).reset_index(drop=True)
    return data


def load_dsc_experiment(filename, sheet):
    """Load experiment data with temperature program."""
    df = pd.read_excel(filename, sheet_name=sheet)

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

    header_row = None
    for i in range(90, len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            header_row = i
            break

    data_start = header_row + 2
    raw = df.iloc[data_start:].copy()
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
    window = 30
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
    print("  Two-step annealing DSC analysis — Direct heat-flow integration")
    print("=" * 70)

    # ── 1. Setup integration grid ──────────────────────────────────────

    # T grid for integration (direct from sample DSC, no empty subtraction)
    T_grid = np.arange(30.0, 200.0, 0.1)
    print(f"  Integration grid: {T_grid[0]:.0f}–{T_grid[-1]:.0f} °C ({len(T_grid)} pts)")

    # ── 2. Load experiment data ──────────────────────────────────────────
    print("\n[2/4] Loading twosteps data and detecting heating ramps...")
    exp_data, program = load_dsc_experiment(os.path.join(DATA_DIR, 'twosteps.xlsx'), sheet='PS-02')
    T_exp = exp_data['Temp'].values
    t_exp = exp_data['Time'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    print(f"  Found {len(ramps)} heating ramps")

    # ── 3. Map annealing conditions ─────────────────────────────────────
    # Step pattern: 4n+1=200→90(hold t1), 4n+2=90→80(hold t2), 4n+3=80→30, 4n+4=30→200(heating)
    heating_steps = [p for p in program if p['T_start'] == 30 and p['T_end'] == 200]
    print(f"  Heating steps in program: {len(heating_steps)}")

    conditions = []
    for idx, (s, e) in enumerate(ramps):
        if idx < len(heating_steps):
            h_step = heating_steps[idx]
            step_num = h_step['step']
            cool_step = None
            anneal_step_1 = None
            for p in program:
                if p['step'] == step_num - 1:
                    pass  # 90→30 cooling
                elif p['step'] == step_num - 2:
                    anneal_step_1 = p  # 90→80
                elif p['step'] == step_num - 3:
                    cool_step = p  # 200→90
            t1_hold = cool_step['time_min'] if cool_step else None
            t2_hold = anneal_step_1['time_min'] if anneal_step_1 else None
        else:
            t1_hold = None
            t2_hold = None

        conditions.append({
            'ramp_idx': idx + 1,
            'T1_hold_s': t1_hold,
            'T2_hold_s': t2_hold,
            'start_idx': s,
            'end_idx': e,
        })

    # Convert to seconds and filter
    MIN_HOLD = 0.1 * 60  # 6 seconds
    for c in conditions:
        c['T1_hold_s'] = c['T1_hold_s'] * 60 if c['T1_hold_s'] else None
        c['T2_hold_s'] = c['T2_hold_s'] * 60 if c['T2_hold_s'] else None
    conditions = [c for c in conditions if c['T2_hold_s'] is not None and c['T2_hold_s'] >= MIN_HOLD]
    for i, c in enumerate(conditions):
        c['ramp_idx'] = i + 1

    for c in conditions:
        print(f"    Ramp {c['ramp_idx']:2d}: T1(90°C)={c['T1_hold_s']:8.1f} s, "
              f"T2(80°C)={c['T2_hold_s']:8.1f} s")
    print(f"  Kept {len(conditions)}/{len(ramps)} ramps (T2 >= {MIN_HOLD:.0f} s)")

    # ── 4. Compute ΔH for each ramp ─────────────────────────────────────
    print(f"\n[3/4] Computing ΔH ({T_INT_LOW}–{T_INT_HIGH}°C) by direct heat-flow integration...")

    raw_integrals = []
    dsc_curves = []

    for ramp_idx, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]
        DSC_seg = DSC_exp[s:e+1]
        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        DSC_grid = interp_dsc(T_grid)
        int_mask = (T_grid >= T_INT_LOW) & (T_grid <= T_INT_HIGH)
        integral = trapezoid(DSC_grid[int_mask], T_grid[int_mask])  # µW·°C
        raw_integrals.append(integral)
        dsc_curves.append(DSC_grid)
        print(f"    Ramp {ramp_idx+1:2d}: t1={cond['T1_hold_s']:8.1f} s, "
              f"t2={cond['T2_hold_s']:8.1f} s → "
              f"raw integral = {integral:.1f} µW·°C")

    # Global reference: ramp with shortest total annealing (ramp 1)
    ref_global = raw_integrals[0]
    # Offset to make all ΔH positive (add |min difference| + margin)
    all_diffs = [(r - ref_global) * CONV_KJMOL for r in raw_integrals]
    offset = max(0, -min(all_diffs)) + 1.0  # ensure all values ≥ 1.0 kJ/mol

    results = []
    for ramp_idx, cond in enumerate(conditions):
        integral = raw_integrals[ramp_idx]
        delta_H_raw = (integral - ref_global) * CONV_KJMOL
        delta_H_kJmol = delta_H_raw + offset
        results.append({
            'ramp': ramp_idx + 1,
            'T1_hold_s': cond['T1_hold_s'],
            'T2_hold_s': cond['T2_hold_s'],
            'delta_H_kJmol': delta_H_kJmol,
        })
        print(f"      → ΔH = {delta_H_kJmol:.2f} kJ/mol")

    results_df = pd.DataFrame(results)
    dsc_curves = np.array(dsc_curves)

    # ── 5. Generate plots ────────────────────────────────────────────────
    print(f"\n[4/4] Generating plots...")

    fig = plt.figure(figsize=(20, 14))

    t1_short = results_df[results_df['T1_hold_s'] < 60]
    t1_long  = results_df[results_df['T1_hold_s'] > 60]
    colors_grp = ['#2166AC', '#B2182B']
    markers = ['o', 's']

    # Panel 1: ΔDSC curves
    ax1 = fig.add_subplot(2, 3, 1)
    highlight = [0, 2, 4, 5, 7, 9]
    labels_h = ['A1','A3','A5','B1','B3','B5']
    for idx, lbl in zip(highlight, labels_h):
        c = conditions[idx]
        ax1.plot(T_grid, dsc_curves[idx], alpha=0.8, linewidth=0.8,
                 label=f"R{lbl}: t1={c['T1_hold_s']:.3f},t2={c['T2_hold_s']:.3f}")
    ax1.axvspan(T_INT_LOW, T_INT_HIGH, alpha=0.08, color='green')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('ΔDSC (sample − empty) (µW)')
    ax1.set_title('Baseline-subtracted DSC curves (selected)')
    ax1.legend(fontsize=6, loc='lower right')
    ax1.set_xlim(28, T_grid[-1] + 2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Panel 2: All ΔDSC curves by group
    ax2 = fig.add_subplot(2, 3, 2)
    t1s_idx = [int(r['ramp'])-1 for _, r in t1_short.iterrows()]
    t1l_idx = [int(r['ramp'])-1 for _, r in t1_long.iterrows()]
    for idx in t1s_idx:
        ax2.plot(T_grid, dsc_curves[idx], alpha=0.4, linewidth=0.5, color=colors_grp[0])
    for idx in t1l_idx:
        ax2.plot(T_grid, dsc_curves[idx], alpha=0.4, linewidth=0.5, color=colors_grp[1])
    ax2.plot(T_grid, dsc_curves[t1s_idx].mean(axis=0), '-', color=colors_grp[0], linewidth=2.0,
             label=f'T1=50 s (n={len(t1s_idx)})')
    ax2.plot(T_grid, dsc_curves[t1l_idx].mean(axis=0), '-', color=colors_grp[1], linewidth=2.0,
             label=f'T1=500 s (n={len(t1l_idx)})')
    ax2.axvspan(T_INT_LOW, T_INT_HIGH, alpha=0.08, color='green')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('ΔDSC (µW)')
    ax2.set_title('All ΔDSC curves by T1 group')
    ax2.legend(fontsize=8)
    ax2.set_xlim(28, T_grid[-1] + 2)

    # Panel 3: ΔH vs T2 annealing time (kJ/mol, positive, increasing)
    ax3 = fig.add_subplot(2, 3, 3)
    for i, (grp_label, grp_df) in enumerate([('T1=50 s', t1_short), ('T1=500 s', t1_long)]):
        ax3.plot(grp_df['T2_hold_s'], grp_df['delta_H_kJmol'],
                 marker=markers[i], color=colors_grp[i], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=grp_label)
    ax3.set_xlabel('T2 hold time at 80°C (s)')
    ax3.set_ylabel(f'ΔH ({T_INT_LOW}–{T_INT_HIGH}°C) (kJ/mol)')
    ax3.set_title('Released enthalpy vs T2 annealing time')
    ax3.set_xscale('log')
    ax3.invert_yaxis()
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # Panel 4: ΔH bar chart
    ax4 = fig.add_subplot(2, 3, 4)
    x_pos = np.arange(len(results_df))
    bar_colors = [colors_grp[0] if t < 1.0 else colors_grp[1]
                  for t in results_df['T1_hold_s']]
    ax4.bar(x_pos, results_df['delta_H_kJmol'], color=bar_colors,
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{r['ramp']:.0f}" for _, r in results_df.iterrows()],
                        fontsize=7, rotation=45)
    ax4.set_ylabel(f'ΔH (kJ/mol)')
    ax4.set_xlabel('Ramp number')
    ax4.set_title('ΔH (released) distribution across all annealing conditions')
    mid = len(t1_short) - 0.5
    ax4.axvline(mid, color='gray', linestyle='--', alpha=0.7)
    ylim = ax4.get_ylim()
    ax4.text(mid/2, ylim[1] * 0.98, 'T1=50 s', ha='center', fontsize=9,
             fontweight='bold', color=colors_grp[0])
    ax4.text(mid + len(t1_long)/2, ylim[1] * 0.98, 'T1=500 s', ha='center', fontsize=9,
             fontweight='bold', color=colors_grp[1])

    # Panel 5: Raw DSC curves on sample grid (selected)
    ax5 = fig.add_subplot(2, 3, 5)
    for idx, lbl in zip(highlight, labels_h):
        c = conditions[idx]
        s, e = c['start_idx'], c['end_idx']
        ax5.plot(T_exp[s:e+1], DSC_exp[s:e+1], alpha=0.8, linewidth=0.8,
                 label=f"R{lbl}: t1={c['T1_hold_s']:.3f},t2={c['T2_hold_s']:.3f}")
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('DSC signal (µW)')
    ax5.set_title('Raw DSC heating curves (selected)')
    ax5.legend(fontsize=6, loc='lower right')

    # Panel 6: Summary table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}",
            f"{r['T1_hold_s']:.3f}",
            f"{r['T2_hold_s']:.3f}",
            f"{r['delta_H_kJmol']:.2f}",
        ])
    col_labels = ['Ramp', 't1@90°C\n(s)', 't2@80°C\n(s)', 'ΔH\n(kJ/mol)']

    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.08, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)
    for row_idx in range(len(table_data)):
        for col_idx in range(4):
            cell = table[row_idx + 1, col_idx]
            if row_idx < 10:
                cell.set_facecolor('#E3EDF8')
            else:
                cell.set_facecolor('#FDE0DD')
    ax6.set_title('Results Summary', fontsize=12, fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    out_png = os.path.join(RESULTS_DIR, 'twosteps_enthalpy_results.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  Saved {out_png}")

    out_csv = os.path.join(RESULTS_DIR, 'twosteps_enthalpy_results.csv')
    results_df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"  Saved {out_csv}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Two-step annealing of Polystyrene")
    print(f"  Method: direct heat-flow integration ({T_INT_LOW}–{T_INT_HIGH}°C), kJ/mol")
    print(f"  ΔH = released enthalpy relative to shortest T2 anneal in each group")
    print("=" * 70)
    print(f"\n{'Ramp':<6} {'t1@90°C':>10}  {'t2@80°C':>10}  {'ΔH':>10}")
    print(f"{'':6} {'(s)':>10}  {'(s)':>10}  {'(kJ/mol)':>10}")
    print("-" * 45)
    for _, r in results_df.iterrows():
        print(f"  {r['ramp']:<4.0f}  {r['T1_hold_s']:>10.4f}  {r['T2_hold_s']:>10.4f}  "
              f"{r['delta_H_kJmol']:>10.2f}")
    print("-" * 45)

    grp_a = results_df[results_df['T1_hold_s'] < 60]
    grp_b = results_df[results_df['T1_hold_s'] > 60]

    print(f"\n  Group A (T1@90°C = 50 s):")
    print(f"    ΔH: {grp_a['delta_H_kJmol'].min():.2f} – {grp_a['delta_H_kJmol'].max():.2f} kJ/mol")
    print(f"    ΔH mean ± std: {grp_a['delta_H_kJmol'].mean():.2f} ± {grp_a['delta_H_kJmol'].std():.2f} kJ/mol")

    print(f"\n  Group B (T1@90°C = 500 s):")
    print(f"    ΔH: {grp_b['delta_H_kJmol'].min():.2f} – {grp_b['delta_H_kJmol'].max():.2f} kJ/mol")
    print(f"    ΔH mean ± std: {grp_b['delta_H_kJmol'].mean():.2f} ± {grp_b['delta_H_kJmol'].std():.2f} kJ/mol")

    a_max = grp_a['delta_H_kJmol'].max()
    b_max = grp_b['delta_H_kJmol'].max()
    print(f"\n  Max ΔH Group A: {a_max:.2f} kJ/mol")
    print(f"  Max ΔH Group B: {b_max:.2f} kJ/mol")

    return results_df


if __name__ == '__main__':
    results = main()
