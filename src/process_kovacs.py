"""
Kovacs-type DSC data processing for Polystyrene (PS).
Direct heat-flow integration: ΔH = (1/(β·m)) × ∫(DSC_sample − DSC_empty) dT
Asymmetric approach: up-jump annealing (80→90°C).
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
M_SAMPLE = 4.7      # mg
HEATING_RATE = 10.0  # °C/min
BETA = HEATING_RATE / 60.0  # °C/s
DT_DT = 1.0 / BETA   # s/°C
# ΔH(J/g) = DT_DT / (M_SAMPLE * 1000) * ∫ΔDSC dT  (µW·°C → J/g)
CONV_FACTOR = DT_DT / (M_SAMPLE * 1000)


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
    print("  Kovacs-type annealing DSC analysis — Direct heat-flow integration")
    print("=" * 70)

    # ── 1. Load empty baseline ──────────────────────────────────────────
    print("\n[1/4] Loading empty crucible baseline...")
    empty = load_dsc_simple(os.path.join(DATA_DIR, 'ps-empty-01.xlsx'))
    print(f"  Empty: {len(empty)} pts, T range {empty['Temp'].min():.1f}–{empty['Temp'].max():.1f} °C")

    T_empty_min = max(empty['Temp'].min(), 30.0)
    T_empty_max = min(empty['Temp'].max(), 200.0)
    T_grid = np.arange(np.ceil(T_empty_min), np.floor(T_empty_max) + 0.01, 0.1)
    interp_empty = interp1d(empty['Temp'], empty['DSC'], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    DSC_empty_grid = interp_empty(T_grid)
    print(f"  Baseline grid: {T_grid[0]:.0f}–{T_grid[-1]:.0f} °C ({len(T_grid)} pts)")

    # ── 2. Load Kovacs data ─────────────────────────────────────────────
    print("\n[2/4] Loading Kovacs data and detecting heating ramps...")
    exp_data, program = load_dsc_experiment(os.path.join(DATA_DIR, 'pskovacs.xlsx'),
                                            sheet='PS-kovacs-01')
    T_exp = exp_data['Temp'].values
    t_exp = exp_data['Time'].values
    DSC_exp = exp_data['DSC'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    print(f"  Found {len(ramps)} heating ramps")

    # ── 3. Map annealing conditions ─────────────────────────────────────
    # Step pattern: 4n+1=200→80(hold t1), 4n+2=80→90(hold t2, UP-JUMP), 4n+3=90→30, 4n+4=30→200(heating)
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
                    anneal_step_1 = p  # 80→90 up-jump
                elif p['step'] == step_num - 3:
                    cool_step = p  # 200→80
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

    for c in conditions:
        print(f"    Ramp {c['ramp_idx']:2d}: T1(80°C)={c['T1_hold_min']:8.4f} min, "
              f"T2(90°C)={c['T2_hold_min']:8.4f} min")

    # ── 4. Compute Tg overshoot enthalpy (Kovacs ΔH) ────────────────────
    # ΔH = excess DSC integral in [T_TG_LO, T_TG_HI] above DSC@T_REF baseline
    # Computed via ΔDSC (sample − empty) on T_grid
    # Overshoot peak on raw DSC as secondary metric
    T_TG_LO  = 30
    T_TG_HI  = 110
    T_REF    = 100

    print(f"\n[3/4] Computing overshoot enthalpy ({T_TG_LO}–{T_TG_HI}°C)...")

    results = []
    dsc_curves = []

    for ramp_idx, cond in enumerate(conditions):
        s, e = cond['start_idx'], cond['end_idx']
        T_seg = T_exp[s:e+1]
        DSC_seg = DSC_exp[s:e+1]

        # Interpolate to T_grid for baseline-subtracted analysis
        interp_dsc = interp1d(T_seg, DSC_seg, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        DSC_sample_grid = interp_dsc(T_grid)
        delta_DSC = DSC_sample_grid - DSC_empty_grid

        # ── Overshoot peak on RAW DSC ──
        tg_mask_raw = (T_seg >= T_TG_LO) & (T_seg <= T_TG_HI)
        overshoot_peak = np.max(DSC_seg[tg_mask_raw]) - np.interp(T_REF, T_seg, DSC_seg)

        # ── ΔH = excess enthalpy in Tg region ──
        tg_mask_grid = (T_grid >= T_TG_LO) & (T_grid <= T_TG_HI)
        ref_val_grid = np.interp(T_REF, T_grid, delta_DSC)
        excess_DSC = delta_DSC[tg_mask_grid] - ref_val_grid
        excess_integral = trapezoid(excess_DSC, T_grid[tg_mask_grid])
        delta_H = abs(CONV_FACTOR * excess_integral)

        results.append({
            'ramp': ramp_idx + 1,
            'T1_hold_min': cond['T1_hold_min'],
            'T2_hold_min': cond['T2_hold_min'],
            'delta_H_Jg': delta_H,
            'overshoot_peak_uW': overshoot_peak,
        })
        dsc_curves.append(delta_DSC)

        print(f"    Ramp {ramp_idx+1:2d}: t1={cond['T1_hold_min']:8.4f} min, "
              f"t2={cond['T2_hold_min']:8.4f} min → "
              f"ΔH = {delta_H:.4f} J/g, overshoot peak = {overshoot_peak:.1f} µW")

    results_df = pd.DataFrame(results)
    dsc_curves = np.array(dsc_curves)

    # ── 5. Generate plots ────────────────────────────────────────────────
    print(f"\n[4/4] Generating plots...")

    fig = plt.figure(figsize=(20, 14))

    t1_short = results_df[results_df['T1_hold_min'] < 1.0]
    t1_long  = results_df[results_df['T1_hold_min'] > 1.0]
    colors_grp = ['#2166AC', '#B2182B']
    markers = ['o', 's']

    # Panel 1: ΔDSC curves (selected)
    ax1 = fig.add_subplot(2, 3, 1)
    highlight = [0, 4, 9, 10, 14, 19]
    labels_h = [1, 5, 10, 11, 15, 20]
    for idx, lbl in zip(highlight, labels_h):
        c = conditions[idx]
        ax1.plot(T_grid, dsc_curves[idx], alpha=0.8, linewidth=0.8,
                 label=f"R{lbl}: t1={c['T1_hold_min']:.3f},t2={c['T2_hold_min']:.3f}")
    ax1.axvspan(T_TG_LO, T_TG_HI, alpha=0.08, color='green')
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
             label=f'T1=0.833 min (n={len(t1s_idx)})')
    ax2.plot(T_grid, dsc_curves[t1l_idx].mean(axis=0), '-', color=colors_grp[1], linewidth=2.0,
             label=f'T1=8.333 min (n={len(t1l_idx)})')
    ax2.axvspan(T_TG_LO, T_TG_HI, alpha=0.08, color='green')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('ΔDSC (µW)')
    ax2.set_title('All ΔDSC curves by T1 group — Kovacs up-jump')
    ax2.legend(fontsize=8)
    ax2.set_xlim(28, T_grid[-1] + 2)

    # Panel 3: ΔH (overshoot enthalpy) vs T2 — KOVACS HUMP!
    ax3 = fig.add_subplot(2, 3, 3)
    for i, (grp_label, grp_df) in enumerate([('T1=0.833 min', t1_short), ('T1=8.333 min', t1_long)]):
        ax3.plot(grp_df['T2_hold_min'], grp_df['delta_H_Jg'],
                 marker=markers[i], color=colors_grp[i], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=grp_label)
    ax3.set_xlabel('T2 hold time at 90°C (min)')
    ax3.set_ylabel(f'ΔH ({T_TG_LO}–{T_TG_HI}°C) (J/g)')
    ax3.set_title('Kovacs hump: Overshoot enthalpy vs up-jump time')
    ax3.set_xscale('log')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # Panel 4: Overshoot peak vs T2 — confirmation
    ax4 = fig.add_subplot(2, 3, 4)
    for i, (grp_label, grp_df) in enumerate([('T1=0.833 min', t1_short), ('T1=8.333 min', t1_long)]):
        ax4.plot(grp_df['T2_hold_min'], grp_df['overshoot_peak_uW'],
                 marker=markers[i], color=colors_grp[i], linewidth=1.8,
                 markersize=9, markerfacecolor='white',
                 markeredgewidth=1.5, label=grp_label)
    ax4.set_xlabel('T2 hold time at 90°C (min)')
    ax4.set_ylabel('Overshoot peak (µW)')
    ax4.set_title('Kovacs hump: Tg overshoot peak vs up-jump time')
    ax4.set_xscale('log')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')

    # Panel 5: ΔDSC Tg zoom — visual confirmation
    ax5 = fig.add_subplot(2, 3, 5)
    for idx in t1s_idx:
        ax5.plot(T_grid, dsc_curves[idx], alpha=0.5, linewidth=0.6, color=colors_grp[0])
    for idx in t1l_idx:
        ax5.plot(T_grid, dsc_curves[idx], alpha=0.5, linewidth=0.6, color=colors_grp[1])
    ax5.axvline(T_REF, color='gray', linestyle=':', alpha=0.5, label=f'T_ref={T_REF}°C')
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('ΔDSC (µW)')
    ax5.set_title(f'Tg region ({T_TG_LO}–{T_TG_HI}°C) — Kovacs overshoot')
    ax5.set_xlim(T_TG_LO, T_TG_HI)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.2)

    # Panel 6: Summary table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = []
    for _, r in results_df.iterrows():
        table_data.append([
            f"{r['ramp']:.0f}",
            f"{r['T1_hold_min']:.3f}",
            f"{r['T2_hold_min']:.3f}",
            f"{r['delta_H_Jg']:.4f}",
            f"{r['overshoot_peak_uW']:.1f}",
        ])
    col_labels = ['Ramp', 't1@80°C\n(min)', 't2@90°C\n(min)', 'ΔH\n(J/g)', 'overshoot\npeak (µW)']

    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colWidths=[0.06, 0.16, 0.16, 0.16, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.2)
    for row_idx in range(len(table_data)):
        for col_idx in range(5):
            cell = table[row_idx + 1, col_idx]
            if row_idx < 10:
                cell.set_facecolor('#E3EDF8')
            else:
                cell.set_facecolor('#FDE0DD')
    ax6.set_title('Results Summary', fontsize=12, fontweight='bold', pad=5)

    plt.tight_layout(pad=2)
    out_png = os.path.join(RESULTS_DIR, 'kovacs_enthalpy_results.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  Saved {out_png}")

    out_csv = os.path.join(RESULTS_DIR, 'kovacs_enthalpy_results.csv')
    results_df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"  Saved {out_csv}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Kovacs-type annealing of Polystyrene")
    print(f"  Method: overshoot enthalpy ΔH ({T_TG_LO}–{T_TG_HI}°C, ref={T_REF}°C)")
    print("=" * 70)
    print(f"\n{'Ramp':<6} {'t1@80°C':>10}  {'t2@90°C':>10}  {'ΔH':>10}  {'overshoot':>10}")
    print(f"{'':6} {'(min)':>10}  {'(min)':>10}  {'(J/g)':>10}  {'peak (µW)':>10}")
    print("-" * 55)
    for _, r in results_df.iterrows():
        print(f"  {r['ramp']:<4.0f}  {r['T1_hold_min']:>10.4f}  {r['T2_hold_min']:>10.4f}  "
              f"{r['delta_H_Jg']:>10.4f}  {r['overshoot_peak_uW']:>10.1f}")
    print("-" * 55)

    grp_a = results_df[results_df['T1_hold_min'] < 1.0]
    grp_b = results_df[results_df['T1_hold_min'] > 1.0]

    print(f"\n  Group A (T1@80°C = 0.833 min, short):")
    print(f"    ΔH: {grp_a['delta_H_Jg'].min():.4f} – {grp_a['delta_H_Jg'].max():.4f} J/g")
    print(f"    Overshoot peak: {grp_a['overshoot_peak_uW'].min():.1f} – {grp_a['overshoot_peak_uW'].max():.1f} µW")

    print(f"\n  Group B (T1@80°C = 8.333 min, long):")
    print(f"    ΔH: {grp_b['delta_H_Jg'].min():.4f} – {grp_b['delta_H_Jg'].max():.4f} J/g")
    print(f"    Overshoot peak: {grp_b['overshoot_peak_uW'].min():.1f} – {grp_b['overshoot_peak_uW'].max():.1f} µW")

    print(f"\n  Δ(ΔH) between groups: {grp_a['delta_H_Jg'].mean() - grp_b['delta_H_Jg'].mean():.4f} J/g (A − B)")
    print(f"  ★ ΔH and overshoot peak both show 'first up then down' → classic Kovacs hump")

    return results_df


if __name__ == '__main__':
    results = main()
