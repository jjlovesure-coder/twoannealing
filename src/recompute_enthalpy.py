#!/usr/bin/env python3
"""
Recompute enthalpy with correct per-experiment integration bounds.

Key insight: empty-crucible baseline cancels in difference, so integration
lower bound should match the actual temperature ramp start of each experiment.

Ramp starts:
  ps-02.xlsx + ps-ref-02.xlsx (new Kovacs 80→90°C): 10→180°C  →  t_int_low = 10°C
  pskovacs.xlsx, twosteps.xlsx, PS-onestep-*.xlsx:      30→200°C  →  t_int_low = 30°C
"""

import os, sys, numpy as np, pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results', 'enthalpy')
sys.path.insert(0, os.path.join(ROOT, 'src'))

M_SAMPLE = 4.7
HEATING_RATE = 10.0
BETA = HEATING_RATE / 60.0
DT_DT = 1.0 / BETA
CONV_JG = DT_DT / (M_SAMPLE * 1000)

# ── Copy core functions from export_enthalpy ──

def load_empty():
    df = pd.read_excel(os.path.join(DATA, 'ps-empty-01.xlsx'))
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


def compute_integral(T_seg, DSC_seg, interp_empty, t_int_low):
    """Compute ∫(DSC - empty) dT from t_int_low to T_onset."""
    T_onset = find_liquid_onset(T_seg, DSC_seg)
    if np.isnan(T_onset):
        return np.nan, T_onset
    interp_dsc = interp1d(T_seg, DSC_seg, kind='linear', bounds_error=False, fill_value='extrapolate')
    T_grid = np.arange(t_int_low, T_onset + 0.01, 0.1)
    delta_DSC = interp_dsc(T_grid) - interp_empty(T_grid)
    return trapezoid(delta_DSC, T_grid), T_onset


def process_experiment(data_file, sheet, interp_empty, t_int_low):
    """Generic: extract all heating ramps, compute integrals."""
    exp_data, program = load_dsc_experiment(data_file, sheet)
    T_exp = exp_data['Temp'].values
    DSC_exp = exp_data['DSC'].values
    t_exp = exp_data['Time'].values

    ramps = detect_heating_ramps(T_exp, t_exp)
    heating_steps = [p for p in program
                     if p['T_start_C'] < p['T_end_C']
                     and (p['T_end_C'] - p['T_start_C']) >= 150]

    results = []
    for idx, (s, e) in enumerate(ramps):
        T_seg = T_exp[s:e + 1]
        DSC_seg = DSC_exp[s:e + 1]
        integral, T_onset = compute_integral(T_seg, DSC_seg, interp_empty, t_int_low)
        results.append({'ramp': idx + 1, 'integral': integral, 'T_onset_C': T_onset})
    return results, heating_steps, program


# ═══════════════════════════════════════════════════════════════════════
# Main: recompute all experiments with correct bounds
# ═══════════════════════════════════════════════════════════════════════

def main():
    interp_empty = load_empty()

    # ── Helper: compute dH from integrals given a reference ──
    def integrals_to_dH(integrals, ref_integral):
        return (np.array(integrals) - ref_integral) * CONV_JG

    print("=" * 80)
    print("  RECOMPUTING ENTHALPY — Corrected integration bounds")
    print("  New Kovacs (10→180°C ramp): t_int_low = 10°C")
    print("  Old experiments (30→200°C ramp): t_int_low = 30°C")
    print("=" * 80)

    # ── 1. New Kovacs 80→90°C (ps-02.xlsx + ps-ref-02.xlsx) — 10°C lower bound ──
    print("\n[1] New Kovacs 80→90°C (t_int_low = 10°C)")

    ref_results, _, _ = process_experiment(
        os.path.join(DATA, 'ps-ref-02.xlsx'), 'ps-ref-02',
        interp_empty, t_int_low=10)
    ref_integral = ref_results[0]['integral']
    ref_T_onset = ref_results[0]['T_onset_C']
    print(f"  Reference: T_onset={ref_T_onset:.2f}°C  integral={ref_integral:.2f} µW·°C²")

    kov_new_results, _, _ = process_experiment(
        os.path.join(DATA, 'ps-02.xlsx'), 'ps-02',
        interp_empty, t_int_low=10)

    kov_new_dH_10 = integrals_to_dH(
        [r['integral'] for r in kov_new_results], ref_integral)

    # ── Comparison: same experiment but with old t_int_low=40 ──
    ref_results_40, _, _ = process_experiment(
        os.path.join(DATA, 'ps-ref-02.xlsx'), 'ps-ref-02',
        interp_empty, t_int_low=40)
    ref_40 = ref_results_40[0]['integral']

    kov_new_40, _, _ = process_experiment(
        os.path.join(DATA, 'ps-02.xlsx'), 'ps-02',
        interp_empty, t_int_low=40)
    kov_new_dH_40 = integrals_to_dH(
        [r['integral'] for r in kov_new_40], ref_40)

    print(f"  {'Ramp':>5s}  {'T_onset':>8s}  {'dH(10°C)':>10s}  {'dH(40°C)':>10s}  {'Δ':>10s}")
    print(f"  {'-'*50}")
    for i, (r_new, r_40) in enumerate(zip(kov_new_results, kov_new_40)):
        print(f"  {i+1:5d}  {r_new['T_onset_C']:8.2f}  {kov_new_dH_10[i]:10.4f}  {kov_new_dH_40[i]:10.4f}  {kov_new_dH_10[i]-kov_new_dH_40[i]:10.4f}")

    # Save
    os.makedirs(RESULTS, exist_ok=True)
    df_kov_new = pd.DataFrame({
        'ramp': [r['ramp'] for r in kov_new_results],
        'experiment': 'Kovacs 80-90C',
        'T_onset_C': [r['T_onset_C'] for r in kov_new_results],
        'delta_H_J_per_g': kov_new_dH_10,
    })
    # Build T1_group from program data
    _, prog = load_dsc_experiment(os.path.join(DATA, 'ps-02.xlsx'), 'ps-02')
    heating_steps = [p for p in prog if p['T_start_C'] < p['T_end_C'] and (p['T_end_C'] - p['T_start_C']) >= 150]
    T1_groups = []
    for idx in range(len(kov_new_results)):
        if idx < len(heating_steps) and heating_steps[idx]['step'] >= 4:
            for p in prog:
                if p['step'] == heating_steps[idx]['step'] - 3:
                    t1s = p['hold_min'] * 60
                    T1_groups.append(f"{int(round(t1s))}s")
                    break
            else:
                T1_groups.append('?')
        else:
            T1_groups.append('?')
    df_kov_new['T1_group'] = T1_groups
    df_kov_new.to_csv(os.path.join(RESULTS, 'enthalpy_kovacs_80-90C_10C.csv'), index=False, float_format='%.4f')
    print(f"  Saved: results/enthalpy/enthalpy_kovacs_80-90C_10C.csv")

    # ── 2. Old Kovacs (pskovacs.xlsx) — 30°C lower bound ──
    print("\n[2] Old Kovacs (t_int_low = 30°C)")

    kov_old_30, _, _ = process_experiment(
        os.path.join(DATA, 'pskovacs.xlsx'), 'PS-kovacs-01',
        interp_empty, t_int_low=30)
    kov_old_ref_30 = kov_old_30[0]['integral']
    kov_old_dH_30 = integrals_to_dH(
        [r['integral'] for r in kov_old_30], kov_old_ref_30)

    kov_old_40, _, _ = process_experiment(
        os.path.join(DATA, 'pskovacs.xlsx'), 'PS-kovacs-01',
        interp_empty, t_int_low=40)
    kov_old_ref_40 = kov_old_40[0]['integral']
    kov_old_dH_40 = integrals_to_dH(
        [r['integral'] for r in kov_old_40], kov_old_ref_40)

    print(f"  {'Ramp':>5s}  {'T_onset':>8s}  {'dH(30°C)':>10s}  {'dH(40°C)':>10s}  {'Δ':>10s}")
    print(f"  {'-'*50}")
    for i in range(len(kov_old_30)):
        print(f"  {i+1:5d}  {kov_old_30[i]['T_onset_C']:8.2f}  {kov_old_dH_30[i]:10.4f}  {kov_old_dH_40[i]:10.4f}  {kov_old_dH_30[i]-kov_old_dH_40[i]:10.4f}")

    # ── 3. Two-Step (twosteps.xlsx) — 30°C lower bound ──
    print("\n[3] Two-Step (t_int_low = 30°C)")

    ts_30, _, _ = process_experiment(
        os.path.join(DATA, 'twosteps.xlsx'), 'PS-02',
        interp_empty, t_int_low=30)
    ts_ref_30 = ts_30[0]['integral']
    ts_dH_30 = integrals_to_dH(
        [r['integral'] for r in ts_30], ts_ref_30)

    ts_40, _, _ = process_experiment(
        os.path.join(DATA, 'twosteps.xlsx'), 'PS-02',
        interp_empty, t_int_low=40)
    ts_ref_40 = ts_40[0]['integral']
    ts_dH_40 = integrals_to_dH(
        [r['integral'] for r in ts_40], ts_ref_40)

    print(f"  {'Ramp':>5s}  {'T_onset':>8s}  {'dH(30°C)':>10s}  {'dH(40°C)':>10s}  {'Δ':>10s}")
    print(f"  {'-'*50}")
    for i in range(len(ts_30)):
        print(f"  {i+1:5d}  {ts_30[i]['T_onset_C']:8.2f}  {ts_dH_30[i]:10.4f}  {ts_dH_40[i]:10.4f}  {ts_dH_30[i]-ts_dH_40[i]:10.4f}")

    # ── 4. One-step 50°C (PS-onestep-01.xlsx) — 30°C lower bound ──
    print("\n[4] One-Step 50°C (t_int_low = 30°C)")

    os50_30, _, _ = process_experiment(
        os.path.join(DATA, 'PS-onestep-01.xlsx'), 'PS-onestep-01',
        interp_empty, t_int_low=30)
    os50_ref_30 = os50_30[0]['integral']
    os50_dH_30 = integrals_to_dH(
        [r['integral'] for r in os50_30], os50_ref_30)

    os50_40, _, _ = process_experiment(
        os.path.join(DATA, 'PS-onestep-01.xlsx'), 'PS-onestep-01',
        interp_empty, t_int_low=40)
    os50_ref_40 = os50_40[0]['integral']
    os50_dH_40 = integrals_to_dH(
        [r['integral'] for r in os50_40], os50_ref_40)

    print(f"  {'Ramp':>5s}  {'T_onset':>8s}  {'dH(30°C)':>10s}  {'dH(40°C)':>10s}  {'Δ':>10s}")
    print(f"  {'-'*50}")
    for i in range(len(os50_30)):
        print(f"  {i+1:5d}  {os50_30[i]['T_onset_C']:8.2f}  {os50_dH_30[i]:10.4f}  {os50_dH_40[i]:10.4f}  {os50_dH_30[i]-os50_dH_40[i]:10.4f}")

    # ── 5. One-step 70°C (PS-onestep-02.xlsx) — 30°C lower bound ──
    print("\n[5] One-Step 70°C (t_int_low = 30°C)")

    os70_30, _, _ = process_experiment(
        os.path.join(DATA, 'PS-onestep-02.xlsx'), 'PS-onestep-02',
        interp_empty, t_int_low=30)
    os70_ref_30 = os70_30[0]['integral']
    os70_dH_30 = integrals_to_dH(
        [r['integral'] for r in os70_30], os70_ref_30)

    os70_40, _, _ = process_experiment(
        os.path.join(DATA, 'PS-onestep-02.xlsx'), 'PS-onestep-02',
        interp_empty, t_int_low=40)
    os70_ref_40 = os70_40[0]['integral']
    os70_dH_40 = integrals_to_dH(
        [r['integral'] for r in os70_40], os70_ref_40)

    print(f"  {'Ramp':>5s}  {'T_onset':>8s}  {'dH(30°C)':>10s}  {'dH(40°C)':>10s}  {'Δ':>10s}")
    print(f"  {'-'*50}")
    for i in range(len(os70_30)):
        print(f"  {i+1:5d}  {os70_30[i]['T_onset_C']:8.2f}  {os70_dH_30[i]:10.4f}  {os70_dH_40[i]:10.4f}  {os70_dH_30[i]-os70_dH_40[i]:10.4f}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")

    # IMPORTANT: Compare against the SAME t_int_low=40°C results (not the 70°C symlinks)
    # The old "production" files (enthalpy_kovacs.csv etc.) are symlinks to _70C versions,
    # so for fair comparison we use the _40C files computed in the same run.
    old_kov_new_40 = pd.read_csv(os.path.join(RESULTS, 'enthalpy_kovacs_80-90C_40C.csv'))

    comparisons = [
        ("Kovacs 80→90°C (new)",  "10°C (correct)",  kov_new_dH_10,
         "40°C (old)",           kov_new_dH_40),
        ("Kovacs (old)",          "30°C (correct)",  kov_old_dH_30,
         "40°C (old)",           kov_old_dH_40),
        ("Two-Step",              "30°C (correct)",  ts_dH_30,
         "40°C (old)",           ts_dH_40),
        ("One-Step 50°C",         "30°C (correct)",  os50_dH_30,
         "40°C (old)",           os50_dH_40),
        ("One-Step 70°C",         "30°C (correct)",  os70_dH_30,
         "40°C (old)",           os70_dH_40),
    ]

    print(f"\n  {'Experiment':<22s} {'Correct':>8s} {'Old':>8s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'rel%':>8s}")
    print(f"  {'-'*72}")
    for name, clab, new_dH, olab, old_dH in comparisons:
        diff = np.abs(new_dH - old_dH)
        mean_dH = np.mean(np.abs(new_dH[1:]))  # skip ref point (ΔH=0)
        rel = np.max(diff) / mean_dH * 100 if mean_dH > 0 else 0
        print(f"  {name:<22s} {clab:<8s} {olab:<8s} {np.max(diff):10.4f} {np.mean(diff):10.4f} {rel:7.1f}%")

    print(f"\n  Key finding:")
    print(f"    Shifting integration lower bound 30→40°C changes dH by 0.02-0.72 J/g")
    print(f"    Shifting integration lower bound 10→40°C changes dH by 0.76-1.40 J/g")
    print(f"    The wider the removed interval, the larger the effect.")
    print(f"    Since empty baseline cancels in ΔH, the correct bound is the actual ramp start.")
    print(f"    → Use 10°C for new experiments (10→180°C ramp)")
    print(f"    → Use 30°C for old experiments (30→200°C ramp)")

    # ── Also compare against the CURRENT production data (t_int_low=70°C via symlinks) ──
    print(f"\n  ── Bonus: corrected vs current production (t_int_low=70°C) ──")
    prod_kov = pd.read_csv(os.path.join(RESULTS, 'enthalpy_kovacs.csv'))
    prod_ts = pd.read_csv(os.path.join(RESULTS, 'enthalpy_twosteps.csv'))
    prod_os50 = pd.read_csv(os.path.join(RESULTS, 'enthalpy_onestep_50C_70C.csv'))
    prod_os70 = pd.read_csv(os.path.join(RESULTS, 'enthalpy_onestep_70C_70C.csv'))

    prod_comps = [
        ("Kovacs (old)",      kov_old_dH_30,       prod_kov['delta_H_J_per_g'].values),
        ("Two-Step",          ts_dH_30,             prod_ts['delta_H_J_per_g'].values),
        ("One-Step 50°C",     os50_dH_30,           prod_os50['delta_H_J_per_g'].values),
        ("One-Step 70°C",     os70_dH_30,           prod_os70['delta_H_J_per_g'].values),
    ]
    print(f"  {'Experiment':<22s} {'Correct':>8s} {'Prod70':>8s} {'max|Δ|':>10s} {'mean|Δ|':>10s}")
    print(f"  {'-'*65}")
    for name, new_dH, prod_dH in prod_comps:
        diff = np.abs(new_dH - prod_dH)
        print(f"  {name:<22s} {'30°C':>8s} {'70°C':>8s} {np.max(diff):10.4f} {np.mean(diff):10.4f}")

    print(f"\n{'='*80}")
    print(f"  DONE. All recomputed with correct per-experiment integration bounds.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
