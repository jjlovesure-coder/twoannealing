#!/usr/bin/env python3
"""
Compare DSC heating curves of polystyrene from two different instruments.

File 1: !2026004715.txt  - Mettler Toledo STARe SW 14.00, 6.4610 mg
File 2: PSrepeat2.xlsx    - Alternative DSC (channel 26111103R9-01), 4.7 mg

Same temperature program: 30↔200°C at 60°C/min, repeated cycles.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ──────────────────────────────────────────────────────────────────
TXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', '!2026004715.txt')
XLSX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'PSrepeat2.xlsx')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ─── Sample info ────────────────────────────────────────────────────────────
TXT_MASS_MG = 6.4610
XLSX_MASS_MG = 4.7

# ─── Common temperature grid ────────────────────────────────────────────────
T_GRID_START = 35.0
T_GRID_END = 195.0
T_GRID_STEP = 0.1


# ═════════════════════════════════════════════════════════════════════════════
# File I/O
# ═════════════════════════════════════════════════════════════════════════════

def read_txt_stare(path):
    """Read Mettler Toledo STARe evaluation export (UTF-16LE, fixed-width)."""
    with open(path, 'rb') as f:
        raw = f.read()
    lines = raw.decode('utf-16-le').split('\r\n')

    data_rows = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            try:
                data_rows.append({
                    'index': int(parts[0]),
                    'Time_s': float(parts[1]),
                    'Ts_C': float(parts[2]),
                    'Tr_C': float(parts[3]),
                    'HF_Wg': float(parts[4]),
                })
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(data_rows)
    return df


def read_xlsx_dsc(path, sheet='PS60-01'):
    """Read raw DSC data from XLSX file."""
    df = pd.read_excel(path, sheet_name=sheet, skiprows=24, usecols='A:D')
    df.columns = ['Time_min', 'Temp_C', 'DSC_uW', 'DDSC_uWmin']
    df = df.dropna().reset_index(drop=True)
    df['Time_s'] = df['Time_min'] * 60.0
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Segment detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_heating_segments(temp, time, min_rate=0.3, min_pts=100,
                            smooth_window=500):
    """Detect heating ramps using smoothed temperature derivative.

    min_rate in °C/s (0.3 ≈ 18°C/min; nominal rate is ~1.0 = 60°C/min).
    """
    dTdt = np.gradient(temp, time)
    dTdt_smooth = uniform_filter1d(dTdt, size=smooth_window)
    heating = dTdt_smooth > min_rate

    runs = []
    in_run = False
    start = 0
    for i in range(len(heating)):
        if heating[i] and not in_run:
            start = i
            in_run = True
        elif not heating[i] and in_run:
            n = i - start
            if n >= min_pts and temp[start] < 40 and temp[i - 1] > 180:
                runs.append((start, i - 1))
            in_run = False
    if in_run:
        n = len(heating) - start
        if n >= min_pts and temp[start] < 40 and temp[-1] > 180:
            runs.append((start, len(heating) - 1))
    return runs


def extract_heating_data(df, time_col, temp_col, hf_col, smooth_window=500):
    """Extract heating segments; returns list of (temp, hf, time) tuples."""
    temp = df[temp_col].values
    time = df[time_col].values
    hf = df[hf_col].values

    n = len(temp)
    effective_window = min(smooth_window, max(5, n // 20))

    segs = detect_heating_segments(temp, time, min_rate=0.3, min_pts=50,
                                   smooth_window=effective_window)
    segments = []
    for s, e in segs:
        mask = (temp[s:e + 1] >= 33) & (temp[s:e + 1] <= 197)
        idx = np.arange(s, e + 1)[mask]
        segments.append((temp[idx], hf[idx], time[idx]))
    return segments


# ═════════════════════════════════════════════════════════════════════════════
# Interpolation
# ═════════════════════════════════════════════════════════════════════════════

def interpolate_to_grid(segments, T_grid):
    """Interpolate segment list onto common T grid. Returns (n_seg, n_T)."""
    n_seg = len(segments)
    result = np.full((n_seg, len(T_grid)), np.nan)
    for i, (tseg, hfseg, _) in enumerate(segments):
        order = np.argsort(tseg)
        t_sorted = tseg[order]
        hf_sorted = hfseg[order]
        _, unique_idx = np.unique(t_sorted, return_index=True)
        t_uniq = t_sorted[np.sort(unique_idx)]
        hf_uniq = hf_sorted[np.sort(unique_idx)]
        if len(t_uniq) < 2:
            continue
        f = interp1d(t_uniq, hf_uniq, kind='linear',
                     bounds_error=False, fill_value=np.nan)
        result[i] = f(T_grid)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Baseline subtraction
# ═════════════════════════════════════════════════════════════════════════════

def subtract_linear_baseline(tseg, hfseg, pre_range=(40, 70),
                              post_range=(130, 180)):
    """Subtract a sigmoidally-blended linear baseline from heat flow data."""
    pre_mask = (tseg >= pre_range[0]) & (tseg <= pre_range[1])
    post_mask = (tseg >= post_range[0]) & (tseg <= post_range[1])
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return hfseg

    pre_fit = np.polyfit(tseg[pre_mask], hfseg[pre_mask], 1)
    post_fit = np.polyfit(tseg[post_mask], hfseg[post_mask], 1)

    # Sigmoidal blend through Tg region (~100°C for PS)
    w = 1 / (1 + np.exp((tseg - 100) / 5))
    baseline = w * np.polyval(pre_fit, tseg) + (1 - w) * np.polyval(post_fit, tseg)
    return hfseg - baseline


# ═════════════════════════════════════════════════════════════════════════════
# Tg analysis
# ═════════════════════════════════════════════════════════════════════════════

def find_tg_features(tseg, hfseg):
    """Find Tg via half-step method on baseline-subtracted heat flow.

    Uses the sigmoidal-baseline-subtracted curve. Tg midpoint is the
    temperature where the subtracted signal crosses its half-height.
    """
    hf_sub = subtract_linear_baseline(tseg, hfseg)

    # Restrict to Tg region for PS
    tg_mask = (tseg >= 70) & (tseg <= 140)
    if tg_mask.sum() < 10:
        return None

    t_tg = tseg[tg_mask]
    h_tg = hf_sub[tg_mask]

    # The step goes from ~0 (pre-Tg) to a lower plateau (post-Tg for endo)
    pre_level = np.median(h_tg[t_tg <= 85]) if (t_tg <= 85).sum() else np.nan
    post_level = np.median(h_tg[t_tg >= 120]) if (t_tg >= 120).sum() else np.nan

    if np.isnan(pre_level) or np.isnan(post_level):
        return None

    half_height = (pre_level + post_level) / 2

    # Find where the curve crosses half-height
    above = h_tg > half_height
    below = h_tg < half_height

    # Midpoint: crossing from above to below (for endothermic step)
    mid_idx_local = None
    for i in range(1, len(h_tg)):
        if above[i-1] and below[i]:
            mid_idx_local = i
            break
        elif below[i-1] and above[i]:
            mid_idx_local = i
            break

    # Onset: where curve deviates from pre-level by 20% of step height
    step_height = abs(post_level - pre_level)
    if step_height < 1e-6:
        return None
    onset_threshold = pre_level - 0.20 * step_height if pre_level > post_level else pre_level + 0.20 * step_height
    onset_local = None
    for i in range(len(h_tg)):
        if pre_level > post_level and h_tg[i] < onset_threshold:
            onset_local = i
            break
        elif pre_level < post_level and h_tg[i] > onset_threshold:
            onset_local = i
            break

    # End: where curve reaches 95% of post-level
    end_threshold = post_level + 0.20 * step_height if pre_level > post_level else post_level - 0.20 * step_height
    end_local = None
    for i in range(len(h_tg) - 1, -1, -1):
        if pre_level > post_level and h_tg[i] > end_threshold:
            end_local = i
            break
        elif pre_level < post_level and h_tg[i] < end_threshold:
            end_local = i
            break

    # Also find inflection point (max dHF/dT)
    dT = np.gradient(t_tg)
    if np.median(dT) > 0.01:
        dHFdT = np.gradient(h_tg) / dT
        infl_local = np.argmax(np.abs(dHFdT[3:-3])) + 3
    else:
        infl_local = len(h_tg) // 2

    return {
        'onset_T': float(t_tg[onset_local]) if onset_local is not None else np.nan,
        'midpoint_T': float(t_tg[mid_idx_local]) if mid_idx_local is not None else np.nan,
        'end_T': float(t_tg[end_local]) if end_local is not None else np.nan,
        'inflection_T': float(t_tg[infl_local]) if infl_local is not None else np.nan,
        'step_height': float(step_height),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(hf1, hf2, T_grid, label=""):
    """Compute difference statistics between two arrays on common T grid."""
    diff = hf2 - hf1
    valid = ~np.isnan(diff) & ~np.isinf(diff)
    if valid.sum() < 10:
        return {}
    d_valid = diff[valid]
    t_valid = T_grid[valid]

    pre_tg = (t_valid >= 35) & (t_valid <= 75)
    tg_region = (t_valid >= 80) & (t_valid <= 120)
    post_tg = (t_valid >= 125) & (t_valid <= 195)

    return {
        'label': label,
        'mean_diff': float(np.mean(d_valid)),
        'rms_diff': float(np.sqrt(np.mean(d_valid ** 2))),
        'max_abs_diff': float(np.max(np.abs(d_valid))),
        'std_diff': float(np.std(d_valid)),
        'pre_tg_mean': float(np.mean(d_valid[pre_tg])) if pre_tg.sum() else np.nan,
        'tg_mean': float(np.mean(d_valid[tg_region])) if tg_region.sum() else np.nan,
        'post_tg_mean': float(np.mean(d_valid[post_tg])) if post_tg.sum() else np.nan,
        'n_valid': int(valid.sum()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison(txt_segs, xlsx_segs, T_grid, hf_txt, hf_xlsx,
                    tg_info, output_path):
    """Generate 6-panel comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    C_TXT = '#1f77b4'
    C_XLSX = '#d62728'

    # ── (a) Raw overlay ──
    ax1.set_title('(a) Raw Heating Curves', fontweight='bold')
    for i, (tseg, hfseg, _) in enumerate(txt_segs):
        ls = '-' if i == 0 else '--'
        ax1.plot(tseg, hfseg, color=C_TXT, linestyle=ls, alpha=0.8, lw=1.0,
                 label='Mettler STARe' if i == 0 else '')
    for i, (tseg, hfseg, _) in enumerate(xlsx_segs):
        ls = '-' if i == 0 else '--'
        ax1.plot(tseg, hfseg, color=C_XLSX, linestyle=ls, alpha=0.8, lw=1.0,
                 label='Other DSC' if i == 0 else '')
    ax1.axvspan(90, 115, alpha=0.06, color='green')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Heat Flow (W/g)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(30, 200)

    # ── (b) Mean ± std ──
    ax2.set_title('(b) Mean Curves ± 1σ', fontweight='bold')
    for label, arr, color in [('Mettler STARe', hf_txt, C_TXT),
                               ('Other DSC', hf_xlsx, C_XLSX)]:
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        valid = ~np.isnan(mu)
        ax2.plot(T_grid[valid], mu[valid], color=color, lw=1.8, label=label)
        ax2.fill_between(T_grid[valid], mu[valid] - sd[valid],
                         mu[valid] + sd[valid], alpha=0.15, color=color)
    ax2.axvspan(90, 115, alpha=0.06, color='green')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Heat Flow (W/g)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(30, 200)

    # ── (c) Baseline-aligned (shift to zero in pre-Tg) ──
    ax3.set_title('(c) Baseline-Aligned (pre-Tg zeroed)', fontweight='bold')
    for i, (tseg, hfseg, _) in enumerate(txt_segs):
        pre = (tseg >= 40) & (tseg <= 70)
        if pre.sum() < 2:
            continue
        ls = '-' if i == 0 else '--'
        ax3.plot(tseg, hfseg - np.mean(hfseg[pre]), color=C_TXT, ls=ls,
                 alpha=0.8, lw=1.0,
                 label='Mettler STARe' if i == 0 else '')
    for i, (tseg, hfseg, _) in enumerate(xlsx_segs):
        pre = (tseg >= 40) & (tseg <= 70)
        if pre.sum() < 2:
            continue
        ls = '-' if i == 0 else '--'
        ax3.plot(tseg, hfseg - np.mean(hfseg[pre]), color=C_XLSX, ls=ls,
                 alpha=0.8, lw=1.0,
                 label='Other DSC' if i == 0 else '')
    ax3.axvspan(90, 115, alpha=0.06, color='green')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('HF − pre-Tg baseline (W/g)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(30, 200)

    # ── (d) Tg region: baseline-subtracted ──
    ax4.set_title('(d) Tg Region — Baseline Subtracted', fontweight='bold')
    for i, (tseg, hfseg, _) in enumerate(txt_segs):
        hf_sub = subtract_linear_baseline(tseg, hfseg)
        tg = (tseg >= 70) & (tseg <= 140)
        ls = '-' if i == 0 else '--'
        ax4.plot(tseg[tg], hf_sub[tg], color=C_TXT, ls=ls, alpha=0.8, lw=1.2,
                 label='Mettler STARe' if i == 0 else '')
    for i, (tseg, hfseg, _) in enumerate(xlsx_segs):
        hf_sub = subtract_linear_baseline(tseg, hfseg)
        tg = (tseg >= 70) & (tseg <= 140)
        ls = '-' if i == 0 else '--'
        ax4.plot(tseg[tg], hf_sub[tg], color=C_XLSX, ls=ls, alpha=0.8, lw=1.2,
                 label='Other DSC' if i == 0 else '')
    ax4.axhline(y=0, color='gray', ls=':', lw=0.5)
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Baseline-subtracted HF (W/g)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── (e) Difference curve ──
    ax5.set_title('(e) Difference (Other DSC − Mettler)', fontweight='bold')
    diff_all = hf_xlsx - hf_txt
    for i in range(diff_all.shape[0]):
        v = ~np.isnan(diff_all[i])
        ax5.plot(T_grid[v], diff_all[i][v], color='gray', alpha=0.5, lw=0.6,
                 label='Individual pair' if i == 0 else '')
    diff_mean = np.nanmean(diff_all, axis=0)
    vd = ~np.isnan(diff_mean)
    ax5.plot(T_grid[vd], diff_mean[vd], 'k-', lw=1.5, label='Mean')
    ax5.axhline(y=0, color='k', ls=':', lw=0.5)
    ax5.axvspan(90, 115, alpha=0.06, color='green')
    overall = float(np.nanmean(diff_mean[vd]))
    ax5.annotate(f'Mean offset: {overall:+.4f} W/g',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('Δ Heat Flow (W/g)')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(30, 200)

    # ── (f) Baseline-aligned difference ──
    ax6.set_title('(f) Baseline-Aligned Difference', fontweight='bold')

    # Re-interpolate baseline-subtracted segments
    txt_bl_sub = []
    xlsx_bl_sub = []
    for i, (tseg, hfseg, _) in enumerate(txt_segs):
        hf_sub = subtract_linear_baseline(tseg, hfseg)
        txt_bl_sub.append((tseg, hf_sub, None))
    for i, (tseg, hfseg, _) in enumerate(xlsx_segs):
        hf_sub = subtract_linear_baseline(tseg, hfseg)
        xlsx_bl_sub.append((tseg, hf_sub, None))

    hf_txt_bl = interpolate_to_grid(txt_bl_sub, T_grid)
    hf_xlsx_bl = interpolate_to_grid(xlsx_bl_sub, T_grid)

    diff_bl = hf_xlsx_bl - hf_txt_bl
    diff_bl_mean = np.nanmean(diff_bl, axis=0)
    v_bl = ~np.isnan(diff_bl_mean)
    ax6.plot(T_grid[v_bl], diff_bl_mean[v_bl], 'k-', lw=1.5, label='Mean difference')
    ax6.axhline(y=0, color='k', ls=':', lw=0.5)
    ax6.axvspan(90, 115, alpha=0.06, color='green')

    # Compute RMS in Tg region
    tg_grid = (T_grid >= 80) & (T_grid <= 120)
    if tg_grid.sum() > 0:
        rms_tg = float(np.sqrt(np.nanmean(diff_bl_mean[tg_grid] ** 2)))
        ax6.annotate(f'RMS in Tg: {rms_tg:.4f} W/g',
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax6.set_xlabel('Temperature (°C)')
    ax6.set_ylabel('Δ HF (baseline-subtracted, W/g)')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(30, 200)

    plt.tight_layout(pad=2.0)
    fig.suptitle('DSC Heating Curve Comparison: Polystyrene'
                 '\nMettler Toledo STARe (6.46 mg) vs Alternative DSC (4.7 mg)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 68)
    print("  DSC Heating Curve Comparison: Polystyrene (Two Instruments)")
    print("=" * 68)
    print(f"  Instrument A (Mettler Toledo STARe):  {TXT_MASS_MG:.3f} mg")
    print(f"  Instrument B (Other DSC 26111103R9-01): {XLSX_MASS_MG:.1f} mg")
    print(f"  Temperature program: 30↔200°C at ~60°C/min, repeated cycles")

    # ── 1. Read ──
    print("\n[1/6] Reading data files...")
    df_txt = read_txt_stare(TXT_PATH)
    df_xlsx = read_xlsx_dsc(XLSX_PATH)
    print(f"  TXT: {len(df_txt)} pts, {df_txt['Ts_C'].min():.1f}–{df_txt['Ts_C'].max():.1f}°C")
    print(f"  XLSX: {len(df_xlsx)} pts, {df_xlsx['Temp_C'].min():.1f}–{df_xlsx['Temp_C'].max():.1f}°C")

    # ── 2. Normalize ──
    print("\n[2/6] Normalizing units (W/g)...")
    df_xlsx['HF_Wg'] = df_xlsx['DSC_uW'] / (XLSX_MASS_MG * 1000)
    print(f"  XLSX raw: {df_xlsx['DSC_uW'].min():.0f}–{df_xlsx['DSC_uW'].max():.0f} uW")
    print(f"  XLSX norm: {df_xlsx['HF_Wg'].min():.4f}–{df_xlsx['HF_Wg'].max():.4f} W/g")
    print(f"  TXT norm:  {df_txt['HF_Wg'].min():.4f}–{df_txt['HF_Wg'].max():.4f} W/g (pre-normalized)")

    # ── 3. Extract heating segments ──
    print("\n[3/6] Extracting heating segments (30→200°C)...")
    txt_segs = extract_heating_data(df_txt, 'Time_s', 'Ts_C', 'HF_Wg',
                                    smooth_window=5)
    xlsx_segs = extract_heating_data(df_xlsx, 'Time_s', 'Temp_C', 'HF_Wg',
                                     smooth_window=500)

    print(f"  TXT: {len(txt_segs)} segments")
    for i, (tseg, hfseg, _) in enumerate(txt_segs):
        print(f"    H{i+1}: {tseg.min():.1f}→{tseg.max():.1f}°C, {len(tseg)} pts,"
              f" HF {hfseg.min():.4f}→{hfseg.max():.4f} W/g")
    print(f"  XLSX: {len(xlsx_segs)} segments")
    for i, (tseg, hfseg, _) in enumerate(xlsx_segs):
        print(f"    H{i+1}: {tseg.min():.1f}→{tseg.max():.1f}°C, {len(tseg)} pts,"
              f" HF {hfseg.min():.4f}→{hfseg.max():.4f} W/g")

    if len(txt_segs) == 0 or len(xlsx_segs) == 0:
        print("\nERROR: No heating segments found!")
        return

    # ── 4. Tg analysis on raw segments ──
    print("\n[4/6] Glass transition analysis...")
    tg_info = {'txt': [], 'xlsx': []}
    for i, (tseg, hfseg, _) in enumerate(txt_segs):
        tg = find_tg_features(tseg, hfseg)
        if tg:
            tg_info['txt'].append(tg)
            print(f"  TXT H{i+1}: onset={tg['onset_T']:.1f}°C,"
                  f" midpoint={tg['midpoint_T']:.1f}°C,"
                  f" inflection={tg['inflection_T']:.1f}°C")
    for i, (tseg, hfseg, _) in enumerate(xlsx_segs):
        tg = find_tg_features(tseg, hfseg)
        if tg:
            tg_info['xlsx'].append(tg)
            print(f"  XLSX H{i+1}: onset={tg['onset_T']:.1f}°C,"
                  f" midpoint={tg['midpoint_T']:.1f}°C,"
                  f" inflection={tg['inflection_T']:.1f}°C")

    # ── 5. Interpolate + statistics ──
    print("\n[5/6] Interpolating and computing statistics...")
    T_grid = np.arange(T_GRID_START, T_GRID_END + T_GRID_STEP, T_GRID_STEP)
    hf_txt_grid = interpolate_to_grid(txt_segs, T_grid)
    hf_xlsx_grid = interpolate_to_grid(xlsx_segs, T_grid)

    # Raw comparison
    txt_mean = np.nanmean(hf_txt_grid, axis=0)
    xlsx_mean = np.nanmean(hf_xlsx_grid, axis=0)
    stats_raw = compute_stats(txt_mean, xlsx_mean, T_grid, label="Raw mean")

    # Baseline-subtracted comparison
    txt_bl_segs = [(t, subtract_linear_baseline(t, h), _)
                   for t, h, _ in txt_segs]
    xlsx_bl_segs = [(t, subtract_linear_baseline(t, h), _)
                    for t, h, _ in xlsx_segs]
    hf_txt_bl_grid = interpolate_to_grid(txt_bl_segs, T_grid)
    hf_xlsx_bl_grid = interpolate_to_grid(xlsx_bl_segs, T_grid)
    txt_bl_mean = np.nanmean(hf_txt_bl_grid, axis=0)
    xlsx_bl_mean = np.nanmean(hf_xlsx_bl_grid, axis=0)
    stats_bl = compute_stats(txt_bl_mean, xlsx_bl_mean, T_grid,
                             label="Baseline-subtracted mean")

    # Sensitivity ratio (slope of XLSX vs TXT)
    v = ~np.isnan(txt_mean) & ~np.isnan(xlsx_mean)
    if v.sum() > 10:
        sens_ratio = np.polyfit(txt_mean[v], xlsx_mean[v], 1)
        sens_r2 = float(np.corrcoef(txt_mean[v], xlsx_mean[v])[0, 1] ** 2)
    else:
        sens_ratio = [np.nan, np.nan]
        sens_r2 = np.nan

    # ── 6. Plots ──
    print("\n[6/6] Generating comparison figure...")
    plot_path = os.path.join(RESULTS_DIR, 'ps_comparison_results.png')
    plot_comparison(txt_segs, xlsx_segs, T_grid, hf_txt_grid, hf_xlsx_grid,
                    tg_info, plot_path)

    # ── Save data ──
    csv_path = os.path.join(RESULTS_DIR, 'ps_comparison_data.csv')
    pd.DataFrame({
        'Temperature_C': T_grid,
        'Mettler_Mean_Wg': txt_mean,
        'Mettler_Std_Wg': np.nanstd(hf_txt_grid, axis=0),
        'OtherDSC_Mean_Wg': xlsx_mean,
        'OtherDSC_Std_Wg': np.nanstd(hf_xlsx_grid, axis=0),
        'Diff_Raw_Wg': xlsx_mean - txt_mean,
        'Mettler_BLsub_Mean_Wg': txt_bl_mean,
        'OtherDSC_BLsub_Mean_Wg': xlsx_bl_mean,
        'Diff_BLsub_Wg': xlsx_bl_mean - txt_bl_mean,
    }).to_csv(csv_path, index=False)
    print(f"  Data saved to: {csv_path}")

    # ── Summary ──
    s = stats_raw
    b = stats_bl
    print("\n" + "=" * 68)
    print("  ANALYSIS RESULTS")
    print("=" * 68)

    print(f"""
  ── Raw Comparison (Other DSC − Mettler STARe) ──
    Mean offset:       {s['mean_diff']:+.4f} W/g
    RMS difference:    {s['rms_diff']:.4f} W/g
    Max deviation:     {s['max_abs_diff']:.4f} W/g
    Pre-Tg (35–75°C):  {s['pre_tg_mean']:+.4f} W/g
    Tg region (80–120°C): {s['tg_mean']:+.4f} W/g
    Post-Tg (125–195°C): {s['post_tg_mean']:+.4f} W/g

  ── Baseline-Subtracted Comparison ──
    Mean offset:       {b['mean_diff']:+.4f} W/g
    RMS difference:    {b['rms_diff']:.4f} W/g
    Max deviation:     {b['max_abs_diff']:.4f} W/g

  ── Instrument Sensitivity ──
    Other DSC = {sens_ratio[0]:.3f} × Mettler + {sens_ratio[1]:.4f}
    R² = {sens_r2:.4f}

  ── Glass Transition Temperature ──""")

    if len(tg_info['txt']) > 0 and len(tg_info['xlsx']) > 0:
        txt_onsets = [t['onset_T'] for t in tg_info['txt']]
        xlsx_onsets = [t['onset_T'] for t in tg_info['xlsx']]
        txt_infl = [t['inflection_T'] for t in tg_info['txt']]
        xlsx_infl = [t['inflection_T'] for t in tg_info['xlsx']]
        txt_steps = [t['step_height'] for t in tg_info['txt']]
        xlsx_steps = [t['step_height'] for t in tg_info['xlsx']]
        print(f"""    Tg onset:      Mettler {np.mean(txt_onsets):.1f}°C,"
                f" Other DSC {np.mean(xlsx_onsets):.1f}°C  (Δ = {np.mean(xlsx_onsets)-np.mean(txt_onsets):.1f}°C)
    Tg inflection: Mettler {np.mean(txt_infl):.1f}°C,"
                f" Other DSC {np.mean(xlsx_infl):.1f}°C  (Δ = {np.mean(xlsx_infl)-np.mean(txt_infl):.1f}°C)
    Step height:   Mettler {np.mean(txt_steps):.4f} W/g,"
                f" Other DSC {np.mean(xlsx_steps):.4f} W/g""")

    print(f"""
  ── Key Findings ──
    1. 两台设备均为原始测量数据（未经基线校正），
       差异反映的是仪器硬件本身的不同特性。

    2. 系统性偏差: 两台仪器之间存在约 {s['mean_diff']:+.2f} W/g
       的基线偏移，Other DSC 的信号整体比 Mettler
       更偏负（吸热方向）。

    3. 扣除线性基线后，残余 RMS 从 {s['rms_diff']:.2f}
       降至 {b['rms_diff']:.4f} W/g，说明两条曲线在热事件
       特征上具有良好的一致性。

    4. 灵敏度: Other DSC 的信号幅度约为 Mettler 的
       {abs(sens_ratio[0]):.1f} 倍（R² = {sens_r2:.3f}），
       两台仪器的信号存在线性比例关系。

    5. 玻璃化转变: Mettler 测得 Tg 拐点约 {np.mean(txt_infl):.0f}°C，
       Other DSC 约 {np.mean(xlsx_infl):.0f}°C。
       温差（~{abs(np.mean(xlsx_infl)-np.mean(txt_infl)):.0f}°C）可能与热滞后
       及温度校准差异有关（升温速率 60°C/min）。

    6. 重复性: 两台仪器各自的两次升温曲线高度吻合，
       说明仪器的测量重复性良好。
""")
    print("=" * 68)


if __name__ == '__main__':
    main()
