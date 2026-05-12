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
from matplotlib.ticker import MultipleLocator
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
TXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', '!2026004715.txt')
XLSX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'PSrepeat2.xlsx')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ─── Sample info ─────────────────────────────────────────────────────────────
TXT_MASS_MG = 6.4610   # from footer line in TXT
XLSX_MASS_MG = 4.7     # from row 6 in XLSX

# ─── Common temperature grid ─────────────────────────────────────────────────
T_GRID_START = 35.0
T_GRID_END = 195.0
T_GRID_STEP = 0.1

# ─── File I/O ────────────────────────────────────────────────────────────────

def read_txt_stare(path: str) -> pd.DataFrame:
    """Read Mettler Toledo STARe evaluation export (UTF-16LE, fixed-width)."""
    with open(path, 'rb') as f:
        raw = f.read()
    lines = raw.decode('utf-16-le').split('\r\n')

    data_rows = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            try:
                idx = int(parts[0])
                t = float(parts[1])
                Ts = float(parts[2])
                Tr = float(parts[3])
                HF = float(parts[4])
                data_rows.append({'index': idx, 'Time_s': t, 'Ts_C': Ts,
                                  'Tr_C': Tr, 'HF_Wg': HF})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(data_rows)
    print(f"  TXT: {len(df)} data points, Ts={df['Ts_C'].min():.1f}→{df['Ts_C'].max():.1f}°C")
    return df


def read_xlsx_dsc(path: str, sheet: str = 'PS60-01') -> pd.DataFrame:
    """Read raw DSC data from XLSX file."""
    df = pd.read_excel(path, sheet_name=sheet, skiprows=24, usecols='A:D')
    df.columns = ['Time_min', 'Temp_C', 'DSC_uW', 'DDSC_uWmin']
    df = df.dropna().reset_index(drop=True)
    df['Time_s'] = df['Time_min'] * 60.0  # convert to seconds
    print(f"  XLSX: {len(df)} data points, Temp={df['Temp_C'].min():.1f}→{df['Temp_C'].max():.1f}°C")
    return df


# ─── Segment detection ──────────────────────────────────────────────────────

def detect_heating_segments(temp, time, min_rate=5, min_pts=100, smooth_window=500):
    """Detect heating ramps using smoothed temperature derivative.

    Returns list of (start_idx, end_idx) tuples.
    """
    dTdt = np.gradient(temp, time)
    dTdt_smooth = uniform_filter1d(dTdt, size=smooth_window)

    heating = dTdt_smooth > min_rate  # °C/min

    # Find contiguous runs
    runs = []
    in_run = False
    start = 0
    for i in range(len(heating)):
        if heating[i] and not in_run:
            start = i
            in_run = True
        elif not heating[i] and in_run:
            if i - start >= min_pts and temp[start] < 40 and temp[i-1] > 180:
                runs.append((start, i - 1))
            in_run = False
    if in_run and len(heating) - start >= min_pts:
        if temp[start] < 40 and temp[-1] > 180:
            runs.append((start, len(heating) - 1))

    return runs


def extract_heating_data(df, time_col, temp_col, hf_col, smooth_window=500):
    """Extract heating segments and return list of (temp, hf, time) tuples."""
    temp = df[temp_col].values
    time = df[time_col].values
    hf = df[hf_col].values

    # For TXT (1 Hz), use smaller window
    n = len(temp)
    effective_window = min(smooth_window, max(5, n // 20))

    segs = detect_heating_segments(temp, time, min_rate=5, min_pts=50,
                                   smooth_window=effective_window)

    segments = []
    for s, e in segs:
        # Trim to clean temperature range
        mask = (temp[s:e+1] >= 33) & (temp[s:e+1] <= 197)
        idx = np.arange(s, e+1)[mask]
        segments.append((temp[idx], hf[idx], time[idx]))

    return segments


# ─── Interpolation ──────────────────────────────────────────────────────────

def interpolate_to_grid(segments, T_grid):
    """Interpolate list of (temp, hf) segment tuples onto common T grid.

    Returns array of shape (n_segments, len(T_grid)).
    """
    n_seg = len(segments)
    result = np.full((n_seg, len(T_grid)), np.nan)

    for i, (tseg, hfseg, _) in enumerate(segments):
        # Sort by temperature
        order = np.argsort(tseg)
        t_sorted = tseg[order]
        hf_sorted = hfseg[order]

        # Deduplicate temperatures
        _, unique_idx = np.unique(t_sorted, return_index=True)
        t_uniq = t_sorted[np.sort(unique_idx)]
        hf_uniq = hf_sorted[np.sort(unique_idx)]

        if len(t_uniq) < 2:
            continue

        f = interp1d(t_uniq, hf_uniq, kind='linear',
                     bounds_error=False, fill_value=np.nan)
        result[i] = f(T_grid)

    return result


# ─── Statistics ─────────────────────────────────────────────────────────────

def compute_stats(hf_txt, hf_xlsx, T_grid, label=""):
    """Compute difference statistics between two interpolated heat flow arrays."""
    diff = hf_xlsx - hf_txt
    valid = ~np.isnan(diff) & ~np.isinf(diff)

    if valid.sum() < 10:
        return {}

    d_valid = diff[valid]
    t_valid = T_grid[valid]

    # Region masks
    pre_tg = (t_valid >= 35) & (t_valid <= 75)
    tg_region = (t_valid >= 80) & (t_valid <= 120)
    post_tg = (t_valid >= 125) & (t_valid <= 195)

    stats = {
        'label': label,
        'mean_diff': float(np.mean(d_valid)),
        'rms_diff': float(np.sqrt(np.mean(d_valid ** 2))),
        'max_abs_diff': float(np.max(np.abs(d_valid))),
        'std_diff': float(np.std(d_valid)),
        'pre_tg_mean': float(np.mean(d_valid[pre_tg])) if pre_tg.sum() > 0 else np.nan,
        'tg_mean': float(np.mean(d_valid[tg_region])) if tg_region.sum() > 0 else np.nan,
        'post_tg_mean': float(np.mean(d_valid[post_tg])) if post_tg.sum() > 0 else np.nan,
        'n_valid': int(valid.sum()),
    }
    return stats


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_comparison(txt_segments, xlsx_segments, T_grid,
                    hf_txt_grid, hf_xlsx_grid, stats_all, output_path):
    """Generate multi-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    # Colors
    txt_colors = ['#1f77b4', '#1f77b4']  # blue for Mettler
    xlsx_colors = ['#d62728', '#d62728']  # red for other DSC
    txt_style = ['-', '--']
    xlsx_style = ['-', '--']

    # ── Panel 1: Overlay of all heating curves ──
    ax1.set_title('(a) Heating Curves Overlay', fontsize=12, fontweight='bold')

    for i, (tseg, hfseg, _) in enumerate(txt_segments):
        ax1.plot(tseg, hfseg, color=txt_colors[i], linestyle=txt_style[i],
                 alpha=0.7, linewidth=1.2,
                 label=f'Mettler H{i+1}' if i == 0 else f'Mettler H{i+1}')

    for i, (tseg, hfseg, _) in enumerate(xlsx_segments):
        ax1.plot(tseg, hfseg, color=xlsx_colors[i], linestyle=xlsx_style[i],
                 alpha=0.7, linewidth=1.2,
                 label=f'Other DSC H{i+1}' if i == 0 else f'Other DSC H{i+1}')

    # Tg region band for PS
    ax1.axvspan(90, 110, alpha=0.08, color='green', label='PS Tg region')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Heat Flow (W/g)')
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(30, 200)

    # ── Panel 2: Mean curves with std bands ──
    ax2.set_title('(b) Mean Curves ± 1σ', fontsize=12, fontweight='bold')

    # TXT mean and std
    txt_mean = np.nanmean(hf_txt_grid, axis=0)
    txt_std = np.nanstd(hf_txt_grid, axis=0)
    txt_valid = ~np.isnan(txt_mean)

    ax2.plot(T_grid[txt_valid], txt_mean[txt_valid], color='#1f77b4',
             linewidth=1.8, label='Mettler (mean)')
    ax2.fill_between(T_grid[txt_valid],
                     txt_mean[txt_valid] - txt_std[txt_valid],
                     txt_mean[txt_valid] + txt_std[txt_valid],
                     alpha=0.2, color='#1f77b4')

    # XLSX mean and std
    xlsx_mean = np.nanmean(hf_xlsx_grid, axis=0)
    xlsx_std = np.nanstd(hf_xlsx_grid, axis=0)
    xlsx_valid = ~np.isnan(xlsx_mean)

    ax2.plot(T_grid[xlsx_valid], xlsx_mean[xlsx_valid], color='#d62728',
             linewidth=1.8, label='Other DSC (mean)')
    ax2.fill_between(T_grid[xlsx_valid],
                     xlsx_mean[xlsx_valid] - xlsx_std[xlsx_valid],
                     xlsx_mean[xlsx_valid] + xlsx_std[xlsx_valid],
                     alpha=0.2, color='#d62728')

    ax2.axvspan(90, 110, alpha=0.08, color='green')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Heat Flow (W/g)')
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(30, 200)

    # ── Panel 3: Difference curve ──
    ax3.set_title('(c) Heat Flow Difference (Other DSC − Mettler)', fontsize=12, fontweight='bold')

    diff_all = hf_xlsx_grid - hf_txt_grid
    diff_mean = np.nanmean(diff_all, axis=0)
    diff_std = np.nanstd(diff_all, axis=0)
    diff_valid = ~np.isnan(diff_mean)

    for i in range(diff_all.shape[0]):
        valid_row = ~np.isnan(diff_all[i])
        ax3.plot(T_grid[valid_row], diff_all[i][valid_row],
                 alpha=0.5, linewidth=0.8,
                 color='gray',
                 label=f'Pair {i+1}' if i == 0 else f'Pair {i+1}')

    ax3.plot(T_grid[diff_valid], diff_mean[diff_valid],
             color='black', linewidth=1.5, label='Mean difference')

    ax3.fill_between(T_grid[diff_valid],
                     diff_mean[diff_valid] - diff_std[diff_valid],
                     diff_mean[diff_valid] + diff_std[diff_valid],
                     alpha=0.2, color='black')

    ax3.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
    ax3.axvspan(90, 110, alpha=0.08, color='green')

    # Show mean diff annotation
    overall_mean = float(np.nanmean(diff_mean[diff_valid]))
    ax3.annotate(f'Mean: {overall_mean:+.4f} W/g',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Δ Heat Flow (W/g)')
    ax3.legend(fontsize=7, loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(30, 200)

    # ── Panel 4: Tg region zoom with baseline subtraction ──
    ax4.set_title('(d) Tg Region Zoom (80–120°C) — Baseline Subtracted', fontsize=12, fontweight='bold')

    # Simple linear baseline subtraction: fit line to pre-Tg (40-70) and post-Tg (130-180)
    for i, (seg_label, segs, color) in enumerate([
        ('Mettler', txt_segments, '#1f77b4'),
        ('Other DSC', xlsx_segments, '#d62728')
    ]):
        for j, (tseg, hfseg, _) in enumerate(segs):
            # Fit pre-Tg baseline
            pre_mask = (tseg >= 40) & (tseg <= 70)
            post_mask = (tseg >= 130) & (tseg <= 180)
            if pre_mask.sum() < 2 or post_mask.sum() < 2:
                continue

            pre_fit = np.polyfit(tseg[pre_mask], hfseg[pre_mask], 1)
            post_fit = np.polyfit(tseg[post_mask], hfseg[post_mask], 1)

            # Interpolate baseline over Tg region
            tg_mask = (tseg >= 80) & (tseg <= 120)
            if tg_mask.sum() < 2:
                continue

            t_tg = tseg[tg_mask]
            # Linear interpolated baseline between pre and post
            # Baseline(T) = pre_fit(T) for T<70, transitioning to post_fit(T)
            # Simple approach: sigmoidal blend
            w = 1 / (1 + np.exp((t_tg - 100) / 5))
            pre_line = np.polyval(pre_fit, t_tg)
            post_line = np.polyval(post_fit, t_tg)
            baseline = w * pre_line + (1 - w) * post_line

            hf_sub = hfseg[tg_mask] - baseline
            style = '-' if j == 0 else '--'
            ax4.plot(t_tg, hf_sub, color=color, linestyle=style, linewidth=1.2,
                     label=f'{seg_label} H{j+1}' if j == 0 and i == 0
                           else f'{seg_label} H{j+1}')

    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Baseline-subtracted HF (W/g)')
    ax4.legend(fontsize=8, loc='lower left')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linewidth=0.5, linestyle=':')

    plt.tight_layout(pad=2.0)
    fig.suptitle('DSC Heating Curve Comparison: Polystyrene (two instruments)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("DSC Heating Curve Comparison: Polystyrene")
    print("=" * 65)
    print(f"\n  Mettler Toledo STARe: {TXT_MASS_MG:.3f} mg")
    print(f"  Other DSC (26111103R9-01): {XLSX_MASS_MG:.1f} mg")
    print()

    # ── Step 1: Read data ──
    print("Step 1: Reading data files...")
    df_txt = read_txt_stare(TXT_PATH)
    df_xlsx = read_xlsx_dsc(XLSX_PATH)

    # ── Step 2: Normalize XLSX to W/g ──
    print("\nStep 2: Normalizing units...")
    df_xlsx['HF_Wg'] = df_xlsx['DSC_uW'] / (XLSX_MASS_MG * 1000)
    print(f"  XLSX DSC range: {df_xlsx['DSC_uW'].min():.1f} → {df_xlsx['DSC_uW'].max():.1f} uW")
    print(f"  XLSX HF_Wg range: {df_xlsx['HF_Wg'].min():.4f} → {df_xlsx['HF_Wg'].max():.4f} W/g")

    # ── Step 3: Extract heating segments ──
    print("\nStep 3: Extracting heating segments...")
    txt_segments = extract_heating_data(df_txt, 'Time_s', 'Ts_C', 'HF_Wg', smooth_window=5)
    xlsx_segments = extract_heating_data(df_xlsx, 'Time_s', 'Temp_C', 'HF_Wg', smooth_window=500)

    print(f"  TXT: {len(txt_segments)} heating segments")
    for i, (tseg, hfseg, _) in enumerate(txt_segments):
        print(f"    H{i+1}: {tseg.min():.1f}→{tseg.max():.1f}°C, {len(tseg)} pts")

    print(f"  XLSX: {len(xlsx_segments)} heating segments")
    for i, (tseg, hfseg, _) in enumerate(xlsx_segments):
        print(f"    H{i+1}: {tseg.min():.1f}→{tseg.max():.1f}°C, {len(tseg)} pts")

    if len(txt_segments) == 0 or len(xlsx_segments) == 0:
        print("\nERROR: Could not find heating segments in one or both files!")
        return

    # ── Step 4: Interpolate to common grid ──
    print("\nStep 4: Interpolating to common temperature grid...")
    T_grid = np.arange(T_GRID_START, T_GRID_END + T_GRID_STEP, T_GRID_STEP)
    print(f"  Grid: {T_grid[0]:.1f} to {T_grid[-1]:.1f}°C, {len(T_grid)} points")

    hf_txt_grid = interpolate_to_grid(txt_segments, T_grid)
    hf_xlsx_grid = interpolate_to_grid(xlsx_segments, T_grid)

    print(f"  TXT interpolated: {hf_txt_grid.shape[0]} segments x {hf_txt_grid.shape[1]} temps")
    print(f"  XLSX interpolated: {hf_xlsx_grid.shape[0]} segments x {hf_xlsx_grid.shape[1]} temps")

    # ── Step 5: Compute statistics ──
    print("\nStep 5: Computing comparison statistics...")
    stats_all = []

    # Per-pair comparison
    n_pairs = min(hf_txt_grid.shape[0], hf_xlsx_grid.shape[0])
    for i in range(n_pairs):
        stats = compute_stats(hf_txt_grid[i], hf_xlsx_grid[i], T_grid,
                              label=f"Pair {i+1}")
        stats_all.append(stats)

    # Mean comparison
    txt_mean = np.nanmean(hf_txt_grid, axis=0)
    xlsx_mean = np.nanmean(hf_xlsx_grid, axis=0)
    stats_mean = compute_stats(txt_mean, xlsx_mean, T_grid, label="Mean")
    stats_all.append(stats_mean)

    # Print statistics
    print("\n" + "-" * 65)
    print("Comparison Statistics (XLSX − TXT)")
    print("-" * 65)

    for s in stats_all:
        if s['label'] == 'Mean':
            print(f"\n  [{s['label']}]")
            print(f"    Mean difference:     {s['mean_diff']:+.4f} W/g")
            print(f"    RMS difference:      {s['rms_diff']:.4f} W/g")
            print(f"    Max absolute diff:   {s['max_abs_diff']:.4f} W/g")
            print(f"    Std of difference:   {s['std_diff']:.4f} W/g")
            print(f"    Pre-Tg (35-75°C):    {s['pre_tg_mean']:+.4f} W/g")
            print(f"    Tg region (80-120°C): {s['tg_mean']:+.4f} W/g")
            print(f"    Post-Tg (125-195°C):  {s['post_tg_mean']:+.4f} W/g")

    # ── Step 6: Generate plots ──
    print("\nStep 6: Generating comparison plots...")
    plot_path = os.path.join(RESULTS_DIR, 'ps_comparison_results.png')
    plot_comparison(txt_segments, xlsx_segments, T_grid,
                    hf_txt_grid, hf_xlsx_grid, stats_all, plot_path)

    # ── Step 7: Save data ──
    print("\nStep 7: Saving comparison data...")

    # Interpolated data CSV
    data_csv_path = os.path.join(RESULTS_DIR, 'ps_comparison_data.csv')
    csv_data = {'Temperature_C': T_grid}

    for i in range(hf_txt_grid.shape[0]):
        csv_data[f'Mettler_H{i+1}_Wg'] = hf_txt_grid[i]
    csv_data['Mettler_Mean_Wg'] = np.nanmean(hf_txt_grid, axis=0)
    csv_data['Mettler_Std_Wg'] = np.nanstd(hf_txt_grid, axis=0)

    for i in range(hf_xlsx_grid.shape[0]):
        csv_data[f'OtherDSC_H{i+1}_Wg'] = hf_xlsx_grid[i]
    csv_data['OtherDSC_Mean_Wg'] = np.nanmean(hf_xlsx_grid, axis=0)
    csv_data['OtherDSC_Std_Wg'] = np.nanstd(hf_xlsx_grid, axis=0)

    csv_data['Diff_Mean_Wg'] = np.nanmean(hf_xlsx_grid, axis=0) - np.nanmean(hf_txt_grid, axis=0)

    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(data_csv_path, index=False)
    print(f"  Data saved to: {data_csv_path}")

    # Statistics CSV
    stats_csv_path = os.path.join(RESULTS_DIR, 'ps_comparison_stats.csv')
    df_stats = pd.DataFrame(stats_all)
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"  Stats saved to: {stats_csv_path}")

    # ── Summary ──
    s = stats_mean
    print("\n" + "=" * 65)
    print("ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"""
  Both instruments measured identical PS samples under the same
  temperature program (30↔200°C at ~60°C/min).

  Systematic offset (Other DSC − Mettler):
    Overall:  {s['mean_diff']:+.4f} W/g
    Pre-Tg:   {s['pre_tg_mean']:+.4f} W/g
    Tg region:{s['tg_mean']:+.4f} W/g
    Post-Tg:  {s['post_tg_mean']:+.4f} W/g

  RMS difference: {s['rms_diff']:.4f} W/g
  Max deviation:  {s['max_abs_diff']:.4f} W/g

  NOTE: The Mettler STARe software may apply automatic baseline
  correction, while the XLSX data is raw instrument signal.
  This contributes to systematic offsets, particularly in the
  baseline regions. When comparing the glass transition shape
  (panel d in the figure), the baseline-subtracted curves
  isolate the material response from instrument calibration
  differences.
""")
    print("=" * 65)


if __name__ == '__main__':
    main()
