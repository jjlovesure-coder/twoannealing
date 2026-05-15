"""
输出原始热流曲线：空坩埚基线 + 各实验所有升温段的 DSC 原始信号
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
OUT_DIR = os.path.join(ROOT_DIR, 'results', 'enthalpy')
os.makedirs(OUT_DIR, exist_ok=True)

CONV = 6 / 4700  # J/g per µW·°C


def load_raw(filename, sheet=None):
    if sheet:
        df = pd.read_excel(filename, sheet_name=sheet)
    else:
        df = pd.read_excel(filename)
    for i in range(len(df)):
        v0 = df.iloc[i, 0]
        if v0 is not None and isinstance(v0, str) and 'Time' in str(v0):
            hr = i
            break
    raw = df.iloc[hr + 2:].copy()
    data = pd.DataFrame({
        'Time': pd.to_numeric(raw.iloc[:, 0], errors='coerce'),
        'Temp': pd.to_numeric(raw.iloc[:, 1], errors='coerce'),
        'DSC':  pd.to_numeric(raw.iloc[:, 2], errors='coerce'),
    }).dropna().reset_index(drop=True).astype(float)
    return data


def detect_heating_ramps(T, t):
    n = len(T)
    window = 30
    dTdt = np.zeros(n)
    for i in range(window, n - window):
        dTdt[i] = (T[i + window] - T[i - window]) / max(t[i + window] - t[i - window], 1e-12)
    is_h = dTdt > 4
    segs = []
    in_seg = False
    start = 0
    for i in range(n):
        if is_h[i] and not in_seg:
            in_seg = True
            start = i
        elif not is_h[i] and in_seg:
            if i - start > 500:
                seg_T = T[start:i]
                if np.min(seg_T) < 40 and np.max(seg_T) > 180:
                    segs.append((start, i - 1))
            in_seg = False
    return segs


# ── 空坩埚 ──
empty = load_raw(os.path.join(DATA_DIR, 'ps-empty-01.xlsx'))
mask = (empty['Temp'] >= 28) & (empty['Temp'] <= 170)
T_empty_plot = empty['Temp'][mask].values
DSC_empty_plot = empty['DSC'][mask].values

# ── 图 1: 空坩埚 ──
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(T_empty_plot, DSC_empty_plot, 'k-', linewidth=0.8)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('DSC (µW)')
ax1.set_title('Empty Crucible Baseline (ps-empty-01)')
ax1.grid(True, alpha=0.3)
fig1.savefig(os.path.join(OUT_DIR, 'raw_empty_crucible.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved raw_empty_crucible.png")


def plot_experiment(data_file, sheet, label, out_name, t2_times, t2_labels, n_ramps=20):
    """为单个实验绘制所有升温段的原始 DSC 曲线"""
    data = load_raw(data_file, sheet=sheet)
    T_all = data['Temp'].values
    DSC_all = data['DSC'].values
    t_all = data['Time'].values
    ramps = detect_heating_ramps(T_all, t_all)
    print(f"  {label}: {len(ramps)} ramps detected")

    # 全范围图
    fig, (ax_full, ax_tg) = plt.subplots(1, 2, figsize=(18, 7))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, min(n_ramps, len(ramps))))

    for i in range(min(n_ramps, len(ramps))):
        s, e = ramps[i]
        T_seg = T_all[s:e + 1]
        DSC_seg = DSC_all[s:e + 1]
        lbl = t2_labels[i] if i < len(t2_labels) else f'R{i+1}'
        ax_full.plot(T_seg, DSC_seg, color=colors[i], linewidth=0.6, alpha=0.8, label=lbl)
        ax_tg.plot(T_seg, DSC_seg, color=colors[i], linewidth=0.6, alpha=0.8, label=lbl)

    ax_full.set_xlabel('Temperature (°C)')
    ax_full.set_ylabel('DSC (µW)')
    ax_full.set_title(f'{label} — Full Range (all {len(ramps)} ramps)')
    ax_full.set_xlim(28, 202)
    ax_full.grid(True, alpha=0.2)

    ax_tg.set_xlabel('Temperature (°C)')
    ax_tg.set_ylabel('DSC (µW)')
    ax_tg.set_title(f'{label} — Tg Region (70–120°C)')
    ax_tg.set_xlim(70, 120)
    ax_tg.legend(fontsize=6, ncol=2, loc='lower right')
    ax_tg.grid(True, alpha=0.2)

    fig.savefig(os.path.join(OUT_DIR, f'raw_{out_name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved raw_{out_name}.png")

    # ── 输出积分值表格 ──
    print(f"\n  {label} — Integrals (30-105°C):")
    print(f"  {'Ramp':<6} {'T2':>10} {'integral':>14} {'|ΔH|(J/g)':>12}")
    for i in range(min(n_ramps, len(ramps))):
        s, e = ramps[i]
        T_seg = T_all[s:e + 1]
        DSC_seg = DSC_all[s:e + 1]
        mask_i = (T_seg >= 30) & (T_seg <= 105)
        integral = trapezoid(DSC_seg[mask_i], T_seg[mask_i])
        abs_H = -integral * CONV
        t2 = t2_times[i] if i < len(t2_times) else '?'
        print(f"  {i+1:<4d}  {str(t2):>10}  {integral:>14.1f}  {abs_H:>12.4f}")


# ── Onestep 50°C ──
t2_all = [0.1002, 0.4998, 0.7980, 1.0002, 4.998, 10.02, 49.98, 100.02, 499.98, 1000.02] * 2
t2_lbl = [f'{t:.0f}s' if t >= 1 else f'{t:.1f}s' for t in t2_all]
plot_experiment(os.path.join(DATA_DIR, 'PS-onestep-01.xlsx'), 'PS-onestep-01',
                'One-step 50°C', 'onestep_50C', t2_all, t2_lbl)

# ── Onestep 70°C ──
plot_experiment(os.path.join(DATA_DIR, 'PS-onestep-02.xlsx'), 'PS-onestep-02',
                'One-step 70°C', 'onestep_70C', t2_all, t2_lbl)

# ── Twosteps ──
plot_experiment(os.path.join(DATA_DIR, 'twosteps.xlsx'), 'PS-02',
                'Two-step (90→80°C)', 'twosteps', t2_all, t2_lbl)

# ── Kovacs ──
plot_experiment(os.path.join(DATA_DIR, 'pskovacs.xlsx'), 'PS-kovacs-01',
                'Kovacs (80→90°C)', 'kovacs', t2_all, t2_lbl)

print(f"\nAll raw curve plots saved to {OUT_DIR}/")
