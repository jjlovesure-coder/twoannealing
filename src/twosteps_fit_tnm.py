#!/usr/bin/env python3
"""
Two-Step Down-Jump (90°C→80°C) — TNM with Cooling Ramps

Key insight: the cooling ramp from T0→T1 (~37s at 60 K/min) is comparable
to the short T1 hold (50s). Without simulating the ramp, Tf at T1 is
unphysically high for the 50s group, causing instant relaxation at T2.

Protocol:
  equil at T0+50 → cool to T1 at 60K/min → hold t1 → cool to T2 → hold t2
"""

import os, warnings, numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
R = 8.314

T1 = 363.15; T2 = 353.15
T1_50S = 49.98; T1_500S = 499.98
Q_COOL = 1.0  # 60 K/min


class TNM_Ramp:
    def __init__(self, A, H_star, x, beta, T0, q=1.0):
        self.A, self.H, self.x, self.b, self.T0, self.q = A, H_star, x, beta, T0, q

    def tau(self, T, Tf):
        e = self.x * self.H / (R * T) + (1 - self.x) * self.H / (R * Tf)
        return self.A * np.exp(np.clip(e, -50, 80))

    def _hold(self, T, t, Tf0):
        if t <= 0:
            return Tf0
        ts = np.unique(np.round(np.logspace(np.log10(t / 100), np.log10(t), 10), 10))
        S, Tf, tp = 0.0, Tf0, 0.0
        for ti in ts:
            S += (ti - tp) / self.tau(T, Tf)
            tp = ti
            Tf = T + (Tf0 - T) * np.exp(-(S ** self.b))
        return Tf

    def _ramp(self, Ta, Tb, rate, Tf0):
        N = max(int(abs(Tb - Ta) / 2.0), 6)
        Tv = np.linspace(Ta, Tb, N + 1)
        Tf = Tf0
        for i in range(N):
            dT = Tv[i + 1] - Tv[i]

            def rhs(T, f):
                tv = self.tau(T, f)
                if tv <= 0 or not np.isfinite(tv): return np.nan
                return (T - f) / (rate * tv)

            k1 = rhs(Tv[i], Tf)
            if np.isnan(k1): return np.nan
            k2 = rhs(Tv[i] + dT / 2, Tf + dT * k1 / 2)
            if np.isnan(k2): return np.nan
            k3 = rhs(Tv[i] + dT / 2, Tf + dT * k2 / 2)
            if np.isnan(k3): return np.nan
            k4 = rhs(Tv[i + 1], Tf + dT * k3)
            if np.isnan(k4): return np.nan
            Tf += dT * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if np.isnan(Tf): return np.nan
        return Tf

    def simulate(self, t1, t2):
        T_start = self.T0 + 50.0
        # Cool from above Tg → T1
        Tfc = self._ramp(T_start, T1, -self.q, T_start)
        if np.isnan(Tfc): return np.nan
        # Hold at T1
        Tf1 = self._hold(T1, t1, Tfc)
        if np.isnan(Tf1): return np.nan
        # Cool T1 → T2
        Tfu = self._ramp(T1, T2, -self.q, Tf1)
        if np.isnan(Tfu): return np.nan
        # Hold at T2
        Tf2 = self._hold(T2, t2, Tfu)
        if np.isnan(Tf2): return np.nan
        dH_norm = (self.T0 - Tf2) / max(self.T0 - T2, 1.0)
        if not np.isfinite(dH_norm): return np.nan
        return dH_norm


def load_targets():
    curve_path = os.path.join(RESULTS, 'twosteps_phenom_curve.csv')
    df_curve = pd.read_csv(curve_path)
    t_target = df_curve['t_s'].values
    dh_target_50 = df_curve['dh_50s_Jg'].values
    dh_target_500 = df_curve['dh_500s_Jg'].values

    exp_path = os.path.join(RESULTS, 'enthalpy', 'enthalpy_twosteps.csv')
    df_exp = pd.read_csv(exp_path)
    grp_50 = df_exp[df_exp['T1_group'] == '50s']
    grp_500 = df_exp[df_exp['T1_group'] == '500s']

    return (t_target, dh_target_50, dh_target_500,
            grp_50['T2_hold_s'].values, grp_50['delta_H_J_per_g'].values,
            grp_500['T2_hold_s'].values, grp_500['delta_H_J_per_g'].values)


def simulate_vector(params, t2_values, t1_hold):
    logA, H_star, x, beta, T0, scale = params
    m = TNM_Ramp(A=10**logA, H_star=H_star, x=x, beta=beta, T0=T0)
    dh = np.array([m.simulate(t1_hold, t) for t in t2_values])
    if np.any(np.isnan(dh)):
        return np.full(len(t2_values), np.nan)
    return scale * dh


def subsample_target(t_full, dh50_full, dh500_full, n_pts=40):
    idx = np.logspace(0, np.log10(len(t_full) - 1), n_pts).astype(int)
    idx = np.unique(np.clip(idx, 0, len(t_full) - 1))
    return t_full[idx], dh50_full[idx], dh500_full[idx]


def compute_cost(params, t_target, dh_target_50, dh_target_500):
    logA, H_star, x, beta, T0, scale = params
    if not (-40 < logA < -5):     return 1e10
    if not (30000 < H_star < 800000): return 1e10
    if not (0.01 < x < 0.99):     return 1e10
    if not (0.01 < beta < 0.95):  return 1e10
    if not (360 < T0 < 480):      return 1e10
    if not (0.5 < scale < 20):    return 1e10
    try:
        p50 = simulate_vector(params, t_target, T1_50S)
        p500 = simulate_vector(params, t_target, T1_500S)
    except Exception:
        return 1e10
    if np.any(np.isnan(p50)) or np.any(np.isnan(p500)):
        return 1e10
    return np.sum((p50 - dh_target_50) ** 2) + \
           np.sum((p500 - dh_target_500) ** 2)


def fit():
    t_full, dh_full_50, dh_full_500, t_exp_50, dh_exp_50, t_exp_500, dh_exp_500 = load_targets()

    # Subsample phenom curve for speed
    t_target, dh_target_50, dh_target_500 = subsample_target(
        t_full, dh_full_50, dh_full_500, n_pts=30)

    bounds = [
        (-38, -10),         # logA
        (50000, 500000),    # H_star J/mol
        (0.05, 0.70),       # x (low = strong Tf coupling for down-jump)
        (0.05, 0.80),       # beta
        (375, 450),         # T0 K
        (2.0, 8.0),         # scale J/g (H_max at 80°C ≈ 4-5)
    ]

    def cost_fn(p):
        return compute_cost(p, t_target, dh_target_50, dh_target_500)

    n_starts = 40
    print(f"Ramp-TNM: {n_starts} multi-starts ({len(t_target)} pts/group)...", flush=True)
    rng = np.random.RandomState(42)
    best_result, best_cost = None, np.inf

    for k in range(n_starts):
        x0 = [rng.uniform(low, high) for low, high in bounds]
        res = minimize(cost_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12})
        if res.fun < best_cost:
            best_cost, best_result = res.fun, res
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{n_starts}, best cost = {best_cost:.4f}", flush=True)

    print(f"  Final best cost = {best_cost:.4f}", flush=True)
    p = best_result.x
    logA, H_star, x, beta, T0, scale = p

    # Final predictions
    pred_50 = simulate_vector(p, t_full, T1_50S)
    pred_500 = simulate_vector(p, t_full, T1_500S)
    pred_exp_50 = simulate_vector(p, t_exp_50, T1_50S)
    pred_exp_500 = simulate_vector(p, t_exp_500, T1_500S)

    res_all = np.concatenate([pred_50 - dh_full_50, pred_500 - dh_full_500])
    rmse = np.sqrt(np.mean(res_all ** 2))

    ssr50 = np.sum((pred_50 - dh_full_50)**2)
    sst50 = np.sum((dh_full_50 - np.mean(dh_full_50))**2)
    r2_50p = 1 - ssr50 / sst50
    ssr500 = np.sum((pred_500 - dh_full_500)**2)
    sst500 = np.sum((dh_full_500 - np.mean(dh_full_500))**2)
    r2_500p = 1 - ssr500 / sst500

    ssr50e = np.sum((pred_exp_50 - dh_exp_50)**2)
    sst50e = np.sum((dh_exp_50 - np.mean(dh_exp_50))**2)
    r2_50e = 1 - ssr50e / sst50e
    ssr500e = np.sum((pred_exp_500 - dh_exp_500)**2)
    sst500e = np.sum((dh_exp_500 - np.mean(dh_exp_500))**2)
    r2_500e = 1 - ssr500e / sst500e

    print(f"\n{'='*60}")
    print(f"TWO-STEP TNM (cooling ramps)")
    print(f"{'='*60}")
    print(f"  logA   = {logA:.4f}    (A = {10**logA:.4e} s)")
    print(f"  H*     = {H_star/1000:.2f} kJ/mol")
    print(f"  x      = {x:.4f}")
    print(f"  β      = {beta:.4f}")
    print(f"  T0     = {T0:.2f} K  ({T0-273.15:.1f} °C)")
    print(f"  scale  = {scale:.4f} J/g")
    print(f"  RMSE phenom = {rmse:.4f}")
    print(f"  R² phenom   = 50s:{r2_50p:.4f}  500s:{r2_500p:.4f}")
    print(f"  R² exp      = 50s:{r2_50e:.4f}  500s:{r2_500e:.4f}")

    print(f"\n  {'t(s)':>8s}  {'Exp50s':>8s}  {'TNM_50':>8s}  {'Phe_50':>8s}  |  "
          f"{'Exp500s':>8s}  {'TNM500':>8s}  {'Phe500':>8s}")
    ph50 = np.interp(t_exp_50, t_full, dh_full_50)
    ph500 = np.interp(t_exp_500, t_full, dh_full_500)
    for i in range(10):
        print(f"  {t_exp_50[i]:8.3f}  {dh_exp_50[i]:8.4f}  {pred_exp_50[i]:8.4f}  "
              f"{ph50[i]:8.4f}  |  {dh_exp_500[i]:8.4f}  {pred_exp_500[i]:8.4f}  {ph500[i]:8.4f}")

    print(f"\n  {'t(s)':>10s}  {'TNM_50':>9s}  {'TNM_500':>9s}  {'diff':>8s}  {'Phe diff':>9s}")
    for t_ext in [1000, 2000, 5000, 10000, 50000, 100000]:
        vt50 = simulate_vector(p, [t_ext], T1_50S)[0]
        vt500 = simulate_vector(p, [t_ext], T1_500S)[0]
        vp50 = np.interp(t_ext, t_full, dh_full_50)
        vp500 = np.interp(t_ext, t_full, dh_full_500)
        print(f"  {t_ext:10.0f}  {vt50:9.4f}  {vt500:9.4f}  {vt500-vt50:8.4f}  {vp500-vp50:9.4f}")

    os.makedirs(os.path.join(RESULTS, 'tnm'), exist_ok=True)
    pd.DataFrame([{
        'logA': logA, 'A_s': 10**logA, 'H_star_kJmol': H_star/1000,
        'x': x, 'beta': beta, 'T0_K': T0, 'T0_C': T0 - 273.15,
        'scale_Jg': scale, 'rmse_phenom': rmse,
        'r2_50s_phenom': r2_50p, 'r2_500s_phenom': r2_500p,
        'r2_50s_exp': r2_50e, 'r2_500s_exp': r2_500e,
    }]).to_csv(os.path.join(RESULTS, 'tnm', 'tnm_twosteps_params.csv'), index=False)

    # ── Plot ──
    t_plt = np.logspace(-1, 6, 300)
    dh_ph50 = np.interp(t_plt, t_full, dh_full_50)
    dh_ph500 = np.interp(t_plt, t_full, dh_full_500)
    dh_t50 = simulate_vector(p, t_plt, T1_50S)
    dh_t500 = simulate_vector(p, t_plt, T1_500S)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(t_exp_50, dh_exp_50, c='#1f77b4', marker='o', s=70, zorder=10,
               edgecolors='k', lw=0.6, label='Exp 50s (T$_1$=50s)')
    ax.scatter(t_exp_500, dh_exp_500, c='#ff7f0e', marker='s', s=70, zorder=10,
               edgecolors='k', lw=0.6, label='Exp 500s (T$_1$=500s)')
    ax.semilogx(t_plt, dh_ph50, '#1f77b4', ls='--', lw=1.5, alpha=0.4,
                label='KWW fit 50s')
    ax.semilogx(t_plt, dh_ph500, '#ff7f0e', ls='--', lw=1.5, alpha=0.4,
                label='KWW fit 500s')
    ax.semilogx(t_plt, dh_t50, '#1f77b4', ls='-', lw=2.2, label='TNM 50s')
    ax.semilogx(t_plt, dh_t500, '#ff7f0e', ls='-', lw=2.2, label='TNM 500s')
    ax.axhline(y=scale, color='red', ls=':', lw=1.2, alpha=0.5)
    ax.axvline(x=1000, color='gray', ls='--', alpha=0.25)
    ax.set_xlabel('$t_2$ (s)', fontsize=13)
    ax.set_ylabel(r'$\Delta H$ (J/g)', fontsize=13)
    ax.set_title('Two-Step (90°C→80°C) — TNM with Cooling Ramps', fontsize=13, fontweight='bold')
    ax.set_xlim(5e-2, 1e6); ax.set_ylim(-0.3, scale * 1.15)
    ax.grid(True, alpha=0.2)

    ins = ax.inset_axes([0.55, 0.10, 0.42, 0.35])
    tz = np.logspace(-1, 1.3, 100)
    ins.semilogx(tz, simulate_vector(p, tz, T1_50S), '#1f77b4', ls='-', lw=1.5)
    ins.semilogx(tz, simulate_vector(p, tz, T1_500S), '#ff7f0e', ls='-', lw=1.5)
    ins.scatter(t_exp_50[t_exp_50<20], dh_exp_50[t_exp_50<20], c='#1f77b4', marker='o',
                s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.scatter(t_exp_500[t_exp_500<20], dh_exp_500[t_exp_500<20], c='#ff7f0e', marker='s',
                s=35, zorder=5, edgecolors='k', lw=0.3)
    ins.set_xlim(0.06, 20); ins.set_ylim(-0.05, scale * 0.75)
    ins.set_title('Short-time detail (0.06–20 s)', fontsize=8.5, pad=3)
    ins.grid(True, alpha=0.2); ins.tick_params(labelsize=7)
    zx0, zx1, zy0, zy1 = 0.06, 20, -0.1, scale * 0.75
    ax.add_patch(Rectangle((zx0, zy0), zx1-zx0, zy1-zy0, lw=1.5, ec='green',
                           fc='green', alpha=0.07, zorder=2))
    fig.add_artist(ConnectionPatch(xyA=(zx1, zy1), coordsA=ax.transData, xyB=(1, 1),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65,
                   arrowstyle='->', mutation_scale=12))
    fig.add_artist(ConnectionPatch(xyA=(zx0, zy0), coordsA=ax.transData, xyB=(0, 0),
                   coordsB=ins.transAxes, color='green', lw=1.2, alpha=0.65,
                   arrowstyle='->', mutation_scale=12))

    leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9,
                  markeredgecolor='k', markeredgewidth=0.5, label='Exp 50s'),
           Line2D([0],[0], marker='s', color='w', markerfacecolor='#ff7f0e', markersize=9,
                  markeredgecolor='k', markeredgewidth=0.5, label='Exp 500s'),
           Line2D([0],[0], color='#1f77b4', lw=2.2, label='TNM (ramp)'),
           Line2D([0],[0], color='#1f77b4', lw=1.5, ls='--', alpha=0.4, label='KWW fit')]
    ax.legend(handles=leg, fontsize=10, loc='lower right')

    ax.text(0.03, 0.97,
        f"TNM parameters (with cooling ramps)\n  log A = {logA:.2f}\n  H* = {H_star/1000:.1f} kJ/mol\n"
        f"  x = {x:.4f}\n  β = {beta:.4f}\n  T₀ = {T0:.1f} K\n"
        f"  scale = {scale:.2f} J/g\n"
        f"R² vs exp: 50s={r2_50e:.3f}  500s={r2_500e:.3f}",
        transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, ec='gray'))

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'tnm', 'twosteps_tnm_fit.png'), dpi=200)
    plt.close()
    print(f"\n  Plot saved to results/tnm/twosteps_tnm_fit.png")

    return p, rmse, (r2_50e, r2_500e)


if __name__ == '__main__':
    fit()
