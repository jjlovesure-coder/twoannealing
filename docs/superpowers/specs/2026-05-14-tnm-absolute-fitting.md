# TNM Absolute Enthalpy Fitting with Extended-Time Prediction

## Goal

Fit TNM model directly to absolute ΔH (kJ/mol), then extend annealing time to 10^5s to verify plateau convergence between paired experimental conditions.

## Phase 1: One-step 70°C

- **Data:** `onestep_70C_v2_results.csv`, run1 (rows 1-10) and run2 (rows 11-20) separately
- **Model:** Instantaneous-quench TNM, 6 parameters: `logA, H*, x, beta, T0, scale`
- **Cost:** `Σ (scale*(T0-Tf) - ΔH_exp)²` across 10 hold times
- **Output:** Single 4-panel figure:
  - Panel 1: Run1 fit + extended prediction to 10^5s, absolute ΔH (kJ/mol) vs log(t)
  - Panel 2: Run2 fit + extended prediction to 10^5s
  - Panel 3: Both runs overlaid, annotated plateau value
  - Panel 4: Tf vs log(t) for both runs
- **Validation:** Run1 and Run2 plateau at same absolute ΔH value

## Phase 2: Two-step 90°C→80°C

- **Data:** `twosteps_v2_results.csv`, Grp A (T1=0.833min) and Grp B (T1=8.333min)
- **Physics:** Two-step ΔH follows S-shaped curve: slow initial increase (memory of T1 pre-annealing) → rapid increase (memory fades, structure responds to T2) → plateau (equilibrium at T2). This differs from one-step's monotonic approach to plateau. The TNM model captures this via the double-exponential two-step formula where the two KWW terms ((T0-T1)*exp(-(S1+S2)^β) and (T1-T2)*exp(-S2^β)) decay at different rates.
- **Model:** Same instantaneous-quench TNM, 6 parameters, fit jointly to both groups
- **Cost:** `Σ (ΔH_model - ΔH_exp)²` across all 20 two-step points
- **Output:** Single 4-panel figure:
  - Panel 1: Grp A fit + extended prediction to 10^5s
  - Panel 2: Grp B fit + extended prediction to 10^5s
  - Panel 3: Both groups overlaid, annotated plateau value
  - Panel 4: Tf vs log(t) for both groups
- **Validation:** Grp A and Grp B converge to same plateau

## Architecture

Two new scripts, reusing existing modules:
- `src/fit_onestep_70C_absolute.py` — Phase 1
- `src/fit_twostep_absolute.py` — Phase 2
- Reuse: `src/tnm_model.py` (instantaneous-quench TNM), `src/tnm_conditions.py` (data loading)

## Key Settings

- Hold times: experimental 0.1-1000s, extended prediction to 10^5s
- Optimization: scipy L-BFGS-B, 25 multi-starts
- Bounds: logA [-25,-12], H* [50-300 kJ/mol], x [0.005-0.9], beta [0.01-0.9], T0 [370-420K], scale [0.1-1000 kJ·mol⁻¹·K⁻¹]
