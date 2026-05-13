# Annealing Data Processing — Empty-Crucible Subtracted Direct Integration

## Goal

Replace the current "direct difference vs shortest-hold reference" method with absolute
enthalpy calculation: subtract empty-crucible heat flow from sample heat flow, then
integrate directly over the Tg recovery region.

## Background

Current processing uses per-group reference ramps (shortest-hold) as baseline, yielding
relative ΔH values. The three-step cp method described in the README requires sapphire
reference data, but the user prefers a simpler approach: empty-crucible subtraction
followed by direct integration.

## Processing Pipeline

```
empty_crucible_data ──┐
                       ├── DSC_sample - DSC_empty ──→ integrate ──→ ΔH (kJ/mol)
sample_data ──────────┘
```

### Per-experiment steps:

1. Load sample data (one-step, two-step, or Kovacs) and detect heating ramps
2. Load empty crucible data (single ramp, 30→200°C)
3. For each sample heating ramp:
   a. Interpolate empty DSC to sample temperature grid
   b. Subtract: `DSC_corrected = DSC_sample - DSC_empty`
   c. Integrate corrected DSC over the chosen temperature range
   d. Convert integrated area to kJ/mol

## Two Temperature-Range Schemes

### Scheme ① — Dynamic upper bound (T_onset)

- Lower bound: 30°C (fixed)
- Upper bound: **T_onset** — detected per-curve as the entry point to the supercooled
  liquid regime. Defined as the temperature above Tg where the baseline-subtracted
  heat flow returns to a stable post-Tg linear trend.
- Detection: fit sigmoid-blended baseline, then scan downward from 150°C to find where
  the excess signal first drops below a noise threshold.

### Scheme ② — Fixed range

- Lower bound: 30°C
- Upper bound: 100°C (covers the main Tg recovery peak for PS at 10°C/min)

Both schemes output to the same results files for side-by-side comparison.

## Validation Criteria

1. **Kovacs hump**: Group A and B ΔH vs T2 curves should show the characteristic
   non-monotonic "dip then rise" pattern (overshoot enthalpy).
2. **One-step monotonic trend**: ΔH should increase monotonically with log(hold time),
   consistent with physical aging theory.
3. **Two-step consistency**: Within each group, longer T2 → larger ΔH release.

## Code Changes

### `src/process_shared.py` — new utilities

- `load_empty_crucible(path)` — load and extract the single heating ramp from the
  empty crucible DSC file
- `subtract_empty(dsc_sample, T_sample, dsc_empty, T_empty)` — interpolate empty
  crucible data to sample temperature grid and subtract
- `detect_t_onset(T, dsc_corrected)` — detect upper integration bound per curve
- `CONV_FACTOR`, `CONV_KJMOL` — already exist

### `src/process_onestep.py` — refactor

- Use empty-crucible subtraction instead of per-run reference difference
- Output both Scheme ① and Scheme ② results
- Dual-panel ΔH vs t plot with both schemes overlaid
- Save two sets of ΔH columns in CSV

### `src/process_twosteps.py` — refactor

- Same as above, adapted for two-step protocol mapping

### `src/process_kovacs.py` — refactor

- Same as above, adapted for Kovacs protocol mapping

## Output Files

Each script produces:
- `results/dsc/{name}_results.png` — multi-panel figure with both schemes
- `results/dsc/{name}_results.csv` — ΔH values for both schemes

## Non-Goals

- TNM model re-fitting (uses existing ΔH CSV outputs, transparent to method change)
- Instrument comparison
- Three-step cp calibration (sapphire reference not used)
