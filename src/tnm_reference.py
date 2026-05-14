"""
Pure isothermal TNM model — exactly reproduces MATLAB reference formulas.
No cooling ramps. Instantaneous temperature jumps only.

TNM equations (Song et al. 2020):
  tau = exp(log(A) + x*H*/(R*T) + (1-x)*H*/(R*Tf))
  Tf(t) = Ta + (T0 - Ta) * exp(-S(t)^beta)   [one-step]
  Tf(t2) = T2 + (T0-T1)*exp(-(S1+S2)^beta) + (T1-T2)*exp(-S2^beta)  [two-step]
"""
import numpy as np

R_GAS = 8.314


def make_tau_func(A, H_star, x):
    """Return tau(T, Tf) function with given parameters."""
    logA = np.log(A)
    def tau(T, Tf):
        exponent = (x * H_star) / (R_GAS * T) + \
                   ((1 - x) * H_star) / (R_GAS * Tf)
        return np.exp(logA + exponent)
    return tau


def one_step(T0, Ta, t_array, A, H_star, x, beta):
    """One-step isothermal annealing.

    Starts at Tf(0) = T0, instant quench to Ta, isothermal hold.
    Returns Tf(t) array.
    """
    tau_func = make_tau_func(A, H_star, x)
    Tf = np.zeros(len(t_array))
    Tf[0] = T0
    S = 0.0

    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i - 1]
        tau_val = tau_func(Ta, Tf[i - 1])
        S += dt / tau_val
        Tf[i] = Ta + (T0 - Ta) * np.exp(-(S ** beta))

    return Tf


def two_step_hi_to_lo(T0, T1, t1_total, T2, t2_array, A, H_star, x, beta):
    """Two-step hi-to-lo: T0 -> T1 (hold t1) -> T2 (hold t2).

    Returns Tf(t2) array for the second hold.
    """
    tau_func = make_tau_func(A, H_star, x)

    # Step 1: hold at T1
    n1 = 500
    t1_sim = np.linspace(0, t1_total, n1)
    Tf_step1 = T0
    S1 = 0.0
    for i in range(1, n1):
        dt = t1_sim[i] - t1_sim[i - 1]
        tau_val = tau_func(T1, Tf_step1)
        S1 += dt / tau_val
        Tf_step1 = T1 + (T0 - T1) * np.exp(-(S1 ** beta))

    # Step 2: hold at T2
    Tf = np.zeros(len(t2_array))
    Tf[0] = Tf_step1
    S2 = 0.0
    for j in range(1, len(t2_array)):
        dt = t2_array[j] - t2_array[j - 1]
        tau_val = tau_func(T2, Tf[j - 1])
        S2 += dt / tau_val
        Tf[j] = T2 + (T0 - T1) * np.exp(-((S1 + S2) ** beta)) + \
                (T1 - T2) * np.exp(-(S2 ** beta))

    return Tf


def two_step_lo_to_hi(T0, T1, t1_total, T2, t2_array, A, H_star, x, beta):
    """Two-step lo-to-hi (Kovacs up-jump): T0 -> T1 (hold t1) -> T2 (hold t2).

    T1 < T2 (up-jump). Same formula as hi-to-lo -- it's general.
    """
    return two_step_hi_to_lo(T0, T1, t1_total, T2, t2_array, A, H_star, x, beta)
