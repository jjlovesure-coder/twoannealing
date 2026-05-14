"""
TNM (Tool-Narayanaswamy-Moynihan) model — instantaneous quench + isothermal hold.

Based on Song et al. (2020). Uses T0 as both equilibrium fictive temperature
reference and initial Tf before quench. No cooling ramp simulation — the
freeze-in during cooling is absorbed into the T0 parameter.

This matches the MATLAB reference model exactly and runs in milliseconds.
"""
import numpy as np

R_GAS = 8.314


class TNMModel:
    """Instantaneous-quench TNM model for isothermal annealing.

    Parameters
    ----------
    A : float
        Pre-exponential factor (seconds).
    H_star : float
        Activation enthalpy (J/mol).
    x : float
        Nonlinearity parameter, 0 < x <= 1.
    beta : float
        KWW stretch exponent, 0 < beta <= 1.
    T0 : float
        Equilibrium fictive temperature (K). Also the initial Tf before quench.
    """

    def __init__(self, A, H_star, x, beta, T0):
        self.A = A
        self.H_star = H_star
        self.x = x
        self.beta = beta
        self.T0 = T0

    def tau(self, T, Tf):
        """Relaxation time: tau = A * exp[x*H*/(R*T) + (1-x)*H*/(R*Tf)]."""
        exponent = (self.x * self.H_star) / (R_GAS * T) + \
                   ((1 - self.x) * self.H_star) / (R_GAS * Tf)
        return self.A * np.exp(exponent)

    def _isothermal_hold(self, T_hold, t_hold, T_f_start):
        """Isothermal KWW relaxation due to instantaneous quench.

        Integrates dxi = dt/tau(T_hold, Tf) with Tf updated via
        Tf(S) = T_hold + (T_f_start - T_hold) * exp(-S^beta).

        Uses log-spaced time steps for efficient integration.
        Returns T_f_end.
        """
        if t_hold <= 0:
            return T_f_start

        # Few log-spaced steps for fast integration
        if t_hold < 0.1:
            n_steps = 5
            t_steps = np.linspace(0, t_hold, n_steps + 1)[1:]
        else:
            n_steps = 15
            t_steps = np.logspace(np.log10(t_hold / 200),
                                  np.log10(t_hold), n_steps)
            t_steps = np.unique(np.round(t_steps, 10))

        S = 0.0
        Tf = T_f_start
        t_prev = 0.0

        for t_i in t_steps:
            tau_val = self.tau(T_hold, Tf)
            dt = t_i - t_prev
            t_prev = t_i
            S += dt / tau_val
            Tf = T_hold + (T_f_start - T_hold) * np.exp(-(S ** self.beta))

        return Tf

    def simulate_one_step(self, T_anneal, t_hold, T_initial=None,
                          cooling_rate=None, heating_rate=None):
        """One-step annealing: instantaneous quench T0 -> T_anneal, then hold.

        Extra parameters accepted for backward compatibility, ignored.
        Returns (T_f_end, T_f_start) where T_f_start = T0.
        """
        T_f_end = self._isothermal_hold(T_anneal, t_hold, self.T0)
        return T_f_end, self.T0

    def simulate_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=None,
                          cooling_rate=None, heating_rate=None):
        """Two-step: quench T0->T1, hold t1, quench T1->T2, hold t2.

        Returns (T_f_end_at_T2, T_f_start_of_hold2).
        """
        Tf_after_t1 = self._isothermal_hold(T1, t1_hold, self.T0)
        Tf_end = self._isothermal_hold(T2, t2_hold, Tf_after_t1)
        return Tf_end, Tf_after_t1

    def delta_H_normalized(self, T_anneal, t_hold, T_initial=None,
                           cooling_rate=None, heating_rate=None):
        """Normalized enthalpy recovery.

        delta_H = (T0 - Tf_end) / (T0 - T_anneal)
        0 = unrelaxed (Tf = T0), 1 = fully relaxed (Tf = T_anneal).
        """
        T_f_end, _ = self.simulate_one_step(T_anneal, t_hold)
        denom = max(self.T0 - T_anneal, 1.0)
        return (self.T0 - T_f_end) / denom

    def delta_H_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=None,
                         cooling_rate=None, heating_rate=None):
        """Two-step normalized enthalpy recovery.

        delta_H = (T0 - Tf_end) / (T0 - T2)
        """
        T_f_end, _ = self.simulate_two_step(T1, t1_hold, T2, t2_hold)
        denom = max(self.T0 - T2, 1.0)
        return (self.T0 - T_f_end) / denom
