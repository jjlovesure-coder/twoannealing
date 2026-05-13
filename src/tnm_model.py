"""
TNM (Tool-Narayanaswamy-Moynihan) model for structural relaxation in glasses.

Based on Song et al. (2020) "Activation Entropy as a Key Factor Controlling
the Memory Effect in Glasses" and Moynihan et al. (1976).

Uses incremental reduced-time summation with Boltzmann superposition
for arbitrary temperature histories. Optimized with two-stage simulation
(separate ramp + hold simulation).
"""

import numpy as np

R_GAS = 8.314  # J/(mol·K)


class TNMModel:
    """TNM model for glass structural relaxation.

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
        Equilibrium fictive temperature reference (K).
    """

    def __init__(self, A, H_star, x, beta, T0):
        self.A = A
        self.H_star = H_star
        self.x = x
        self.beta = beta
        self.T0 = T0

    def tau(self, T, Tf):
        """Relaxation time at temperature T with fictive temperature Tf.

        Song et al. Eq. 3 / Suppl. Eq. 12:
        tau = A * exp[ x*H*/(R*T) + (1-x)*H*/(R*Tf) ]
        """
        exponent = (self.x * self.H_star) / (R_GAS * T) + \
                   ((1 - self.x) * self.H_star) / (R_GAS * Tf)
        return self.A * np.exp(exponent)

    def _kww(self, xi):
        """KWW stretched exponential: phi(xi) = exp(-xi^beta)."""
        return np.exp(-(xi ** self.beta))

    def _simulate_ramp(self, T_start, T_end, rate, T_f_init, xi_init=0.0):
        """Simulate a temperature ramp, return final state."""
        dT_total = abs(T_end - T_start)
        if dT_total < 0.1 or rate <= 0:
            return T_f_init, xi_init
        duration = dT_total / rate
        n_steps = max(10, int(np.ceil(dT_total / 2.0)))
        dt_step = duration / n_steps
        T_vals = np.linspace(T_start, T_end, n_steps + 1)[1:]

        T_f = np.zeros(n_steps + 1)
        xi = np.zeros(n_steps + 1)
        T_f[0] = T_f_init
        xi[0] = xi_init
        T_hist = np.concatenate([[T_start], T_vals])

        for i in range(1, n_steps + 1):
            T_i = T_vals[i - 1]
            T_f_prev = T_f[i - 1]
            tau_i = self.tau(T_i, T_f_prev)
            dxi = dt_step / tau_i
            xi[i] = xi[i - 1] + dxi
            T_f_i = T_i
            for j in range(i - 1, -1, -1):
                dxi_j = xi[i] - xi[j]
                kww_val = self._kww(dxi_j)
                if kww_val < 1e-8:
                    break
                dT = T_hist[j + 1] - T_hist[j]
                T_f_i -= dT * kww_val
            T_f[i] = T_f_i
        return T_f[-1], xi[-1]

    def _simulate_hold(self, T_hold, t_hold, T_f_init, xi_init=0.0):
        """Isothermal hold, returns (T_f_end, xi_end, T_f_start)."""
        if t_hold <= 0:
            return T_f_init, xi_init, T_f_init
        if t_hold < 1.0:
            n_steps = max(3, min(30, int(t_hold / 0.005) + 3))
            t_steps = np.linspace(0, t_hold, n_steps + 1)[1:]
        else:
            n_steps = max(8, min(50, int(np.log10(t_hold) * 20 + 10)))
            t_start_log = max(0.02, t_hold / 200)
            t_steps = np.logspace(np.log10(t_start_log),
                                  np.log10(t_hold), n_steps)
            t_steps = np.unique(np.round(t_steps, 8))

        n = len(t_steps)
        T_f = np.zeros(n + 1)
        xi = np.zeros(n + 1)
        T_f[0] = T_f_init
        xi[0] = xi_init

        t_prev = 0.0
        for i in range(n):
            dt = t_steps[i] - t_prev
            t_prev = t_steps[i]
            T_f_prev = T_f[i]
            tau_i = self.tau(T_hold, T_f_prev)
            dxi = dt / tau_i
            xi[i + 1] = xi[i] + dxi
            dxi_from_start = xi[i + 1] - xi[0]
            kww_val = self._kww(dxi_from_start)
            T_f[i + 1] = T_hold + (T_f_init - T_hold) * kww_val
        return T_f[-1], xi[-1], T_f[0]

    def simulate_one_step(self, T_anneal, t_hold, T_initial=473.15,
                          cooling_rate=1.0):
        """One-step annealing: cool -> hold.
        Returns (T_f_end, T_f_hold_start) in Kelvin.
        """
        T_f_after_cool, xi_after_cool = self._simulate_ramp(
            T_initial, T_anneal, cooling_rate, T_initial, 0.0)
        T_f_end, xi_end, T_f_hold_start = self._simulate_hold(
            T_anneal, t_hold, T_f_after_cool, xi_after_cool)
        return T_f_end, T_f_hold_start

    def simulate_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=473.15,
                          cooling_rate=1.0, rate_T1_to_T2=None):
        """Two-step annealing: cool->hold1->ramp->hold2.

        For down-jump (T2 < T1): ramp is cooling at cooling_rate.
        For up-jump (T2 > T1, e.g. Kovacs 80->90°C): use rate_T1_to_T2
        if provided (typically heating_rate = 0.1667 K/s).

        Returns (T_f_end_at_T2, T_f_start_of_hold2).
        """
        if rate_T1_to_T2 is None:
            rate_T1_to_T2 = cooling_rate
        T_f_1, xi_1 = self._simulate_ramp(T_initial, T1, cooling_rate,
                                           T_initial, 0.0)
        T_f_1e, xi_1e, _ = self._simulate_hold(T1, t1_hold, T_f_1, xi_1)
        T_f_2, xi_2 = self._simulate_ramp(T1, T2, rate_T1_to_T2, T_f_1e, xi_1e)
        T_f_2e, xi_2e, T_f_hold_start = self._simulate_hold(
            T2, t2_hold, T_f_2, xi_2)
        return T_f_2e, T_f_hold_start

    def delta_H_raw(self, T_anneal, t_hold, T_initial=473.15, cooling_rate=1.0):
        """One-step delta_H as (T0 - T_f_end) in Kelvin — for direct kJ/mol fitting."""
        T_f_end, T_f_start = self.simulate_one_step(
            T_anneal, t_hold, T_initial, cooling_rate)
        return self.T0 - T_f_end

    def delta_H_normalized(self, T_anneal, t_hold, T_initial=473.15,
                           cooling_rate=1.0):
        """One-step delta_H normalized to [0, 1] using T0 reference.
        delta_H = (T0 - T_f_end) / (T0 - T_anneal).
        """
        return self.delta_H_raw(T_anneal, t_hold, T_initial, cooling_rate) \
               / (self.T0 - T_anneal)

    def delta_H_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=473.15,
                         cooling_rate=1.0, rate_T1_to_T2=None):
        """Two-step delta_H as (T0 - T_f_end_at_T2) in Kelvin."""
        T_f_end, T_f_start = self.simulate_two_step(
            T1, t1_hold, T2, t2_hold, T_initial, cooling_rate, rate_T1_to_T2)
        return self.T0 - T_f_end

    def delta_H_two_step_normalized(self, T1, t1_hold, T2, t2_hold,
                                    T_initial=473.15, cooling_rate=1.0,
                                    rate_T1_to_T2=None):
        """Two-step delta_H normalized by (T0 - T2)."""
        return self.delta_H_two_step(T1, t1_hold, T2, t2_hold,
                                     T_initial, cooling_rate, rate_T1_to_T2) \
               / (self.T0 - T2)
