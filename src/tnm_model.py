"""
TNM (Tool-Narayanaswamy-Moynihan) model for structural relaxation in glasses.

Based on Song et al. (2020) "Activation Entropy as a Key Factor Controlling
the Memory Effect in Glasses" and Moynihan et al. (1976).

Implements finite-rate cooling/heating with correct Boltzmann superposition
throughout the full thermal history.
"""

import numpy as np

R_GAS = 8.314  # J/(mol*K)


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
        Equilibrium fictive temperature reference (K), typically near Tg.
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

    @staticmethod
    def _kww(xi, beta):
        """KWW stretched exponential: phi(xi) = exp(-xi^beta)."""
        return np.exp(-(xi ** beta))

    @staticmethod
    def _compute_Tf(T_hist, xi_hist, beta):
        """Boltzmann superposition: compute Tf from full temperature + reduced time history.

        Tf(T_n, xi_n) = T_n - SUM_j (T_{j+1} - T_j) * exp(-(xi_n - xi_j)^beta)
        """
        n = len(xi_hist) - 1
        xi_n = xi_hist[n]
        Tf = T_hist[n]
        for j in range(n - 1, -1, -1):
            dxi = xi_n - xi_hist[j]
            kww_val = TNMModel._kww(dxi, beta)
            if kww_val < 1e-10:
                break
            dT = T_hist[j + 1] - T_hist[j]
            Tf -= dT * kww_val
        return Tf

    def _simulate_ramp(self, T_start, T_end, rate, T_f_init, xi_init=0.0):
        """Simulate a temperature ramp with Boltzmann superposition.

        Returns (T_f_end, xi_end, T_hist, xi_hist).
        """
        dT_total = abs(T_end - T_start)
        if dT_total < 0.1 or rate <= 0:
            T_hist = np.array([T_start])
            xi_hist = np.array([xi_init])
            return T_f_init, xi_init, T_hist, xi_hist

        duration = dT_total / rate
        n_steps = max(10, int(np.ceil(dT_total / 2.0)))
        T_vals = np.linspace(T_start, T_end, n_steps + 1)[1:]

        T_hist = np.zeros(n_steps + 1)
        xi_hist = np.zeros(n_steps + 1)
        T_hist[0] = T_start
        xi_hist[0] = xi_init

        for i in range(n_steps):
            T_i = T_vals[i]
            T_f_prev = self._compute_Tf(T_hist[:i + 1], xi_hist[:i + 1], self.beta)
            tau_i = self.tau(T_i, T_f_prev)
            dt = duration / n_steps
            xi_hist[i + 1] = xi_hist[i] + dt / tau_i
            T_hist[i + 1] = T_i

        T_f_end = self._compute_Tf(T_hist, xi_hist, self.beta)
        return T_f_end, xi_hist[-1], T_hist, xi_hist

    def _simulate_hold(self, T_hold, t_hold, T_f_init, xi_init=0.0,
                       T_hist_prev=None, xi_hist_prev=None):
        """Isothermal hold with correct Boltzmann superposition.

        If T_hist_prev/xi_hist_prev are provided, the hold continues from
        the preceding thermal history.

        Returns (T_f_end, xi_end, T_f_start, T_hist_full, xi_hist_full).
        """
        if T_hist_prev is not None and xi_hist_prev is not None:
            T_hist = np.array(list(T_hist_prev), dtype=float)
            xi_hist = np.array(list(xi_hist_prev), dtype=float)
        else:
            T_hist = np.array([T_f_init], dtype=float)
            xi_hist = np.array([xi_init], dtype=float)

        T_f_start = self._compute_Tf(T_hist, xi_hist, self.beta)

        if t_hold <= 0:
            return T_f_start, xi_init, T_f_start, T_hist, xi_hist

        if t_hold < 1.0:
            n_steps = max(3, min(30, int(t_hold / 0.005) + 3))
            t_steps = np.linspace(0, t_hold, n_steps + 1)[1:]
        else:
            n_steps = max(8, min(50, int(np.log10(t_hold) * 20 + 10)))
            t_start_log = max(0.02, t_hold / 200)
            t_steps = np.logspace(np.log10(t_start_log),
                                  np.log10(t_hold), n_steps)
            t_steps = np.unique(np.round(t_steps, 8))

        n_existing = len(T_hist)
        T_hist = np.concatenate([T_hist, np.full(len(t_steps), T_hold)])
        xi_hist = np.concatenate([xi_hist, np.zeros(len(t_steps))])

        t_prev = 0.0
        for k, t_k in enumerate(t_steps):
            idx = n_existing + k
            Tf_prev = self._compute_Tf(T_hist[:idx], xi_hist[:idx], self.beta)
            tau_i = self.tau(T_hold, Tf_prev)
            dt = t_k - t_prev
            t_prev = t_k
            xi_hist[idx] = xi_hist[idx - 1] + dt / tau_i

        T_f_end = self._compute_Tf(T_hist, xi_hist, self.beta)
        return T_f_end, xi_hist[-1], T_f_start, T_hist, xi_hist

    def simulate_one_step(self, T_anneal, t_hold, T_initial=473.15,
                          cooling_rate=1.0, heating_rate=None):
        """One-step annealing: cool -> hold.

        Returns (T_f_end, T_f_hold_start).
        """
        T_f_after_cool, xi_after_cool, T_hist_cool, xi_hist_cool = \
            self._simulate_ramp(T_initial, T_anneal, cooling_rate, T_initial, 0.0)
        T_f_end, xi_end, T_f_hold_start, _, _ = self._simulate_hold(
            T_anneal, t_hold, T_f_after_cool, xi_after_cool,
            T_hist_prev=T_hist_cool, xi_hist_prev=xi_hist_cool)
        return T_f_end, T_f_hold_start

    def simulate_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=473.15,
                          cooling_rate=1.0, heating_rate=None):
        """Two-step annealing: cool->hold1->cool->hold2.

        Returns (T_f_end_at_T2, T_f_start_of_hold2).
        """
        T_f_1, xi_1, T_hist_1, xi_hist_1 = self._simulate_ramp(
            T_initial, T1, cooling_rate, T_initial, 0.0)
        T_f_1e, xi_1e, _, T_hist_1e, xi_hist_1e = self._simulate_hold(
            T1, t1_hold, T_f_1, xi_1,
            T_hist_prev=T_hist_1, xi_hist_prev=xi_hist_1)
        T_f_2, xi_2, T_hist_2, xi_hist_2 = self._simulate_ramp(
            T1, T2, cooling_rate, T_f_1e, xi_1e)
        T_hist_full = np.concatenate([T_hist_1e, T_hist_2[1:]])
        xi_hist_full = np.concatenate([xi_hist_1e, xi_hist_2[1:]])
        T_f_2e, xi_2e, T_f_hold_start, _, _ = self._simulate_hold(
            T2, t2_hold, T_f_2, xi_2,
            T_hist_prev=T_hist_full, xi_hist_prev=xi_hist_full)
        return T_f_2e, T_f_hold_start

    def delta_H_normalized(self, T_anneal, t_hold, T_initial=473.15,
                           cooling_rate=1.0, heating_rate=None):
        """One-step delta_H normalized.

        delta_H = (T0 - Tf_end) / (T0 - T_anneal)
        0 = unrelaxed (Tf = T0), 1 = fully relaxed (Tf = T_anneal).
        Longer annealing -> larger Tf drop -> larger delta_H.
        """
        T_f_end, T_f_start = self.simulate_one_step(
            T_anneal, t_hold, T_initial, cooling_rate, heating_rate)
        return (self.T0 - T_f_end) / max(self.T0 - T_anneal, 1.0)

    def delta_H_two_step(self, T1, t1_hold, T2, t2_hold, T_initial=473.15,
                         cooling_rate=1.0, heating_rate=None):
        """Two-step delta_H normalized.

        delta_H = (T0 - Tf_end) / (T0 - T2)
        """
        T_f_end, T_f_start = self.simulate_two_step(
            T1, t1_hold, T2, t2_hold, T_initial, cooling_rate, heating_rate)
        return (self.T0 - T_f_end) / max(self.T0 - T2, 1.0)
