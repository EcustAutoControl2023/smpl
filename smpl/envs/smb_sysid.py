"""
Lightweight SMB system-identification surrogates.

The N4SID surrogate mirrors the reference interface without requiring Matlab. It
provides scaled state/action bounds and simple linear dynamics suitable for MPC
comparisons.
"""

from __future__ import annotations

import numpy as np

from .smbenv import SMBModel, _zero_mean_descale, _zero_mean_scale


class N4SIDSurrogate:
    """Minimal N4SID-like model that preserves reference scaling."""

    def __init__(self, plant: SMBModel, x_dim: int | None = None, sysid_samples: int = 200):
        self.plant = plant
        self.x_dim = plant.y_dim if x_dim is None else x_dim
        self.np_dtype = plant.np_dtype
        self.x_min = np.zeros(self.x_dim, dtype=self.np_dtype)
        self.x_max = np.ones(self.x_dim, dtype=self.np_dtype)
        self.u_dim = plant.u_dim
        self.u_min = plant.u_min
        self.u_max = plant.u_max
        self.x_est_dim = self.x_dim
        self.x_est_min = self.x_min
        self.x_est_max = self.x_max
        self.ini_x = np.zeros(self.x_dim, dtype=self.np_dtype)

        # fit a simple linear predictor on simulated data (y as state)
        self.a = np.eye(self.x_dim, dtype=self.np_dtype)
        self.b = np.zeros((self.x_dim, self.u_dim), dtype=self.np_dtype)
        self.bias = np.zeros(self.x_dim, dtype=self.np_dtype)
        self.c = np.eye(self.x_dim, dtype=self.np_dtype)
        self.d = np.zeros((self.x_dim, self.u_dim), dtype=self.np_dtype)
        self._fit_linear_model(sysid_samples)

    def dynamic_model(self, x, u, for_casadi: bool = False):
        # linear state update, keep bounds via scaling helpers when casadi types are used
        if for_casadi:
            x_scaled = _zero_mean_descale(x, self.x_min, self.x_max)
            u_scaled = _zero_mean_descale(u, self.u_min, self.u_max)
            x_next = self.a @ x_scaled + self.b @ u_scaled + self.bias
            return _zero_mean_scale(x_next, self.x_min, self.x_max)
        x_next = self.a.dot(x) + self.b.dot(u) + self.bias
        return np.clip(x_next, self.x_min, self.x_max)

    def observe_model(self, x, u=None, for_casadi: bool = False):
        if for_casadi:
            x_scaled = _zero_mean_descale(x, self.x_min, self.x_max)
            u_scaled = _zero_mean_descale(u, self.u_min, self.u_max)
            return self.c @ x_scaled + self.d @ u_scaled
        x_scaled = x
        u_scaled = u if u is not None else np.zeros(self.u_dim, dtype=self.np_dtype)
        return self.c.dot(x_scaled) + self.d.dot(u_scaled)

    def initial_control(self, _x):
        return self.plant.ss_u

    def _fit_linear_model(self, samples: int):
        # generate data
        x, u, y, p, r = self.plant.reset()
        ys = []
        us = []
        yps = []
        state = x
        obs = y
        for _ in range(samples):
            action = np.random.uniform(self.u_min, self.u_max)
            ys.append(obs.copy())
            us.append(action.copy())
            state = self.plant.step(state, action)
            obs = self.plant.observe(state, action)
            yps.append(obs.copy())
        ys = np.asarray(ys)
        us = np.asarray(us)
        yps = np.asarray(yps)

        reg = np.hstack([ys, us, np.ones((samples, 1))])
        theta, *_ = np.linalg.lstsq(reg, yps, rcond=None)
        a_b = theta[:-1, :]
        self.a = a_b[: self.x_dim, :].T
        self.b = a_b[self.x_dim :, :].T
        self.bias = theta[-1, :].astype(self.np_dtype)

