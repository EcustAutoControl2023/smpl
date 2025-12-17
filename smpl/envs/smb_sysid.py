"""
Lightweight SMB system-identification surrogates.

The N4SID surrogate mirrors the reference interface without requiring Matlab. It
provides scaled state/action bounds and simple linear dynamics suitable for MPC
comparisons.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .smbenv import SMBModel, _zero_mean_descale, _zero_mean_scale

# Paths for reference sysid imports
REF_ROOT = Path(__file__).resolve().parents[2] / "ref" / "Quantitative-Comparison-of-RL-and-MPC"


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


class MatlabNNARXSurrogate:
    """
    Wrapper around the reference NNARX sysid (TensorFlow) with SMBModel compatibility.
    Generates data from the SMB model, trains the NNARX, and exposes dynamic/observe models.
    """

    def __init__(self, plant: SMBModel, data_length: int = 2000, seed: int = 0):
        self.plant = plant
        self.seed = seed
        self.np_dtype = plant.np_dtype

        import sys

        if str(REF_ROOT) not in sys.path:
            sys.path.insert(0, str(REF_ROOT))

        from smb_config import SmbConfig  # noqa: E402
        from Sysids import sysid as ref_sysid  # noqa: E402
        from Utility import utility as ut  # noqa: E402

        self.config = SmbConfig()
        self.config.seed = seed
        self.config.sysid_method = "NNARX"
        self.sysid = ref_sysid.SYSID(plant=self.plant, config=self.config)

        rng = np.random.default_rng(seed)
        steps = data_length
        x = np.zeros((steps + 1, self.plant.x_dim), dtype=self.np_dtype)
        y = np.zeros((steps + 1, self.plant.y_dim), dtype=self.np_dtype)
        u = np.zeros((steps + 1, self.plant.u_dim), dtype=self.np_dtype)
        p = np.zeros((steps + 1, self.plant.p_dim), dtype=self.np_dtype)
        r = np.zeros((steps + 1, self.plant.r_dim), dtype=self.np_dtype)
        x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = self.plant.do_reset()

        random_signal = ut.random_step_signal_generation(
            signal_dim=self.plant.u_dim,
            signal_length=steps + 1,
            signal_min=self.plant.u_min,
            signal_max=self.plant.u_max,
            signal_interval=self.config.SYSID_signal_interval,
        )

        for k in range(steps):
            r[k, :] = self.plant.ref
            u[k, :] = random_signal[k, :]
            p[k, :] = self.plant.get_feed_concentration()
            x[k + 1, :] = self.plant.go_step(x[k, :], u[k, :])
            y[k + 1, :] = self.plant.get_observation(x[k + 1, :], u[k, :])

        self.sysid.add_data_and_scale(u, y)
        self.sysid.do_identification(data_length)

        self.dynamic_model = self.sysid.dynamic_model
        self.observe_model = self.sysid.observe_model
        self.ini_x = self.sysid.ini_x
        self.x_est_dim = self.sysid.x_est_dim
        self.x_est_min = self.sysid.x_est_min
        self.x_est_max = self.sysid.x_est_max
        self.u_bias = self.sysid.u_bias
        self.u_scale = self.sysid.u_scale
        self.y_bias = self.sysid.y_bias
        self.y_scale = self.sysid.y_scale
        self.u_dim = self.plant.u_dim
        self.u_min = self.plant.u_min
        self.u_max = self.plant.u_max

    def initial_control(self, _x):
        return self.plant.ss_u
