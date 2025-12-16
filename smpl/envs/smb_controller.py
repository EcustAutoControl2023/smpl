"""
Baseline SMB controllers for the SMPL environment.

These controllers avoid Matlab dependencies by relying purely on the Python SMB model
and a lightweight surrogate system-identification model for MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import casadi as ca
import numpy as np

from .smbenv import SMBModel, _zero_mean_descale, _zero_mean_scale


def _mpc_stage_cost(
    ref: np.ndarray, y: ca.SX, u: ca.SX, extract_weight: float = 0.03, raffinate_weight: float = 0.03
):
    """Stage/terminal cost mirroring the reference SMB config."""
    puri_e = y[0] / (y[0] + y[1] + 1e-10)
    puri_r = y[3] / (y[2] + y[3] + 1e-10)
    u_cost = -1 * (u[2] - u[1])
    y_cost = extract_weight * (ref[0] - puri_e) + raffinate_weight * (ref[1] - puri_r)
    return 0.002 + y_cost + u_cost


@dataclass
class SMBBaselineSysId:
    """Minimal surrogate sysid used for MPC when full system-ID is unavailable."""

    plant: SMBModel

    def __post_init__(self):
        self.x_est_dim = self.plant.y_dim
        self.x_est_min = self.plant.y_min
        self.x_est_max = self.plant.y_max
        self.u_dim = self.plant.u_dim
        self.u_min = self.plant.u_min
        self.u_max = self.plant.u_max
        self.ini_x = np.zeros(self.x_est_dim, dtype=self.plant.np_dtype)

    def dynamic_model(self, x, u, for_casadi: bool = False):
        # Keep prediction state as observation-level surrogate.
        if for_casadi:
            return x
        return x

    def observe_model(self, x, u=None, for_casadi: bool = False):
        if for_casadi:
            return x
        return x

    def initial_control(self, x):
        return self.plant.ss_u


# Alias to allow swapping in richer sysid (e.g., N4SIDSurrogate) without changing controller usage.
SMBSystemId = SMBBaselineSysId


class SMBMPCController:
    """Lightweight MPC baseline using a surrogate sysid and CasADi solver."""

    def __init__(
        self,
        sysid: SMBBaselineSysId,
        prediction_horizon: int = 10,
        stage_cost: Callable[[ca.SX, ca.SX, ca.SX], ca.SX] = None,
        terminal_cost: Callable[[ca.SX, ca.SX, ca.SX], ca.SX] = None,
        ref: np.ndarray | None = None,
    ):
        self.sysid = sysid
        self.prediction_horizon = prediction_horizon
        self.ref = ref if ref is not None else self.sysid.plant.ref
        if stage_cost is None:
            stage_cost = lambda x, u: _mpc_stage_cost(self.ref, self.sysid.observe_model(x, u, True), u)
        if terminal_cost is None:
            terminal_cost = stage_cost
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost

    def control(self, current_state: np.ndarray) -> np.ndarray:
        # Descaled in, descaled out
        ini_x, ini_u = self._guess_generation(current_state)

        ca_x = ca.SX.sym("x", self.sysid.x_est_dim)
        ca_u = ca.SX.sym("u", self.sysid.u_dim)
        ca_x_d = _zero_mean_descale(ca_x, self.sysid.x_est_min, self.sysid.x_est_max)
        ca_u_d = _zero_mean_descale(ca_u, self.sysid.u_min, self.sysid.u_max)

        next_ca_x = self.sysid.dynamic_model(ca_x_d, ca_u_d, for_casadi=True)
        next_ca_x = _zero_mean_scale(next_ca_x, self.sysid.x_est_min, self.sysid.x_est_max)
        step = ca.Function("Step", [ca_x, ca_u], [next_ca_x], ["x", "u"], ["x_plus"])

        stage_cost = self.stage_cost(ca_x_d, ca_u_d)
        terminal_cost = self.terminal_cost(ca_x_d, ca_u_d)
        stage_cost_func = ca.Function("L", [ca_x, ca_u], [stage_cost], ["x", "u"], ["Lc"])
        terminal_cost_func = ca.Function("V", [ca_x, ca_u], [terminal_cost], ["x", "u"], ["Vc"])

        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        cost = 0
        x_plot, u_plot = [], []

        xk = ca.MX.sym("x0", self.sysid.x_est_dim)
        w.append(xk)
        lbw = np.append(lbw, -np.ones((self.sysid.x_est_dim, 1)))
        ubw = np.append(ubw, np.ones((self.sysid.x_est_dim, 1)))
        w0 = np.append(w0, _zero_mean_scale(ini_x[:, 0], self.sysid.x_est_min, self.sysid.x_est_max))
        x_plot.append(xk)

        g.append(xk - _zero_mean_scale(ini_x[:, 0], self.sysid.x_est_min, self.sysid.x_est_max))
        lbg = np.append(lbg, np.zeros((self.sysid.x_est_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.sysid.x_est_dim, 1)))

        for k in range(self.prediction_horizon):
            uk = ca.MX.sym(f"u{k}", self.sysid.u_dim)
            w.append(uk)
            lbw = np.append(lbw, -np.ones((self.sysid.u_dim, 1)))
            ubw = np.append(ubw, np.ones((self.sysid.u_dim, 1)))
            w0 = np.append(w0, _zero_mean_scale(ini_u[:, k], self.sysid.u_min, self.sysid.u_max))
            u_plot.append(uk)

            cost += stage_cost_func(xk, uk)

            g.append(ca.MX.zeros(0))  # no inequality constraints
            lbg = np.append(lbg, [])
            ubg = np.append(ubg, [])

            xk_predict = step(xk, uk)
            xk = ca.MX.sym(f"x{k+1}", self.sysid.x_est_dim)
            w.append(xk)
            lbw = np.append(lbw, -np.ones((self.sysid.x_est_dim, 1)))
            ubw = np.append(ubw, np.ones((self.sysid.x_est_dim, 1)))
            w0 = np.append(w0, _zero_mean_scale(ini_x[:, k + 1], self.sysid.x_est_min, self.sysid.x_est_max))
            x_plot.append(xk)

            g.append(xk - xk_predict)
            lbg = np.append(lbg, np.zeros((self.sysid.x_est_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.sysid.x_est_dim, 1)))

        uk = ca.MX.sym(f"u{self.prediction_horizon}", self.sysid.u_dim)
        w.append(uk)
        lbw = np.append(lbw, -np.ones((self.sysid.u_dim, 1)))
        ubw = np.append(ubw, np.ones((self.sysid.u_dim, 1)))
        w0 = np.append(w0, _zero_mean_scale(ini_u[:, self.prediction_horizon], self.sysid.u_min, self.sysid.u_max))
        u_plot.append(uk)

        cost += terminal_cost_func(xk, uk)

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)

        prob = {"f": cost, "x": w, "g": g}
        opts = {"print_time": False, "ipopt": {"print_level": 0}}
        solver = ca.nlpsol("solver", "ipopt", prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        trajectories = ca.Function("trajectories", [w], [x_plot, u_plot], ["w"], ["x", "u"])
        x_opt, u_opt = trajectories(sol["x"])
        descale_u_opt = _zero_mean_descale(np.transpose(u_opt.full()), self.sysid.u_min, self.sysid.u_max)
        control_value = descale_u_opt[0, :]
        return control_value

    def _guess_generation(self, current_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        guess_x = np.zeros((self.sysid.x_est_dim, self.prediction_horizon + 1))
        guess_u = np.zeros((self.sysid.u_dim, self.prediction_horizon + 1))
        guess_x[:, 0] = current_state
        for k in range(self.prediction_horizon):
            guess_u[:, k] = self.sysid.initial_control(guess_x[:, k])
            guess_u[:, k] = np.clip(guess_u[:, k], self.sysid.u_min, self.sysid.u_max)
            guess_x[:, k + 1] = self.sysid.dynamic_model(guess_x[:, k], guess_u[:, k], for_casadi=False)
            guess_x[:, k + 1] = np.clip(guess_x[:, k + 1], self.sysid.x_est_min, self.sysid.x_est_max)
        guess_u[:, -1] = self.sysid.initial_control(guess_x[:, -1])
        return guess_x, guess_u
