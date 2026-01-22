"""
Baseline CSTR controllers for the SMPL environment.

These controllers mirror the reference MPC implementation while keeping the
environment side Matlab-free. System identification can still rely on Matlab
for reproduction if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import casadi as ca
import numpy as np

from .cstrenv import _zero_mean_descale, _zero_mean_scale, CSTRModel


@dataclass
class CSTRBaselineSysId:
    """Minimal surrogate sysid used for MPC when full system-ID is unavailable."""

    plant: CSTRModel

    def __post_init__(self):
        self.x_est_dim = self.plant.y_dim
        self.x_est_min = self.plant.y_min
        self.x_est_max = self.plant.y_max
        self.u_dim = self.plant.u_dim
        self.u_min = self.plant.u_min
        self.u_max = self.plant.u_max
        self.ini_x = np.zeros(self.x_est_dim, dtype=self.plant.np_dtype)

    def dynamic_model(self, x, u, for_casadi: bool = False):
        _ = u
        if for_casadi:
            return x
        return x

    def observe_model(self, x, u=None, for_casadi: bool = False):
        _ = u
        if for_casadi:
            return x
        return x

    def initial_control(self, x):
        _ = x
        return self.plant.ss_u


# Alias for consistency with other envs
CSTRSystemId = CSTRBaselineSysId


class CSTRMPCController:
    """MPC controller matching the reference MPC formulation."""

    def __init__(self, sysid, config):
        self.config = config
        self.sysid = sysid

        self.dynamic_model = self.sysid.dynamic_model
        self.observe_model = self.sysid.observe_model
        self.abstract_stage_cost = self.config.mpc_abstract_stage_cost
        self.abstract_terminal_cost = self.config.mpc_abstract_terminal_cost
        self.abstract_stage_constraint = self.config.mpc_abstract_stage_constraint
        self.abstract_terminal_constraint = self.config.mpc_abstract_terminal_constraint
        self.initial_control = self.config.mpc_initial_control

        self.x_dim = sysid.x_est_dim
        self.x_min = sysid.x_est_min
        self.x_max = sysid.x_est_max

        self.u_dim = sysid.plant.u_dim
        self.u_min = sysid.plant.u_min
        self.u_max = sysid.plant.u_max

        self.g_dim = config.MPC_g_dim
        self.h_dim = config.MPC_h_dim
        self.prediction_horizon = config.MPC_prediction_horizon
        self.u_guess = np.zeros((self.u_dim, self.prediction_horizon + 1))

    def control(self, current_state):
        ini_x, ini_u = self._guess_generation(current_state)

        ca_x = ca.SX.sym("x", self.x_dim)
        ca_u = ca.SX.sym("u", self.u_dim)
        ca_x_d = _zero_mean_descale(ca_x, self.x_min, self.x_max)
        ca_u_d = _zero_mean_descale(ca_u, self.u_min, self.u_max)

        next_ca_x = self.dynamic_model(ca_x_d, ca_u_d, for_casadi=True)
        next_ca_x = _zero_mean_scale(next_ca_x, self.x_min, self.x_max)
        step = ca.Function("Step", [ca_x, ca_u], [next_ca_x], ["x", "u"], ["x_plus"])

        stage_constraint = self._stage_constraint(ca_x_d, ca_u_d, for_casadi=True)
        terminal_constraint = self._terminal_constraint(ca_x_d, ca_u_d, for_casadi=True)
        stage_constraint_func = ca.Function("g", [ca_x, ca_u], [stage_constraint], ["x", "u"], ["gc"])
        terminal_constraint_func = ca.Function("h", [ca_x, ca_u], [terminal_constraint], ["x", "u"], ["hc"])

        stage_cost = self._stage_cost(ca_x_d, ca_u_d, for_casadi=True)
        terminal_cost = self._terminal_cost(ca_x_d, ca_u_d, for_casadi=True)
        stage_cost_func = ca.Function("L", [ca_x, ca_u], [stage_cost], ["x", "u"], ["Lc"])
        terminal_cost_func = ca.Function("V", [ca_x, ca_u], [terminal_cost], ["x", "u"], ["Vc"])

        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        cost = 0

        x_plot, u_plot = [], []

        xk = ca.MX.sym("x0", self.x_dim)
        w.append(xk)
        lbw = np.append(lbw, -np.ones((self.x_dim, 1)))
        ubw = np.append(ubw, np.ones((self.x_dim, 1)))
        w0 = np.append(w0, _zero_mean_scale(ini_x[:, 0], self.x_min, self.x_max))
        x_plot.append(xk)

        g.append(xk - _zero_mean_scale(ini_x[:, 0], self.x_min, self.x_max))
        lbg = np.append(lbg, np.zeros((self.x_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.x_dim, 1)))

        for k in range(self.prediction_horizon):
            uk = ca.MX.sym(f"u{k}", self.u_dim)
            w.append(uk)
            lbw = np.append(lbw, -np.ones((self.u_dim, 1)))
            ubw = np.append(ubw, np.ones((self.u_dim, 1)))
            w0 = np.append(w0, _zero_mean_scale(ini_u[:, k], self.u_min, self.u_max))
            u_plot.append(uk)

            cost += stage_cost_func(xk, uk)

            g.append(stage_constraint_func(xk, uk))
            lbg = np.append(lbg, -np.inf * np.ones((self.g_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.g_dim, 1)))

            xk_predict = step(xk, uk)
            xk = ca.MX.sym(f"x{k+1}", self.x_dim)
            w.append(xk)
            lbw = np.append(lbw, -np.ones((self.x_dim, 1)))
            ubw = np.append(ubw, np.ones((self.x_dim, 1)))
            w0 = np.append(w0, _zero_mean_scale(ini_x[:, k + 1], self.x_min, self.x_max))
            x_plot.append(xk)

            g.append(xk - xk_predict)
            lbg = np.append(lbg, np.zeros((self.x_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.x_dim, 1)))

        uk = ca.MX.sym(f"u{self.prediction_horizon}", self.u_dim)
        w.append(uk)
        lbw = np.append(lbw, -np.ones((self.u_dim, 1)))
        ubw = np.append(ubw, np.ones((self.u_dim, 1)))
        w0 = np.append(w0, _zero_mean_scale(ini_u[:, self.prediction_horizon], self.u_min, self.u_max))
        u_plot.append(uk)

        cost += terminal_cost_func(xk, uk)

        g.append(terminal_constraint_func(xk, uk))
        lbg = np.append(lbg, -np.inf * np.ones((self.h_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.h_dim, 1)))

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)

        prob = {"f": cost, "x": w, "g": g}
        opts = {"print_time": False, "ipopt": {"print_level": 0}}
        solver = ca.nlpsol("solver", "ipopt", prob, opts)

        trajectories = ca.Function("trajectories", [w], [x_plot, u_plot], ["w"], ["x", "u"])
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt, u_opt = trajectories(sol["x"])

        descale_u_opt = _zero_mean_descale(np.transpose(u_opt.full()), self.u_min, self.u_max)
        return descale_u_opt[0, :]

    def save_controller(self, directory, name):
        import pickle

        control_parameters = [self.sysid, self.config]
        directory = Path(directory)
        with open(Path.joinpath(directory, name + "-controller_parameters.pickle"), "wb") as handle:
            pickle.dump(control_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_controller(self, directory, name):
        import pickle

        directory = Path(directory)
        with open(Path.joinpath(directory, name + "-controller_parameters.pickle"), "rb") as handle:
            control_parameters = pickle.load(handle)
        self.__init__(control_parameters[0], control_parameters[1])

    def _stage_cost(self, x, u, for_casadi):
        return self.abstract_stage_cost(x, u, for_casadi, observe_model=self.observe_model)

    def _terminal_cost(self, x, u, for_casadi):
        return self.abstract_terminal_cost(x, u, for_casadi, observe_model=self.observe_model)

    def _stage_constraint(self, x, u, for_casadi):
        return self.abstract_stage_constraint(x, u, for_casadi, observe_model=self.observe_model)

    def _terminal_constraint(self, x, u, for_casadi):
        return self.abstract_terminal_constraint(x, u, for_casadi, observe_model=self.observe_model)

    def _guess_generation(self, current_state) -> Tuple[np.ndarray, np.ndarray]:
        guess_x = np.zeros((self.x_dim, self.prediction_horizon + 1))
        guess_u = np.zeros((self.u_dim, self.prediction_horizon + 1))
        guess_x[:, 0] = current_state
        for k in range(self.prediction_horizon):
            guess_u[:, k] = self.initial_control(guess_x[:, k])
            guess_u[:, k] = np.clip(guess_u[:, k], self.u_min, self.u_max)
            guess_x[:, k + 1] = self.dynamic_model(guess_x[:, k], guess_u[:, k], for_casadi=False)
            guess_x[:, k + 1] = np.clip(guess_x[:, k + 1], self.x_min, self.x_max)
        guess_u[:, -1] = self.initial_control(guess_x[:, -1])
        return guess_x, guess_u
