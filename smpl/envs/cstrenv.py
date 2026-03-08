"""
Continuous stirred-tank reactor (CSTR) environment in SMPL style.

This mirrors the reference implementation in
ref/Quantitative-Comparison-of-RL-and-MPC/Systems/sys_cstr.py
and avoids any MATLAB dependency on the environment side.
"""

from __future__ import annotations

from typing import Literal, Tuple

import casadi as ca
import numpy as np

from .utils import denormalize_spaces, normalize_spaces, smplEnvBase


def _zero_mean_scale(
    var: np.ndarray, var_min: np.ndarray, var_max: np.ndarray
) -> np.ndarray:
    """Scale var to [-1, 1] using min/max bounds."""
    return (2.0 * var - var_max - var_min) / (var_max - var_min)


def _zero_mean_descale(
    var: np.ndarray, var_min: np.ndarray, var_max: np.ndarray
) -> np.ndarray:
    """Invert `_zero_mean_scale`."""
    return (var_max - var_min) / 2.0 * var + (var_max + var_min) / 2.0


class CSTRModel:
    """Core CSTR simulator that mirrors the reference SysCSTR implementation."""

    def __init__(
        self,
        seed: int = 0,
        state_disturb: bool = False,
        measure_disturb: bool = True,
        para_disturb: bool = True,
        np_dtype=np.float64,
    ):
        self.process_type = "continuous"
        self.seed = seed
        self.state_disturb = state_disturb
        self.measure_disturb = measure_disturb
        self.para_disturb = para_disturb
        self.np_dtype = np_dtype

        self.x_dim = 4
        self.u_dim = 2
        self.y_dim = 2
        self.p_dim = 1
        self.r_dim = 2

        self.state_std = np.zeros(self.x_dim, dtype=self.np_dtype)
        self.measure_std = np.zeros(self.y_dim, dtype=self.np_dtype)
        self.para_std = np.zeros(self.p_dim, dtype=self.np_dtype)
        if self.state_disturb:
            self.state_std = np.array([0.001, 0.001, 0.001, 0.001], dtype=self.np_dtype)
        if self.measure_disturb:
            self.measure_std = np.array([0.003, 0.05], dtype=self.np_dtype)
        if self.para_disturb:
            self.para_std = np.array([0.05], dtype=self.np_dtype)

        self.time_interval = self.np_dtype(1 / 60)

        self.x_min = np.array([0.0, 0.0, 80.0, 80.0], dtype=self.np_dtype)
        self.x_max = np.array([3.0, 3.0, 130.0, 130.0], dtype=self.np_dtype)
        self.u_min = np.array([2.0, -9000.0], dtype=self.np_dtype)
        self.u_max = np.array([35.0, 0.0], dtype=self.np_dtype)
        self.y_min = np.array([0.0, 80.0], dtype=self.np_dtype)
        self.y_max = np.array([3.0, 130.0], dtype=self.np_dtype)
        self.p_min = np.array([100.0], dtype=self.np_dtype)
        self.p_max = np.array([110.0], dtype=self.np_dtype)

        # y = ax + b, a = 2/(max - min), dy/dt = a * dx/dt
        self.scale_grad = 2.0 / (self.x_max - self.x_min)

        # initial value
        self.ini_x = np.array([2.14, 1.09, 114.2, 112.9], dtype=self.np_dtype)
        self.ini_u = np.array([14.19, -1113.5], dtype=self.np_dtype)
        self.ini_y = np.array([1.09, 114.2], dtype=self.np_dtype)
        self.ini_p = np.array([105.0], dtype=self.np_dtype)

        # steady-state value
        self.ss_x = np.array([2.14, 1.09, 114.2, 112.9], dtype=self.np_dtype)
        self.ss_u = np.array([14.19, -1113.5], dtype=self.np_dtype)
        self.ss_y = np.array([1.09, 114.2], dtype=self.np_dtype)
        self.ss_p = np.array([105.0], dtype=self.np_dtype)

        # reference
        self.ref1 = np.array([1.09, 114.2], dtype=self.np_dtype)
        self.ref2 = np.array([1.04, 94.2], dtype=self.np_dtype)

        # feed temperature
        self.p_now = self.ini_p.copy()

        # fixed parameters
        self.ca0 = 5.10
        self.k10 = 1.287 * 10**12
        self.k20 = 1.287 * 10**12
        self.k30 = 9.043 * 10**9
        self.E1 = -9758.3
        self.E2 = -9758.3
        self.E3 = -8560.0
        self.Hab = 4.2
        self.Hbc = -11.0
        self.Had = -41.85
        self.rho_Cp = 2.8119
        self.kw_AR = 866.88
        self.VR = 10.0
        self.mk = 5.0
        self.Cpk = 2.0

        self.step_fcn = self._make_step_function()
        self.xdot_fcn = self._make_xdot_function()

    def reset(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        self.p_now = self.np_dtype(105.0)
        return (
            self.ini_x.copy(),
            self.ini_u.copy(),
            self.ini_y.copy(),
            self.ini_p.copy(),
            self.ref1.copy(),
        )

    # Compatibility helpers for reference utilities/controllers
    def do_reset(self):
        return self.reset()

    def step(self, x: np.ndarray, u: np.ndarray, *, return_p: bool = False):
        self.p_now, p_bdd = self._disturbance_generation(self.p_now)
        scaled_x = _zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = _zero_mean_scale(u, self.u_min, self.u_max)
        scaled_p = _zero_mean_scale(p_bdd, self.p_min, self.p_max)
        scaled_up = np.hstack((scaled_u, scaled_p))

        result = self.step_fcn(x0=scaled_x, p=scaled_up)
        scaled_next_x = np.squeeze(np.array(result["xf"]))
        next_x = _zero_mean_descale(scaled_next_x, self.x_min, self.x_max)
        if return_p:
            return next_x, float(np.array(p_bdd).squeeze())
        return next_x

    # Compatibility helper for reference naming
    def go_step(self, x, u):
        return self.step(x, u)

    def observe(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        _ = u
        y = x[1:3] + np.random.normal(0, self.measure_std)
        return y

    def get_observation(self, x):
        return self.observe(x)

    def cost(
        self, y: np.ndarray, u: np.ndarray, ref: np.ndarray | None = None
    ) -> float:
        ref = self.ref1 if ref is None else ref
        y_weight = np.diag([1.0, 0.01 * 0.01])
        y_cost = (y - ref).T @ y_weight @ (y - ref)
        u_cost = 0.01 * 0.01 * 0.1 * (u[0] + np.abs(0.001 * u[1]))
        return float(y_cost + u_cost)

    def get_cost(self, y, u, ref):
        return self.cost(y, u, ref)

    def get_steady_state(self):
        return (
            self.ss_x.copy(),
            self.ss_u.copy(),
            self.ss_y.copy(),
            self.ss_p.copy(),
            self.ref1.copy(),
        )

    def get_feed_temperature(self):
        return self.p_now

    def _system_dynamics(self, x, u, p):
        x1, x2, x3, x4 = ca.vertsplit(x)
        u1, u2 = ca.vertsplit(u)
        p1 = p

        k1 = self.k10 * ca.exp(self.E1 / (x3 + 273.15))
        k2 = self.k20 * ca.exp(self.E2 / (x3 + 273.15))
        k3 = self.k30 * ca.exp(self.E3 / (x3 + 273.15))

        heat_by_rxns = k1 * x1 * self.Hab + k2 * x2 * self.Hbc + k3 * (x1**2) * self.Had

        x1dot = u1 * (self.ca0 - x1) - k1 * x1 - k3 * (x1**2)
        x2dot = -u1 * x2 + k1 * x1 - k2 * x2
        x3dot = (
            u1 * (p1 - x3)
            + self.kw_AR * (x4 - x3) / (self.rho_Cp * self.VR)
            - heat_by_rxns / self.rho_Cp
        )
        x4dot = (u2 + self.kw_AR * (x3 - x4)) / (self.mk * self.Cpk)
        return ca.vcat([x1dot, x2dot, x3dot, x4dot])

    def _make_step_function(self):
        x_ca = ca.SX.sym("x", self.x_dim)
        u_ca = ca.SX.sym("u", self.u_dim)
        p_ca = ca.SX.sym("p", self.p_dim)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = _zero_mean_descale(x_ca, self.x_min, self.x_max)
        u_d = _zero_mean_descale(u_ca, self.u_min, self.u_max)
        p_d = _zero_mean_descale(p_ca, self.p_min, self.p_max)

        xdot = self._system_dynamics(x_d, u_d, p_d)
        xdot = np.multiply(xdot, self.scale_grad)
        ode = {"x": x_ca, "p": up_ca, "ode": xdot}
        return ca.integrator("Integrator", "cvodes", ode, 0, [self.time_interval], {})

    def _make_xdot_function(self):
        x_ca = ca.SX.sym("x", self.x_dim)
        u_ca = ca.SX.sym("u", self.u_dim)
        p_ca = ca.SX.sym("p", self.p_dim)
        xdot = self._system_dynamics(x_ca, u_ca, p_ca)
        return ca.Function("xdot", [x_ca, u_ca, p_ca], [xdot])

    def reactor_temp_rate(
        self, x: np.ndarray, u: np.ndarray, p: np.ndarray | None = None
    ) -> float:
        p = self.p_now if p is None else p
        xdot = self.xdot_fcn(x, u, p)
        return float(np.array(xdot[2]).squeeze())

    def _disturbance_generation(self, p):
        p_next = (
            p
            - (p - self.ss_p) * self.time_interval
            + np.random.normal(0, self.para_std, 1)
        )
        p_bdd = np.clip(p_next, self.p_min, self.p_max)
        return p_next, p_bdd


class CSTREnv(smplEnvBase):
    """Gym-style CSTR environment following SMPL conventions.

    Safety detection (conservative defaults):
        - action bounds: terminate when action is outside `[u_min, u_max]`
          if `enable_action_bounds_safety` is True.
        - thermal runaway: terminate when `enable_thermal_safety` is True and:
          (1) `T_next >= hard_temp_threshold`, or
          (2) reactor temperature rate exceeds `runaway_temp_rate_threshold`
              while reactor temperature is above `runaway_temp_threshold`.
    """

    def __init__(
        self,
        dense_reward: bool = True,
        normalize: bool = True,
        debug_mode: bool = False,
        max_steps: int = 60 * 3,
        error_reward: float = -100.0,
        model: CSTRModel | None = None,
        seed: int = 0,
        runaway_temp_threshold: float | None = None,
        runaway_temp_rate_threshold: float | None = None,
        hard_temp_threshold: float | None = None,
        enable_action_bounds_safety: bool = True,
        enable_thermal_safety: bool = True,
        reference_schedule_mode: Literal[
            "fixed_ref1", "fixed_ref2", "parity_hourly"
        ] = "parity_hourly",
    ):
        self.model = model if model is not None else CSTRModel(seed=seed)
        self.seed = seed
        self.runaway_temp_threshold = (
            115.0 if runaway_temp_threshold is None else runaway_temp_threshold
        )
        self.runaway_temp_rate_threshold = (
            0.5 if runaway_temp_rate_threshold is None else runaway_temp_rate_threshold
        )
        self.hard_temp_threshold = (
            130.0 if hard_temp_threshold is None else hard_temp_threshold
        )
        self.enable_action_bounds_safety = bool(enable_action_bounds_safety)
        self.enable_thermal_safety = bool(enable_thermal_safety)
        self.reference_schedule_mode: Literal[
            "fixed_ref1", "fixed_ref2", "parity_hourly"
        ] = (
            "parity_hourly"
        )
        self.set_reference_mode(reference_schedule_mode)
        self.current_reference = np.asarray(
            self.model.ref1, dtype=self.model.np_dtype
        ).copy()
        super().__init__(
            dense_reward=dense_reward,
            normalize=normalize,
            debug_mode=debug_mode,
            action_dim=self.model.u_dim,
            observation_dim=self.model.y_dim,
            max_observations=self.model.y_max,
            min_observations=self.model.y_min,
            max_actions=self.model.u_max,
            min_actions=self.model.u_min,
            np_dtype=self.model.np_dtype,
            max_steps=max_steps,
            error_reward=error_reward,
        )
        self.reward_function = self._reward_from_cost
        self.done_calculator = self.done_calculator_standard
        self.reset()

    def set_reference_mode(
        self, mode: Literal["fixed_ref1", "fixed_ref2", "parity_hourly"] | str
    ) -> None:
        allowed = {"fixed_ref1", "fixed_ref2", "parity_hourly"}
        if mode not in allowed:
            raise ValueError(
                f"reference_schedule_mode must be one of {sorted(allowed)}, got {mode!r}."
            )
        self.reference_schedule_mode = mode  # type: ignore[assignment]

    def get_reference_for_step(self, step_index: int) -> np.ndarray:
        if self.reference_schedule_mode == "fixed_ref1":
            return np.asarray(self.model.ref1, dtype=self.model.np_dtype)
        if self.reference_schedule_mode == "fixed_ref2":
            return np.asarray(self.model.ref2, dtype=self.model.np_dtype)
        if (int(step_index) // 60) % 2 == 0:
            return np.asarray(self.model.ref1, dtype=self.model.np_dtype)
        return np.asarray(self.model.ref2, dtype=self.model.np_dtype)

    def reset(self, *, seed=None, options=None, initial_state=None):
        if seed is not None:
            self.seed = seed
            self.model.seed = seed
        x, u, y, _, _ = self.model.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.current_reference = self.get_reference_for_step(0).copy()
        self.previous_state = x
        self.previous_action = u
        observation = y.copy()
        if initial_state is not None:
            observation = np.array(initial_state, dtype=self.np_dtype)
        if self.normalize:
            observation, _, _ = normalize_spaces(
                observation, self.max_observations, self.min_observations
            )
        observation = observation.astype(self.observation_space.dtype)
        self.previous_observation = observation
        return observation, {}

    def step(self, action, normalize=None):
        reward = None
        done = None
        done_info = {"terminal": False, "timeout": False}
        normalize = self.normalize if normalize is None else normalize
        action = np.array(action, dtype=self.np_dtype)
        if normalize:
            action, _, _ = denormalize_spaces(
                action, self.max_actions, self.min_actions
            )


        unsafe_reason = None
        prev_state = self.previous_state
        if self.enable_action_bounds_safety and (
            np.any(action < self.model.u_min) or np.any(action > self.model.u_max)
        ):
            unsafe_reason = "action_out_of_bounds"

        next_state, p_bdd = self.model.step(prev_state, action, return_p=True)

        dt = float(self.model.time_interval)
        dT_avg = (next_state[2] - prev_state[2]) / dt

        # instantaneous rate under the SAME p_bdd used by the integrator
        dT_inst = self.model.reactor_temp_rate(prev_state, action, p=p_bdd)

        T_prev, T_next = float(prev_state[2]), float(next_state[2])

        if self.enable_thermal_safety and unsafe_reason is None:
            if T_next >= self.hard_temp_threshold:
                unsafe_reason = "thermal_runaway"
            elif (
                max(dT_avg, dT_inst) > self.runaway_temp_rate_threshold
                and max(T_prev, T_next) >= self.runaway_temp_threshold
            ):
                unsafe_reason = "thermal_runaway"
        observation = self.model.observe(next_state, action)
        reference = self.get_reference_for_step(self.step_count)
        self.current_reference = reference.copy()
        cost = self.model.cost(observation, action, ref=reference)

        reward = -cost
        if unsafe_reason is not None:
            reward = self.error_reward
            done = True
            done_info = {
                "terminal": True,
                "timeout": False,
                "unsafe_reason": unsafe_reason,
            }

        obs_for_check = observation.astype(self.observation_space.dtype)
        if normalize:
            obs_for_check, _, _ = normalize_spaces(
                obs_for_check, self.max_observations, self.min_observations
            )
            obs_for_check = obs_for_check.astype(self.observation_space.dtype)
        reward = self.reward_function(
            self.previous_observation, action, obs_for_check, reward=reward
        )
        done, done_info = self.done_calculator(
            obs_for_check, self.step_count, reward, done=done, done_info=done_info
        )

        self.previous_state = next_state
        self.previous_action = action
        self.previous_observation = obs_for_check
        self.step_count += 1
        self.total_reward += reward

        if not self.dense_reward and not done:
            reward = 0.0
        elif not self.dense_reward:
            reward = self.total_reward

        observation = observation.clip(self.min_observations, self.max_observations)
        if normalize:
            observation, _, _ = normalize_spaces(
                observation, self.max_observations, self.min_observations
            )
        info = {
            "cost": cost,
            "ref_ca": float(reference[0]),
            "ref_temp": float(reference[1]),
            "reference_mode": self.reference_schedule_mode,
            "feed_temperature": float(self.model.get_feed_temperature()),
            "unsafe_reason": done_info.get("unsafe_reason"),
            "unsafe": bool(done_info.get("unsafe_reason")),
            "safety_thresholds": {
                "hard_temp_threshold": self.hard_temp_threshold,
                "runaway_temp_threshold": self.runaway_temp_threshold,
                "runaway_temp_rate_threshold": self.runaway_temp_rate_threshold,
                "enable_action_bounds_safety": self.enable_action_bounds_safety,
                "enable_thermal_safety": self.enable_thermal_safety,
            },
        }
        info.update(done_info)
        observation = observation.astype(self.observation_space.dtype)
        reward = float(reward)
        return observation, reward, done, done, info

    def _reward_from_cost(
        self, previous_observation, action, current_observation, reward=None
    ):
        if reward is None:
            return self.error_reward
        if np.any(np.isnan(current_observation)) or np.any(
            np.isinf(current_observation)
        ):
            return self.error_reward
        return reward

    def observation_beyond_box(self, observation):
        observation = np.asarray(observation, dtype=self.observation_space.dtype)
        return (
            (not self.observation_space.contains(observation))
            or np.any(np.isnan(observation))
            or np.any(np.isinf(observation))
        )
