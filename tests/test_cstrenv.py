import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np

from smpl.envs.cstrenv import CSTRModel, CSTREnv
from smpl.envs.cstr_controller import CSTRMPCController

REF_SYS_PATH = (
    Path(__file__).resolve().parent.parent
    / "ref"
    / "Quantitative-Comparison-of-RL-and-MPC"
    / "Systems"
    / "sys_cstr.py"
)


def load_reference_model(seed=0):
    spec = importlib.util.spec_from_file_location("ref_sys_cstr", REF_SYS_PATH)
    module = importlib.util.module_from_spec(spec)
    ref_root = REF_SYS_PATH.parent.parent
    ref_root_str = str(ref_root)
    if ref_root_str not in sys.path:
        sys.path.insert(0, ref_root_str)
    if "tensorflow" not in sys.modules:
        import types

        tf_stub = types.SimpleNamespace(
            constant=lambda value, dtype=None: np.array(value, dtype=np.float64),
            random=types.SimpleNamespace(set_seed=lambda *_: None),
        )
        sys.modules["tensorflow"] = tf_stub
    cwd = Path.cwd()
    import os

    os.chdir(ref_root_str)
    try:
        spec.loader.exec_module(module)
        config = SimpleNamespace(
            seed=seed,
            system_state_disturb=False,
            system_measure_disturb=False,
            system_para_disturb=False,
        )
        return module.SysCSTR(config)
    finally:
        os.chdir(cwd)


def rollout_reference(model, action, steps, seed):
    np.random.seed(seed)
    x, _, _, _, ref = model.do_reset()
    trajectory = []
    for _ in range(steps):
        y = model.get_observation(x)
        cost = model.get_cost(y, action, ref)
        trajectory.append((y.copy(), cost))
        x = model.go_step(x, action)
    return trajectory


def rollout_ported(model, action, steps, seed):
    np.random.seed(seed)
    x, _, _, _, ref = model.reset()
    trajectory = []
    for _ in range(steps):
        y = model.observe(x, action)
        cost = model.cost(y, action, ref)
        trajectory.append((y.copy(), cost))
        x = model.step(x, action)
    return trajectory


class DummySafetyModel:
    """Deterministic CSTR-like model used to test env safety logic."""

    def __init__(self, temp_delta=0.0, temp_rate=0.0):
        self.np_dtype = np.float64
        self.x_dim = 4
        self.u_dim = 2
        self.y_dim = 2
        self.p_dim = 1
        self.r_dim = 2

        self.u_min = np.array([2.0, -9000.0], dtype=self.np_dtype)
        self.u_max = np.array([35.0, 0.0], dtype=self.np_dtype)
        self.y_min = np.array([0.0, 80.0], dtype=self.np_dtype)
        self.y_max = np.array([3.0, 130.0], dtype=self.np_dtype)

        self.time_interval = self.np_dtype(1.0 / 60.0)
        self.ini_x = np.array([2.14, 1.09, 114.2, 112.9], dtype=self.np_dtype)
        self.ini_u = np.array([14.19, -1113.5], dtype=self.np_dtype)
        self.ini_y = np.array([1.09, 114.2], dtype=self.np_dtype)
        self.ini_p = np.array([105.0], dtype=self.np_dtype)
        self.ref1 = np.array([1.09, 114.2], dtype=self.np_dtype)

        self.temp_delta = float(temp_delta)
        self.temp_rate = float(temp_rate)
        self.p_now = self.ini_p.copy()

    def reset(self):
        self.p_now = self.ini_p.copy()
        return (
            self.ini_x.copy(),
            self.ini_u.copy(),
            self.ini_y.copy(),
            self.ini_p.copy(),
            self.ref1.copy(),
        )

    def step(self, x, u, *, return_p=False):
        _ = u
        next_x = np.array(x, dtype=self.np_dtype).copy()
        next_x[2] += self.temp_delta
        p_bdd = float(self.ini_p[0])
        self.p_now = np.array([p_bdd], dtype=self.np_dtype)
        if return_p:
            return next_x, p_bdd
        return next_x

    def observe(self, x, u=None):
        _ = u
        return np.array([x[1], x[2]], dtype=self.np_dtype)

    def cost(self, y, u, ref=None):
        _ = y, u, ref
        return 1.0

    def reactor_temp_rate(self, x, u, p=None):
        _ = x, u, p
        return self.temp_rate

    def get_feed_temperature(self):
        return self.p_now


def test_model_matches_reference_rollout():
    seed = 3
    steps = 3
    action = np.array([14.19, -1113.5], dtype=np.float64)

    ref_model = load_reference_model(seed=seed)
    ported_model = CSTRModel(seed=seed, measure_disturb=False, para_disturb=False)

    ref_traj = rollout_reference(ref_model, action, steps, seed)
    ported_traj = rollout_ported(ported_model, action, steps, seed)

    for (y_ref, c_ref), (y_new, c_new) in zip(ref_traj, ported_traj):
        np.testing.assert_allclose(y_new, y_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(c_new, c_ref, rtol=1e-7, atol=1e-9)


def test_env_step_returns_cost_and_reward_alignment():
    env = CSTREnv(normalize=True, dense_reward=True, seed=0, max_steps=8)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    action = np.zeros(env.action_space.shape, dtype=np.float64)
    next_obs, reward, done, done, info = env.step(action)

    assert env.observation_space.contains(next_obs)
    assert not done
    assert "cost" in info
    assert np.isclose(reward, -info["cost"])


def test_action_bounds_safety_toggle():
    valid_action = np.array([14.19, -1113.5], dtype=np.float64)
    out_of_bounds_action = np.array([100.0, -1113.5], dtype=np.float64)

    env_enabled = CSTREnv(
        model=DummySafetyModel(),
        normalize=False,
        enable_action_bounds_safety=True,
        enable_thermal_safety=False,
    )
    env_enabled.reset()
    _, _, done_enabled, _, info_enabled = env_enabled.step(out_of_bounds_action)
    assert done_enabled
    assert info_enabled["unsafe_reason"] == "action_out_of_bounds"

    env_disabled = CSTREnv(
        model=DummySafetyModel(),
        normalize=False,
        enable_action_bounds_safety=False,
        enable_thermal_safety=False,
    )
    env_disabled.reset()
    _, _, done_disabled, _, info_disabled = env_disabled.step(out_of_bounds_action)
    assert not done_disabled
    assert info_disabled["unsafe_reason"] is None
    assert not info_disabled["unsafe"]

    _, _, done_valid, _, info_valid = env_disabled.step(valid_action)
    assert not done_valid
    assert info_valid["unsafe_reason"] is None


def test_thermal_hard_cap_is_configurable():
    action = np.array([14.19, -1113.5], dtype=np.float64)

    env_strict = CSTREnv(
        model=DummySafetyModel(temp_delta=10.0, temp_rate=0.0),
        normalize=False,
        enable_action_bounds_safety=False,
        enable_thermal_safety=True,
        hard_temp_threshold=120.0,
        runaway_temp_threshold=200.0,
        runaway_temp_rate_threshold=1e9,
    )
    env_strict.reset()
    _, _, done_strict, _, info_strict = env_strict.step(action)
    assert done_strict
    assert info_strict["unsafe_reason"] == "thermal_runaway"

    env_relaxed = CSTREnv(
        model=DummySafetyModel(temp_delta=10.0, temp_rate=0.0),
        normalize=False,
        enable_action_bounds_safety=False,
        enable_thermal_safety=True,
        hard_temp_threshold=150.0,
        runaway_temp_threshold=200.0,
        runaway_temp_rate_threshold=1e9,
    )
    env_relaxed.reset()
    _, _, done_relaxed, _, info_relaxed = env_relaxed.step(action)
    assert not done_relaxed
    assert info_relaxed["unsafe_reason"] is None


def test_thermal_safety_toggle():
    action = np.array([14.19, -1113.5], dtype=np.float64)

    env_enabled = CSTREnv(
        model=DummySafetyModel(temp_delta=10.0, temp_rate=2.0),
        normalize=False,
        enable_action_bounds_safety=False,
        enable_thermal_safety=True,
        hard_temp_threshold=120.0,
        runaway_temp_threshold=110.0,
        runaway_temp_rate_threshold=0.1,
    )
    env_enabled.reset()
    _, _, done_enabled, _, info_enabled = env_enabled.step(action)
    assert done_enabled
    assert info_enabled["unsafe_reason"] == "thermal_runaway"

    env_disabled = CSTREnv(
        model=DummySafetyModel(temp_delta=10.0, temp_rate=2.0),
        normalize=False,
        enable_action_bounds_safety=False,
        enable_thermal_safety=False,
        hard_temp_threshold=120.0,
        runaway_temp_threshold=110.0,
        runaway_temp_rate_threshold=0.1,
    )
    env_disabled.reset()
    _, _, done_disabled, _, info_disabled = env_disabled.step(action)
    assert not done_disabled
    assert info_disabled["unsafe_reason"] is None


def test_info_safety_thresholds_schema():
    action = np.array([14.19, -1113.5], dtype=np.float64)
    env = CSTREnv(
        model=DummySafetyModel(),
        normalize=False,
        hard_temp_threshold=140.0,
        runaway_temp_threshold=118.0,
        runaway_temp_rate_threshold=0.7,
        enable_action_bounds_safety=False,
        enable_thermal_safety=True,
    )
    env.reset()
    _, _, _, _, info = env.step(action)

    thresholds = info["safety_thresholds"]
    assert "input_rate_threshold" not in thresholds
    assert np.isclose(thresholds["hard_temp_threshold"], 140.0)
    assert np.isclose(thresholds["runaway_temp_threshold"], 118.0)
    assert np.isclose(thresholds["runaway_temp_rate_threshold"], 0.7)
    assert thresholds["enable_action_bounds_safety"] is False
    assert thresholds["enable_thermal_safety"] is True


def test_mpc_aligns_with_reference_controller():
    ref_root = REF_SYS_PATH.parent.parent
    ref_root_str = str(ref_root)
    if ref_root_str not in sys.path:
        sys.path.insert(0, ref_root_str)
    from Controls import control_mpc
    from cstr_config import CstrConfig

    model = CSTRModel(seed=0, measure_disturb=False, para_disturb=False)

    class DummySysId:
        def __init__(self, plant):
            self.plant = plant
            self.x_est_dim = plant.y_dim
            self.x_est_min = plant.y_min
            self.x_est_max = plant.y_max
            self.ini_x = np.zeros(self.x_est_dim, dtype=plant.np_dtype)

        def dynamic_model(self, x, u, for_casadi=False):
            _ = u
            if for_casadi:
                return x
            return x

        def observe_model(self, x, for_casadi=False, u=None):
            _ = u
            if for_casadi:
                return x
            return x

    sysid = DummySysId(model)
    config = CstrConfig()
    config.MPC_prediction_horizon = 4

    ref_ctrl = control_mpc.MPC(sysid, config)
    smpl_ctrl = CSTRMPCController(sysid, config)

    x0 = sysid.ini_x
    np.testing.assert_allclose(smpl_ctrl.control(x0), ref_ctrl.control(x0), rtol=1e-8, atol=1e-10)
