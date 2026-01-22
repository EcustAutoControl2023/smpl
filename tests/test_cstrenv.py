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
