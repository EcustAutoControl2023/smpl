import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smpl.envs.smbenv import SMBEnv, SMBModel

REF_SYS_PATH = (
    Path(__file__).resolve().parent.parent
    / "ref"
    / "Quantitative-Comparison-of-RL-and-MPC"
    / "Systems"
    / "sys_smb.py"
)


def load_reference_model(seed=0):
    spec = importlib.util.spec_from_file_location("ref_sys_smb", REF_SYS_PATH)
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
    Path(ref_root_str).mkdir(exist_ok=True, parents=True)
    import os

    os.chdir(ref_root_str)
    try:
        spec.loader.exec_module(module)
        config = SimpleNamespace(
            seed=seed,
            system_state_disturb=False,
            system_measure_disturb=False,
            system_para_disturb=True,
        )
        return module.SysSMB(config)
    finally:
        os.chdir(cwd)


def rollout_reference(model, action, steps, seed):
    np.random.seed(seed)
    x, _, _, _, ref = model.do_reset()
    trajectory = []
    for _ in range(steps):
        y = model.get_observation(x, action)
        cost = model.get_cost(y, action, ref)
        trajectory.append((y.copy(), cost))
        x = model.go_step(x, action)
    return trajectory


def rollout_ported(model, action, steps, seed):
    np.random.seed(seed)
    x, _, _, _, _ = model.reset()
    trajectory = []
    for _ in range(steps):
        y = model.observe(x, action)
        cost = model.cost(y, action)
        trajectory.append((y.copy(), cost))
        x = model.step(x, action)
    return trajectory


def test_model_matches_reference_rollout():
    seed = 3
    steps = 3
    action = np.array([0.013, 0.013, 0.014, 0.014], dtype=np.float64)

    ref_model = load_reference_model(seed=seed)
    ported_model = SMBModel(seed=seed)

    ref_traj = rollout_reference(ref_model, action, steps, seed)
    ported_traj = rollout_ported(ported_model, action, steps, seed)

    for (y_ref, c_ref), (y_new, c_new) in zip(ref_traj, ported_traj):
        np.testing.assert_allclose(y_new, y_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(c_new, c_ref, rtol=1e-7, atol=1e-9)


def test_env_step_returns_cost_and_reward_alignment():
    env = SMBEnv(normalize=True, dense_reward=True, seed=0, max_steps=8)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    action = np.zeros(env.action_space.shape, dtype=np.float64)
    next_obs, reward, done, info = env.step(action)

    assert env.observation_space.contains(next_obs)
    assert not done
    assert "cost" in info
    assert np.isclose(reward, -info["cost"])
