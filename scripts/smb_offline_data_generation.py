"""
Offline dataset generation for SMBEnv.

Generates d4rl-style datasets under offline_datasets/smb using either
an MPC policy (default) or random actions. Intended for offline RL training.

Usage (from repo root per AGENTS.md):
  UV_CACHE_DIR=.uv-cache uv run scripts/smb_offline_data_generation.py \\
    --episodes 10 --horizon 192 --policy mpc --normalize False
"""

import argparse
import pickle
from pathlib import Path
import sys
import random

import numpy as np

from smpl.envs import SMBEnv
from smpl.envs.smb_controller import SMBMPCController
from smpl.envs.smb_sysid import N4SIDSurrogate, MatlabNNARXSurrogate

REF_ROOT = Path(__file__).resolve().parents[1] / "ref" / "Quantitative-Comparison-of-RL-and-MPC"
sys.path.insert(0, str(REF_ROOT))
from smb_config import SmbConfig  # noqa: E402
from Utility import utility as ut  # noqa: E402
from Sysids import sysid as ref_sysid  # noqa: E402
from Estimators import estimator as ref_estimator  # noqa: E402
from Controls import controller as ref_controller  # noqa: E402


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, _obs):
        return self.action_space.sample()


class MPCPolicy:
    def __init__(self, env: SMBEnv, prediction_horizon: int = 10, sysid_samples: int = 400):
        model = env.model
        sysid = N4SIDSurrogate(model, x_dim=model.y_dim, sysid_samples=sysid_samples)
        self.controller = SMBMPCController(sysid=sysid, prediction_horizon=prediction_horizon)
        self.state = sysid.ini_x.copy()
        self.sysid = sysid

    def predict(self, obs):
        # obs unused because sysid state is propagated internally
        action = self.controller.control(self.state)
        self.state = self.sysid.dynamic_model(self.state, action, for_casadi=False)
        return action


class MatlabMPCPolicy:
    """Uses Matlab-backed N4SID + MPC from the reference stack on the SMPL SMB model."""

    def __init__(self, env: SMBEnv, offline_switch: int = 20):
        self.env = env
        self.plant = env.model
        self.config = SmbConfig()
        self.config.plot_bool = False
        self.config.sysid_method = "N4SID"
        self.config.estimate_method = "KF"
        self.config.control_method = "MPC"
        self.config.seed = 0
        self.offline_switch = offline_switch

        self.sysid = self._fit_sysid()
        self.estimator = ref_estimator.ESTIMATOR(sysid=self.sysid, config=self.config)
        self.controller = ref_controller.CONTROLLER(self.sysid, self.config)
        self.x_est = self.sysid.ini_x.copy()

    def _fit_sysid(self):
        steps = 32 * self.offline_switch
        learning_offline_ini_horizon = 1
        smb_sysid = ref_sysid.SYSID(plant=self.plant, config=self.config)

        x = np.zeros((steps + 1, self.plant.x_dim), dtype=np.float64)
        y = np.zeros((steps + 1, self.plant.y_dim), dtype=np.float64)
        u = np.zeros((steps + 1, self.plant.u_dim), dtype=np.float64)
        p = np.zeros((steps + 1, self.plant.p_dim), dtype=np.float64)
        r = np.zeros((steps + 1, self.plant.r_dim), dtype=np.float64)
        x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = self.plant.do_reset()

        random_signal = ut.random_step_signal_generation(
            signal_dim=self.plant.u_dim,
            signal_length=self.config.SYSID_signal_length,
            signal_min=self.config.SYSID_signal_min,
            signal_max=self.config.SYSID_signal_max,
            signal_interval=self.config.SYSID_signal_interval,
        )

        for k in range(steps):
            r[k, :] = self.plant.ref
            if k > 32 * learning_offline_ini_horizon:
                u[k, :] = random_signal[k, :]
            else:
                u[k, :] = self.plant.ss_u
            p[k, :] = self.plant.get_feed_concentration()
            x[k + 1, :] = self.plant.go_step(x[k, :], u[k, :])
            y[k + 1, :] = self.plant.get_observation(x[k + 1, :], u[k, :])
        smb_sysid.add_data_and_scale(u, y)
        smb_sysid.do_identification(32 * self.offline_switch)
        return smb_sysid

    def predict(self, obs, mode=None):
        # obs is denormalized observation
        if mode is None:
            mode = 0
        action = self.controller.control_without_exploration(self.x_est)
        self.last_action = action
        self.last_mode = mode
        return action

    def update_estimate(self, next_obs, mode):
        # uses last_action stored in predict
        self.x_est = self.estimator.estimate(self.x_est, self.last_action, next_obs, mode)


class MatlabNNARXMPCPolicy:
    """MPC using Matlab NNARX sysid for SMB."""

    def __init__(self, env: SMBEnv, data_length: int = 2000):
        self.env = env
        self.plant = env.model
        self.sysid = MatlabNNARXSurrogate(self.plant, data_length=data_length, seed=0)
        self.config = SmbConfig()
        self.config.plot_bool = False
        self.config.sysid_method = "NNARX"
        self.config.estimate_method = "STACKING"
        self.config.control_method = "MPC"
        self.config.seed = 0
        self.config.STACK_o_dim = 0
        self.config.STACK_o_min = np.array([], dtype=np.float64)
        self.config.STACK_o_max = np.array([], dtype=np.float64)
        self.estimator = ref_estimator.ESTIMATOR(sysid=self.sysid, config=self.config)
        self.controller = ref_controller.CONTROLLER(self.sysid, self.config)
        self.x_est = self.sysid.ini_x.copy()

    def predict(self, obs, mode=None):
        if mode is None:
            mode = 0
        action = self.controller.control_without_exploration(self.x_est)
        self.last_action = action
        self.last_mode = mode
        return action

    def update_estimate(self, next_obs, mode):
        self.x_est = self.estimator.estimate(self.x_est, self.last_action, next_obs, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to generate")
    parser.add_argument("--horizon", type=int, default=192, help="Steps per episode")
    parser.add_argument(
        "--policy",
        type=str,
        default="mpc",
        choices=["mpc", "random", "mpc_n4sid_matlab", "mpc_nnarx_matlab"],
    )
    parser.add_argument("--normalize", type=str, default="False", help="Whether to normalize env (True/False)")
    parser.add_argument("--prediction-horizon", type=int, default=10, help="MPC prediction horizon")
    parser.add_argument("--sysid-samples", type=int, default=400, help="Samples to fit surrogate sysid")
    parser.add_argument("--offline-switch", type=int, default=20, help="Offline switch hours for Matlab N4SID fit")
    parser.add_argument("--nnarx-data-length", type=int, default=2000, help="Samples for Matlab NNARX fit")
    parser.add_argument("--outdir", type=Path, default=Path("offline_datasets/smb"))
    args = parser.parse_args()

    normalize_flag = args.normalize.lower() == "true"
    env = SMBEnv(normalize=normalize_flag, dense_reward=True, max_steps=args.horizon)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.policy == "mpc":
        policy = MPCPolicy(env, prediction_horizon=args.prediction_horizon, sysid_samples=args.sysid_samples)
        policy_name = f"mpc_ph{args.prediction_horizon}"
        dataset = env.generate_dataset_with_algorithm(
            policy,
            normalize=normalize_flag,
            num_episodes=args.episodes,
            error_reward=env.error_reward,
            initial_states=None,
            format="d4rl",
        )
    elif args.policy == "mpc_n4sid_matlab":
        policy = MatlabMPCPolicy(env, offline_switch=args.offline_switch)
        policy_name = f"mpc_n4sid_matlab_off{args.offline_switch}"
        obs_all, act_all, rew_all, term_all, tout_all = [], [], [], [], []
        for epi in range(args.episodes):
            obs, _ = env.reset()
            policy.x_est = policy.sysid.ini_x.copy()
            done = False
            for step in range(args.horizon):
                action = policy.predict(obs, mode=None)
                obs_next, reward, done_flag, _done2, info = env.step(action)
                mode = info.get("mode", 0)
                policy.update_estimate(obs_next, mode)
                timeout = info.get("timeout", False) or step == args.horizon - 1
                done = done_flag or timeout
                obs_all.append(obs)
                act_all.append(action)
                rew_all.append(reward)
                term_all.append(done)
                tout_all.append(timeout)
                obs = obs_next
                if done:
                    break
        # convert lists to numpy arrays for d4rl compatibility
        dataset = {
            "observations": np.asarray(obs_all, dtype=np.float32),
            "actions": np.asarray(act_all, dtype=np.float32),
            "rewards": np.asarray(rew_all, dtype=np.float32),
            "terminals": np.asarray(term_all, dtype=bool),
            "timeouts": np.asarray(tout_all, dtype=bool),
        }
    elif args.policy == "mpc_nnarx_matlab":
        policy = MatlabNNARXMPCPolicy(env, data_length=args.nnarx_data_length)
        policy_name = f"mpc_nnarx_matlab_len{args.nnarx_data_length}"
        dataset = {"observations": [], "actions": [], "rewards": [], "terminals": [], "timeouts": []}
        for epi in range(args.episodes):
            obs, _ = env.reset()
            policy.x_est = policy.sysid.ini_x.copy()
            done = False
            step = 0
            while not done and step < args.horizon:
                action = policy.predict(obs, mode=None)
                obs_next, reward, done, _done2, info = env.step(action)
                mode = info.get("mode", 0)
                policy.update_estimate(obs_next, mode)
                dataset["observations"].append(obs)
                dataset["actions"].append(action)
                dataset["rewards"].append(reward)
                dataset["terminals"].append(done)
                dataset["timeouts"].append(info.get("timeout", False))
                obs = obs_next
                step += 1
        dataset["observations"] = np.asarray(dataset["observations"], dtype=np.float32)
        dataset["actions"] = np.asarray(dataset["actions"], dtype=np.float32)
        dataset["rewards"] = np.asarray(dataset["rewards"], dtype=np.float32)
        dataset["terminals"] = np.asarray(dataset["terminals"], dtype=bool)
        dataset["timeouts"] = np.asarray(dataset["timeouts"], dtype=bool)
    else:
        policy = RandomPolicy(env.action_space)
        policy_name = "random"
        dataset = env.generate_dataset_with_algorithm(
            policy,
            normalize=normalize_flag,
            num_episodes=args.episodes,
            error_reward=env.error_reward,
            initial_states=None,
            format="d4rl",
        )

    fname = f"smb_{policy_name}_episodes{args.episodes}_h{args.horizon}_norm={normalize_flag}.pkl"
    with open(args.outdir / fname, "wb") as fp:
        pickle.dump(dataset, fp)
    print(f"Saved dataset to {args.outdir / fname}")


if __name__ == "__main__":
    main()
