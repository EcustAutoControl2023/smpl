"""
SMPL-style SMB training/evaluation script (Matlab-free).

Run with uv from repo root (see AGENTS.md):
    UV_CACHE_DIR=.uv-cache uv run scripts/smb_train.py --seeds 1 --horizon 192 --warmup 32

This mirrors the reference `smb_main.py` MPC flow using:
- SMBEnv (normalize=False) for plant simulation.
- N4SIDSurrogate + SMBMPCController for MPC without Matlab.
Outputs:
- cost trajectories per seed in Data/smb_smpl/*txt.
- stdout summary matching the reference reporting style.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from smpl.envs import SMBEnv, SMBModel
from smpl.envs.smb_controller import SMBMPCController, _mpc_stage_cost
from smpl.envs.smb_sysid import N4SIDSurrogate


def run_episode(env: SMBEnv, controller: SMBMPCController, sysid: N4SIDSurrogate, horizon: int, warmup: int):
    obs, _ = env.reset()
    x_est = sysid.ini_x.copy()
    costs = np.zeros(horizon, dtype=np.float64)
    for t in range(horizon):
        if t < warmup:
            action = sysid.plant.ss_u
        else:
            action = controller.control(x_est)
        obs, reward, done, info = env.step(action)
        cost = info.get("cost", -reward)
        costs[t] = cost
        x_est = sysid.dynamic_model(x_est, action, for_casadi=False)
        if done:
            break
    return costs


def build_controller(model: SMBModel, prediction_horizon: int):
    sysid = N4SIDSurrogate(model, x_dim=model.y_dim, sysid_samples=400)

    def stage_cost(x, u, for_casadi=False, observe_model=None):
        obs_model = observe_model if observe_model is not None else sysid.observe_model
        y = obs_model(x, u, for_casadi)
        return _mpc_stage_cost(model.ref, y, u)

    controller = SMBMPCController(
        sysid=sysid,
        prediction_horizon=prediction_horizon,
        stage_cost=lambda x, u: stage_cost(x, u, for_casadi=True),
        terminal_cost=lambda x, u: stage_cost(x, u, for_casadi=True),
        ref=model.ref,
    )
    return controller, sysid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--horizon", type=int, default=32 * 6, help="Episode horizon (steps)")
    parser.add_argument("--warmup", type=int, default=32, help="Steps using steady-state input before MPC")
    parser.add_argument("--prediction-horizon", type=int, default=10, help="MPC prediction horizon")
    parser.add_argument("--outdir", type=Path, default=Path("Data/smb_smpl"), help="Output directory")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    online_costs = np.zeros((args.seeds, args.horizon))

    for seed in range(args.seeds):
        np.random.seed(seed)
        random.seed(seed)
        model = SMBModel(seed=seed)
        env = SMBEnv(normalize=False, dense_reward=True, seed=seed, max_steps=args.horizon)
        controller, sysid = build_controller(model, args.prediction_horizon)

        costs = run_episode(env, controller, sysid, args.horizon, args.warmup)
        online_costs[seed, : len(costs)] = costs
        np.savetxt(args.outdir / f"smb_costs_seed{seed}.txt", costs, fmt="%12.8f")
        print(f"Seed {seed} total cost: {np.sum(costs):.4f}")

    np.savetxt(args.outdir / "smb_costs_all.txt", online_costs, fmt="%12.8f")
    mean_final_cost = float(np.mean(online_costs[:, args.horizon - 1]))
    summary = {
        "seeds": args.seeds,
        "horizon": args.horizon,
        "warmup": args.warmup,
        "prediction_horizon": args.prediction_horizon,
        "mean_final_cost": mean_final_cost,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
