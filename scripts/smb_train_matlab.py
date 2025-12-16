"""
SMPL SMB training using Matlab N4SID (reference stack) on the SMPL environment.

Requires a working Matlab Engine. Run from repo root per AGENTS.md:
  UV_CACHE_DIR=.uv-cache LD_PRELOAD=$PWD/ref/Quantitative-Comparison-of-RL-and-MPC/libexecstackshim.so \\
  uv run scripts/smb_train_matlab.py --seeds 1 --offline-switch 20

Outputs cost trajectories to Data/smb_matlab/ similar to smb_results_offline_cost_1.txt.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

from smpl.envs import SMBModel

ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = ROOT / "ref" / "Quantitative-Comparison-of-RL-and-MPC"
sys.path.insert(0, str(REF_ROOT))

from smb_config import SmbConfig  # noqa: E402
from Utility import utility as ut  # noqa: E402
from Sysids import sysid as ref_sysid  # noqa: E402
from Estimators import estimator as ref_estimator  # noqa: E402
from Controls import controller as ref_controller  # noqa: E402
from smb_test import test as ref_test  # noqa: E402

OFFLINE_SAVE_INDEX = np.array(
    [20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int
)


def offline_learning(config, plant, offline_switch, directory, file_name):
    learning_offline_ini_horizon = 1
    file_name_tag = file_name + "-offline"
    sysid_err_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size, dtype=np.float64)

    smb_sysid = ref_sysid.SYSID(plant=plant, config=config)

    steps = 32 * offline_switch
    x = np.zeros((steps + 1, plant.x_dim), dtype=np.float64)
    y = np.zeros((steps + 1, plant.y_dim), dtype=np.float64)
    u = np.zeros((steps + 1, plant.u_dim), dtype=np.float64)
    p = np.zeros((steps + 1, plant.p_dim), dtype=np.float64)
    c = np.zeros((steps + 1, 1), dtype=np.float64)
    r = np.zeros((steps + 1, plant.r_dim), dtype=np.float64)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    random_signal = ut.random_step_signal_generation(
        signal_dim=plant.u_dim,
        signal_length=config.SYSID_signal_length,
        signal_min=config.SYSID_signal_min,
        signal_max=config.SYSID_signal_max,
        signal_interval=config.SYSID_signal_interval,
    )

    for k in range(steps):
        r[k, :] = plant.ref
        if k > 32 * learning_offline_ini_horizon:
            u[k, :] = random_signal[k, :]
        else:
            u[k, :] = plant.ss_u
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_concentration()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :], u[k, :])

    r[-1, :] = r[-2, :]
    u[-1, :] = u[-2, :]
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.p_now

    smb_sysid.add_data_and_scale(u, y)

    for idx, operated_time in enumerate(OFFLINE_SAVE_INDEX):
        if operated_time > offline_switch:
            break
        smb_sysid.do_identification(32 * operated_time)
        sysid_err_trajectory[idx] = smb_sysid.sysid_error
        smb_ctrl = ref_controller.CONTROLLER(smb_sysid, config)
        smb_ctrl.save_controller(directory, file_name_tag + "-ctrl-" + str(operated_time))
    return sysid_err_trajectory, smb_sysid


def offline_test(config, smb_sysid, directory, file_name, offline_switch):
    file_name_tag = file_name + "-offline"
    cost_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size)
    smb_ctrl = ref_controller.CONTROLLER(smb_sysid, config)
    for idx, operated_time in enumerate(OFFLINE_SAVE_INDEX):
        if operated_time > offline_switch:
            break
        smb_ctrl.load_controller(directory, file_name_tag + "-ctrl-" + str(operated_time))
        smb_sysid = smb_ctrl.controller.sysid
        smb_est = ref_estimator.ESTIMATOR(sysid=smb_sysid, config=config)
        cost_trajectory[idx] = ref_test(
            directory,
            file_name_tag + "-test-" + str(operated_time),
            config,
            smb_ctrl,
            smb_sysid,
            smb_est,
            False,
            True,
        )
    return cost_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--offline-switch", type=int, default=20, help="Hours (multiplied by 32)")
    parser.add_argument("--outdir", type=Path, default=Path("Data/smb_matlab"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    offline_costs = np.zeros((args.seeds, OFFLINE_SAVE_INDEX.size), dtype=np.float64)
    sysid_errors = np.zeros_like(offline_costs)

    for seed in range(args.seeds):
        random.seed(seed)
        np.random.seed(seed)

        config = SmbConfig()
        config.plot_bool = False
        config.sysid_method = "N4SID"
        config.estimate_method = "KF"
        config.control_method = "MPC"
        config.seed = seed

        directory = args.outdir / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)

        plant = SMBModel(seed=seed)

        file_name = f"MPC-N4SID-KF-SEED{seed}-{args.offline_switch}-MATLAB"
        config.save_settings(directory, file_name)

        errors, smb_sysid = offline_learning(config, plant, args.offline_switch, directory, file_name)
        sysid_errors[seed, :] = errors
        offline_costs[seed, :] = offline_test(config, smb_sysid, directory, file_name, args.offline_switch)

    np.savetxt(args.outdir / "smb_results_offline_cost_matlab.txt", offline_costs, fmt="%12.8f")
    np.savetxt(args.outdir / "smb_results_sysid_errors_matlab.txt", sysid_errors, fmt="%12.8f")
    print("Offline costs (last):", offline_costs[:, -1])
    print("SysID errors (last):", sysid_errors[:, -1])


if __name__ == "__main__":
    main()
