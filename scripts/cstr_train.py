"""
SMPL-style CSTR training/evaluation script.

This mirrors the reference `cstr_main.py` flow while using the SMPL CSTR model
as the plant. System identification and controllers can still rely on Matlab
for reproduction if desired (e.g., LSTM sysid).
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from smpl.envs import CSTRModel
from smpl.envs.cstr_controller import CSTRMPCController


def _ensure_ref_path():
    ref_root = Path(__file__).resolve().parent.parent / "ref" / "Quantitative-Comparison-of-RL-and-MPC"
    ref_root_str = str(ref_root)
    if ref_root_str not in os.sys.path:
        os.sys.path.insert(0, ref_root_str)
    return ref_root


def _build_controller(sysid, config, controller_type: str):
    if controller_type == "smpl":
        return CSTRMPCController(sysid, config)
    from Controls import control_mpc

    return control_mpc.MPC(sysid, config)


def offline_learning(
    plant,
    config1,
    config2,
    file_name,
    directory,
    offline_hour,
    save_index,
    controller_type,
    save_bool,
):
    from Utility import utility as ut
    from Sysids import sysid

    learning_offline_ini_horizon = 1
    file_name += "-offline"
    sysid_err_trajectory = np.zeros(save_index.size, dtype=np.float64)

    cstr_sysid = sysid.SYSID(plant=plant, config=config1)

    x = np.zeros((60 * offline_hour + 1, plant.x_dim), dtype=np.float64)
    y = np.zeros((60 * offline_hour + 1, plant.y_dim), dtype=np.float64)
    u = np.zeros((60 * offline_hour + 1, plant.u_dim), dtype=np.float64)
    p = np.zeros((60 * offline_hour + 1, plant.p_dim), dtype=np.float64)
    c = np.zeros((60 * offline_hour + 1, 1), dtype=np.float64)
    r = np.zeros((60 * offline_hour + 1, plant.y_dim), dtype=np.float64)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    random_signal = ut.random_step_signal_generation(
        signal_dim=plant.u_dim,
        signal_length=config1.SYSID_signal_length,
        signal_min=config1.SYSID_signal_min,
        signal_max=config1.SYSID_signal_max,
        signal_interval=config1.SYSID_signal_interval,
    )

    for k in range(60 * offline_hour):
        if np.mod(np.floor(k / 60), 2) == 0:
            r[k, :] = plant.ref1
        else:
            r[k, :] = plant.ref2
        if k > 60 * learning_offline_ini_horizon:
            u[k, :] = random_signal[k, :]
        else:
            u[k, :] = plant.ss_u
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :])
    r[-1, :] = r[-2, :]
    u[-1, :] = u[-2, :]
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.p_now

    if save_bool:
        config1.save_data(directory, file_name, config1.seed, x, x, u, y, p, r, c)

    cstr_sysid.add_data_and_scale(u, y)

    for k, operated_time in enumerate(save_index):
        print("seed:", config1.seed, "current offline step:", operated_time)
        cstr_sysid.do_identification(60 * operated_time)
        sysid_err_trajectory[k] = cstr_sysid.sysid_error
        controller1 = _build_controller(cstr_sysid, config1, controller_type)
        controller2 = _build_controller(cstr_sysid, config2, controller_type)
        if save_bool:
            controller1.save_controller(directory, file_name + "-ctrl1-" + str(operated_time))
            controller2.save_controller(directory, file_name + "-ctrl2-" + str(operated_time))
    return sysid_err_trajectory, cstr_sysid


def online_learning(
    plant,
    config1,
    config2,
    file_name,
    directory,
    online_hour,
    save_index,
    controller_type,
    save_bool,
):
    from Sysids import sysid
    from Estimators import estimator

    learning_online_ini_horizon = 1
    file_name += "-online"

    cstr_sysid = sysid.SYSID(plant=plant, config=config1)
    cstr_estimator = estimator.ESTIMATOR(sysid=cstr_sysid, config=config1)

    controller1 = _build_controller(cstr_sysid, config1, controller_type)
    controller2 = _build_controller(cstr_sysid, config2, controller_type)

    x = np.zeros((60 * online_hour + 1, plant.x_dim), dtype=np.float64)
    y = np.zeros((60 * online_hour + 1, plant.y_dim), dtype=np.float64)
    u = np.zeros((60 * online_hour + 1, plant.u_dim), dtype=np.float64)
    p = np.zeros((60 * online_hour + 1, plant.p_dim), dtype=np.float64)
    c = np.zeros((60 * online_hour + 1, 1), dtype=np.float64)
    r = np.zeros((60 * online_hour + 1, plant.y_dim), dtype=np.float64)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()
    x_est = np.zeros((60 * online_hour + 1, cstr_sysid.x_est_dim), dtype=np.float64)

    for k in range(60 * online_hour):
        if np.mod(np.floor(k / 60), 2) == 0:
            r[k, :] = plant.ref1
        else:
            r[k, :] = plant.ref2

        if k < 60 * learning_online_ini_horizon:
            u[k, :] = plant.ss_u
        elif np.mod(np.floor(k / 60), 2) == 0:
            u[k, :] = controller1.control(x_est[k, :])
        else:
            u[k, :] = controller2.control(x_est[k, :])
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :])
        x_est[k + 1, :] = cstr_estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :])

        if int(k + 1) in 60 * save_index:
            print("seed:", config1.seed, "current online step:", int((k + 1) / 60))
            if save_bool:
                config1.save_data(directory, file_name, config1.seed, x, x_est, u, y, p, r, c)
            controller1.save_controller(directory, file_name + "-ctrl1-" + str(int((k + 1) / 60)))
            controller2.save_controller(directory, file_name + "-ctrl2-" + str(int((k + 1) / 60)))
    r[-1, :] = r[-2, :]
    u[-1, :] = controller1.control(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_temperature()

    if save_bool:
        config1.save_data(directory, file_name, config1.seed, x, x_est, u, y, p, r, c)

    return cstr_sysid


def run_test(
    config1,
    controller1,
    controller2,
    sysid,
    estimator,
    horizon=60 * 3,
    init_horizon=20,
    safety_cfg=None,
):
    test_seed = config1.seed + 12345
    np.random.seed(test_seed)
    random.seed(test_seed)

    plant = CSTRModel(seed=test_seed)
    x = np.zeros((horizon, plant.x_dim), dtype=np.float64)
    y = np.zeros((horizon, plant.y_dim), dtype=np.float64)
    u = np.zeros((horizon, plant.u_dim), dtype=np.float64)
    p = np.zeros((horizon, plant.p_dim), dtype=np.float64)
    r = np.zeros((horizon, plant.r_dim), dtype=np.float64)
    c = np.zeros((horizon, 1), dtype=np.float64)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    x_est = np.zeros((horizon, sysid.x_est_dim), dtype=np.float64)

    unsafe = False
    unsafe_reason = None
    if safety_cfg is None:
        safety_cfg = {}
    runaway_temp_threshold = safety_cfg.get("runaway_temp_threshold", 115.0)
    runaway_temp_rate_threshold = safety_cfg.get("runaway_temp_rate_threshold", 0.5)
    input_rate_threshold = safety_cfg.get("input_rate_threshold", np.array([3.0, 500.0]))
    input_rate_threshold = np.array(input_rate_threshold, dtype=np.float64)
    for k in range(horizon - 1):
        if np.mod(np.floor(k / 60), 2) == 0:
            r[k, :] = plant.ref1
        else:
            r[k, :] = plant.ref2

        if k < init_horizon:
            u[k, :] = plant.ss_u
        elif np.mod(np.floor(k / 60), 2) == 0:
            u[k, :] = controller1.control(x_est[k, :])
        else:
            u[k, :] = controller2.control(x_est[k, :])
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :])
        x_est[k + 1, :] = estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :])
        if np.any(u[k, :] < plant.u_min) or np.any(u[k, :] > plant.u_max):
            unsafe = True
            unsafe_reason = "action_out_of_bounds"
            break
        if np.any(np.abs(u[k, :] - u[k - 1, :]) > input_rate_threshold) and k > 0:
            unsafe = True
            unsafe_reason = "action_rate_exceeded"
            break
        temp_rate = plant.reactor_temp_rate(x[k, :], u[k, :])
        if temp_rate > runaway_temp_rate_threshold and x[k, 2] >= runaway_temp_threshold:
            unsafe = True
            unsafe_reason = "thermal_runaway"
            break

    r[-1, :] = plant.ref1
    u[-1, :] = controller1.control(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_temperature()
    return float(np.sum(c)), unsafe, unsafe_reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--online-hour", type=int, default=1001)
    parser.add_argument("--offline-hour", type=int, default=0)
    parser.add_argument("--controller", choices=["ref", "smpl"], default="ref")
    parser.add_argument("--sysid-method", default="LSTM")
    parser.add_argument("--estimate-method", default="STACKING")
    parser.add_argument("--control-method", default="MPC")
    parser.add_argument("--extra-name", default="SMPL")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("Data/cstr_smpl"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _ensure_ref_path()
    from cstr_config import CstrConfig

    total_computation_time = datetime.now()
    execute_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.outdir.mkdir(parents=True, exist_ok=True)
    directory = args.outdir / (execute_time + "-" + args.control_method + "-" + args.extra_name)
    directory.mkdir(parents=True, exist_ok=True)

    online_save_index = np.array(
        [5, 10, 20, 30, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int
    )
    offline_save_index = np.array(
        [5, 10, 20, 30, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int
    )

    online_costs = np.zeros((args.seeds, online_save_index.size), dtype=np.float64)
    offline_costs = np.zeros((args.seeds, offline_save_index.size), dtype=np.float64)
    sysid_errors = np.zeros((args.seeds, offline_save_index.size), dtype=np.float64)
    online_failures = np.zeros((args.seeds, online_save_index.size), dtype=np.int64)
    offline_failures = np.zeros((args.seeds, offline_save_index.size), dtype=np.int64)

    for seed in range(args.seeds):
        random.seed(seed)
        np.random.seed(seed)
        print("Current run is", seed)

        config1 = CstrConfig()
        config1.plot_bool = False
        config1.ref = np.array([1.09, 114.2], dtype=np.float64)
        config1.sysid_method = args.sysid_method
        config1.estimate_method = args.estimate_method
        config1.control_method = args.control_method
        config1.seed = seed

        config2 = CstrConfig()
        config2.plot_bool = False
        config2.ref = np.array([1.04, 94.2], dtype=np.float64)
        config2.QMPC_prediction_horizon = 2
        config2.sysid_method = args.sysid_method
        config2.estimate_method = args.estimate_method
        config2.control_method = args.control_method
        config2.seed = seed

        file_name = f"{args.control_method}-{args.sysid_method}-{args.estimate_method}"
        file_name += f"-SEED{seed}-{args.online_hour}-{args.offline_hour}-{args.extra_name}"

        if args.save:
            config1.save_settings(directory, file_name + "-1")
            config2.save_settings(directory, file_name + "-2")

        plant = CSTRModel(seed=seed)

        if args.offline_hour > 0:
            errors, cstr_sysid = offline_learning(
                plant,
                config1,
                config2,
                file_name,
                directory,
                args.offline_hour,
                offline_save_index,
                args.controller,
                args.save,
            )
            sysid_errors[seed, :] = errors
            if not args.skip_eval:
                from Controls import controller as ref_controller
                from Estimators import estimator

                file_offline = file_name + "-offline"
                for k, operated_time in enumerate(offline_save_index):
                    ctrl1 = ref_controller.CONTROLLER(cstr_sysid, config1)
                    ctrl2 = ref_controller.CONTROLLER(cstr_sysid, config2)
                    ctrl1.load_controller(directory, file_offline + "-ctrl1-" + str(operated_time))
                    ctrl2.load_controller(directory, file_offline + "-ctrl2-" + str(operated_time))
                    cstr_sysid = ctrl1.controller.sysid
                    cstr_estimator = estimator.ESTIMATOR(sysid=cstr_sysid, config=config1)
                    offline_costs[seed, k], unsafe, _ = run_test(
                        config1, ctrl1.controller, ctrl2.controller, cstr_sysid, cstr_estimator
                    )
                    offline_failures[seed, k] = int(unsafe)

        if args.online_hour > 0:
            cstr_sysid = online_learning(
                plant,
                config1,
                config2,
                file_name,
                directory,
                args.online_hour,
                online_save_index,
                args.controller,
                args.save,
            )
            if not args.skip_eval:
                from Controls import controller as ref_controller
                from Estimators import estimator

                file_online = file_name + "-online"
                for k, operated_time in enumerate(online_save_index):
                    ctrl1 = ref_controller.CONTROLLER(cstr_sysid, config1)
                    ctrl2 = ref_controller.CONTROLLER(cstr_sysid, config2)
                    ctrl1.load_controller(directory, file_online + "-ctrl1-" + str(operated_time))
                    ctrl2.load_controller(directory, file_online + "-ctrl2-" + str(operated_time))
                    cstr_sysid = ctrl1.controller.sysid
                    cstr_estimator = estimator.ESTIMATOR(sysid=cstr_sysid, config=config1)
                    online_costs[seed, k], unsafe, _ = run_test(
                        config1, ctrl1.controller, ctrl2.controller, cstr_sysid, cstr_estimator
                    )
                    online_failures[seed, k] = int(unsafe)

        np.savetxt(fname=directory / "cstr_results_online_cost.txt", X=online_costs, fmt="%12.8f")
        np.savetxt(fname=directory / "cstr_results_offline_cost.txt", X=offline_costs, fmt="%12.8f")
        np.savetxt(fname=directory / "cstr_results_sysid_errors.txt", X=sysid_errors, fmt="%12.8f")
        np.savetxt(fname=directory / "cstr_results_online_failures.txt", X=online_failures, fmt="%d")
        np.savetxt(fname=directory / "cstr_results_offline_failures.txt", X=offline_failures, fmt="%d")

    elapsed = datetime.now() - total_computation_time
    print("Total computation time:", elapsed)


if __name__ == "__main__":
    main()
