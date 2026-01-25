"""
Evaluate trained CSTR controllers and generate a comparison figure.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from smpl.envs import CSTRModel
from smpl.envs.cstr_controller import CSTRBaselineSysId


@dataclass
class SeriesSpec:
    label: str
    ctrl1_prefix: str
    ctrl2_prefix: str


def _ensure_ref_path():
    ref_root = Path(__file__).resolve().parent.parent / "ref" / "Quantitative-Comparison-of-RL-and-MPC"
    ref_root_str = str(ref_root)
    if ref_root_str not in os.sys.path:
        os.sys.path.insert(0, ref_root_str)
    return ref_root


def _load_controllers(series: SeriesSpec, config1, config2):
    from Controls import controller as ref_controller

    dummy_sysid = CSTRBaselineSysId(CSTRModel(seed=config1.seed))
    ctrl1 = ref_controller.CONTROLLER(dummy_sysid, config1)
    ctrl2 = ref_controller.CONTROLLER(dummy_sysid, config2)

    ctrl1_dir, ctrl1_name = _split_prefix(series.ctrl1_prefix)
    ctrl2_dir, ctrl2_name = _split_prefix(series.ctrl2_prefix)
    ctrl1.load_controller(ctrl1_dir, ctrl1_name)
    ctrl2.load_controller(ctrl2_dir, ctrl2_name)
    return ctrl1.controller, ctrl2.controller


def _split_prefix(prefix: str) -> Tuple[Path, str]:
    path = Path(prefix)
    return path.parent, path.name


def _run_episode(controller1, controller2, horizon: int, init_horizon: int):
    from Estimators import estimator

    cfg = controller1.config
    plant = CSTRModel(seed=cfg.seed)
    x = np.zeros((horizon, plant.x_dim), dtype=np.float64)
    y = np.zeros((horizon, plant.y_dim), dtype=np.float64)
    u = np.zeros((horizon, plant.u_dim), dtype=np.float64)
    p = np.zeros((horizon, plant.p_dim), dtype=np.float64)
    r = np.zeros((horizon, plant.r_dim), dtype=np.float64)
    c = np.zeros((horizon, 1), dtype=np.float64)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    sysid = controller1.sysid
    x_est = np.zeros((horizon, sysid.x_est_dim), dtype=np.float64)
    cstr_estimator = estimator.ESTIMATOR(sysid=sysid, config=cfg)
    safety_cfg = {
        "runaway_temp_threshold": 115.0,
        "runaway_temp_rate_threshold": 0.5,
        "hard_temp_threshold": 130.0,
    }
    if hasattr(cfg, "runaway_temp_threshold"):
        safety_cfg["runaway_temp_threshold"] = cfg.runaway_temp_threshold
    if hasattr(cfg, "runaway_temp_rate_threshold"):
        safety_cfg["runaway_temp_rate_threshold"] = cfg.runaway_temp_rate_threshold
    if hasattr(cfg, "hard_temp_threshold"):
        safety_cfg["hard_temp_threshold"] = cfg.hard_temp_threshold

    unsafe = False
    unsafe_reason = None
    terminal_step = horizon - 1
    max_temp_rate = -np.inf
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
        if np.any(u[k, :] < plant.u_min) or np.any(u[k, :] > plant.u_max):
            unsafe = True
            unsafe_reason = "action_out_of_bounds"
            terminal_step = k
            break
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()
        next_x, p_bdd = plant.step(x[k, :], u[k, :], return_p=True)
        x[k + 1, :] = next_x
        y[k + 1, :] = plant.get_observation(x[k + 1, :])
        x_est[k + 1, :] = cstr_estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :])

        rate_thr = float(safety_cfg["runaway_temp_rate_threshold"])
        temp_thr = float(safety_cfg.get("runaway_temp_threshold", -np.inf))
        hard_thr = float(safety_cfg.get("hard_temp_threshold", np.inf))
        dt = float(plant.time_interval)
        dT_avg = (next_x[2] - x[k, 2]) / dt
        dT_inst = plant.reactor_temp_rate(x[k, :], u[k, :], p=p_bdd)
        max_temp_rate = max(max_temp_rate, max(dT_avg, dT_inst))
        T_prev = float(x[k, 2])
        T_next = float(next_x[2])
        if T_next >= hard_thr:
            unsafe = True
            unsafe_reason = "thermal_runaway"
            terminal_step = k
            break
        if max(dT_avg, dT_inst) > rate_thr and max(T_prev, T_next) >= temp_thr:
            unsafe = True
            unsafe_reason = "thermal_runaway"
            terminal_step = k
            break

    r[-1, :] = plant.ref1
    u[-1, :] = controller1.control(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_temperature()
    if unsafe:
        y[terminal_step + 1 :] = np.nan
        u[terminal_step + 1 :] = np.nan
        r[terminal_step + 1 :] = np.nan
        c[terminal_step + 1 :] = np.nan
    debug_info = {
        "max_temp_rate": max_temp_rate if max_temp_rate != -np.inf else None,
        "max_du": None,
    }
    return y, u, r, c, unsafe, unsafe_reason, terminal_step, debug_info


def _plot_series(series_results, horizon, out_path: Path):
    fig = plt.figure(figsize=(12, 8))
    x_time = np.arange(horizon)
    ref = series_results[0][1]["r"]

    ax1 = plt.subplot(221)
    for label, data in series_results:
        ax1.plot(x_time, data["y"][:, 0], label=label)
        if data.get("unsafe"):
            idx = data.get("terminal_step", horizon - 1)
            ax1.plot(idx, data["y"][idx, 0], "x", color=ax1.lines[-1].get_color())
        if data.get("max_temp_rate") is not None:
            ax1.text(
                0.02,
                0.95,
                f"max dT/dt={data['max_temp_rate']:.3f}",
                transform=ax1.transAxes,
                fontsize=8,
                color=ax1.lines[-1].get_color(),
            )
    ax1.plot(x_time, ref[:, 0], "--k", label="Reference")
    ax1.set_title("y1")
    ax1.set_xlabel("Time (Min)")
    ax1.set_ylabel("Value")

    ax2 = plt.subplot(222)
    for label, data in series_results:
        ax2.plot(x_time, data["y"][:, 1], label=label)
        if data.get("unsafe"):
            idx = data.get("terminal_step", horizon - 1)
            ax2.plot(idx, data["y"][idx, 1], "x", color=ax2.lines[-1].get_color())
    ax2.plot(x_time, ref[:, 1], "--k", label="Reference")
    ax2.set_title("y2")
    ax2.set_xlabel("Time (Min)")
    ax2.set_ylabel("Value")

    ax3 = plt.subplot(223)
    for label, data in series_results:
        ax3.plot(x_time, data["u"][:, 0], label=label)
        if data.get("unsafe"):
            idx = data.get("terminal_step", horizon - 1)
            ax3.plot(idx, data["u"][idx, 0], "x", color=ax3.lines[-1].get_color())
    ax3.set_title("u1")
    ax3.set_xlabel("Time (Min)")
    ax3.set_ylabel("Value")

    ax4 = plt.subplot(224)
    for label, data in series_results:
        ax4.plot(x_time, data["u"][:, 1], label=label)
        if data.get("unsafe"):
            idx = data.get("terminal_step", horizon - 1)
            ax4.plot(idx, data["u"][idx, 1], "x", color=ax4.lines[-1].get_color())
        if data.get("max_du") is not None:
            ax4.text(
                0.02,
                0.95 - 0.08 * len(ax4.texts),
                f"max |du|={data['max_du'][0]:.2f},{data['max_du'][1]:.2f}",
                transform=ax4.transAxes,
                fontsize=8,
                color=ax4.lines[-1].get_color(),
            )
    ax4.set_title("u2")
    ax4.set_xlabel("Time (Min)")
    ax4.set_ylabel("Value")
    ax4.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_series(raw_series: List[str]) -> List[SeriesSpec]:
    series_specs = []
    for item in raw_series:
        parts = item.split(",")
        if len(parts) != 3:
            raise ValueError("Series should be formatted as label,ctrl1_prefix,ctrl2_prefix")
        series_specs.append(SeriesSpec(label=parts[0], ctrl1_prefix=parts[1], ctrl2_prefix=parts[2]))
    return series_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, help="label,ctrl1_prefix,ctrl2_prefix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=60 * 3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("figures/cstr_eval.png"))
    args = parser.parse_args()

    _ensure_ref_path()
    from cstr_config import CstrConfig

    random.seed(args.seed)
    np.random.seed(args.seed)

    config1 = CstrConfig()
    config1.seed = args.seed
    config1.control_method = "MPC"
    config1.sysid_method = "LSTM"
    config1.estimate_method = "STACKING"

    config2 = CstrConfig()
    config2.seed = args.seed
    config2.control_method = "MPC"
    config2.sysid_method = "LSTM"
    config2.estimate_method = "STACKING"
    config2.ref = np.array([1.04, 94.2], dtype=np.float64)

    series_specs = _parse_series(args.series)
    series_results = []
    for spec in series_specs:
        ctrl1, ctrl2 = _load_controllers(spec, config1, config2)
        y, u, r, c, unsafe, unsafe_reason, terminal_step, debug_info = _run_episode(
            ctrl1, ctrl2, args.horizon, args.warmup
        )
        print(
            f"{spec.label}: unsafe={unsafe} reason={unsafe_reason} "
            f"terminal_step={terminal_step} max_dTdt={debug_info['max_temp_rate']}"
        )
        series_results.append(
            (
                spec.label,
                {
                    "y": y,
                    "u": u,
                    "r": r,
                    "c": c,
                    "unsafe": unsafe,
                    "unsafe_reason": unsafe_reason,
                    "terminal_step": terminal_step,
                    "max_temp_rate": debug_info["max_temp_rate"],
                    "max_du": debug_info["max_du"],
                },
            )
        )

    _plot_series(series_results, args.horizon, args.out)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
