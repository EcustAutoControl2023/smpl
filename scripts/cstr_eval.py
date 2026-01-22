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
        x_est[k + 1, :] = cstr_estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :])

    r[-1, :] = plant.ref1
    u[-1, :] = controller1.control(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_temperature()
    return y, u, r, c


def _plot_series(series_results, horizon, out_path: Path):
    fig = plt.figure(figsize=(12, 8))
    x_time = np.arange(horizon)
    ref = series_results[0][1]["r"]

    ax1 = plt.subplot(221)
    for label, data in series_results:
        ax1.plot(x_time, data["y"][:, 0], label=label)
    ax1.plot(x_time, ref[:, 0], "--k", label="Reference")
    ax1.set_title("y1")
    ax1.set_xlabel("Time (Min)")
    ax1.set_ylabel("Value")

    ax2 = plt.subplot(222)
    for label, data in series_results:
        ax2.plot(x_time, data["y"][:, 1], label=label)
    ax2.plot(x_time, ref[:, 1], "--k", label="Reference")
    ax2.set_title("y2")
    ax2.set_xlabel("Time (Min)")
    ax2.set_ylabel("Value")

    ax3 = plt.subplot(223)
    for label, data in series_results:
        ax3.plot(x_time, data["u"][:, 0], label=label)
    ax3.set_title("u1")
    ax3.set_xlabel("Time (Min)")
    ax3.set_ylabel("Value")

    ax4 = plt.subplot(224)
    for label, data in series_results:
        ax4.plot(x_time, data["u"][:, 1], label=label)
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
        y, u, r, c = _run_episode(ctrl1, ctrl2, args.horizon, args.warmup)
        series_results.append((spec.label, {"y": y, "u": u, "r": r, "c": c}))

    _plot_series(series_results, args.horizon, args.out)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
