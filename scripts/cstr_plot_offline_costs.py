"""
Plot offline MPC comparison curves and training progress for CSTR.

Example:
  uv run scripts/cstr_plot_offline_costs.py \\
    --series "LSTM,Data/cstr_smpl/2026-01-22-13-35-44-MPC-SMPL/cstr_results_offline_cost.txt" \\
    --out figures/cstr_offline_compare.png \\
    --last-run Data/cstr_smpl/2026-01-22-13-35-44-MPC-SMPL/cstr_results_offline_cost.txt \\
    --last-out figures/cstr_last_run.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_HOURS = np.array(
    [5, 10, 20, 30, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    dtype=int,
)


def _parse_series(items: List[str]) -> List[Tuple[str, Path]]:
    series = []
    for item in items:
        if "," not in item:
            raise ValueError("Each --series must be 'label,path'")
        label, path = item.split(",", 1)
        series.append((label.strip(), Path(path.strip())))
    return series


def _load_costs(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _format_axes(ax, hours: np.ndarray, updates: np.ndarray):
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Cost")
    ax.set_xlim(hours.min(), hours.max())

    secax = ax.secondary_xaxis(
        "bottom",
        functions=(lambda x: x * 60.0, lambda x: x / 60.0),
    )
    secax.set_xlabel("Number of Update [Below]")
    secax.set_xticks(updates)
    secax.set_xticklabels([f"{int(u)}" for u in updates])


def plot_offline_comparison(
    series: List[Tuple[str, Path]],
    out_path: Path,
    hours: np.ndarray,
    optimal: float | None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    updates = hours * 60

    for label, path in series:
        data = _load_costs(path)
        mean = data.mean(axis=0)
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        ax.plot(hours, mean, marker="o", label=label)
        ax.fill_between(hours, lo, hi, alpha=0.15)

    if optimal is not None:
        ax.axhline(optimal, color="k", linestyle="--", linewidth=1.0, label="Optimal")

    _format_axes(ax, hours, updates)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_last_run(
    cost_path: Path,
    out_path: Path,
    hours: np.ndarray,
    label: str,
):
    data = _load_costs(cost_path)
    mean = data.mean(axis=0)
    lo = data.min(axis=0)
    hi = data.max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hours, mean, marker="o", label=label)
    if data.shape[0] > 1:
        ax.fill_between(hours, lo, hi, alpha=0.15)

    updates = hours * 60
    _format_axes(ax, hours, updates)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", default=[], help="label,path")
    parser.add_argument("--out", type=Path, default=Path("figures/cstr_offline_compare.png"))
    parser.add_argument("--optimal", type=float, default=None)
    parser.add_argument("--last-run", type=Path, default=None)
    parser.add_argument("--last-out", type=Path, default=Path("figures/cstr_last_run.png"))
    parser.add_argument("--last-label", type=str, default="CSTR MPC LSTM (last run)")
    parser.add_argument("--hours", type=str, default=None, help="Comma-separated hours matching saved checkpoints")
    args = parser.parse_args()

    hours = DEFAULT_HOURS
    if args.hours:
        hours = np.array([int(x.strip()) for x in args.hours.split(",")], dtype=int)

    if args.series:
        series = _parse_series(args.series)
        plot_offline_comparison(series, args.out, hours, args.optimal)

    if args.last_run is not None:
        plot_last_run(args.last_run, args.last_out, hours, args.last_label)


if __name__ == "__main__":
    main()
