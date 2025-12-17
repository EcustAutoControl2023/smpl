"""
Plot SMB offline dataset trajectories (purity, actions, cost).

Usage (from repo root per AGENTS.md):
  UV_CACHE_DIR=.uv-cache uv run scripts/smb_plot_dataset.py \\
    --dataset offline_datasets/smb/smb_mpc_n4sid_matlab_off20_episodes1_h192_norm=False.pkl \\
    --label off20 --ref 0.99 --out smb_plot.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(path: Path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    required_keys = ["observations", "actions", "rewards", "terminals"]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"Dataset missing key: {k}")
    return data


def split_episodes(dataset):
    obs = np.asarray(dataset["observations"])
    acts = np.asarray(dataset["actions"])
    rewards = np.asarray(dataset["rewards"])
    terminals = np.asarray(dataset["terminals"])
    timeouts = np.asarray(dataset.get("timeouts", np.zeros_like(terminals)), dtype=bool)

    episodes = []
    start = 0
    for idx, term in enumerate(terminals):
        if term:
            end = idx + 1
            episodes.append(
                {
                    "observations": obs[start:end],
                    "actions": acts[start:end],
                    "rewards": rewards[start:end],
                    "terminals": terminals[start:end],
                    "timeouts": timeouts[start:end],
                }
            )
            start = end
    if start < len(obs):
        episodes.append(
            {
                "observations": obs[start:],
                "actions": acts[start:],
                "rewards": rewards[start:],
                "terminals": terminals[start:],
                "timeouts": timeouts[start:],
            }
        )
    return episodes


def plot_dataset(episodes, ref=0.99, label="dataset", out_path=None):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs = axs.flatten()

    # Purities
    ax_ex, ax_ra = axs[0], axs[1]
    for i, epi in enumerate(episodes):
        obs = epi["observations"]
        acts = epi["actions"]
        rewards = epi["rewards"]
        y0, y1, y2, y3 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        pur_ex = y0 / (y0 + y1 + 1e-9)
        pur_ra = y3 / (y2 + y3 + 1e-9)
        t = np.arange(len(obs))
        color = colors[i % len(colors)]
        ax_ex.plot(t, pur_ex, label=f"{label}-epi{i}", color=color)
        ax_ra.plot(t, pur_ra, label=f"{label}-epi{i}", color=color)
    for ax, title in zip([ax_ex, ax_ra], ["Extract purity", "Raffinate purity"]):
        ax.axhline(ref, linestyle="--", color="k")
        ax.set_ylabel(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Actions u1..u4
    for j in range(4):
        ax = axs[2 + j]
        for i, epi in enumerate(episodes):
            acts = epi["actions"]
            t = np.arange(len(acts))
            color = colors[i % len(colors)]
            ax.step(t, acts[:, j], where="post", label=f"{label}-epi{i}" if j == 0 else None, color=color)
        ax.set_ylabel(f"u{j+1}")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    if axs[2].get_legend():
        axs[2].legend()

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Path to offline dataset pickle")
    parser.add_argument("--ref", type=float, default=0.99, help="Reference purity line")
    parser.add_argument("--label", type=str, default=None, help="Label for legend (e.g., offline-switch)")
    parser.add_argument("--out", type=Path, default=None, help="Output image path; show if omitted")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    episodes = split_episodes(dataset)
    label = args.label if args.label else args.dataset.stem
    plot_dataset(episodes, ref=args.ref, label=label, out_path=args.out)


if __name__ == "__main__":
    main()

