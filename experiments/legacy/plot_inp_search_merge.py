import os
import re
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update({"text.usetex": True})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--rm_slowest", action="store_true", help="Remove the slowest run"
    )
    args = parser.parse_args()

    REGEX_PATTERN = "(\d+)-model-(\d+)-node-exp"
    node_sizes = []

    # Plot data
    # X: Time
    # Y: Succ. Rate

    def sort_by_time(time, succ_rate):
        time = np.array(time)
        succ_rate = np.array(succ_rate)
        sort_idx = time.argsort()
        return time[sort_idx], succ_rate[sort_idx]

    sampling_res = []
    grad_res = []
    proxy_res = []

    for nsize_folder in args.root:
        res = re.match(REGEX_PATTERN, nsize_folder)
        n_model, n_nodes = res.groups()
        node_sizes.append(int(n_nodes))

        sampling_time = []
        sampling_succ_rate = []

        grad_time = []
        grad_succ_rate = []

        proxy_time = []
        proxy_succ_rate = []

        for f in os.listdir(nsize_folder):
            if f.endswith(".csv") and f != "model_info.csv":
                data = pd.read_csv(os.path.join(nsize_folder, f))

                # Do not count the slowest iteration (1st iter usually)
                # as initialization takes some time.
                last_idx = -2 if args.rm_slowest else -1

                idx = data["sampling-time"].to_numpy().argsort()[:last_idx]
                sampling_time.append(data["sampling-time"][idx].mean())
                sampling_succ_rate.append(data["sampling-succ"][idx].mean())

                idx = data["grad-time"].to_numpy().argsort()[:last_idx]
                grad_time.append(data["grad-time"][idx].mean())
                grad_succ_rate.append(data["grad-succ"][idx].mean())

                idx = data["proxy-time"].to_numpy().argsort()[:last_idx]
                proxy_time.append(data["proxy-time"][idx].mean())
                proxy_succ_rate.append(data["proxy-succ"][idx].mean())
            elif f == "model_info.csv":
                data = pd.read_csv(os.path.join(nsize_folder, f))
                gentime = data["gen_time"]
                print(f"{gentime.mean()=}, {gentime.min()=}, {gentime.max()=}")

        # sort succ rate by time
        sampling_res.append(sort_by_time(sampling_time, sampling_succ_rate))
        grad_res.append(sort_by_time(grad_time, grad_succ_rate))
        proxy_res.append(sort_by_time(proxy_time, proxy_succ_rate))

    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)

    colors = ["dodgerblue", "violet", "green"]  # ['b', 'r', 'g']
    markers = ["1", ".", "+"]
    markercolor = "k"
    markersize = 10
    markeredgewidth = 1.2
    lw = 1.5

    max_time = 0

    for i in range(3):
        c = colors[i]
        alpha = 1 - 0.36 * i

        sampling_time, sampling_succ_rate = sampling_res[i]
        grad_time, grad_succ_rate = grad_res[i]
        proxy_time, proxy_succ_rate = proxy_res[i]

        # self *= 1000: sec -> milli
        sampling_time *= 1000
        grad_time *= 1000
        proxy_time *= 1000

        ax.plot(
            proxy_time,
            proxy_succ_rate,
            marker=markers[0],
            markeredgecolor=markercolor,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            linestyle="--",
            color=c,
            lw=lw,
        )
        ax.plot(
            grad_time,
            grad_succ_rate,
            marker=markers[1],
            markeredgecolor=markercolor,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            linestyle="--",
            color=c,
            lw=lw,
        )
        ax.plot(
            sampling_time,
            sampling_succ_rate,
            marker=markers[2],
            markeredgecolor=markercolor,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            linestyle="--",
            color=c,
            lw=lw,
        )

        max_time = max(max_time, sampling_time.max())

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    lines = ax.get_lines()
    legend1 = plt.legend(
        [lines[i] for i in [0, 1, 2]],
        ["Gradient (Proxy Deriv.)", "Gradient", "Sampling"],
        loc="upper right",
        title="Searching Method",
    )

    patches = []
    for i in range(3):
        patches.append(mpatches.Patch(color=colors[i], label=node_sizes[i]))
    legend2 = plt.legend(
        handles=patches,
        loc="center right",
        bbox_to_anchor=(1, 0.36),
        title="Model Size",
    )

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_yticks(np.arange(0.6, 1.1, 0.1))
    ax.set_ylim(0.6, 1.0)

    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlim(0, max_time + 0.5)

    ax.set_xlabel("Avg. Searching Time (millisecond)", fontweight="bold")
    ax.set_ylabel("Success Rate", fontweight="bold")

    plt.savefig(os.path.join(args.output, f"input-search-merge.pdf"))
    plt.savefig(os.path.join(args.output, f"input-search-merge.png"))
