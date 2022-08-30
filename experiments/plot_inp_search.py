import os
import re
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    REGEX_PATTERN = "(\d+)-model-(\d+)-node-exp"
    res = re.match(REGEX_PATTERN, args.root)
    n_model, n_nodes = res.groups()

    # Plot data
    # X: Time
    # Y: Succ. Rate

    sampling_time = []
    sampling_succ_rate = []

    grad_time = []
    grad_succ_rate = []

    proxy_time = []
    proxy_succ_rate = []

    for f in os.listdir(args.root):
        if f.endswith(".csv") and f != "model_info.csv":
            data = pd.read_csv(os.path.join(args.root, f))

            sampling_time.append(data["sampling-time"].mean())
            sampling_succ_rate.append(data["sampling-succ"].mean())
            grad_time.append(data["grad-time"].mean())
            grad_succ_rate.append(data["grad-succ"].mean())
            proxy_time.append(data["proxy-time"].mean())
            proxy_succ_rate.append(data["proxy-succ"].mean())
        elif f == "model_info.csv":
            data = pd.read_csv(os.path.join(args.root, f))
            gentime = data["gen_time"]
            print(f"{gentime.mean()=}, {gentime.min()=}, {gentime.max()=}")

    def sort_by_time(time, succ_rate):
        time = np.array(time)
        succ_rate = np.array(succ_rate)
        sort_idx = time.argsort()
        return time[sort_idx], succ_rate[sort_idx]

    # sort succ rate by time
    sampling_time, sampling_succ_rate = sort_by_time(sampling_time, sampling_succ_rate)
    grad_time, grad_succ_rate = sort_by_time(grad_time, grad_succ_rate)
    proxy_time, proxy_succ_rate = sort_by_time(proxy_time, proxy_succ_rate)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    fig.subplots_adjust(hspace=0.1)

    for ax in [ax1, ax2]:
        ax.plot(
            sampling_time * 1000,
            sampling_succ_rate,
            markersize=10,
            marker="x",
            linestyle="--",
            label="Sampling",
        )
        ax.plot(
            grad_time * 1000,
            grad_succ_rate,
            marker="*",
            markersize=10,
            linestyle="--",
            label="Gradient",
        )
        ax.plot(
            proxy_time * 1000,
            proxy_succ_rate,
            marker="^",
            markersize=10,
            linestyle="--",
            label="Gradient (Proxy Deriv.)",
        )
        ax.grid(True, linestyle="--", linewidth=0.5)

    ax1.set_ylim(0.66, 1)
    ax2.set_ylim(0.29, 0.43)

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.legend(loc="lower right")
    ax1.set_yticks([0.7, 0.8, 0.9, 1])
    ax2.set_yticks([0.3, 0.4])

    ax2.set_xlabel("Avg. Searching Time (millisecond)", fontweight="bold")
    ax2.xaxis.set_label_coords(0.5, 0.05, transform=fig.transFigure)
    ax2.set_ylabel("Success Rate", fontweight="bold")
    ax2.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

    plt.savefig(os.path.join(args.output, f"input-search{n_model}-{n_nodes}.pdf"))
    plt.savefig(os.path.join(args.output, f"input-search{n_model}-{n_nodes}.png"))
