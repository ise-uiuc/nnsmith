import os
import re
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

# plt.rc('hatch', linewidth=)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()

    REGEX_PATTERN = '(\d+)-model-(\d+)-node-exp'
    res = re.match(REGEX_PATTERN, args.root)
    n_model, n_nodes = res.groups()

    # Plot data
    # X: Time
    # Y: Succ. Rate

    sampling_time = []
    sampling_succ_rate = []

    grad_time = []
    grad_succ_rate = []

    for f in os.listdir(args.root):
        if f.endswith('.csv') and f != 'model_info.csv':
            data = pd.read_csv(os.path.join(args.root, f))
            sampling_time.append(data['sampling-time'].mean())
            sampling_succ_rate.append(data['sampling-succ'].mean())
            grad_time.append(data['grad-time'].mean())
            grad_succ_rate.append(data['grad-succ'].mean())

    # sort succ rate by time
    sampling_time, sampling_succ_rate = (
        np.array(sampling_time) * 1000, np.array(sampling_succ_rate))
    sort_idx = np.argsort(sampling_time)
    sampling_time, sampling_succ_rate = (
        sampling_time[sort_idx], sampling_succ_rate[sort_idx])

    grad_time, grad_succ_rate = (
        np.array(grad_time) * 1000, np.array(grad_succ_rate))
    sort_idx = np.argsort(grad_time)
    grad_time, grad_succ_rate = (
        grad_time[sort_idx], grad_succ_rate[sort_idx])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.1)

    ax1.plot(sampling_time, sampling_succ_rate, markersize=10,
             marker='x', linestyle='--', label='Sampling')
    ax1.plot(grad_time, grad_succ_rate, marker='*', markersize=10,
             linestyle='--', label='Gradient')

    ax2.plot(sampling_time, sampling_succ_rate, markersize=10,
             marker='x', linestyle='--', label='Sampling')
    ax2.plot(grad_time, grad_succ_rate, marker='*', markersize=10,
             linestyle='--', label='Gradient')

    ax1.set_ylim(.69, 1)
    ax2.set_ylim(.3, .41)

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.grid(True, linestyle='--', linewidth=.5)
    ax2.grid(True, linestyle='--', linewidth=.5)

    plt.legend(loc='lower right')
    ax1.set_yticks([.7, .8, .9, 1])

    ax2.set_xlabel('Avg. Searching Time (millisecond)', fontweight='bold')
    ax2.xaxis.set_label_coords(0.5, 0.05, transform=fig.transFigure)
    ax2.set_ylabel('Success Rate', fontweight='bold')
    ax2.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

    plt.savefig(f'input-search{n_model}-{n_nodes}.pdf')
    plt.savefig(f'input-search{n_model}-{n_nodes}.png')
