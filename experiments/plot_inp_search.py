import os
import re
import argparse
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

# plt.rc('hatch', linewidth=)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
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

    fig, ax = plt.subplots(
        1, 1, constrained_layout=True, figsize=(7, 3))

    ax.plot(sampling_time, sampling_succ_rate, markersize=10,
            marker='o', linestyle='--', label='Sampling')
    ax.plot(grad_time, grad_succ_rate, marker='*', markersize=10,
            linestyle='--', label='Gradient')

    plt.legend(loc='lower right')
    plt.xlabel('Avg. Searching Time (millisecond)', fontweight='bold')
    plt.ylabel('Success Rate', fontweight='bold')

    plt.savefig(f'input-search{n_model}-{n_nodes}.pdf')
    plt.savefig(f'input-search{n_model}-{n_nodes}.png')
