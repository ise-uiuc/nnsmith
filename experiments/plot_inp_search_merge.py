import os
import re
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    REGEX_PATTERN = '(\d+)-model-(\d+)-node-exp'
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
            if f.endswith('.csv') and f != 'model_info.csv':
                data = pd.read_csv(os.path.join(nsize_folder, f))

                sampling_time.append(data['sampling-time'].mean())
                sampling_succ_rate.append(data['sampling-succ'].mean())
                grad_time.append(data['grad-time'].mean())
                grad_succ_rate.append(data['grad-succ'].mean())
                proxy_time.append(data['proxy-time'].mean())
                proxy_succ_rate.append(data['proxy-succ'].mean())
            elif f == 'model_info.csv':
                data = pd.read_csv(os.path.join(nsize_folder, f))
                gentime = data['gen_time']
                print(f'{gentime.mean()=}, {gentime.min()=}, {gentime.max()=}')

        # sort succ rate by time
        sampling_res.append(sort_by_time(sampling_time, sampling_succ_rate))
        grad_res.append(sort_by_time(grad_time, grad_succ_rate))
        proxy_res.append(sort_by_time(proxy_time, proxy_succ_rate))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    colors = ['b', 'r', 'g']
    for i in range(3):
        c = colors[i]
        alpha = 1 - 0.36 * i

        sampling_time, sampling_succ_rate = sampling_res[i]
        grad_time, grad_succ_rate = grad_res[i]
        proxy_time, proxy_succ_rate = proxy_res[i]

        ax.plot(proxy_time * 1000, proxy_succ_rate,
                linestyle='-', color=c, alpha=alpha)
        ax.plot(grad_time * 1000, grad_succ_rate,
                linestyle=':', color=c, alpha=alpha)
        ax.plot(sampling_time * 1000, sampling_succ_rate,
                linestyle='-.', color=c, alpha=alpha)

    ax.grid(True, linestyle=':', linewidth=.5, alpha=0.5)

    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [0, 1, 2]], [
                         "Gradient (Proxy Deriv.)", "Gradient", "Sampling"],
                         loc=8, bbox_to_anchor=(0.6, 0.), title='Searching Method')
    legend2 = plt.legend([lines[i] for i in [0, 3, 6]],
                         node_sizes, loc=4, title='Model Size')
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_yticks(np.arange(0.3, 1.1, 0.1))
    ax.set_ylim(0.3, 1.0)

    ax.set_xticks(np.arange(0, 26, 5))
    ax.set_xlim(0, 25)

    ax.set_xlabel('Avg. Searching Time (millisecond)', fontweight='bold')
    ax.set_ylabel('Success Rate', fontweight='bold')

    plt.savefig(os.path.join(
        args.output, f'input-search-merge.pdf'))
    plt.savefig(os.path.join(
        args.output, f'input-search-merge.png'))
