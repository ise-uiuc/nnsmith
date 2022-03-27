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

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_model', type=int, default=None)
    args = parser.parse_args()

    MODEL_SIZE_TO_GLOB = args.n_model
    REGEX_PATTERN = 'r(\d+)-model(\d+)-node(\d+)-inp-search.csv'

    plot_data = {}

    files_to_plot = {}

    for fname in os.listdir('.'):
        res = re.match(REGEX_PATTERN, fname)
        if res:
            _, n_model, n_nodes = res.groups()

            if MODEL_SIZE_TO_GLOB is None:
                MODEL_SIZE_TO_GLOB = n_model
                print(
                    '`--n_model` is not set, using the model size from the 1st globbed file:', MODEL_SIZE_TO_GLOB)
            elif MODEL_SIZE_TO_GLOB != n_model:
                continue

            files_to_plot.setdefault(n_nodes, []).append(fname)

    v3_times = {}
    grad_times = {}

    for n_nodes, flist in files_to_plot.items():
        for fname in flist:
            data = pd.read_csv(fname)
            v3_times.setdefault(n_nodes, []).extend(data["v3-time"][1:]) # first item takes lib init time which is unfair to be considered.
            grad_times.setdefault(n_nodes, []).extend(data["grad-time"][1:])
            plot_data.setdefault('base', {}).setdefault(n_nodes, []).append(sum(data['v3-succ']))
            plot_data.setdefault('gradient', {}).setdefault(n_nodes, []).append(sum(data['grad-succ']))

    for n_nodes in v3_times.keys():
        print(f'{n_nodes} nodes: v3 mean time: {1000 * np.mean(v3_times[n_nodes]):.2f}ms')
        print(f'{n_nodes} nodes: grad mean time: {1000 * np.mean(grad_times[n_nodes]):.2f}ms')

    print(plot_data)

    means = {}
    stds = {}

    for idx, (label, data_dict) in enumerate(plot_data.items()):
        for n_node, vs in data_dict.items():
            means.setdefault(label, {})[n_node] = np.mean(vs)
            stds.setdefault(label, {})[n_node] = np.std(vs)

    if not plot_data:
        print('No data found, please check the regex pattern', REGEX_PATTERN)
        exit(1)

    col_width = 0.8
    bar_width = col_width / len(plot_data)
    base_x = np.arange(len(plot_data['base']))
    
    fig, ax = plt.subplots(
        1, 1, constrained_layout=True, figsize=(9, 5.5))

    legends = []
    keys = sorted(plot_data['base'].keys())
    HATCHES = ['X', '*', '|', '-', '.', '/', 'O', 'o', 'x', '\\']

    for idx, (label, data_dict) in enumerate(plot_data.items()):
        legends.append(label)
        # raw_data = [data_dict[n_nodes] for n_nodes in keys]
        # ax.bar(x_pos, raw_data,
        #        width=bar_width, label=label, align='center', hatch=HATCHES[idx], edgecolor='violet', color='lavender')
        mean_val = [means[label][n_nodes] for n_nodes in keys]
        std_val = [stds[label][n_nodes] for n_nodes in keys]

        print(mean_val)
        print(std_val)

        x_pos = base_x - 0.5 * col_width + (idx + 0.5) * bar_width
        container = ax.bar(x_pos, mean_val, error_kw=dict(lw=2, capsize=10, capthick=1),
               width=bar_width, label=label, yerr=std_val, align='center', hatch=HATCHES[idx], edgecolor='violet', color='lavender')

        for x, v in zip(x_pos, mean_val):
            ax.text(x, v + .25, str(v), fontweight='bold', fontsize=MEDIUM_SIZE)
        
        _, _, (vline,) = container.errorbar.lines
        vline.set_color('darkgrey')

    plt.legend(legends)
    # plt.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #            fancybox=True, shadow=True, ncol=5)
    plt.xticks(base_x, [f'{k}-node model' for k in keys])
    # plt.xlabel('# Operators in Models w/ Vulnerable Op.', fontweight='bold')
    plt.ylabel('# Tests w/o NaN/Inf', fontweight='bold')

    plt.savefig(f'grad-{MODEL_SIZE_TO_GLOB}.pdf')
    plt.savefig(f'grad-{MODEL_SIZE_TO_GLOB}.png')
