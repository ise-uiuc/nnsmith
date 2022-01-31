import os
import re
import argparse
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
    file_to_plot_date = {}

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

            # use the latest data
            if n_nodes not in files_to_plot or modification_date(fname) > file_to_plot_date[n_nodes]:
                file_to_plot_date[n_nodes] = modification_date(fname)
                files_to_plot[n_nodes] = fname

    for n_nodes, fname in files_to_plot.items():
        data = pd.read_csv(fname)
        plot_data.setdefault('sampling', {})[
            n_nodes] = sum(data['v3-succ'])
        plot_data.setdefault('sampling + gradient', {}
                             )[n_nodes] = sum(data['grad-succ'])

    if not plot_data:
        print('No data found, please check the regex pattern', REGEX_PATTERN)
        exit(1)

    col_width = 0.8
    bar_width = col_width / len(plot_data)
    base_x = np.arange(len(plot_data['sampling']))

    ax = plt.subplot(111)
    legends = []
    keys = sorted(plot_data['sampling'].keys())
    HATCHES = ['+', '*', '|', '-', '.', '/', 'O', 'o', 'x', '\\']

    for idx, (label, data_dict) in enumerate(plot_data.items()):
        legends.append(label)
        raw_data = [data_dict[n_nodes] for n_nodes in keys]
        x_pos = base_x - 0.5 * col_width + (idx + 0.5) * bar_width
        ax.bar(x_pos, raw_data,
               width=bar_width, label=label, align='center', hatch=HATCHES[idx], edgecolor='violet', color='lavender')
        for x, v in zip(x_pos, raw_data):
            ax.text(x - 0.3 * bar_width, v + .25, str(v), fontweight='bold')

    # plt.legend(legends)
    plt.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 1.1),
               fancybox=True, shadow=True, ncol=5)
    plt.xticks(base_x, keys)
    plt.xlabel('# Operators / Model', fontweight='bold')
    plt.ylabel('# Models w/ Valid Inputs', fontweight='bold')

    plt.savefig(f'plot-inp-search-{MODEL_SIZE_TO_GLOB}.pdf')
    plt.savefig(f'plot-inp-search-{MODEL_SIZE_TO_GLOB}.png')
