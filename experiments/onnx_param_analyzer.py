"""Just to figure out operators types and connections.
"""

from collections import Counter
import os
from multiprocessing import Pool, cpu_count
import traceback
from typing import Dict

import onnx
import pandas as pd
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from onnx import helper, shape_inference
from tqdm import tqdm
from onnx_graph_analyzer import analyze_one_relay


def analyze_one(model_path):
    try:
        return analyze_one_relay(model_path, use_counter=True)
    except:
        print('-------------> Skip model', model_path, 'due to exception:')
        traceback.print_exc()
        return None


def analyze_folders(folders, cache_dir=None, force=False, n_limit=None):
    res = []

    __CACHE_FILE__ = 'onnx_param_cache.pkl'
    for folder in folders:
        if os.path.exists(os.path.join(cache_dir, __CACHE_FILE__)) and not force and cache_dir is not None:
            print('===> {} already exists.'.format(
                os.path.join(folder, __CACHE_FILE__)))
            with open(os.path.join(cache_dir, __CACHE_FILE__), 'rb') as fp:
                return pickle.load(fp)

    times = []
    file_hubs = []
    least_time = None
    assert n_limit is None or len(n_limit) == len(folders)
    for i, folder in enumerate(folders):
        df = pd.read_csv(os.path.join(folder, 'gentime.csv'),
                         usecols=[0, 1], header=None)
        ts = df[0].to_numpy()

        times.append(ts)
        files = df[1].tolist()
        if n_limit is not None:
            files = files[:n_limit[i]]
        files = [os.path.join(folder, f) for f in files]
        file_hubs.append(files)

        time_cost = ts.sum()
        if least_time is None or time_cost < least_time:
            least_time = time_cost

    # only consider rows that < least_time
    for i, ts in enumerate(times):
        file_hubs[i] = file_hubs[i][:(ts < least_time).sum()]

    for files in file_hubs:
        cnts = {}
        with Pool(min(cpu_count(), len(files))) as p:
            for new in tqdm(p.imap_unordered(analyze_one, files), total=len(files)):
                if new is None:
                    continue
                for op_name, cnt in new.items():
                    if op_name not in cnts:
                        cnts[op_name] = Counter()
                    cnts[op_name].update(cnt)

        res.append(cnts)

    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, __CACHE_FILE__), 'wb') as fp:
            pickle.dump(res, fp)
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=str, nargs='+', required=True)
    parser.add_argument('--tags', type=str, nargs='+', default=None)
    # should compare models within same generation duration.
    parser.add_argument('--nlim', type=int, nargs='+', default=None)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ops', type=str, nargs='+',
                        default=None, help='Pass operator names to show.')
    args = parser.parse_args()

    ops = args.ops
    if ops is None:
        ops = ['add', 'nn.avg_pool2d', 'nn.conv2d',
               'nn.dense', 'strided_slice', 'reshape']
    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split('-')[0] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    results = analyze_folders(
        args.folders, cache_dir=args.output, force=args.force, n_limit=args.nlim)

    def to_df(cnts, suffix):
        cnts = {k[:-len(suffix)]: v for k,
                v in cnts.items() if k.endswith(suffix)}
        return pd.DataFrame({
            'name': cnts.keys(),
            'count': [len(cnt) for cnt in cnts.values()],
            'ratio': [len(cnt) / sum(cnt.values()) for cnt in cnts.values()],
            'cat': [suffix] * len(cnts)
        })
    df = pd.DataFrame()
    for tag, cnts in zip(args.tags, results):
        df_op_attr = to_df(cnts, '_attr')
        df_op_inp = to_df(cnts, '_inp')
        df_op_inp_attr = to_df(cnts, '_inp_attr')
        df1 = pd.concat([df_op_attr, df_op_inp, df_op_inp_attr])
        df1['fuzzers'] = tag
        df = df.append(df1, ignore_index=True)

# Example data frame:
#              name  count     ratio    cat        fuzzers
# 0        ceil_inp     17  0.708333  _attr  models_random
# 1            ceil      1  0.043478  _attr  models_random
# 2         tan_inp      6  0.750000  _attr  models_random
# 3             tan      1  0.125000  _attr  models_random
# 4    multiply_inp     88  0.926316  _attr  models_random
# ..            ...    ...       ...    ...            ...
# 357           cos      1  0.333333  _attr  models_guided
# 358       tan_inp      3  1.000000  _attr  models_guided
# 359           tan      1  0.333333  _attr  models_guided
# 360      atan_inp      2  1.000000  _attr  models_guided
# 361          atan      1  0.500000  _attr  models_guided
# `cat` means what information is hashed, can be
#   '_attr' -> attributes only,
#   '_inp' -> input shape && dtype,
#   '_inp_attr' -> input shape && dtype && attributes.

    def print_most_common(d):
        # Examine the hash_str sorted by count
        # Example usage: print_most_common(results[0]['nn.conv2d_inp'])
        # Example usage: print_most_common(results[1]['nn.conv2d_inp'])
        s = sum(d.values())
        for k, v in d.most_common():
            print(f'`{k}`: count={v} ratio={v / s}')

    def plot_one_cat(df, cat, name):
        df = df[df['cat'] == cat]
        for c in ['Count', 'Ratio']:
            _c = c.lower()
            df_ops = df[df.name.map(lambda x: x in ops)]
            plt.title(
                f"{c} of unqiue {name} combination for different ONNX operators")
            sns.barplot(x='name', y=f'{_c}', hue='fuzzers', data=df_ops)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, f'{_c}_selected{cat}.png'))
            plt.close()

            df_ops = df
            fig, ax = plt.subplots(figsize=(20, 6))
            plt.title(
                f"{c} of unqiue {name} combination for different ONNX operators")
            sns.barplot(x='name', y=f'{_c}', hue='fuzzers', data=df_ops, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, f'{_c}_all{cat}.png'))
            plt.close()

    plot_one_cat(df, '_attr', 'attribute')
    plot_one_cat(df, '_inp', 'input shapes and dtypes')
    plot_one_cat(df, '_inp_attr', 'attribute, input shapes and dtypes')
