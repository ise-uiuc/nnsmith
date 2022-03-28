"""Just to figure out operators types and connections.
"""

from collections import Counter
import os
from multiprocessing import cpu_count, Process
from time import sleep, time
import traceback
from typing import List, Tuple

from uuid import uuid4
import pandas as pd
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from tqdm import tqdm
from onnx_graph_analyzer import analyze_one_relay

__TMP_FOLDER__ = 'param_analyzer_tmp'


def analyze_one(model_path, verbose=False) -> Tuple[str, Process]:
    def execute(model_path, target_path):
        try:
            res = analyze_one_relay(model_path, use_counter=True)
            if not os.path.exists(__TMP_FOLDER__):
                os.makedirs(__TMP_FOLDER__)
            with open(target_path, 'wb') as f:
                pickle.dump(res, f)
            return target_path
        except:
            print('-------------> Skip model', model_path, 'due to exception:')
            if verbose:
                traceback.print_exc()
            return None

    target_path = os.path.join(__TMP_FOLDER__, f'{uuid4()}.pkl')
    p = Process(target=execute, args=(model_path, target_path))
    p.start()
    return target_path, p


def analyze_folders(folders, cache_dir=None, force=False, n_limit=None, resume=False, write_freq=250):
    res = []

    def write_once(res):
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, __CACHE_FILE__), 'wb') as fp:
                pickle.dump(res, fp)

    __CACHE_FILE__ = 'onnx_param_cache.pkl'
    use_cache = os.path.exists(os.path.join(
        cache_dir, __CACHE_FILE__)) and cache_dir is not None
    if resume:
        assert use_cache
    if use_cache and not force:
        print('===> {} already exists.'.format(
            os.path.join(cache_dir, __CACHE_FILE__)))
        with open(os.path.join(cache_dir, __CACHE_FILE__), 'rb') as fp:
            if resume:
                res = pickle.load(fp)
            else:
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

    first_hub_resume = False
    if resume:
        new_file_hub = []
        if len(res) > 0:
            last_job_idx = len(res) - 1
            first_hub_done = res[last_job_idx]['nfiles']
            if first_hub_done < len(file_hubs[last_job_idx]):
                new_file_hub.append(file_hubs[last_job_idx][first_hub_done:])
                first_hub_resume = True
        new_file_hub.extend(file_hubs[len(res):])
        file_hubs = new_file_hub
        print(f'===> Resume from {len(file_hubs)} jobs.')

    def update_once(path):
        if os.path.exists(path):
            with open(path, 'rb') as fp:
                new = pickle.load(fp)
                for op_name, cnt in new.items():
                    if op_name not in res[-1]['param_map']:
                        res[-1]['param_map'][op_name] = Counter()
                    res[-1]['param_map'][op_name].update(cnt)
            os.remove(path)

    for files in file_hubs:
        if not first_hub_resume:
            res.append({
                'nfiles': 0,
                'param_map': {},
            })

        n_pararllel = min(cpu_count() + 4, len(files))
        jobs: List[Tuple[str, Process]] = []

        def try_pop(timeout=None):
            if len(jobs) == 0:
                return False
            tar, p = jobs[0]
            p.join(timeout=timeout)
            if p.is_alive():
                return False
            if p.exitcode != 0:
                print('-------------> Skip model', path,
                      'due to crash | exit code', p.exitcode)
            res[-1]['nfiles'] += 1
            pbar.update()
            update_once(tar)
            jobs.pop(0)
            return True

        with tqdm(total=len(files)) as pbar:
            for path in files:
                if len(jobs) < n_pararllel:
                    jobs.append(analyze_one(path, verbose=False))
                else:
                    try_pop()
                    while try_pop(timeout=0.1):
                        pass

                if pbar.n % write_freq == 0:
                    write_once(res)

            while try_pop():
                pass

    write_once(res)
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
    parser.add_argument('--resume', action='store_true')
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
        args.folders, cache_dir=args.output, force=args.force, n_limit=args.nlim, resume=args.resume)

    for tag, single_res in zip(args.tags, results):
        param_map = single_res['param_map']
        pass
