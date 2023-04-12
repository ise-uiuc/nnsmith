"""Just to figure out operators types and connections.
"""

from collections import Counter
import os
from multiprocessing import cpu_count, Process
import traceback
from typing import List, Tuple

from uuid import uuid4
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from onnx_graph_analyzer import analyze_one_relay

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
SUPER_BIG = 22

plt.rcParams.update({"text.usetex": True})

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SUPER_BIG)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SUPER_BIG)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

__TMP_FOLDER__ = "param_analyzer_tmp"


def analyze_one(model_path, verbose=False) -> Tuple[str, Process]:
    def execute(model_path, target_path):
        try:
            res = analyze_one_relay(model_path, use_counter=True)
            if not os.path.exists(__TMP_FOLDER__):
                os.makedirs(__TMP_FOLDER__)
            with open(target_path, "wb") as f:
                pickle.dump(res, f)
            return target_path
        except:
            print("-------------> Skip model", model_path, "due to exception:")
            if verbose:
                traceback.print_exc()
            return None

    target_path = os.path.join(__TMP_FOLDER__, f"{uuid4()}.pkl")
    p = Process(target=execute, args=(model_path, target_path))
    p.start()
    return target_path, p


def analyze_folders(
    folders, cache_dir=None, force=False, n_limit=None, resume=False, write_freq=250
):
    res = []

    def write_once(res):
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, __CACHE_FILE__), "wb") as fp:
                pickle.dump(res, fp)

    __CACHE_FILE__ = "onnx_param_cache.pkl"
    use_cache = (
        os.path.exists(os.path.join(cache_dir, __CACHE_FILE__))
        and cache_dir is not None
    )
    if resume:
        assert use_cache
    if use_cache and not force:
        print("===> {} already exists.".format(os.path.join(cache_dir, __CACHE_FILE__)))
        with open(os.path.join(cache_dir, __CACHE_FILE__), "rb") as fp:
            if resume:
                res = pickle.load(fp)
            else:
                return pickle.load(fp)

    times = []
    file_hubs = []
    least_time = None
    assert n_limit is None or len(n_limit) == len(folders)
    for i, folder in enumerate(folders):
        df = pd.read_csv(
            os.path.join(folder, "gentime.csv"), usecols=[0, 1], header=None
        )
        ts = df[0].to_numpy()

        times.append(ts)
        files = df[1].tolist()
        if n_limit is not None:
            files = files[: n_limit[i]]
        files = [os.path.join(folder, f) for f in files]
        file_hubs.append(files)

        time_cost = ts.sum()
        if least_time is None or time_cost < least_time:
            least_time = time_cost

    # only consider rows that < least_time
    for i, ts in enumerate(times):
        file_hubs[i] = file_hubs[i][: (ts < least_time).sum()]

    first_hub_resume = False
    if resume:
        new_file_hub = []
        if len(res) > 0:
            last_job_idx = len(res) - 1
            first_hub_done = res[last_job_idx]["nfiles"]
            if first_hub_done < len(file_hubs[last_job_idx]):
                new_file_hub.append(file_hubs[last_job_idx][first_hub_done:])
                first_hub_resume = True
        new_file_hub.extend(file_hubs[len(res) :])
        file_hubs = new_file_hub
        print(f"===> Resume from {len(file_hubs)} jobs.")

    def update_once(path):
        if os.path.exists(path):
            with open(path, "rb") as fp:
                new = pickle.load(fp)
                for op_name, cnt in new.items():
                    if op_name not in res[-1]["param_map"]:
                        res[-1]["param_map"][op_name] = Counter()
                    res[-1]["param_map"][op_name].update(cnt)
            os.remove(path)

    for files in file_hubs:
        if not first_hub_resume:
            res.append(
                {
                    "nfiles": 0,
                    "param_map": {},
                }
            )
        else:
            first_hub_resume = False

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
                print(
                    "-------------> Skip model",
                    path,
                    "due to crash | exit code",
                    p.exitcode,
                )
            res[-1]["nfiles"] += 1
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", type=str, nargs="+", required=True)
    parser.add_argument("--tags", type=str, nargs="+", default=None)
    # should compare models within same generation duration.
    parser.add_argument("--nlim", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ops", type=str, nargs="+", default=None, help="Pass operator names to show."
    )
    args = parser.parse_args()

    ops = args.ops
    if ops is None:
        ops = [
            "add",
            "nn.avg_pool2d",
            "nn.conv2d",
            "nn.dense",
            "strided_slice",
            "reshape",
        ]
    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split("-")[0] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    results = analyze_folders(
        args.folders,
        cache_dir=args.output,
        force=args.force,
        n_limit=args.nlim,
        resume=args.resume,
    )

    mutual_keys = set.intersection(*[set(r["param_map"].keys()) for r in results])

    # dyn -> dynamic op not considered
    # copy -> memory op not considered
    BLACK_LIST = ["dyn.", "copy"]
    # You at least to have 5 instances otherwise too trivial to show.
    NUM_THRESH = 5

    for k in list(mutual_keys):
        for bk in BLACK_LIST:
            if bk in k:
                mutual_keys -= set([k])

    same_keys = []
    # Don't plot if the item num is the same (meaningless to compare)
    for k in mutual_keys:
        if len(set([len(r["param_map"][k]) for r in results])) == 1:
            same_keys.append(k)

    for res in results:
        for k in list(mutual_keys):
            if len(res["param_map"][k]) < NUM_THRESH:
                mutual_keys -= set([k])

    mutual_keys -= set(same_keys)

    # only show representative ops
    #   merge redundancy;
    #      -> param-free element wise ops
    unary_pf_ew = [
        "abs",
        "acos",
        "cos",
        "asin",
        "sin",
        "ceil",
        "floor",
        "log",
        "logical_not",
        "atan",
        "tan",
        "erf",
        "negative",
        "nn.relu",
        "round",
        "sigmoid",
        "sqrt",
        "clip",
        "cast",
        "nn.prelu",
        "nn.leaky_relu",
    ]
    bin_pf_ew = [
        "add",
        "divide",
        "equal",
        "floor_mod",
        "greater",
        "multiply",
        "subtract",
        "less",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "power",
    ]
    simple_reduce_ops = ["sum", "min", "max", "mean"]
    UEW_KEY = "$\\bf unary$ elem.-wise"
    BEW_KEY = "$\\bf bin.$ elem.-wise"
    REDUCE_KEY = "common $\\bf reduce$"
    for res in results:
        res["param_map"][UEW_KEY] = Counter()
        for uk in unary_pf_ew:
            if uk in res["param_map"]:
                res["param_map"][UEW_KEY] += res["param_map"][uk]
        res["param_map"][BEW_KEY] = Counter()
        for bk in bin_pf_ew:
            if bk in res["param_map"]:
                res["param_map"][BEW_KEY] += res["param_map"][bk]
        res["param_map"][REDUCE_KEY] = Counter()
        for rk in simple_reduce_ops:
            if rk in res["param_map"]:
                res["param_map"][REDUCE_KEY] += res["param_map"][rk]

    mutual_keys -= set(unary_pf_ew)
    mutual_keys -= set(bin_pf_ew)
    mutual_keys -= set(simple_reduce_ops)

    mutual_keys.add(UEW_KEY)
    mutual_keys.add(BEW_KEY)
    mutual_keys.add(REDUCE_KEY)

    mutual_keys = sorted(list(mutual_keys))

    col_width = 3
    bar_width = (col_width * 0.5) / len(results)
    base_x = np.arange(len(mutual_keys))

    vals = []
    for single_res in results:
        param_map = single_res["param_map"]
        v = []
        for k in mutual_keys:
            v.append(len(param_map[k]))
        vals.append(v)

    vals = np.array(vals)

    legends = []
    HATCHES = ["x", None]
    COLORS = ["lightblue", "dodgerblue"]

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(13, 5))

    print(mutual_keys)

    for idx, (tag, single_res) in enumerate(zip(args.tags, results)):
        param_map = single_res["param_map"]
        print(f"{single_res['nfiles']=}")
        padding = int((10 - len(tag)) * 1.2)
        blank = r"\ "
        legends.append(
            f'{tag + "".join([blank for _ in range(padding)])} (total: {vals[idx].sum()})'
        )

    normalized = vals.max(axis=0) / vals.min(axis=0)
    indices = np.argsort(normalized)

    for idx, (tag, single_res) in enumerate(zip(args.tags, results)):
        pv = vals[idx] / np.min(vals, axis=0)

        x_pos = base_x  # - 0.5 * col_width + (idx + 0.5) * bar_width
        ax.bar(
            x_pos,
            pv[indices],
            width=bar_width,
            label=tag,
            color=COLORS[idx],
            align="center",
            hatch=HATCHES[idx],
            edgecolor="k",
            linewidth=3,
        )

    for x, v in zip(base_x, normalized[indices]):
        ax.text(
            x - bar_width * 0.54,
            v + 0.13,
            f"{v:.1f}$\\times$",
            fontweight="bold",
            fontsize=MEDIUM_SIZE,
            fontname="Times New Roman",
        )

    ax.get_yaxis().set_ticks([])
    plt.legend(legends)

    xticks = []
    for idx in indices:
        k = mutual_keys[idx]
        xticks.append(k if k.startswith("$") else k.split(".")[-1])
    plt.xticks(base_x, xticks, rotation=35, ha="right", rotation_mode="anchor")
    plt.xlim([base_x[0] - 0.5, base_x[-1] + 0.5])
    plt.ylim([0, normalized.max() + 0.8])
    # plt.gcf().subplots_adjust(left=base_x[0] - 1, right=base_x[-1] + 1)
    # fig.subplots_adjust(left=base_x[0] - 1, right=base_x[-1] + 1)
    # plt.xlabel('# Operators in Models w/ Vulnerable Op.', fontweight='bold')
    plt.ylabel("Unique Operator Instances", fontweight="bold")
    plt.savefig(os.path.join(args.output, "param_diff.pdf"))
    plt.savefig(os.path.join(args.output, "param_diff.png"))
