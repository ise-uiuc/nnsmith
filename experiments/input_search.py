"""Evaluating input searching algorithms.
simply generate a bunch of models and see if the can find viable inputs.
"""

from multiprocessing import Process
import shutil
import traceback
import uuid
import time
import argparse
import random
import os
import pickle
import psutil

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx


from nnsmith.materialize.onnx.export import torch2onnx
from nnsmith.util import mkdir
from nnsmith.graph_gen import random_model_gen, SymbolNet, random_tensor
from nnsmith.materialize.torch.input_gen import GradSearch, SamplingSearch
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.abstract.op import ALL_OP_TYPES


def mknet(args, differentiable_ops=None):
    model_seed = random.getrandbits(32)
    gen, solution = random_model_gen(
        mode=args.mode,
        seed=model_seed,
        max_nodes=args.max_nodes,
        candidates_overwrite=differentiable_ops,
        init_fp=True,
        skip=args.skip,
    )
    net = SymbolNet(
        gen.abstract_graph,
        solution,
        verbose=args.verbose,
        alive_shapes=gen.alive_shapes,
    )
    net.eval()
    return net, gen.num_op(), model_seed


def mknets(args, exp_seed):
    np.random.seed(exp_seed)
    random.seed(exp_seed)
    torch.manual_seed(exp_seed)

    model_path = os.path.join(args.root, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    __DIFF_CACHE__ = "config/diff.pkl"
    differentiable_ops = rewrite_op_dtype(
        ALL_OP_TYPES, factory=None, diff=True, verbose=True, cache=__DIFF_CACHE__
    )
    print(differentiable_ops)
    results = {
        "model_seed": [],
        "n_nodes": [],
        "gen_time": [],
    }
    for model_id in tqdm(range(args.n_model)):
        gen_time = None
        while True:
            try:
                gen_tstart = time.time()
                net, num_op, model_seed = mknet(args, differentiable_ops)
            except:
                traceback.print_exc()
                continue
            # break # NOTE: uncomment this line to see how serious the issue is.
            if net.n_vulnerable_op > 0:
                try:
                    torch2onnx(net, os.path.join(model_path, f"{model_id}.onnx"))
                except Exception as e:
                    print(e)
                    print("Failed to convert to onnx")
                gen_time = time.time() - gen_tstart
                break
        if hasattr(net, "graph"):
            nx.drawing.nx_pydot.to_pydot(net.graph).write_png(
                os.path.join(model_path, f"{model_id}-graph.png")
            )
            net.to_picklable()
        cloudpickle.dump(
            net, open(os.path.join(model_path, f"{model_id}-net.pkl"), "wb"), protocol=4
        )
        results["n_nodes"].append(num_op)
        results["model_seed"].append(model_seed)
        results["gen_time"].append(gen_time)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.root, "model_info.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_nodes", type=int, default=20)
    parser.add_argument("--exp-seed", type=int)
    parser.add_argument("--max_sample", type=int, default=1)
    parser.add_argument("--max_time_ms", type=int, default=500)
    parser.add_argument("--n_model", type=int, default=50)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--root", type=str, help="save models and results to this path")
    parser.add_argument("--skip")
    # for reproducibility
    parser.add_argument(
        "--result", help="Name of the grad search result csv (exp_name).", default=None
    )
    parser.add_argument("--print_grad", type=int, default=0)
    args = parser.parse_args()

    MAX_MEM = (
        psutil.virtual_memory().total
        if not args.use_cuda
        else torch.cuda.get_device_properties(0).total_memory
    )

    del_root = False
    if args.root is None:
        args.root = "input_search_root_" + str(uuid.uuid4())
        del_root = True

    exp_seed = args.exp_seed
    if exp_seed is None:
        exp_seed = random.getrandbits(32)
    print(f"Using seed {exp_seed}")

    # mkdir and generate models if not exists.
    if not os.path.exists(args.root):
        mkdir(args.root)
        p = Process(target=mknets, args=(args, exp_seed))
        p.start()
        p.join()
    else:
        print(f"{args.root} exists, skip model generation phase.")

    np.random.seed(exp_seed)
    random.seed(exp_seed)
    torch.manual_seed(exp_seed)

    ref_df = pd.read_csv(os.path.join(args.root, "model_info.csv"))  # load models

    exp_name = (
        args.result
        if args.result
        else f"r{exp_seed}-model{args.n_model}-node{args.max_nodes}-inp-search.csv"
    )
    if not exp_name.endswith(".csv"):
        exp_name += ".csv"

    results = {
        "model_seed": [],
        "n_nodes": [],
        "naive-time": [],
        "naive-try": [],
        "naive-succ": [],
        "sampling-time": [],
        "sampling-try": [],
        "sampling-succ": [],
        "grad-time": [],
        "grad-try": [],
        "grad-succ": [],
        "proxy-time": [],
        "proxy-try": [],
        "proxy-succ": [],
    }

    with tqdm(range(args.n_model)) as pbar:
        for model_id in pbar:
            model_seed = ref_df["model_seed"][model_id]
            pbar.set_description(f"id={model_id}, seed={model_seed}")

            net = pickle.load(
                open(os.path.join(args.root, "model", f"{model_id}-net.pkl"), "rb")
            )
            net = SymbolNet(
                net.concrete_graph,
                None,
                verbose=args.verbose,
                megabyte_lim=net.megabyte_lim,
                print_grad=args.print_grad,
            )
            net.eval()
            net.use_gradient = False
            num_op = ref_df["n_nodes"][model_id]

            results["n_nodes"].append(num_op)
            results["model_seed"].append(model_seed)

            def seedme():
                nseed = (model_seed + 1) % (2**32)
                random.seed(nseed)
                np.random.seed(nseed)
                torch.manual_seed(nseed)

            # ------------------------------------------------------------
            # Test naive:
            # how other fuzzers do: just randomly init tensors using torch.random -> 0~1 by `once`
            seedme()
            strt_time = time.time()

            if args.use_cuda:
                net.use_cuda()

            # ------------------------------------------------------------
            # Estimate model size | avoid OOM
            nbytes = 0
            for ii in net.input_info:
                dtype = ii.op.asb_tensor.dtype.torch()
                nbytes += (
                    ii.op.asb_tensor.nelement()
                    * torch.tensor(0, dtype=dtype).element_size()
                )
            for name, param in net.named_parameters():
                nbytes += param.numel() * param.element_size()

            MEM_FACTOR = 0.4
            if nbytes * args.max_sample > MAX_MEM * MEM_FACTOR:
                max_sample = int(MAX_MEM * MEM_FACTOR / nbytes)
                print(
                    f"{args.max_sample} x weights require {nbytes * args.max_sample / (1024**3)}GB"
                    f" which is more than what you have. Downgrade to {max_sample} samples."
                )
            else:
                max_sample = args.max_sample
            # ------------------------------------------------------------
            seedme()

            net.check_intermediate_numeric = True
            with torch.no_grad():
                _ = net(
                    *net.get_random_inps(
                        base="center", margin=1, use_cuda=args.use_cuda
                    )
                )
                results["naive-succ"].append(not net.invalid_found_last)

            results["naive-time"].append(time.time() - strt_time)
            results["naive-try"].append(1)  # naive method always try once
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # Test sampling:
            # sync input & weight between `sampling` and `grad`
            seedme()
            init_tensor_samples = []
            for i in range(max_sample):
                data = net.get_random_inps(use_cuda=args.use_cuda)
                init_tensor_samples.append(data)

            init_weight_samples = []
            with torch.no_grad():
                for i in range(max_sample):
                    weight_sample = {}
                    for name, param in net.named_parameters():
                        weight_sample[name] = random_tensor(
                            param.shape, dtype=param.dtype, use_cuda=args.use_cuda
                        )
                    init_weight_samples.append(weight_sample)

            def apply_weights(net, weight_sample):
                with torch.no_grad():
                    for name, param in net.named_parameters():
                        param.copy_(weight_sample[name])

            if args.use_cuda:
                net.use_cuda()

            strt_time = time.time()
            searcher = SamplingSearch(
                net, init_tensor_samples, init_weight_samples, use_cuda=args.use_cuda
            )

            n_try, sat_inputs = searcher.search(
                max_time_ms=args.max_time_ms, max_sample=args.max_sample
            )

            results["sampling-time"].append(time.time() - strt_time)
            results["sampling-try"].append(n_try)
            results["sampling-succ"].append(sat_inputs is not None)
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # Test grad
            # If sampling can succeed, grad can succeed too as their initial input are the same.
            seedme()

            strt_time = time.time()

            searcher = GradSearch(
                net, init_tensor_samples, init_weight_samples, use_cuda=args.use_cuda
            )
            n_try, sat_inputs = searcher.search(
                max_time_ms=args.max_time_ms, max_sample=args.max_sample
            )

            results["grad-time"].append(time.time() - strt_time)
            results["grad-try"].append(n_try)
            results["grad-succ"].append(sat_inputs is not None)
            # --------------------------------------------------------------------

            # ------------------------------------------------------------
            # Test grad + proxy
            # If sampling can succeed, grad can succeed too as their initial input are the same.
            # Proxy makes some operators differentiable.
            seedme()

            strt_time = time.time()

            net.enable_proxy_grad()
            searcher = GradSearch(
                net, init_tensor_samples, init_weight_samples, use_cuda=args.use_cuda
            )
            n_try, sat_inputs = searcher.search(
                max_time_ms=args.max_time_ms, max_sample=args.max_sample
            )

            results["proxy-time"].append(time.time() - strt_time)
            results["proxy-try"].append(n_try)
            results["proxy-succ"].append(sat_inputs is not None)
            # --------------------------------------------------------------------

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.root, exp_name), index=False)
    with open(os.path.join(args.root, exp_name + "-stats.log"), "w") as f:
        f.write(str(df.mean()) + "\n")
    with pd.option_context("display.float_format", "{:0.3f}".format):
        print(df.drop("model_seed", axis=1).mean())
    if del_root:
        shutil.rmtree(args.root)
