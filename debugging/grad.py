"""Evaluating input searching algorithms.
simply generate a bunch of models and see if the can find viable inputs.
"""

import psutil
from nnsmith.input_gen import GradSearch, SamplingSearch
from nnsmith.graph_gen import random_tensor, SymbolNet
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.abstract.op import ALL_OP_TYPES

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
import argparse
import random
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', help='Path to net')
    parser.add_argument('model_seed', type=int,
                        help='Random seed used for model generation')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print_grad', type=int, default=0)
    parser.add_argument('--max_sample', type=int, default=1)
    parser.add_argument('--max_time_ms', type=int, default=500)
    parser.add_argument(
        '--mode', choices=['proxy', 'grad', 'all'], default='proxy')
    args = parser.parse_args()

    MAX_MEM = psutil.virtual_memory(
    ).total if not args.use_cuda else torch.cuda.get_device_properties(0).total_memory

    __DIFF_CACHE__ = 'config/diff.pkl'
    differentiable_ops = rewrite_op_dtype(
        ALL_OP_TYPES, factory=None, diff=True, verbose=args.verbose, cache=__DIFF_CACHE__)
    print(differentiable_ops)

    results = {
        'model_seed': [],
        'n_nodes': [],

        'naive-time': [],
        'naive-try': [],
        'naive-succ': [],

        'sampling-time': [],
        'sampling-try': [],
        'sampling-succ': [],

        'grad-time': [],
        'grad-try': [],
        'grad-succ': [],

        'proxy-time': [],
        'proxy-try': [],
        'proxy-succ': [],
    }

    net, model_seed = pickle.load(open(args.net, 'rb')), args.model_seed
    net = SymbolNet(net.concrete_graph, None,
                    megabyte_lim=net.megabyte_lim, verbose=args.verbose, print_grad=args.print_grad)
    net.eval()

    net.use_gradient = False

    results['model_seed'].append(model_seed)

    def seedme():
        nseed = (model_seed + 1) % (2 ** 32)
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
    inputs = net.get_random_inps(use_cuda=args.use_cuda)
    for tensor in inputs:
        nbytes += tensor.numel() * tensor.element_size()
    for name, param in net.named_parameters():
        nbytes += param.numel() * param.element_size()

    MEM_FACTOR = 0.4
    if nbytes * args.max_sample > MAX_MEM * MEM_FACTOR:
        max_sample = int(MAX_MEM * MEM_FACTOR / nbytes)
        print(
            f'{args.max_sample} x weights require {nbytes * args.max_sample / (1024**3)}GB'
            f' which is more than what you have. Downgrade to {max_sample} samples.')
    else:
        max_sample = args.max_sample
    # ------------------------------------------------------------
    seedme()

    net.check_intermediate_numeric = True
    with torch.no_grad():
        _ = net(*net.get_random_inps(base=0,
                margin=1, use_cuda=args.use_cuda))
        results['naive-succ'].append(not net.invalid_found_last)

    results['naive-time'].append(time.time() - strt_time)
    results['naive-try'].append(1)  # naive method always try once
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Test sampling:
    # sync input & weight between `sampling` and `grad`
    seedme()
    init_tensor_samples = []
    # 5 samples look like: [0, +-1, +-2] + uniform(-0.5, 0.5)
    for i in range(max_sample):
        data = net.get_random_inps(
            base=((i + 1) // 2) * (-1) ** i, margin=1, use_cuda=args.use_cuda)
        init_tensor_samples.append(data)

    init_weight_samples = []
    with torch.no_grad():
        for i in range(max_sample):
            weight_sample = {}
            for name, param in net.named_parameters():
                weight_sample[name] = random_tensor(
                    param.shape, dtype=param.dtype, use_cuda=args.use_cuda,
                    base=((i + 1) // 2) * (-1) ** i, margin=1)
            init_weight_samples.append(weight_sample)

    def apply_weights(net, weight_sample):
        with torch.no_grad():
            for name, param in net.named_parameters():
                param.copy_(weight_sample[name])

    if args.use_cuda:
        net.use_cuda()

    strt_time = time.time()
    searcher = SamplingSearch(
        net, init_tensor_samples, init_weight_samples, use_cuda=args.use_cuda)

    n_try, sat_inputs = searcher.search(
        max_time_ms=args.max_time_ms, max_sample=args.max_sample)

    results['sampling-time'].append(time.time() - strt_time)
    results['sampling-try'].append(n_try)
    results['sampling-succ'].append(sat_inputs is not None)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Test grad
    # If sampling can succeed, grad can succeed too as their initial input are the same.
    if args.mode == 'all' or args.mode == 'grad':
        seedme()

        strt_time = time.time()

        searcher = GradSearch(
            net, init_tensor_samples, init_weight_samples, use_cuda=args.use_cuda)
        n_try, sat_inputs = searcher.search(
            max_time_ms=args.max_time_ms, max_sample=args.max_sample)

        results['grad-time'].append(time.time() - strt_time)
        results['grad-try'].append(n_try)
        results['grad-succ'].append(sat_inputs is not None)
    # --------------------------------------------------------------------

    # ------------------------------------------------------------
    # Test grad + proxy
    # If sampling can succeed, grad can succeed too as their initial input are the same.
    # Proxy makes some operators differentiable.
    if args.mode == 'all' or args.mode == 'proxy':
        seedme()

        strt_time = time.time()

        net.enable_proxy_grad()
        searcher = GradSearch(
            net, init_tensor_samples, init_weight_samples, use_cuda=args.use_cuda)
        n_try, sat_inputs = searcher.search(
            max_time_ms=args.max_time_ms, max_sample=args.max_sample)

        results['proxy-time'].append(time.time() - strt_time)
        results['proxy-try'].append(n_try)
        results['proxy-succ'].append(sat_inputs is not None)
    # --------------------------------------------------------------------
    print(results)
