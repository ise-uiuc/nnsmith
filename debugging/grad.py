"""Evaluating input searching algorithms.
simply generate a bunch of models and see if the can find viable inputs.
"""

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
    parser.add_argument('--n_inp_sample', type=int, default=1)
    args = parser.parse_args()

    __DIFF_CACHE__ = 'config/diff.pkl'
    differentiable_ops = rewrite_op_dtype(
        ALL_OP_TYPES, backend=None, diff=True, verbose=True, cache=__DIFF_CACHE__)
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
    }

    net, model_seed = pickle.load(open(args.net, 'rb')), args.model_seed
    net = SymbolNet(net.concrete_graph, None, megabyte_lim=net.megabyte_lim)
    net.eval()
    print('model_seed=', model_seed)

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
    for _ in range(args.n_inp_sample):
        init_tensor_samples.append(net.get_random_inps(
            base=0, margin=10, use_cuda=args.use_cuda))

    init_weight_samples = []
    with torch.no_grad():
        for _ in range(args.n_inp_sample):
            weight_sample = {}
            for name, param in net.named_parameters():
                weight_sample[name] = random_tensor(
                    param.shape, dtype=param.dtype, use_cuda=args.use_cuda)
            init_weight_samples.append(weight_sample)

    def apply_weights(net, weight_sample):
        with torch.no_grad():
            for name, param in net.named_parameters():
                param.copy_(weight_sample[name])

    if args.use_cuda:
        net.use_cuda()

    strt_time = time.time()
    succ_sampling = False
    try_times_sampling = 0

    net.check_intermediate_numeric = True
    with torch.no_grad():
        for inp_sample, w_sample in zip(init_tensor_samples, init_weight_samples):
            try_times_sampling += 1
            apply_weights(net, w_sample)
            _ = net(*inp_sample)
            if not net.invalid_found_last:
                succ_sampling = True
                break

    results['sampling-time'].append(time.time() - strt_time)
    results['sampling-try'].append(try_times_sampling)
    results['sampling-succ'].append(succ_sampling)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Test grad
    # If sampling can succeed, grad can succeed too as their initial input are the same.
    seedme()

    strt_time = time.time()
    succ_grad = False
    try_times_grad = 0

    for inp_sample, w_sample in zip(init_tensor_samples, init_weight_samples):
        try_times_grad += 1
        try:
            apply_weights(net, w_sample)
            sat_inputs = net.grad_input_gen(
                init_tensors=inp_sample, use_cuda=args.use_cuda)
        except RuntimeError as e:
            if 'element 0 of tensors does not require grad and does not have a grad_fn' in str(e):
                # means some op are not differentiable.
                succ_grad = succ_sampling
                try_times_grad = try_times_sampling
                break
            raise e
        if sat_inputs is not None:
            succ_grad = True
            break

    # Some operator is not differentiable that will fall back to v3.
    results['grad-succ'].append(succ_grad)
    results['grad-try'].append(try_times_grad)
    results['grad-time'].append(time.time() - strt_time)
    # --------------------------------------------------------------------

    print(results)
