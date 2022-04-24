"""Evaluating input searching algorithms.
simply generate a bunch of models and see if the can find viable inputs.
"""

from nnsmith.graph_gen import random_model_gen, SymbolNet
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.abstract.op import ALL_OP_TYPES

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
import argparse
import random
import os
from nnsmith.export import torch2onnx
from nnsmith.util import mkdir
import cloudpickle
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', help='Path to net')
    parser.add_argument('model_seed', type=int,
                        help='Random seed used for model generation')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print_grad', action='store_true')
    args = parser.parse_args()

    __DIFF_CACHE__ = 'config/diff.pkl'
    differentiable_ops = rewrite_op_dtype(
        ALL_OP_TYPES, backend=None, diff=True, verbose=args.verbose, cache=__DIFF_CACHE__)
    print(differentiable_ops)

    results = {}
    net, model_seed = pickle.load(open(args.net, 'rb')), args.model_seed
    net = SymbolNet(net.concrete_graph, None,
                    megabyte_lim=net.megabyte_lim, verbose=args.verbose, print_grad=args.print_grad)
    net.eval()
    print('model_seed=', model_seed)

    init_tensor_samples = []
    nseed = (model_seed + 1) % (2 ** 32)
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    init_tensors = net.get_random_inps(
        use_cuda=args.use_cuda)
    init_tensor_samples.append(init_tensors)

    # Test v3
    strt_time = time.time()
    succ_v3 = False
    try_times_v3 = 0

    if args.use_cuda:
        net.use_cuda()

    net.check_intermediate_numeric = True
    with torch.no_grad():
        for init_tensors in init_tensor_samples:
            try_times_v3 += 1
            _ = net(*init_tensors)
            if not net.invalid_found_last:
                succ_v3 = True
                break

    results['v3-time'] = time.time() - strt_time
    results['v3-try'] = try_times_v3
    results['v3-succ'] = succ_v3

    # Test grad
    # If v3 can succeed, grad can succeed too as their initial input are the same.
    strt_time = time.time()
    succ_grad = False
    try_times_grad = 0
    for init_tensors in init_tensor_samples:
        try_times_grad += 1
        try:
            sat_inputs = net.grad_input_gen(
                init_tensors=init_tensors, use_cuda=args.use_cuda)
        except RuntimeError as e:
            if 'element 0 of tensors does not require grad and does not have a grad_fn' in str(e):
                # means some op are not differentiable.
                succ_grad = succ_v3
                try_times_grad = try_times_v3
                break
            raise e
        if sat_inputs is not None:
            succ_grad = True
            break

    # Some operator is not differentiable that will fall back to v3.
    results['grad-succ'] = succ_grad
    results['grad-try'] = try_times_grad
    results['grad-time'] = time.time() - strt_time

    print(results)
