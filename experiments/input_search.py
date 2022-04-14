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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nodes', type=int, default=25)
    parser.add_argument('--exp-seed', type=int)
    parser.add_argument('--n_inp_sample', type=int, default=1)
    parser.add_argument('--n_model', type=int, default=50)
    parser.add_argument('--min_dims', type=list, default=[1, 3, 48, 48])
    parser.add_argument('--timeout', type=int, default=5000)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_path', type=str, default='output.onnx')
    parser.add_argument('--mode', type=str, default='random')
    parser.add_argument('--save_model', type=str,
                        help='save models to this path')
    args = parser.parse_args()

    if args.save_model is not None:
        mkdir(args.save_model)

    __DIFF_CACHE__ = 'config/diff.pkl'
    differentiable_ops = rewrite_op_dtype(
        ALL_OP_TYPES, backend=None, diff=True, verbose=True, cache=__DIFF_CACHE__)
    print(differentiable_ops)

    exp_seed = args.exp_seed
    if exp_seed is None:
        exp_seed = random.getrandbits(32)
    print(f"Using seed {exp_seed}")

    np.random.seed(exp_seed)
    random.seed(exp_seed)
    torch.manual_seed(exp_seed)

    exp_name = f'r{exp_seed}-model{args.n_model}-node{args.max_nodes}-inp-search.csv'
    results = {
        'model_seed': [],
        'n_nodes': [],

        'v3-time': [],
        'v3-try': [],
        'v3-succ': [],

        'grad-time': [],
        'grad-try': [],
        'grad-succ': [],
    }

    def mknet():
        model_seed = random.getrandbits(32)
        gen, solution = random_model_gen(
            mode=args.mode, min_dims=args.min_dims, seed=model_seed, max_nodes=args.max_nodes,
            timeout=args.timeout, candidates_overwrite=differentiable_ops, init_fp=True)
        gen.viz('debug.png')
        net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose,
                        alive_shapes=gen.alive_shapes)
        net.eval()
        return net, gen.num_op(), model_seed

    for _ in tqdm(range(args.n_model)):
        while True:
            net, num_op, model_seed = mknet()
            # break # NOTE: uncomment this line to see how serious the issue is.
            if net.n_vulnerable_op > 0:
                break

        if args.save_model is not None:
            torch2onnx(net, os.path.join(args.save_model, f'{_}.onnx'))

        results['n_nodes'].append(num_op)
        results['model_seed'].append(model_seed)

        init_tensor_samples = []
        n_step = args.n_inp_sample
        interval = 1 / n_step
        for v in np.linspace(-10, 10, n_step):
            init_tensors = [(v + torch.rand(ii.op.shape_var.shape)
                            * interval).to(dtype=ii.op.shape_var.dtype.value) for ii in net.input_info]
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

        results['v3-time'].append(time.time() - strt_time)
        results['v3-try'].append(try_times_v3)
        results['v3-succ'].append(succ_v3)

        # Test grad
        # If v3 can succeed, grad can succeed too as their initial input are the same.
        strt_time = time.time()
        succ_grad = False
        try_times_grad = 0
        for init_tensors in init_tensor_samples:
            try_times_grad += 1
            try:
                sat_inputs = net.grad_input_gen(use_cuda=args.use_cuda)
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
        results['grad-succ'].append(succ_grad)
        results['grad-try'].append(try_times_grad)
        results['grad-time'].append(time.time() - strt_time)

    pd.DataFrame(results).to_csv(exp_name, index=False)
    if args.save_model is not None:
        os.system(
            f'cp {exp_name} {os.path.join(args.save_model, "model_info.csv")}')
