"""Evaluating input searching algorithms.
simply generate a bunch of models and see if the can find viable inputs.
"""

from nnsmith.error import ConstraintCheck, ConstraintError
from nnsmith.graph_gen import random_model_gen, SymbolNet

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nodes', type=int, default=25)
    parser.add_argument('--exp-seed', type=int)
    parser.add_argument('--n_model', type=int, default=50)
    parser.add_argument('--min_dims', type=list, default=[1, 3, 48, 48])
    parser.add_argument('--timeout', type=int, default=50000)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_bitvec', action='store_true')
    parser.add_argument('--viz_graph', action='store_true')
    parser.add_argument('--output_path', type=str, default='output.onnx')
    args = parser.parse_args()

    exp_seed = args.exp_seed
    if exp_seed is None:
        exp_seed = random.getrandbits(32)
    print(f"Using seed {exp_seed}")

    np.random.seed(exp_seed)
    random.seed(exp_seed)
    torch.manual_seed(exp_seed)

    exp_name = f'r{exp_seed}-msize{args.n_model}-mnode{args.max_nodes}-inp-search.csv'
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

    for _ in tqdm(range(args.n_model)):
        model_seed = random.getrandbits(32)
        gen, solution = random_model_gen(min_dims=args.min_dims, seed=model_seed, max_nodes=args.max_nodes,
                                         use_bitvec=args.use_bitvec, timeout=args.timeout)
        net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose,
                        alive_shapes=gen.alive_shapes)

        results['n_nodes'].append(len(net.graph.nodes))
        results['model_seed'].append(model_seed)

        init_tensor_samples = []
        for v in np.linspace(-1, 1, 10):
            init_tensors = [torch.ones(ii.op.shape)
                            * v for ii in net.input_info]
            init_tensor_samples.append(init_tensors)

        # Test v3
        strt_time = time.time()
        succ_v3 = False
        try_times = 0

        net.check_intermediate_numeric = True
        with torch.no_grad():
            for init_tensors in init_tensor_samples:
                try_times += 1
                _ = net(*init_tensors)
                if not net.invalid_found_last:
                    succ_v3 = True
                    break

        net.check_intermediate_numeric = False
        results['v3-time'].append(time.time() - strt_time)
        results['v3-try'].append(try_times)
        results['v3-succ'].append(succ_v3)

        # Test grad
        strt_time = time.time()
        try_times = 0
        succ_grad = False
        for init_tensors in init_tensor_samples:
            try_times += 1
            sat_inputs = net.grad_input_gen(init_tensors=init_tensors)
            if sat_inputs is not None:
                succ_grad = True
                break

        results['grad-succ'].append(succ_grad)
        results['grad-try'].append(try_times)
        results['grad-time'].append(time.time() - strt_time)

    pd.DataFrame(results).to_csv(exp_name, index=False)
