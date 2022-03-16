from nnsmith.graph_gen import random_model_gen, SymbolNet
from nnsmith.export import torch2onnx
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.abstract.op import ALL_OP_TYPES
from nnsmith.util import mkdir

from experiments.graphfuzz import GraphFuzz

import os
import random
import argparse
import time
import warnings

from tqdm import tqdm
import torch


def nnsmith_gen_once(path, seed, max_nodes, candidates_overwrite=None):
    torch.manual_seed(seed)
    gen, solution = random_model_gen(
        min_dims=[1, 3, 48, 48],  # Only rank useful. Dim sizes means nothing.
        seed=seed, max_nodes=max_nodes, candidates_overwrite=candidates_overwrite)
    net = SymbolNet(gen.abstract_graph, solution,
                    verbose=False, alive_shapes=gen.alive_shapes)
    with torch.no_grad():
        net.eval()
        torch2onnx(net, path, verbose=False, use_cuda=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_dir', type=str, required=True)
    parser.add_argument('--time_budget', type=int, default=60 * 60 * 4)
    parser.add_argument('--max_nodes', type=int, default=10)
    parser.add_argument('--graphfuzz_ops', action='store_true') 
    parser.add_argument('--ort_cache', type=str, default=None)
    args = parser.parse_args()

    mkdir(args.onnx_dir)

    if args.ort_cache:
        if not os.path.exists(args.ort_cache):
            print(f'Please first generate cache! (mkdir config first)')
            print(f'python nnsmith/dtype_test.py --cache {args.ort_cache}')
            exit(1)
        # must pre run this. otherwise using ort will slow down generation.
        rewrite_op_dtype(ALL_OP_TYPES, backend=None, cache=args.ort_cache)

    if args.graphfuzz_ops:
        candidates_overwrite = GraphFuzz.get_available_op_ts()
    else:
        candidates_overwrite = None

    # FORMAT: {generation time cost in seconds}, {model relative path}
    # MUST RANK by GENERATION ORDER.
    config_file = open(os.path.join(args.onnx_dir, 'gentime.csv'), 'w')

    start_time = time.time()
    gen_cnt = 0
    valid_cnt = 0

    with tqdm(total=args.time_budget) as pbar:
        while time.time() - start_time < args.time_budget:
            seed = random.getrandbits(32)
            to_name = f'{valid_cnt}.onnx'

            tstart = time.time()
            try:
                with warnings.catch_warnings():  # just shutup.
                    warnings.simplefilter("ignore")
                    nnsmith_gen_once(os.path.join(
                        args.onnx_dir, to_name), seed, max_nodes=10,
                        candidates_overwrite=candidates_overwrite)
                label = to_name
                valid_cnt += 1
            except Exception as e:
                print(f'Fail when seed={seed}')
                print(e)
                raise e
                label = 'FAILURE'

            time_diff = time.time() - tstart
            config_file.write(f'{time_diff:.5f},{label}\n')

            gen_cnt += 1
            config_file.flush()

            pbar.update(int(time.time() - start_time) - pbar.n)
            pbar.set_description(f'valid={valid_cnt},fail={gen_cnt-valid_cnt}')
            pbar.refresh()
        config_file.close()
