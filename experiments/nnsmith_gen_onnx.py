from nnsmith.graph_gen import random_model_gen, SymbolNet
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.onnx.export import torch2onnx
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.abstract.op import ALL_OP_TYPES
from nnsmith.util import mkdir

from experiments.graphfuzz import GraphFuzz

import pickle
import os
import random
import argparse
import time
import warnings
import tarfile

from tqdm import tqdm
import torch


def nnsmith_gen_once(
    path_prefix, seed, max_nodes, candidates_overwrite=None, mode="random"
):
    if mode == "hybrid":
        mode = random.choice(["random", "guided"])

    torch.manual_seed(seed)
    gen_tstart = time.time()
    gen, solution = random_model_gen(
        init_rank=4,
        seed=seed,
        max_nodes=max_nodes,
        candidates_overwrite=candidates_overwrite,
        mode=mode,
    )
    net = SymbolNet(
        gen.abstract_graph, solution, verbose=False, alive_shapes=gen.alive_shapes
    )
    gen_time = time.time() - gen_tstart

    net.enable_proxy_grad()
    net.eval()  # otherwise BN wants batch > 1
    searcher = PracticalHybridSearch(net)
    n_try, sat_inputs = searcher.search(
        max_time_ms=gen_time * 0.02 * 1000, max_sample=2, return_list=True
    )
    net.disable_proxy_grad()

    with torch.no_grad():
        net.eval()

        test_inputs = sat_inputs if sat_inputs else net.get_random_inps(use_cuda=False)

        outputs = net.forward(*test_inputs)

        inames, onames, oidx = torch2onnx(
            net,
            path_prefix + ".onnx",
            verbose=False,
            use_cuda=False,
            dummy_inputs=test_inputs,
        )

        inputs = [t.cpu().numpy() for t in test_inputs]

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs.cpu().numpy()]
        else:
            outputs = [o.cpu().numpy() for o in outputs]

        input_dict = {ina: inp for ina, inp in zip(inames, inputs)}
        output_dict = {oname: outputs[i] for oname, i in zip(onames, oidx)}

        with open(path_prefix + ".pkl", "wb") as f:
            pickle.dump((input_dict, output_dict), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, required=True)
    parser.add_argument("--time_budget", type=int, default=60 * 60 * 4)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--graphfuzz_ops", action="store_true")
    parser.add_argument("--ort_cache", type=str, default=None)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--tar", action="store_true")
    args = parser.parse_args()

    mkdir(args.onnx_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.ort_cache:
        print(args.ort_cache)
        if not os.path.exists(args.ort_cache):
            print(f"Please first generate cache! (mkdir config first)")
            print(f"python nnsmith/dtype_test.py --cache {args.ort_cache}")
            exit(1)
        # must pre run this. otherwise using ort will slow down generation.
        rewrite_op_dtype(ALL_OP_TYPES, factory=None, cache=args.ort_cache)

    if args.graphfuzz_ops:
        candidates_overwrite = GraphFuzz.get_available_op_ts()
    else:
        candidates_overwrite = None

    # FORMAT: {generation time cost in seconds}, {model relative path}
    # MUST RANK by GENERATION ORDER.
    config_file = open(os.path.join(args.onnx_dir, "gentime.csv"), "w")

    start_time = time.time()
    gen_cnt = 0
    valid_cnt = 0

    if args.tar:
        tar = tarfile.open(os.path.join(args.onnx_dir, "models.tar"), "w")

    with tqdm(total=args.time_budget) as pbar:
        while time.time() - start_time < args.time_budget:
            seed = random.getrandbits(32)

            tstart = time.time()
            try:
                with warnings.catch_warnings():  # just shutup.
                    warnings.simplefilter("ignore")
                    nnsmith_gen_once(
                        os.path.join(args.onnx_dir, f"{valid_cnt}"),
                        seed,
                        max_nodes=10,
                        candidates_overwrite=candidates_overwrite,
                        mode=args.mode,
                    )
                to_name = f"{valid_cnt}.onnx"
                label = to_name
                valid_cnt += 1
                if args.tar:
                    tar.add(
                        os.path.join(args.onnx_dir, to_name),
                        arcname="models/" + to_name,
                    )
                    os.unlink(os.path.join(args.onnx_dir, to_name))
            except Exception as e:
                print(f"Fail when seed={seed}")
                print(e)  # Skip a few errors.
                label = "FAILURE"

            time_diff = time.time() - tstart
            config_file.write(f"{time_diff:.5f},{label}\n")

            gen_cnt += 1
            config_file.flush()

            pbar.update(int(time.time() - start_time) - pbar.n)
            pbar.set_description(f"valid={valid_cnt},fail={gen_cnt-valid_cnt}")
            pbar.refresh()
        config_file.close()
