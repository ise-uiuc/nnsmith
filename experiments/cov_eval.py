"""Evaluation steps:
1. Get onnx models:
    - LEMON:
        Run LEMON to generate models (https://github.com/ganler/LEMON);
        python experiments/lemon_tf2onnx.py --lemon_output_dir /.../LEMON/lemon_outputs/ --onnx_dir ...
    - NNSMITH: TBD
    - GRAPH-FUZZ: TBD
2. Get source-level coverage: https://github.com/ganler/tvm/tree/coverage
    python experiments/cov_eval.py --model_dir ONNX_DIR --report_folder REPORT_FOLDER
"""

import os
import sys
import random
from time import time
import argparse
import subprocess

from tqdm import tqdm
import numpy as np

from nnsmith.util import mkdir

# CMD EXAMPLE:
# python experiments/cov_eval.py --model_dir lemon --report_folder test-cov --backend tvm --lib ../tvm/build/libtvm.so --llvm-version 14
# python experiments/cov_eval.py --model_dir test-onnx --report_folder test-cov-q --backend ort \
#  --lib '../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime_providers_shared.so ../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so'  --llvm-version 14

# Coverage lcov:          {i}.lcov
# Timing / # model:       stats.csv


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Folder to the onnx models."
    )
    parser.add_argument("--report_folder", type=str, required=True)
    parser.add_argument(
        "--backend", type=str, default="tvm", help="One of ort, trt, tvm, and xla"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dp", type=int, default=250, help="# data point you want.")
    parser.add_argument("--dev", type=str, default="cpu", help="cpu/gpu")
    parser.add_argument("--sum", action="store_true", help="Use summary.")
    parser.add_argument("--resume", action="store_true", help="Resume eval.")
    parser.add_argument(
        "--seed", type=int, default=233, help="to generate random input data"
    )
    parser.add_argument("--memcov", action="store_true", help="Use memcov.")
    parser.add_argument(
        "--lib", type=str, default=None, help="path to instrumented library"
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=60 * 60 * 4,
        help="max time in seconds for coverage evaluation",
    )
    parser.add_argument(
        "--llvm-version",
        type=str,
        default="",
        help="version of llvm during coverage stuff. must align w/ tvm.",
    )
    parser.add_argument("-y", action="store_true")
    parser.add_argument("--keep_raw", action="store_true")
    args = parser.parse_args()

    # Set global seed
    print(f"==> Setting global seed to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    HAS_LZ4 = os.system("lz4 --version > /dev/null 2>&1") == 0
    if not HAS_LZ4 and not args.memcov:
        print(
            "==> lz4 not found. Storing lcov w/o compression is disk-killing. Please install lz4!"
        )
        exit(1)

    if args.lib:
        lib_expr = ""
        for lib in args.lib.split():
            assert os.path.exists(lib), f"{lib} does not exist!"
            lib_expr += f" -object {os.path.realpath(lib)} "
    else:
        assert (
            args.memcov
        ), "you need to provide either --lib for ast cov or --memcov for memcov."

    if args.memcov and args.dp < 1000:
        print(
            "==> For memcov, you need at least 1k data points as we lose cov when it crashes."
        )
        print(f"==> Setting data point from {args.dp} to 1000.")
        args.dp = 1000

    if not args.resume:
        mkdir(args.report_folder, yes=args.y)
        # FORMAT:
        #   batch time (gen time + eval time)
        #   seed
        #   # models
        config_file = open(os.path.join(args.report_folder, "stats.csv"), "w")
        stderr_file = open(os.path.join(args.report_folder, "stderr.txt"), "w")

        next_batch_since_hist = 0
        process_time_sum = 0  # sum of all btime
    else:
        print("==> Resuming eval...")
        assert os.path.exists(
            os.path.join(args.report_folder, "stats.csv")
        ), "stats.csv does not exist!"
        import pandas as pd

        btimes = (
            pd.read_csv(
                os.path.join(args.report_folder, "stats.csv"), usecols=[0], header=None
            )
            .to_numpy()
            .squeeze()
        )
        process_time_sum = btimes.sum()

        history = open(os.path.join(args.report_folder, "stats.csv"), "r").readlines()
        if len(history) > 0 and not history[-1]:
            history.pop()

        next_batch_since_hist = (
            int(history[-1].rstrip().split(",")[-1].split(".")[0]) + 1
        )

        config_file = open(os.path.join(args.report_folder, "stats.csv"), "a")
        stderr_file = open(os.path.join(args.report_folder, "stderr.txt"), "a")

    def record(btime, n_models, seed, cov_name):
        config_file.write(f"{btime},{seed},{n_models},{cov_name}\n")
        config_file.flush()

    lines = open(os.path.join(args.model_dir, "gentime.csv"), "r").readlines()
    batch_size = max(len(lines) // args.dp, 1)

    print(f"==> Setting batch size: {batch_size}")
    batch_list = list(batched(lines, n=batch_size))

    # sometimes stuff got crashed and we will attribute the cost to the later batch.
    lagged_time = 0
    lagged_n_model = 0

    btime_begin = int(process_time_sum)
    start_time = time()

    with tqdm(total=args.max_time) as pbar:
        pbar.update(process_time_sum)
        pbar.refresh()

        for i in range(next_batch_since_hist, len(batch_list)):
            if process_time_sum > args.max_time:
                print(f"==> Timeout!")
                break

            batch = batch_list[i]
            btime = 0
            sum_path = os.path.join(args.report_folder, f"{i}.txt")

            # for source cov;
            lcov_name = f"{i}.lcov"
            lcov_path = os.path.join(args.report_folder, lcov_name)

            # for memcov
            memcov_name = f"{i}.memcov"
            memcov_path = os.path.join(args.report_folder, memcov_name)

            seed = random.getrandbits(32)

            model_batch = []
            for line in batch:
                tstr, mstr = line.rstrip("\n").split(",")
                btime += float(tstr)  # Generation time
                if mstr != "FAILURE":
                    model_batch.append(os.path.join(args.model_dir, mstr))

            profraw_path = os.path.join(args.report_folder, f"{i}.profraw")

            # Execute batch evaluation
            copied_env = os.environ.copy()
            # Path to store llvm profile.
            copied_env["LLVM_PROFILE_FILE"] = str(profraw_path)

            tstart = time()  # <=== START
            arguments = [
                "python",
                "experiments/batch_eval.py",
                "--models",
                *model_batch,
                "--backend",
                args.backend,
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--fuzz_report_folder",
                args.report_folder,
            ]
            if args.memcov:
                arguments += ["--memcov", memcov_path]
            p = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=copied_env,
            )
            _, errs = p.communicate()
            errs = errs.decode()
            exit_code = p.returncode

            if exit_code != 0:
                print(f"==> Batch {i} process crashed!")

            # Write stderr
            stderr_file.write(f"iter {i}: =================> EXIT CODE {exit_code}\n")
            if errs:
                stderr_file.write(errs)
            stderr_file.flush()

            if "$ORT.SKIP$" in errs:  # all models are unsupported by ort. skip it.
                if os.path.exists(profraw_path) and not args.keep_raw:
                    os.remove(profraw_path)
                continue

            btime += time() - tstart  # <=== ENDING
            process_time_sum += btime

            # Wrap up this batch.

            # Get coverage report
            if not args.memcov:
                if os.path.exists(profraw_path):
                    llvm_profdata = "llvm-profdata"
                    llvm_cov = "llvm-cov"
                    if args.llvm_version and str(args.llvm_version).isnumeric():
                        llvm_profdata += f"-{args.llvm_version}"
                        llvm_cov += f"-{args.llvm_version}"

                    profdata_path = os.path.join(args.report_folder, f"{i}.profdata")
                    # summary might be useless as it does not consider prior runs.
                    if 0 != os.system(
                        f"{llvm_profdata} merge -sparse {profraw_path} -o {profdata_path}"
                    ) or 0 != os.system(
                        f"{llvm_cov} export -instr-profile={profdata_path} -format=lcov {lib_expr} > {lcov_path}"
                    ):
                        print(f"Getting coverage failed!!", file=sys.stderr)
                    else:  # clean temporary files
                        if args.sum:
                            os.system(
                                f"{llvm_cov} report -instr-profile={profdata_path} {lib_expr} > {sum_path}"
                            )
                        assert 0 == os.system(f"lz4 {lcov_path} {lcov_path}.lz4")
                        if not args.keep_raw:
                            os.remove(profraw_path)
                            os.remove(profdata_path)
                            os.remove(lcov_path)
                else:
                    print(f"{profraw_path} does not exist...", file=sys.stderr)

            if os.path.exists(lcov_path + ".lz4"):
                record(
                    btime + lagged_time,
                    len(model_batch) + lagged_n_model,
                    seed,
                    cov_name=lcov_name,
                )
                lagged_time = 0
                lagged_n_model = 0
            elif os.path.exists(memcov_path + ".pkl"):
                record(
                    btime + lagged_time,
                    len(model_batch) + lagged_n_model,
                    seed,
                    cov_name=memcov_name,
                )
                lagged_time = 0
                lagged_n_model = 0
            else:
                # Means no lcov or memcov due to some LLVM issues.
                lagged_time += btime
                lagged_n_model += len(model_batch)

            pbar.update(int(time() - start_time + btime_begin) - pbar.n)
            pbar.set_description(f"batch tasks: {i}/{len(batch_list)}")
            pbar.refresh()

        config_file.close()
        stderr_file.close()
