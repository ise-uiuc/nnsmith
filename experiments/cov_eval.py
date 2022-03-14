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
import numpy as np
import argparse
import subprocess

from tqdm import tqdm

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
        yield iterable[ndx:min(ndx + n, l)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Folder to the onnx models.')
    parser.add_argument('--report_folder', type=str, required=True)
    parser.add_argument('--backend', type=str, default='tvm', help='One of ort, trt, tvm, and xla')
    parser.add_argument('--dp', type=int, default=100, help='# data point you want.')
    parser.add_argument('--dev', type=str, default='cpu', help='cpu/gpu')
    parser.add_argument('--sum', action='store_true', help='Use summary.')
    parser.add_argument('--seed', type=int, default=233, help='to generate random input data')
    parser.add_argument('--lib', type=str, required=True, help='path to instrumented library')
    parser.add_argument('--max_time', type=int, default=60*60*4, help='max time in seconds for coverage evaluation')
    parser.add_argument('--llvm-version', type=str, default='', help='version of llvm during coverage stuff. must align w/ tvm.')
    args = parser.parse_args()

    # Set global seed
    print(f'==> Setting global seed to {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    HAS_LZ4 = os.system('lz4 --version > /dev/null 2>&1') == 0
    if not HAS_LZ4:
        print('==> lz4 not found. Storing lcov w/o compression is disk-killing. Please install lz4!')
        exit(1)

    lib_expr = ''
    for lib in args.lib.split():
        assert os.path.exists(lib), f'{lib} does not exist!'
        lib_expr += f' -object {os.path.realpath(lib)} '

    mkdir(args.report_folder)

    # FORMAT:
    #   batch time (gen time + eval time)
    #   seed
    #   # models
    config_file = open(os.path.join(args.report_folder, 'stats.csv'), 'w')
    stderr_file = open(os.path.join(args.report_folder, 'stderr.txt'), 'w')

    def record(btime, n_models, seed):
        config_file.write(f'{btime},{seed},{n_models}\n')
        config_file.flush()

    lines = open(os.path.join(args.model_dir, 'gentime.csv'), 'r').readlines()
    batch_size = max(len(lines) // args.dp, 1)

    print(f'==> Setting batch size: {batch_size}')
    batch_list = list(batched(lines, n=batch_size))
    process_start_time = time()

    for i in tqdm(range(len(batch_list))):
        if time() - process_start_time > args.max_time:
            print(f'==> Timeout!')
            break

        batch = batch_list[i]
        btime = 0
        sum_path = os.path.join(args.report_folder, f'{i}.txt')
        lcov_path = os.path.join(args.report_folder, f'{i}.lcov')
        seed = random.getrandbits(32)

        model_batch = []
        for line in batch:
            tstr, mstr = line.rstrip('\n').split(',')
            btime += float(tstr) # Generation time
            if mstr != 'FAILURE':
                model_batch.append(os.path.join(args.model_dir, mstr))

        profraw_path = os.path.join(args.report_folder, f'{i}.profraw')

        # Execute batch evaluation
        copied_env = os.environ.copy()
        # Path to store llvm profile.
        copied_env["LLVM_PROFILE_FILE"] = str(profraw_path)

        tstart = time()          # <=== START
        p = subprocess.Popen([
            'python', 'experiments/batch_eval.py', 
            '--models', *model_batch, 
            '--backend', args.backend,
            '--dev', args.dev,
            '--seed', str(seed),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=copied_env)
        _, errs = p.communicate()
        errs = errs.decode()
        exit_code = p.returncode

        # Write stderr
        stderr_file.write(f'iter {i}: {model_batch} ~ exit_code={exit_code}\n')
        if errs:
            stderr_file.write(errs)
        stderr_file.flush()

        if '$ORT.SKIP$' in errs: # all models are unsupported by ort. skip it.
            if os.path.exists(profraw_path):
                os.remove(profraw_path)
            continue

        btime += time() - tstart # <=== ENDING

        # Wrap up this batch.

        # Get coverage report
        if os.path.exists(profraw_path):
            llvm_profdata = 'llvm-profdata'
            llvm_cov = 'llvm-cov'
            if args.llvm_version and str(args.llvm_version).isnumeric():
                llvm_profdata += f'-{args.llvm_version}'
                llvm_cov += f'-{args.llvm_version}'

            profdata_path = os.path.join(args.report_folder, f'{i}.profdata')
            # summary might be useless as it does not consider prior runs.
            if 0 != os.system(f'{llvm_profdata} merge -sparse {profraw_path} -o {profdata_path}') or \
                0 != os.system(f'{llvm_cov} export -instr-profile={profdata_path} -format=lcov {lib_expr} > {lcov_path}'):
                print(f'Getting coverage failed!!', file=sys.stderr)
            else: # clean temporary files
                if args.sum:
                    os.system(f'{llvm_cov} report -instr-profile={profdata_path} {lib_expr} > {sum_path}')
                if 0 == os.system(f'lz4 {lcov_path}'):
                    os.remove(lcov_path)
                os.remove(profraw_path)
                os.remove(profdata_path)
        else:
            print(f'{profraw_path} does not exist...', file=sys.stderr)

        record(btime, len(model_batch), seed)

    config_file.close()
    stderr_file.close()
