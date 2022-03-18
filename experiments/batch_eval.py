import os
import sys
import random
import numpy as np

from nnsmith.backends import DiffTestBackend
from nnsmith.input_gen import gen_one_input
import traceback

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='List to ONNX model paths')
    parser.add_argument('--backend', type=str, default='tvm', help='One of ort, trt, tvm, and xla')
    parser.add_argument('--dev', type=str, default='cpu', help='cpu/gpu')
    parser.add_argument('--seed', type=int, default=233, help='to generate random input data')
    args = parser.parse_args()

    # Set global seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = None
    if args.backend == 'tvm':
        from nnsmith.backends.tvm_graph import TVMExecutor
        backend = TVMExecutor(opt_level=4)
    elif args.backend == 'ort':
        from nnsmith.backends.ort_graph import ORTExecutor
        backend = ORTExecutor(opt_level=3)
    else:
        raise NotImplementedError("Other backends not supported yet.")

    n_unsupported = 0
    for path in args.models:
        onnx_model = DiffTestBackend.get_onnx_proto(path)
        is_diff_test = os.path.exists(path + '.inp.pkl') # TODO: Check if needs to run diff test
        
        try:
            if is_diff_test:
                # path + '.inp.pkl' -> input tensor dictionary
                # path + '.out.pkl' -> output tensor dictionary (from PyTorch)
                # Run diff test to verify.
                pass # TODO: Implement diff test
            else:
                input_spec, onames = DiffTestBackend.analyze_onnx_io(onnx_model)
                eval_inputs = gen_one_input(input_spec, 0, 1)
                backend.predict(onnx_model, eval_inputs)
        except Exception as e:
            if 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented' in str(type(e)) or \
                "Unexpected data type for" in str(e):
                # OK we hit an unsupported but valid op in ORT.
                # For simplicity, and we don't want to change `in/out_dtypes`, we just skip it w/o counting time.
                n_unsupported += 1
                continue
            print("==============================================================", file=sys.stderr)
            print(f"Failed execution at {path}", file=sys.stderr)
            traceback.print_exc()
            # Done!

    if n_unsupported == len(args.models):
        print("$ORT.SKIP$ all ORT models are not supported. just don't count this.", file=sys.stderr)
