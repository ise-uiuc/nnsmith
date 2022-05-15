import os
import sys
import random
import numpy as np
import pickle
import traceback

from nnsmith.backends import DiffTestBackend
from nnsmith.difftest import assert_allclose
from nnsmith.util import gen_one_input


def mcov_write(path):
    if path:
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(backend._coverage_install().get_hitmap(), f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True,
                        help='List to ONNX model paths')
    parser.add_argument('--backend', type=str, default='tvm',
                        help='One of ort, trt, tvm, and xla')
    parser.add_argument('--memcov', type=str, default=None,
                        help='Path to store memcov.')
    parser.add_argument('--dev', type=str, default='cpu', help='cpu/gpu')
    parser.add_argument('--seed', type=int, default=233,
                        help='to generate random input data')
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

    if args.memcov:
        assert backend._coverage_install().get_now() is not None, "Memcov unavailable!"

    n_unsupported = 0
    for i, path in enumerate(args.models):
        print(f'-> {path}', flush=True, file=sys.stderr)
        onnx_model = DiffTestBackend.get_onnx_proto(path)
        # TODO: Check if needs to run diff test
        oracle_path = path.replace('.onnx', '.pkl')

        try:
            if os.path.exists(oracle_path):
                with open(oracle_path, 'rb') as f:
                    eval_inputs, eval_outputs = pickle.load(f)
                predicted = backend.predict(onnx_model, eval_inputs)
                try:
                    assert_allclose(predicted, eval_outputs)
                except Exception as e:
                    pass
            else:
                input_spec, onames = DiffTestBackend.analyze_onnx_io(
                    onnx_model)
                eval_inputs = gen_one_input(input_spec, 1, 2)
                backend.predict(onnx_model, eval_inputs)

            mcov_write(args.memcov)

        except Exception as e:
            if 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented' in str(type(e)) or \
                    "Unexpected data type for" in str(e):
                # OK we hit an unsupported but valid op in ORT.
                # For simplicity, and we don't want to change `in/out_dtypes`, we just skip it w/o counting time.
                n_unsupported += 1
                continue
            print(
                "==============================================================", file=sys.stderr)
            print(f"Failed execution at {path}", file=sys.stderr)
            traceback.print_exc()
            # Done!

    if n_unsupported == len(args.models):
        print("$ORT.SKIP$ all ORT models are not supported. just don't count this.", file=sys.stderr)

    mcov_write(args.memcov)
