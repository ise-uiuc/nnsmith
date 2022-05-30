import glob
import os
import shutil
import sys
import random
import numpy as np
import pickle
import traceback

from nnsmith.backends import DiffTestBackend
from nnsmith.difftest import assert_allclose
from nnsmith.util import gen_one_input
from nnsmith.fuzz import simple_bug_report


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
    parser.add_argument('--seed', type=int, default=233,
                        help='to generate random input data')
    parser.add_argument('--fuzz_max_nodes', type=int, required=True,
                        help='parameter from fuzzer')
    parser.add_argument('--fuzz_seed', type=int, required=True,
                        help='parameter from fuzzer')
    parser.add_argument('--fuzz_report_folder', type=str, required=True,
                        help='parameter from fuzzer')
    # add fuzz_timeout?
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
        oracle_path = path.replace('.onnx', '.pkl')
        if os.path.exists(oracle_path):
            with open(oracle_path, 'rb') as f:
                eval_inputs, eval_outputs = pickle.load(f)
            try:
                predicted = backend.predict(onnx_model, eval_inputs)
                assert_allclose(predicted, eval_outputs,
                                args.backend, "PyTorch")
            except Exception as e:
                if 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented' in str(type(e)) or \
                        "Unexpected data type for" in str(e):
                    # OK we hit an unsupported but valid op in ORT.
                    # For simplicity, and we don't want to change `in/out_dtypes`, we just skip it w/o counting time.
                    n_unsupported += 1
                    # continue
                # failed... report this.
                to_repro = f'python nnsmith/graph_gen.py --max_nodes {args.fuzz_max_nodes} --seed {args.fuzz_seed} --viz_graph'
                # TODO: don't report nanerror if input search failed.
                simple_bug_report(
                    report_folder=args.fuzz_report_folder,
                    buggy_onnx_path=path,
                    oracle_path=oracle_path,
                    message=to_repro + '\n' + str(e),
                    bug_type=type(e).__name__,
                )
        else:
            # TODO: Delete if not needed.
            raise NotImplementedError(f'No oracle for {path}')
            input_spec, onames = DiffTestBackend.analyze_onnx_io(
                onnx_model)
            eval_inputs = gen_one_input(input_spec, 1, 2)
            backend.predict(onnx_model, eval_inputs)
        if os.path.exists(path):
            # remove after the model is tested. useful for locating the crashed model in the batch.
            os.unlink(path)
            os.unlink(oracle_path)

        mcov_write(args.memcov)

    if n_unsupported == len(args.models):
        print("$ORT.SKIP$ all ORT models are not supported. just don't count this.", file=sys.stderr)

    mcov_write(args.memcov)
