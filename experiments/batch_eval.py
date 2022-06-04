import os
import sys
import random
import numpy as np
import pickle

from nnsmith.error import IncorrectResult
from nnsmith.backends import DiffTestBackend
from nnsmith.difftest import assert_allclose
from nnsmith.util import gen_one_input
from nnsmith.fuzz import simple_bug_report


def mcov_write(path):
    if path:
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(backend._coverage_install().get_hitmap(), f)


def verify(backend, backend_name, oracle_name, inputs, oracle=None):
    unsupported = 0
    e_ret = predicted = None
    try:
        predicted = backend.predict(onnx_model, inputs)
        if oracle is not None:
            assert_allclose(predicted, oracle,
                            backend_name, oracle_name)
    except Exception as e:
        e_ret = e
        if 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented' in str(type(e)) or \
                "Unexpected data type for" in str(e):
            # OK we hit an unsupported but valid op in ORT.
            # For simplicity, and we don't want to change `in/out_dtypes`, we just skip it w/o counting time.
            unsupported = 1
            e = None
            # continue
    return e_ret, unsupported, predicted


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
    parser.add_argument('--fuzz_max_nodes', type=int,
                        help='parameter from fuzzer')
    parser.add_argument('--fuzz_seed', type=int,
                        help='seed parameter from fuzzer')
    parser.add_argument('--fuzz_report_folder', type=str,
                        help='parameter from fuzzer')
    parser.add_argument('--clean_after_eval', action='store_true',
                        help='rm models/oracle after eval')
    # add fuzz_timeout?
    args = parser.parse_args()

    if args.fuzz_report_folder is None:
        print(
            '[WARNING] Bug report is not enabled as fuzzer parameters are not provided.', file=sys.stderr)

    # Set global seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = None
    if args.backend == 'tvm':
        from nnsmith.backends.tvm_graph import TVMExecutor
        backend = TVMExecutor(opt_level=4)
        backend_unopt = TVMExecutor(opt_level=0)
    elif args.backend == 'ort':
        from nnsmith.backends.ort_graph import ORTExecutor
        backend = ORTExecutor(opt_level=3)
        backend_unopt = ORTExecutor(opt_level=0)
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
        else:
            print(f'No oracle found for model `{path}`', file=sys.stderr)
            input_spec, onames = DiffTestBackend.analyze_onnx_io(
                onnx_model)
            eval_inputs = gen_one_input(input_spec, 1, 2)
            eval_outputs = None  # No oracle.

        e_vs_tch, unsup, predicted = verify(
            backend, args.backend, "PyTorch", eval_inputs, eval_outputs)
        n_unsupported += unsup
        if e_vs_tch is not None:  # bug found
            loc = 'opt_bug'
            numeric_valid = all(np.isfinite(v).all()
                                for v in eval_outputs.values())
            # confirm the location
            if isinstance(e_vs_tch, IncorrectResult) and numeric_valid:
                assert predicted is not None, "Predicted is None but IncorrectResult caught!"
                e_vs_unopt = verify(
                    backend_unopt, args.backend + "_UnOpt", args.backend, eval_inputs, predicted)[0]
                if e_vs_unopt is None:
                    loc = 'other_bug'

            if args.fuzz_report_folder is not None:
                # failed... report this.
                to_repro = f'python nnsmith/graph_gen.py --max_nodes {args.fuzz_max_nodes} --seed {args.fuzz_seed} --viz_graph'

                # For inconsistency bugs, we only consisder pure-finite number computation.
                if not isinstance(e_vs_tch, IncorrectResult) or numeric_valid:
                    simple_bug_report(
                        report_folder=args.fuzz_report_folder,
                        buggy_onnx_path=path,
                        oracle_path=oracle_path,
                        message=to_repro + '\n' + str(e_vs_tch),
                        bug_type=loc + "-" + type(e_vs_tch).__name__,
                    )

        # remove after the model is tested. useful for locating the crashed model in the batch.
        if args.clean_after_eval:
            if os.path.exists(path):
                os.unlink(path)
            if os.path.exists(oracle_path):
                os.unlink(oracle_path)

        mcov_write(args.memcov)

    if n_unsupported == len(args.models):
        print("$ORT.SKIP$ all ORT models are not supported. just don't count this.", file=sys.stderr)

    mcov_write(args.memcov)
