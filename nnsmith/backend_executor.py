import random
import dill as pickle
from pathlib import Path
import time

import onnx
import onnx.checker

from nnsmith import difftest
from nnsmith.util import is_invalid
from nnsmith.backends import BackendFactory, gen_one_input_rngs, mk_factory

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help=f"One of `tvm`, `ort`, `trt`, and `xla`",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--optmin", action="store_true")
    parser.add_argument(
        "--model", type=str, help="For debugging purpose: path to onnx model;"
    )
    parser.add_argument("--dump_raw", help="Dumps the raw output to the specified path")
    parser.add_argument(
        "--raw_input",
        type=str,
        help="When specified, the model will be fed with the specified input. Otherwise, input will be generated on the fly.",
    )
    parser.add_argument("--oracle", type=str, help="Path to the oracle")
    parser.add_argument("--seed", type=int, help="to generate random input data")
    parser.add_argument(
        "--cmp_with", type=str, default=None, help="the backend to compare with"
    )
    parser.add_argument("--print_output", action="store_true")

    # TODO: Add support for passing backend-specific options
    args = parser.parse_args()

    st = time.time()
    if args.seed is None:
        seed = random.getrandbits(32)
    else:
        seed = args.seed
    print("Using seed:", seed)

    onnx_model = onnx.load(args.model)
    onnx.checker.check_model(onnx_model, full_check=True)

    # Step 1: Generate input
    oracle = None
    oracle_outputs = None
    # -- oracle:
    if args.oracle == "auto":
        args.oracle = args.model.replace("model.onnx", "oracle.pkl")

    if args.oracle is not None:
        print("Using oracle from:", args.oracle)
        res = pickle.load(Path(args.oracle).open("rb"))
        test_inputs, oracle_outputs = res[0], res[1]
        num_must_valid = res[2] if len(res) == 3 else False
        if not num_must_valid:
            print("Inputs from oracle might cause NaN/Inf compute.")
        else:
            print("Inputs from oracle should be valid.")
    # -- raw_input:
    else:
        if args.raw_input is not None:
            print("Using raw input pkl file from:", args.raw_input)
            test_inputs = pickle.load(Path(args.raw_input).open("rb"))
        # -- randomly generated input:
        else:
            print("No raw input or oracle found. Generating input on the fly.")
            inp_spec = BackendFactory.analyze_onnx_io(onnx_model)[0]
            test_inputs = gen_one_input_rngs(inp_spec, None, seed)

    # Step 2: Run backend
    # -- reference backend:
    if args.cmp_with is not None:
        print(f"Using {args.cmp_with} as the reference backend/oracle")
        # use optmin for the reference backend
        ref_backend = mk_factory(
            args.cmp_with, device=args.device, optmax=False
        ).mk_backend(onnx_model)
        oracle_outputs = ref_backend(test_inputs)
        if is_invalid(oracle_outputs):
            print(f"[WARNING] Backend {args.cmp_with} produces nan/inf in output.")

    # -- this backend:
    this_backend = mk_factory(
        args.backend, device=args.device, optmax=not args.optmin
    ).mk_backend(onnx_model)
    this_outputs = this_backend(test_inputs)
    if is_invalid(this_outputs):
        print(f"[WARNING] Backend {args.backend} produces nan/inf in output.")

    if args.dump_raw is not None:
        print("Storing (input,output,oracle_outputs) pair to:", args.dump_raw)
        pickle.dump(
            (test_inputs, this_outputs, oracle_outputs), open(args.dump_raw, "wb")
        )

    if args.print_output:
        print("this_output=", this_outputs)
        print("oracle_output=", oracle_outputs)
        try:
            difftest.assert_allclose(
                this_outputs,
                oracle_outputs,
                args.backend,
                args.cmp_with if args.cmp_with else "oracle",
                atol=0,
                rtol=0,
            )
        except Exception as e:
            print("Errors=")
            print(e)

    # Step 3: Compare
    if oracle_outputs is not None:
        difftest.assert_allclose(
            this_outputs,
            oracle_outputs,
            args.backend,
            args.cmp_with if args.cmp_with else "oracle",
        )
        print("Differential testing passed!")

    print(f"Total time: {time.time() - st}")
