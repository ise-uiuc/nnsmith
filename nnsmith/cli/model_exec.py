"""Directly perform model execution from a bare model, NNSmith model folder, or NNSmith bug report folder.
Example usage:
# compile + run the model (in ORT by default); verify with the oracle under `nnsmith_output`;
 nnsmith.model_exec model.path=nnsmith_output/model.onnx
# compile + run the model; verify with the oracle with fallback mode (random input)
 nnsmith.model_exec model.path=nnsmith_output/model.onnx oracle=null
# differential testing with tvm
 nnsmith.model_exec model.path=nnsmith_output/model.onnx cmp_with='{type:tvm, optmax:true, device:cpu}'
"""

import os
import pickle
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig

from nnsmith.backends import BackendFactory
from nnsmith.materialize import BugReport, Model, Oracle, TestCase
from nnsmith.util import fail_print, note_print, succ_print


@hydra.main(version_base=None, config_path="../config", config_name="model_exec")
def main(cfg: DictConfig):
    seed = random.getrandbits(32) if cfg["seed"] is None else cfg["seed"]
    note_print(f"Using seed {seed}")

    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"])
    ModelType.add_seed_setter()

    if isinstance(model_cfg["path"], ListConfig):
        model_paths = model_cfg["path"]
        if cfg["output"] is not None:
            assert isinstance(cfg["output"], ListConfig), (
                "With multiple models compiled-and-executed together,"
                " the `output_dir` must also be a list of paths."
            )
    else:
        model_paths = [model_cfg["path"]]

    output_dirs = cfg["output"]
    if isinstance(output_dirs, (int, float, str)):
        output_dirs = [Path(output_dirs)]

    for i, model_path in enumerate(model_paths):
        model = ModelType.load(model_path)
        model_basename = os.path.basename(os.path.normpath(model_path))

        test_inputs = None
        test_outputs = None
        provider = "unknown"

        # 1. Use raw_input as test_inputs if specified;
        if cfg["raw_input"] is not None:
            note_print("Using raw input from {} as oracle".format(cfg["raw_input"]))
            test_inputs = pickle.load(Path(cfg["raw_input"]).open("rb"))
            assert isinstance(
                test_inputs, dict
            ), "Raw input type should be Dict[str, np.ndarray]"
            provider = "raw input from {}".format(cfg["raw_input"])

        # 2. Otherwise, use existing or generated oracle;
        if test_inputs is None:
            oracle_path = None
            # 1. check if we can directly use oracle from `oracle.pkl`
            if "auto" == cfg["oracle"]:
                oracle_path = model_path.replace(model_basename, "oracle.pkl")
                if not os.path.exists(oracle_path):
                    oracle_path = None
            elif cfg["oracle"] is not None:
                oracle_path = cfg["oracle"]

            if oracle_path is not None:
                note_print("Using oracle from {}".format(oracle_path))
                res = pickle.load(Path(oracle_path).open("rb"))
                test_inputs = res["input"]
                test_outputs = res["output"]
                provider = res["provider"]

        if test_inputs is None:
            note_print("Generating input data from BackendFactory.make_random_input")
            test_inputs = BackendFactory.make_random_input(model.input_like)
            provider = f"random inputs"

        # the oracle might not have oracle outputs.
        oracle = Oracle(test_inputs, test_outputs, provider)
        testcase = TestCase(model, oracle)

        this_fac = BackendFactory.init(
            cfg["backend"]["type"],
            device=cfg["backend"]["device"],
            optmax=cfg["backend"]["optmax"],
            catch_process_crash=False,
        )

        def emit_testcase(bug_report_or, msg=None) -> bool:  # succ?
            msg = "" if msg is None else msg
            if not isinstance(bug_report_or, BugReport):
                succ_print(f"[PASS] {msg}")
                return True
            else:
                bug_report = bug_report_or
                fail_print("[FAIL] ")
                fail_print(bug_report.log)
                if output_dirs is not None:
                    note_print("Saving bug report to {}".format(output_dirs[i]))
                    bug_report.dump(output_dirs[i])
                return False

        bug_or_res = this_fac.checked_compile_and_exec(testcase)
        if emit_testcase(bug_or_res, msg="Compile + Execution"):
            if test_outputs is not None:  # we have output results && no bug yet
                # do result verification
                emit_testcase(
                    this_fac.verify_results(
                        bug_or_res,
                        testcase,
                        equal_nan=cfg["equal_nan"],
                    ),
                    msg="Result Verification w/ Oracle",
                )

        # Compare with
        if cfg["cmp_with"] is not None:
            cmp_fac = BackendFactory.init(
                cfg["cmp_with"]["type"],
                device=cfg["cmp_with"]["device"],
                optmax=cfg["cmp_with"]["optmax"],
                catch_process_crash=False,
            )
            cmp_testcase = cmp_fac.make_testcase(model, input=test_inputs)
            if emit_testcase(cmp_testcase, "Compile + Execution (`cmp_with`)"):
                emit_testcase(
                    this_fac.verify_results(
                        bug_or_res,
                        cmp_testcase,
                        equal_nan=cfg["equal_nan"],
                    ),
                    msg="Result Verification w/ Reference Backend",
                )


if __name__ == "__main__":
    main()
