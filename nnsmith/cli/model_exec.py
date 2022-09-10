"""Directly perform model execution from a bare model, NNSmith model folder, or NNSmith bug report folder.
Example usage:
# compile + run the model (in ORT by default); verify with the oracle under `nnsmith_output`;
 nnsmith.model_exec model.path=nnsmith_output/model.onnx
# compile + run the model; verify with the oracle with fallback mode (random input)
 nnsmith.model_exec model.path=nnsmith_output/model.onnx cmp.oracle=null
# differential testing with tvm
 nnsmith.model_exec model.path=nnsmith_output/model.onnx cmp.with='{type:tvm, optmax:true, device:cpu}'
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


def verify_testcase(
    cmp_cfg: DictConfig,
    factory: BackendFactory,
    testcase: TestCase,
    output_dir: os.PathLike,
) -> bool:
    def check_result(bug_report_or, msg=None) -> bool:  # succ?
        msg = "" if msg is None else msg
        if not isinstance(bug_report_or, BugReport):
            if cmp_cfg["verbose"]:
                succ_print(f"[PASS] {msg}")
            return True
        else:
            bug_report = bug_report_or
            fail_print("[FAIL] ")
            fail_print(bug_report.log)
            if output_dir is not None:
                note_print("Saving bug report to {}".format(output_dir))
                bug_report.dump(output_dir)
            return False

    bug_or_res = factory.checked_compile_and_exec(testcase)
    if check_result(bug_or_res, msg="Compile + Execution"):
        if testcase.oracle.output is not None:  # we have output results && no bug yet
            # do result verification
            if not check_result(
                factory.verify_results(
                    bug_or_res,
                    testcase,
                    equal_nan=cmp_cfg["equal_nan"],
                ),
                msg="Result Verification w/ Oracle",
            ):
                return False
    else:
        return False

    # Compare with
    if cmp_cfg["with"] is not None:
        cmp_fac = BackendFactory.init(
            cmp_cfg["with"]["type"],
            device=cmp_cfg["with"]["device"],
            optmax=cmp_cfg["with"]["optmax"],
            catch_process_crash=False,
        )
        cmp_testcase = cmp_fac.make_testcase(
            testcase.model, input=testcase.oracle.input
        )
        if check_result(cmp_testcase, "Compile + Execution (`cmp.with`)"):
            if not check_result(
                factory.verify_results(
                    bug_or_res,
                    cmp_testcase,
                    equal_nan=cmp_cfg["equal_nan"],
                ),
                msg="Result Verification w/ Reference Backend",
            ):
                return False
        else:
            return False

    return True


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    cmp_cfg = cfg["cmp"]
    seed = random.getrandbits(32) if cmp_cfg["seed"] is None else cmp_cfg["seed"]
    note_print(f"Using seed {seed}")

    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"])
    ModelType.add_seed_setter()

    if isinstance(model_cfg["path"], ListConfig):
        model_paths = model_cfg["path"]
        if cmp_cfg["save"] is not None:
            assert isinstance(cmp_cfg["save"], ListConfig), (
                "With multiple models compiled-and-executed together,"
                " the `save` must also be a list of paths."
            )
    else:
        model_paths = [model_cfg["path"]]

    output_dirs = cmp_cfg["save"]
    if isinstance(output_dirs, (int, float, str)):
        output_dirs = [Path(output_dirs)]

    for i, model_path in enumerate(model_paths):
        model = ModelType.load(model_path)
        model_basename = os.path.basename(os.path.normpath(model_path))

        test_inputs = None
        test_outputs = None
        provider = "unknown"

        # 1. Use raw_input as test_inputs if specified;
        if cmp_cfg["raw_input"] is not None:
            note_print("Using raw input from {} as oracle".format(cmp_cfg["raw_input"]))
            test_inputs = pickle.load(Path(cmp_cfg["raw_input"]).open("rb"))
            assert isinstance(
                test_inputs, dict
            ), "Raw input type should be Dict[str, np.ndarray]"
            provider = "raw input from {}".format(cmp_cfg["raw_input"])

        # 2. Otherwise, use existing or generated oracle;
        if test_inputs is None:
            oracle_path = None
            # 1. check if we can directly use oracle from `oracle.pkl`
            if "auto" == cmp_cfg["oracle"]:
                oracle_path = model_path.replace(model_basename, "oracle.pkl")
                if not os.path.exists(oracle_path):
                    oracle_path = None
            elif cmp_cfg["oracle"] is not None:
                oracle_path = cmp_cfg["oracle"]

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

        output_dir = None if output_dirs is None else output_dirs[i]
        verify_testcase(
            cmp_cfg,
            this_fac,
            testcase,
            output_dir,
        )


if __name__ == "__main__":
    main()
