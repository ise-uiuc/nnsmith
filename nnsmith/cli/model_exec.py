"""Directly perform model execution from a bare model, NNSmith model folder, or NNSmith bug report folder.
Example usage:
# compile + run the model (in ORT by default); verify with the oracle under `nnsmith_output`;
 nnsmith.model_exec model.path=nnsmith_output/model.onnx
# compile + run the model; verify with the oracle with fallback mode (random input)
 nnsmith.model_exec model.path=nnsmith_output/model.onnx cmp.oracle=null
# differential testing with tvm
 nnsmith.model_exec model.path=nnsmith_output/model.onnx cmp.with='{type:tvm, optmax:true, target:cpu}'
"""

import os
import pickle
import random
from pathlib import Path
from types import FunctionType
from typing import List, Optional

import hydra
from omegaconf import DictConfig, ListConfig

from nnsmith.backends import BackendFactory
from nnsmith.logging import EXEC_LOG
from nnsmith.macro import NNSMITH_BUG_PATTERN_TOKEN
from nnsmith.materialize import BugReport, Model, Oracle, TestCase


def verify_testcase(
    cmp_cfg: DictConfig,
    factory: BackendFactory,
    testcase: TestCase,
    output_dir: os.PathLike,
    filters: List = None,
    supress_succ=True,
    crash_safe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    if filters is None:
        filters = []

    def check_result(bug_report_or, odir, msg=None) -> bool:  # succ?
        msg = "" if msg is None else msg
        if not isinstance(bug_report_or, BugReport):
            if not supress_succ:
                EXEC_LOG.info(f"[PASS] {msg}")
            return True
        else:
            bug_report = bug_report_or

            for f in filters:
                if f(bug_report):  # filter: no log & dump. but still a bug.
                    filter_name = (
                        f.__name__ if isinstance(f, FunctionType) else type(f).__name__
                    )
                    EXEC_LOG.info(
                        f"Filter [{filter_name}] {bug_report.symptom} at {bug_report.stage}"
                    )
                    return False

            EXEC_LOG.warning("[FAIL] ")
            EXEC_LOG.warning(bug_report.log)
            if odir is not None:
                odir = str(odir).replace(
                    NNSMITH_BUG_PATTERN_TOKEN,
                    f"{bug_report.symptom}-{bug_report.stage}",
                )
                EXEC_LOG.warning("Saving bug report to {}".format(odir))
                bug_report.dump(odir)
            return False

    bug_or_res = factory.checked_compile_and_exec(
        testcase, crash_safe=crash_safe, timeout=timeout
    )
    if check_result(
        bug_or_res, odir=output_dir, msg=f"Compile + Execution [{factory}]"
    ):
        if testcase.oracle.output is not None:  # we have output results && no bug yet
            # do result verification
            if not check_result(
                factory.verify_results(
                    bug_or_res,
                    testcase,
                    equal_nan=cmp_cfg["equal_nan"],
                ),
                odir=output_dir,
                msg=f"Result Verification w/ Oracle from {testcase.oracle.provider}",
            ):
                return False
    else:
        return False

    # Compare with
    if cmp_cfg["with"] is not None and cmp_cfg["with"]["type"] is not None:
        cmp_fac = BackendFactory.init(
            cmp_cfg["with"]["type"],
            target=cmp_cfg["with"]["target"],
            optmax=cmp_cfg["with"]["optmax"],
        )
        cmp_testcase = cmp_fac.make_testcase(
            testcase.model,
            input=testcase.oracle.input,
            crash_safe=crash_safe,
            timeout=timeout,
        )
        if check_result(
            cmp_testcase,
            odir=output_dir,
            msg=f"Compile + Execution [`cmp.with`: {cmp_fac}]",
        ):
            if not check_result(
                factory.verify_results(
                    bug_or_res,
                    cmp_testcase,
                    equal_nan=cmp_cfg["equal_nan"],
                ),
                odir=output_dir,
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
    EXEC_LOG.info(f"Using seed {seed}")

    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"], cfg["backend"]["target"])
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
            EXEC_LOG.info(
                "Using raw input from {} as oracle".format(cmp_cfg["raw_input"])
            )
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
                EXEC_LOG.info("Using oracle from {}".format(oracle_path))
                res = pickle.load(Path(oracle_path).open("rb"))
                test_inputs = res["input"]
                test_outputs = res["output"]
                provider = res["provider"]

        if test_inputs is None:
            EXEC_LOG.info("Generating input data from BackendFactory.make_random_input")
            test_inputs = BackendFactory.make_random_input(model.input_like)
            provider = f"random inputs"

        # the oracle might not have oracle outputs.
        oracle = Oracle(test_inputs, test_outputs, provider)
        testcase = TestCase(model, oracle)

        this_fac = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
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
