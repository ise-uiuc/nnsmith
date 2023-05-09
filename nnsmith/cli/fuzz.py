import os
import random
import time
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import FunctionType
from typing import Type

import hydra
from omegaconf import DictConfig

from nnsmith.abstract.extension import activate_ext
from nnsmith.backends.factory import BackendFactory
from nnsmith.cli.model_exec import verify_testcase
from nnsmith.error import InternalError
from nnsmith.filter import FILTERS
from nnsmith.graph_gen import model_gen
from nnsmith.logging import FUZZ_LOG
from nnsmith.macro import NNSMITH_BUG_PATTERN_TOKEN
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import (
    hijack_patch_requires,
    mkdir,
    op_filter,
    parse_timestr,
    set_seed,
)


class StatusCollect:
    def __init__(self, root):
        self.root = Path(root)
        mkdir(self.root)
        self.n_testcases = 0
        self.n_bugs = 0
        self.n_fail_make_test = 0

    def get_next_bug_path(self):
        return self.root / f"bug-{NNSMITH_BUG_PATTERN_TOKEN}-{self.n_bugs}"


class FuzzingLoop:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.cfg = cfg

        # FIXME(@ganler): well-form the fix or report to TF
        # Dirty fix for TFLite on CUDA-enabled systems.
        # If CUDA is not needed, disable them all.
        cmpwith = cfg["cmp"]["with"]
        if cfg["backend"]["type"] == "tflite" and (
            cmpwith is None or cmpwith["target"] != "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if (
            cfg["fuzz"]["crash_safe"]
            and cfg["backend"]["type"] == "xla"
            and cfg["backend"]["target"] == "cuda"
        ) or (
            cmpwith is not None
            and cmpwith["type"] == "xla"
            and cmpwith["target"] == "cuda"
        ):
            raise ValueError(
                "Please set `fuzz.crash_safe=false` for XLA on CUDA. "
                "Also see https://github.com/ise-uiuc/nnsmith/blob/main/doc/known-issues.md"
            )

        self.crash_safe = bool(cfg["fuzz"]["crash_safe"])
        self.test_timeout = cfg["fuzz"]["test_timeout"]
        if self.test_timeout is not None:
            if isinstance(self.test_timeout, str):
                self.test_timeout = parse_timestr(self.test_timeout)
            assert isinstance(
                self.test_timeout, int
            ), "`fuzz.test_timeout` must be an integer (or with `s` (default), `m`/`min`, or `h`/`hr`)."

        if not self.crash_safe and self.test_timeout is not None:
            # user enabled test_timeout but not crash_safe.
            FUZZ_LOG.warning(
                "`fuzz.crash_safe` is automatically enabled given `fuzz.test_timeout` is set."
            )

        self.filters = []
        # add filters.
        filter_types = (
            [cfg["filter"]["type"]]
            if isinstance(cfg["filter"]["type"], str)
            else cfg["filter"]["type"]
        )
        if filter_types:
            patches = (
                [cfg["filter"]["patch"]]
                if isinstance(cfg["filter"]["patch"], str)
                else cfg["filter"]["patch"]
            )
            for f in patches:
                assert os.path.isfile(
                    f
                ), "filter.patch must be a list of file locations."
                assert "@filter(" in open(f).read(), f"No filter found in the {f}."
                spec = spec_from_file_location("module.name", f)
                spec.loader.exec_module(module_from_spec(spec))
                FUZZ_LOG.info(f"Imported filter patch: {f}")
            for filter in filter_types:
                filter = str(filter)
                if filter not in FILTERS:
                    raise ValueError(
                        f"Filter {filter} not found. Available filters: {FILTERS.keys()}"
                    )
                fn_or_cls = FILTERS[filter]
                if isinstance(fn_or_cls, Type):
                    self.filters.append(fn_or_cls())
                elif isinstance(fn_or_cls, FunctionType):
                    self.filters.append(fn_or_cls)
                else:
                    raise InternalError(
                        f"Invalid filter type: {fn_or_cls} (aka {filter})"
                    )
                FUZZ_LOG.info(f"Filter enabled: {filter}")

        self.status = StatusCollect(cfg["fuzz"]["root"])

        self.factory = BackendFactory.init(
            cfg["backend"]["type"],
            ad=cfg["ad"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
            parse_name=True,
        )

        model_cfg = self.cfg["model"]
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )
        self.ModelType.add_seed_setter()

        self.opset = op_filter(
            auto_opset(
                self.ModelType,
                self.factory,
                vulops=cfg["mgen"]["vulops"],
                grad=cfg["mgen"]["grad_check"],
            ),
            cfg["mgen"]["include"],
            cfg["mgen"]["exclude"],
        )

        hijack_patch_requires(cfg["mgen"]["patch_requires"])
        activate_ext(opset=self.opset, factory=self.factory)

        seed = cfg["fuzz"]["seed"] or random.getrandbits(32)
        set_seed(seed)

        FUZZ_LOG.info(
            f"Test success info supressed -- only showing logs for failed tests"
        )

        # Time budget checking.
        self.timeout_s = self.cfg["fuzz"]["time"]
        if isinstance(self.timeout_s, str):
            self.timeout_s = parse_timestr(self.timeout_s)
        assert isinstance(
            self.timeout_s, int
        ), "`fuzz.time` must be an integer (with `s` (default), `m`/`min`, or `h`/`hr`)."

        self.save_test = cfg["fuzz"]["save_test"]
        if isinstance(self.save_test, str):  # path of root dir.
            FUZZ_LOG.info(f"Saving all intermediate testcases to {self.save_test}")
            mkdir(self.save_test)

    def make_testcase(self, seed) -> TestCase:
        mgen_cfg = self.cfg["mgen"]
        gen = model_gen(
            opset=self.opset,
            method=mgen_cfg["method"],
            seed=seed,
            max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
            max_nodes=mgen_cfg["max_nodes"],
            timeout_ms=mgen_cfg["timeout_ms"],
            rank_choices=mgen_cfg["rank_choices"],
            dtype_choices=mgen_cfg["dtype_choices"],
        )

        ir = gen.make_concrete()
        model = self.ModelType.from_gir(ir)
        if self.cfg["debug"]["viz"]:
            model.attach_viz(ir)

        model.refine_weights()  # either random generated or gradient-based.
        model.set_grad_check(self.cfg["mgen"]["grad_check"])
        oracle = model.make_oracle()
        return TestCase(model, oracle)

    def validate_and_report(self, testcase: TestCase) -> bool:
        if not verify_testcase(
            self.cfg["cmp"],
            factory=self.factory,
            testcase=testcase,
            output_dir=self.status.get_next_bug_path(),
            filters=self.filters,
            crash_safe=self.crash_safe,
            timeout=self.test_timeout,
        ):
            self.status.n_bugs += 1
            return False
        return True

    def run(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout_s:
            seed = random.getrandbits(32)
            FUZZ_LOG.debug(f"Making testcase with seed: {seed}")

            time_stat = {}

            gen_start = time.time()
            try:
                testcase = self.make_testcase(seed)
            except InternalError as e:
                raise e  # propagate internal errors
            except Exception:
                FUZZ_LOG.error(
                    f"`make_testcase` failed with seed {seed}. It can be NNSmith or Generator ({self.cfg['model']['type']}) bug."
                )
                FUZZ_LOG.error(traceback.format_exc())
                self.status.n_fail_make_test += 1
                continue
            time_stat["gen"] = time.time() - gen_start

            eval_start = time.time()
            if not self.validate_and_report(testcase):
                FUZZ_LOG.warning(f"Failed model seed: {seed}")
            time_stat["eval"] = time.time() - eval_start

            if self.save_test:
                save_start = time.time()
                testcase_dir = os.path.join(
                    self.save_test, f"{time.time() - start_time:.3f}"
                )
                mkdir(testcase_dir)
                tmp, testcase.model.dotstring = testcase.model.dotstring, None
                testcase.dump(testcase_dir)
                testcase.model.dotstring = tmp
                time_stat["save"] = time.time() - save_start

            FUZZ_LOG.info(
                f"Timing: { ''.join(f'{k}: {1000 * v:.1f}ms, ' for k, v in time_stat.items()) }"
            )
            self.status.n_testcases += 1
        FUZZ_LOG.info(f"Total {self.status.n_testcases} testcases generated.")
        FUZZ_LOG.info(f"Total {self.status.n_bugs} bugs found.")
        FUZZ_LOG.info(f"Total {self.status.n_fail_make_test} failed to make testcases.")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    FuzzingLoop(cfg).run()


if __name__ == "__main__":
    main()
