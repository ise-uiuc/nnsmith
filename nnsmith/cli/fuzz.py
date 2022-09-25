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

from nnsmith.backends.factory import BackendFactory
from nnsmith.cli.model_exec import verify_testcase
from nnsmith.error import InternalError
from nnsmith.filter import FILTERS
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.logging import FUZZ_LOG
from nnsmith.macro import NNSMITH_BUG_PATTERN_TOKEN
from nnsmith.materialize import Model, Schedule, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import mkdir, parse_timestr, set_seed


class StatusCollect:
    def __init__(self, root):
        self.root = Path(root)
        mkdir(self.root)
        self.n_testcases = 0
        self.n_bugs = 0

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
        if cfg["backend"]["type"] == "tflite" and (
            cfg["cmp"]["with"] is None or cfg["cmp"]["with"]["target"] != "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
        )

        model_cfg = self.cfg["model"]
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )
        self.ModelType.add_seed_setter()
        self.opset = auto_opset(self.ModelType, self.factory)

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

    def make_testcase(self, seed) -> TestCase:
        mgen_cfg = self.cfg["mgen"]
        gen = random_model_gen(
            opset=self.opset,
            init_rank=mgen_cfg["init_rank"],
            seed=seed,
            max_nodes=mgen_cfg["max_nodes"],
            timeout_ms=mgen_cfg["timeout_ms"],
        )

        fixed_graph, concrete_abstensors = concretize_graph(
            gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
        )

        schedule = Schedule.init(fixed_graph, concrete_abstensors)

        model = self.ModelType.from_schedule(schedule)
        if self.cfg["debug"]["viz"]:
            model.attach_viz(fixed_graph)

        model.refine_weights()  # either random generated or gradient-based.
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
            try:
                testcase = self.make_testcase(seed)
            except InternalError as e:
                raise e  # propagate internal errors
            except Exception:
                FUZZ_LOG.error(
                    f"`make_testcase` failed. It could be a NNSmith bug or Generator bug (e.g., {self.cfg['model']['type']})."
                )
                FUZZ_LOG.error(traceback.format_exc())
                repro = "nnsmith.model_gen"
                repro += f" mgen.seed={seed}"
                repro += f" mgen.max_nodes={self.cfg['mgen']['max_nodes']}"
                repro += f" model.type={self.cfg['model']['type']}"
                repro += f" backend.target={self.cfg['backend']['target']}"
                FUZZ_LOG.error(f"repro with: {repro}")
                continue

            if not self.validate_and_report(testcase):
                FUZZ_LOG.warning(f"Failed model seed: {seed}")
            self.status.n_testcases += 1


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    FuzzingLoop(cfg).run()


if __name__ == "__main__":
    main()
