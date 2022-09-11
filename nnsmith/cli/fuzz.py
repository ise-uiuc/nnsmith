import random
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from nnsmith.backends.factory import BackendFactory
from nnsmith.cli.model_exec import verify_testcase
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.logging import FUZZ_LOG
from nnsmith.macro import NNSMITH_BUG_PATTERN_TOKEN
from nnsmith.materialize import Model, Schedule, TestCase
from nnsmith.narrow_spec import opset_from_auto_cache
from nnsmith.util import mkdir, set_seed


class Reporter:
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

        self.reporter = Reporter(cfg["fuzz"]["root"])

        self.factory = BackendFactory.init(
            cfg["backend"]["type"],
            device=cfg["backend"]["device"],
            optmax=cfg["backend"]["optmax"],
            catch_process_crash=False,
        )

        model_cfg = self.cfg["model"]
        self.ModelType = Model.init(model_cfg["type"])
        self.ModelType.add_seed_setter()
        self.opset = opset_from_auto_cache(self.ModelType, self.factory)

        seed = cfg["fuzz"]["seed"] or random.getrandbits(32)
        set_seed(seed)

        FUZZ_LOG.info(
            f"Test success info supressed -- only showing logs for failed tests"
        )

        # Time budget checking.
        self.timeout_s = self.cfg["fuzz"]["time"]
        if isinstance(self.timeout_s, str):
            if self.timeout_s.endswith("hr") or self.timeout_s.endswith("h"):
                self.timeout_s = int(self.timeout_s[:-2]) * 3600
            elif self.timeout_s.endswith("min") or self.timeout_s.endswith("m"):
                self.timeout_s = int(self.timeout_s[:-3]) * 60
            elif self.timeout_s.endswith("s"):
                self.timeout_s = int(self.timeout_s[:-1])
        assert isinstance(
            self.timeout_s, int
        ), "Time budget must be an integer (with `s` (default), `m`/`min`, or `h`/`hr`)."

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
            output_dir=self.reporter.get_next_bug_path(),
        ):
            self.reporter.n_bugs += 1
            return False
        return True

    def run(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout_s:
            seed = random.getrandbits(32)
            FUZZ_LOG.debug(f"Making testcase with seed: {seed}")
            testcase = self.make_testcase(seed)
            if not self.validate_and_report(testcase):
                FUZZ_LOG.warning(f"Failed model seed: {seed}")
            self.reporter.n_testcases += 1


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    FuzzingLoop(cfg).run()


if __name__ == "__main__":
    main()
