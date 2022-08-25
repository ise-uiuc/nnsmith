from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional
import traceback

import numpy as np

from nnsmith.abstract.tensor import AbsTensor
from nnsmith.materialize import BugReport, Oracle, Stage, Symptom, TestCase, Model
from nnsmith.difftest import assert_allclose


class BackendFactory(ABC):
    def __init__(self, device="cpu", optmax=True, catch_process_crash=True):
        super().__init__()
        self.device = device
        self.optmax = optmax
        # If true, will run the compilation and execution in a subprocess.
        # and catch segfaults returned as BugReport.
        self.catch_process_crash = catch_process_crash

    @property
    @abstractmethod
    def system_name(self) -> str:
        pass

    def __str__(self) -> str:
        return (
            f"{self.system_name} (opt={'max' if self.optmax else 'min'}-{self.device})"
        )

    @staticmethod
    def make_random_input(
        input_like: Dict[str, AbsTensor], low=1, high=2
    ) -> Dict[str, np.ndarray]:
        return {
            name: np.random.uniform(low=low, high=high, size=aten.shape).astype(
                aten.dtype.numpy()
            )
            for name, aten in input_like.items()
        }

    @abstractmethod
    def mk_backend(
        self, model: Model
    ) -> Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        raise NotImplementedError

    def verify_testcase(self, testcase: TestCase) -> Optional[BugReport]:
        # model = testcase.model.native_model

        # TODO(@ganler): impl fault catching in subprocess
        assert not self.catch_process_crash, "not implemented"

        try:  # compilation
            print(testcase.model)
            executable = self.mk_backend(testcase.model)
        except Exception:
            return BugReport(
                testcase=testcase,
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.COMPILATION,
                log=traceback.format_exc(),
            )

        if not testcase.oracle:
            # generate inputs automatically.
            input = self.make_random_input(testcase.model.input_like())
        else:
            input = testcase.oracle.input

        try:  # execution
            output = executable(input)
        except Exception:
            return BugReport(
                testcase=testcase,
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.EXECUTION,
                log=traceback.format_exc(),
            )

        if not testcase.oracle.output:
            try:  # verification
                assert_allclose(
                    output,
                    testcase.oracle.output,
                    self.__str__(),
                    testcase.oracle.provider,
                )
            except AssertionError:
                return BugReport(
                    testcase=testcase,
                    system=self.system_name,
                    symptom=Symptom.INCONSISTENCY,
                    stage=Stage.VERIFICATION,
                    log=traceback.format_exc(),
                )

        return None

    def bump_testcase(
        self, model: Model, input: Dict[str, np.ndarray] = None
    ) -> TestCase:
        try:  # compilation
            executable = self.mk_backend(model)
        except Exception:
            raise BugReport(
                TestCase(model=model, oracle=None),  # None means no oracle
                Symptom.EXCEPTION,
                Stage.COMPILATION,
                traceback.format_exc(),
            )

        if not input:
            # generate inputs automatically.
            input = self.make_random_input(model.input_like())

        try:  # execution
            output = executable(input)
        except Exception:
            return BugReport(
                TestCase(model, Oracle(input=input, output=None)),
                Symptom.EXCEPTION,
                Stage.EXECUTION,
                traceback.format_exc(),
            )

        return TestCase(model, Oracle(input=input, output=output))
