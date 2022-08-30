from typing import Callable, Dict, Optional, Any
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union, List
import traceback

import numpy as np
from nnsmith.abstract.dtype import DType

from nnsmith.abstract.tensor import AbsTensor
from nnsmith.materialize import BugReport, Oracle, Stage, Symptom, TestCase, Model
from nnsmith.difftest import assert_allclose

BackendCallable = Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]


class BackendFactory(ABC):
    def __init__(self, device="cpu", optmax: bool = False, catch_process_crash=True):
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
        return f"{self.system_name} ({self.device}  opt: {self.optmax})"

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

    @classmethod
    def skip_dtypes(cls) -> List[DType]:
        return []

    @abstractmethod
    def make_backend(self, model: Model) -> BackendCallable:
        raise NotImplementedError

    def verify_testcase(self, testcase: TestCase) -> Optional[BugReport]:
        # TODO(@ganler): impl fault catching in subprocess
        assert not self.catch_process_crash, "not implemented"

        try:  # compilation
            executable = self.make_backend(testcase.model)
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
            input = self.make_random_input(testcase.model.input_like)
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

    def make_testcase(
        self, model: Model, input: Dict[str, np.ndarray] = None
    ) -> Union[BugReport, TestCase]:
        try:  # compilation
            executable = self.make_backend(model)
        except Exception:
            return BugReport(
                TestCase(model=model, oracle=None),  # None means no oracle
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.COMPILATION,
                log=traceback.format_exc(),
            )

        if not input:
            # generate inputs automatically.
            input = self.make_random_input(model.input_like)

        try:  # execution
            output = executable(input)
        except Exception:
            return BugReport(
                TestCase(model, Oracle(input=input, output=None)),
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.EXECUTION,
                log=traceback.format_exc(),
            )

        return TestCase(model, Oracle(input=input, output=output))

    @staticmethod
    def init(name, device="cpu", optmax=True, catch_process_crash=False, **kwargs):
        if name == "onnxruntime":
            from nnsmith.backends.onnxruntime import ORTFactory

            return ORTFactory(
                device=device,
                optmax=optmax,
                catch_process_crash=catch_process_crash,
                **kwargs,
            )
        elif name == "tvm":
            from nnsmith.backends.tvm import TVMFactory

            # default executor is graph
            kwargs["executor"] = kwargs.get("executor", "graph")
            return TVMFactory(
                device=device,
                optmax=optmax,
                catch_process_crash=catch_process_crash,
                **kwargs,
            )
        elif name == "tensorrt":
            from nnsmith.backends.tensorrt import TRTFactory

            return TRTFactory(
                device=device,
                optmax=optmax,
                catch_process_crash=catch_process_crash,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown backend: {name}")
