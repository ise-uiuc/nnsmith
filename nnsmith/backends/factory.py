import multiprocessing as mp
import os
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.extension import BACKEND_REQUIRES
from nnsmith.abstract.op import AbsOpBase
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.difftest import assert_allclose
from nnsmith.error import InternalError
from nnsmith.logging import CORE_LOG
from nnsmith.materialize import BugReport, Model, Oracle, Stage, Symptom, TestCase

BackendCallable = Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]


class BackendFactory(ABC):
    def __init__(self, target="cpu", optmax: bool = False):
        super().__init__()
        self.target = target
        self.optmax = optmax

    @property
    @abstractmethod
    def system_name(self) -> str:
        pass

    def __str__(self) -> str:
        return f"{self.system_name} ({self.target} opt: {self.optmax})"

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

    def critical_assert_dispatchable(self, model: Model):
        if not self.make_backend.dispatch(type(model)):
            CORE_LOG.critical(
                f"[Not implemented] {type(self).__name__} for {type(model).__name__}!\n"
                "Check https://github.com/ise-uiuc/nnsmith#backend-model-support for compatile `model.type` and `backend.type`."
            )
            sys.exit(1)

    def checked_make_backend(self, model: Model) -> BackendCallable:
        self.critical_assert_dispatchable(model)
        return self.make_backend(model)

    def checked_compile(self, testcase: TestCase) -> Union[BackendCallable, BugReport]:
        try:  # compilation
            return self.checked_make_backend(testcase.model)
        except InternalError as e:
            raise e
        except Exception:
            return BugReport(
                testcase=testcase,
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.COMPILATION,
                log=traceback.format_exc(),
            )

    def checked_exec(
        self, executable: BackendCallable, testcase: TestCase
    ) -> Union[Dict[str, np.ndarray], BugReport]:
        input = testcase.oracle.input
        if input is None:
            input = self.make_random_input(testcase.model.input_like)
            testcase = TestCase(
                model=testcase.model, oracle=Oracle(input=input, output=None)
            )

        try:  # execution
            return executable(input)
        except InternalError as e:
            raise e
        except Exception:
            return BugReport(
                testcase=testcase,
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.EXECUTION,
                log=traceback.format_exc(),
            )

    def checked_compile_and_exec(
        self, testcase: TestCase, crash_safe=False, timeout=None
    ) -> Union[Dict[str, np.ndarray], BugReport]:
        # pre-check model dispatchability
        self.critical_assert_dispatchable(testcase.model)
        if (
            not crash_safe and timeout is None
        ):  # not crash safe, compile & exec natively in current process.
            bug_or_exec = self.checked_compile(testcase)
            if isinstance(bug_or_exec, BugReport):
                return bug_or_exec
            return self.checked_exec(bug_or_exec, testcase)
        else:  # crash safe, compile & exec in a separate process.
            if timeout is not None:
                assert isinstance(
                    timeout, int
                ), "timeout are `seconds` => must be an integer."

        # TODO: optimize to shared memory in the future (Python 3.8+)
        # https://docs.python.org/3/library/multiprocessing.shared_memory.html
        # NOTE: Similar implementation as Tzer.
        with mp.Manager() as manager:
            shared_dict = manager.dict(
                {
                    "symptom": None,
                    "stage": Stage.COMPILATION,
                    "log": None,
                    "output": None,
                    "uncaught_exception": None,
                }
            )

            def crash_safe_compile_exec(sdict):
                try:
                    bug_or_exec = self.checked_compile(testcase)
                    if isinstance(bug_or_exec, BugReport):
                        sdict["symptom"] = bug_or_exec.symptom
                        sdict["log"] = bug_or_exec.log
                        return

                    sdict["stage"] = Stage.EXECUTION
                    bug_or_result = self.checked_exec(bug_or_exec, testcase)
                    if isinstance(bug_or_result, BugReport):
                        sdict["symptom"] = bug_or_result.symptom
                        sdict["log"] = bug_or_result.log
                        return

                    sdict["output"] = bug_or_result
                except Exception as e:
                    sdict["uncaught_exception"] = e

            p = mp.Process(target=crash_safe_compile_exec, args=(shared_dict,))

            p.start()
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                assert not p.is_alive()
                return BugReport(
                    testcase=testcase,
                    system=self.system_name,
                    symptom=Symptom.TIMEOUT,
                    stage=shared_dict["stage"],
                    log=f"Timeout after {timeout} seconds.",
                )

            if shared_dict["output"] is not None:
                return shared_dict["output"]

            if shared_dict["uncaught_exception"] is not None:
                CORE_LOG.critical(
                    f"Found uncaught {type(shared_dict['uncaught_exception'])} in crash safe mode."
                )
                raise shared_dict["uncaught_exception"]

            if p.exitcode != 0:
                return BugReport(
                    testcase=testcase,
                    system=self.system_name,
                    symptom=Symptom.SEGFAULT,
                    stage=shared_dict["stage"],
                    log=f"Process crashed with exit code: {p.exitcode}",
                )
            else:
                return BugReport(
                    testcase=testcase,
                    system=self.system_name,
                    symptom=shared_dict["symptom"],
                    stage=shared_dict["stage"],
                    log=shared_dict["log"],
                )

    def verify_results(
        self, output: Dict[str, np.ndarray], testcase: TestCase, equal_nan=True
    ) -> Optional[BugReport]:
        try:  # verification
            assert_allclose(
                output,
                testcase.oracle.output,
                self.__str__(),
                testcase.oracle.provider,
                equal_nan=equal_nan,
            )
        except AssertionError:
            return BugReport(
                testcase=testcase,
                system=self.system_name,
                symptom=Symptom.INCONSISTENCY,
                stage=Stage.VERIFICATION,
                log=traceback.format_exc(),
            )

    def verify_testcase(
        self, testcase: TestCase, equal_nan=True
    ) -> Optional[BugReport]:
        executable = self.checked_compile(testcase)
        if isinstance(executable, BugReport):
            return executable

        output = self.checked_exec(executable, testcase)
        if isinstance(output, BugReport):
            return output

        if testcase.oracle.output is not None:
            return self.verify_results(output, testcase, equal_nan=equal_nan)

        return None

    def make_testcase(
        self,
        model: Model,
        input: Dict[str, np.ndarray] = None,
        crash_safe=False,
        timeout=None,
    ) -> Union[BugReport, TestCase]:
        if input is None:
            input = self.make_random_input(model.input_like)

        partial_testcase = TestCase(
            model=model, oracle=Oracle(input=input, output=None)
        )
        bug_or_res = self.checked_compile_and_exec(
            partial_testcase, crash_safe=crash_safe, timeout=timeout
        )
        if isinstance(bug_or_res, BugReport):
            return bug_or_res
        else:
            partial_testcase.oracle.output = bug_or_res

        try:  # compilation
            executable = self.checked_make_backend(model)
        except InternalError as e:
            raise e
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
        except InternalError as e:
            raise e
        except Exception:
            return BugReport(
                TestCase(model, Oracle(input=input, output=None)),
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.EXECUTION,
                log=traceback.format_exc(),
            )

        return TestCase(
            model, Oracle(input=input, output=output, provider=self.system_name)
        )

    @staticmethod
    def init(name, target="cpu", optmax=True, **kwargs):
        if name is None:
            raise ValueError(
                "Backend type cannot be None. Specify via `backend.type=[onnxruntime | tvm | tensorrt | tflite | xla | iree]`"
            )

        if target == "gpu":
            target = "cuda"  # `gpu` means `cuda` by default.

        if name == "onnxruntime":
            from nnsmith.backends.onnxruntime import ORTFactory

            return ORTFactory(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "tvm":
            from nnsmith.backends.tvm import TVMFactory

            # default executor is graph
            kwargs["executor"] = kwargs.get("executor", "graph")
            return TVMFactory(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "tensorrt":
            from nnsmith.backends.tensorrt import TRTFactory

            return TRTFactory(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "tflite":
            from nnsmith.backends.tflite import TFLiteFactory

            return TFLiteFactory(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "xla":
            from nnsmith.backends.xla import XLAFactory

            return XLAFactory(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "iree":
            from nnsmith.backends.iree import IREEFactory

            return IREEFactory(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown backend: {name}")

    def add_constraints(self, op_types: List[Type[AbsOpBase]]) -> List[Type[AbsOpBase]]:

        for optype in op_types:
            if optype.name() in BACKEND_REQUIRES[self.system_name]:
                optype.requires = BACKEND_REQUIRES[self.system_name][optype.name()]

        return op_types
