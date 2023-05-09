import multiprocessing as mp
import re
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.difftest import assert_allclose
from nnsmith.error import InternalError
from nnsmith.logging import CORE_LOG
from nnsmith.materialize import BugReport, Model, Oracle, Stage, Symptom, TestCase

BackendCallable = Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]


def parse_name_kwargs(text):
    # with parse_name, we will extract kwargs from parts that guarded by "[" and "]" in `name`.
    # the grammar is:
    fmt = r"<NAME> <key1>@<value1> <key2>@<value2> ..."
    # where:
    #    <NAME> is the name of the backend.
    #    <key?> is the key of the kwargs.
    #    <value?> is the value corresponding to <key?> in the kwargs.
    # Those must comply with the regex pattern of r"[a-zA-Z0-9_]+".
    tokens = text.strip().split()
    if len(tokens) == 0:
        raise ValueError(f"Invalid backend: {text}. Expected format: {fmt}")

    pattern = re.compile(r"^[a-zA-Z0-9_]+$")

    name = tokens[0]
    if not pattern.match(name):
        raise ValueError(f"Invalid backend: {text}. Expected format: {fmt}")

    kvs = {}
    for token in tokens[1:]:
        kv = token.split("@")
        if len(kv) == 2:
            k, v = kv
            if pattern.match(k) and pattern.match(v):
                kvs[k] = v
                continue
        raise ValueError(f"Invalid backend: {text}. Expected format: {fmt}")

    return name, kvs


class BackendFactory(ABC):
    def __init__(self, target="cpu", optmax: bool = True):
        super().__init__()
        self.target = target
        self.optmax = optmax

    @property
    @abstractmethod
    def system_name(self) -> str:
        pass

    @property
    def version(self) -> str:
        return "unknown"

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
                version=self.version,
            )

    def checked_exec(
        self, executable: BackendCallable, testcase: TestCase
    ) -> Union[Dict[str, np.ndarray], BugReport]:
        input = None if testcase.oracle is None else testcase.oracle.input
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
                version=self.version,
            )

    def checked_compile_and_exec(
        self, testcase: TestCase, crash_safe=False, timeout=None
    ) -> Union[Dict[str, np.ndarray], BugReport]:
        # pre-check if model is dispatchable
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
                    CORE_LOG.debug(f"[FORK] Compiling.")
                    bug_or_exec = self.checked_compile(testcase)
                    if isinstance(bug_or_exec, BugReport):
                        sdict["symptom"] = bug_or_exec.symptom
                        sdict["log"] = bug_or_exec.log
                        return

                    CORE_LOG.debug(f"[FORK] Executing.")
                    sdict["stage"] = Stage.EXECUTION
                    bug_or_result = self.checked_exec(bug_or_exec, testcase)
                    CORE_LOG.debug(f"[FORK] Done.")
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
                    version=self.version,
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
                    version=self.version,
                )
            else:
                return BugReport(
                    testcase=testcase,
                    system=self.system_name,
                    symptom=shared_dict["symptom"],
                    stage=shared_dict["stage"],
                    log=shared_dict["log"],
                    version=self.version,
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
                version=self.version,
            )
        except Exception:
            return BugReport(
                testcase=testcase,
                system=self.system_name,
                symptom=Symptom.EXCEPTION,
                stage=Stage.VERIFICATION,
                log=traceback.format_exc(),
                version=self.version,
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

        if testcase.oracle is not None and testcase.oracle.output is not None:
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
                version=self.version,
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
                version=self.version,
            )

        return TestCase(
            model, Oracle(input=input, output=output, provider=self.system_name)
        )

    @property
    @abstractmethod
    def import_libs(self) -> List[str]:
        pass

    def emit_compile(
        self, opt_name: str, mod_name: str, inp_name: Optional[str] = None
    ) -> str:
        raise NotImplementedError

    def emit_run(self, out_name: str, opt_name: str, inp_name: str) -> str:
        raise NotImplementedError

    @staticmethod
    def init(
        name: str,
        target: str = "cpu",
        ad: str = None,
        optmax: bool = True,
        parse_name=False,
        **kwargs,
    ):
        if name is None:
            raise ValueError(
                "Backend type cannot be None. Use `backend.type=[onnxruntime|tvm|tensorrt|xla|tflite|pt2|torchjit]`"
            )

        if target == "gpu":
            target = "cuda"  # `gpu` means `cuda` by default.

        if parse_name:
            name, kw_dict = parse_name_kwargs(name)
            kwargs.update(kw_dict)

        if name == "onnxruntime":
            from nnsmith.backends.onnxruntime import ORT

            return ORT(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "tvm":
            from nnsmith.backends.tvm import TVM

            # default executor is graph
            kwargs["executor"] = kwargs.get("executor", "graph")
            return TVM(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "tensorrt":
            from nnsmith.backends.tensorrt import TRT

            return TRT(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "tflite":
            from nnsmith.backends.tflite import TFLite

            return TFLite(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "xla":
            from nnsmith.backends.xla import XLA

            return XLA(
                target=target,
                optmax=optmax,
                **kwargs,
            )
        elif name == "torchjit":
            from nnsmith.backends.torchjit import TorchJIT

            return TorchJIT(target=target, optmax=optmax, **kwargs)
        elif name == "torchjitAD":
            from nnsmith.backends.torchjitAD import TorchJITAD

            return TorchJITAD(target=target, optmax=optmax, ad=ad, **kwargs)
        elif name == "pt2":
            from nnsmith.backends.pt2 import PT2

            return PT2(target=target, optmax=optmax, ad=ad, **kwargs)
        else:
            raise ValueError(f"unknown backend: {name}")
