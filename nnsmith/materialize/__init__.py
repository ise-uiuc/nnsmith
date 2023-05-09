from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# Enables type checking while avoiding circular imports.
if TYPE_CHECKING:
    from nnsmith.backends import BackendFactory

import numpy as np
from multipledispatch import dispatch

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsOpBase, Constant
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import SanityCheck
from nnsmith.gir import GraphIR
from nnsmith.util import HAS_PYGRAPHVIZ, viz_dot


def framework_operator_impl(
    framework_realizable_ops: List[Type[AbsOpBase]],
    all_framework_ops: List[Type[AbsOpBase]],
    op_type: AbsOpBase,
    *args,
    **kwargs,
):
    """When implementing `forward_fn` for an operator class, add this operator into all_framework_ops list.

    Usage:
        In `forward.py`, define `operator_impl = partial(framework_operator_impl, FW_REALIZABLE_OPS, ALL_FM_OPS)`.
        Then add `@operator_impl(OpClass)` when implementing `forward_fn` for `OpClass`.

    Args:
        framework_realizable_ops (List[Type[AbsOpBase]]): all realizable ops in the framework. Usually it can be obtained by FULL_OPERATOR_SETS["core"].union(FULL_OPERATOR_SETS["framework_name"])
        all_framework_ops (List[Type[AbsOpBase]]): list of operator classes that are implemented `forward_fn` in the framework.
        op_type (AbsOpBase): operator class
    """
    SanityCheck.true(
        issubclass(op_type, AbsOpBase),
        f"Decorator operator_impl takes AbsOpBase subclass, but got {op_type}",
    )
    if op_type is not Constant:  # Constant comes from placeholder.
        dispatchables = [
            rtype for rtype in framework_realizable_ops if issubclass(rtype, op_type)
        ]
        for rtype in dispatchables:
            all_framework_ops.append(rtype)

        SanityCheck.true(
            len(dispatchables) != 0,
            f"Decorator operator_impl only take types decorated by `mark_realize`, but got {op_type}",
        )
    return dispatch(op_type, *args, **kwargs)


class Oracle:
    def __init__(
        self,
        input: Dict[str, np.ndarray],
        output: Dict[str, np.ndarray],
        provider: str = "unknown",
    ) -> None:
        self.input = input
        self.output = output
        self._provider = provider

    def __repr__(self) -> str:
        return f"input={self.input}, output={self.output}"

    @staticmethod
    def name() -> str:
        return "oracle.pkl"

    @property
    def provider(self) -> str:
        return self._provider

    def dump(self, path: PathLike) -> None:
        with open(path, "wb") as f:
            to_dump = {
                "input": self.input,
                "output": self.output,
                "provider": self.provider,
            }
            pickle.dump(to_dump, f)

    @staticmethod
    def load(path: PathLike) -> "Oracle":
        with open(path, "rb") as f:
            to_load = pickle.load(f)
            return Oracle(to_load["input"], to_load["output"], to_load["provider"])


MT = TypeVar("MT", bound="Model")


class Model(ABC):
    def __init__(self):
        self.dotstring: Optional[str] = None
        self._grad_check: bool = False

    def needs_grad_check(self) -> bool:
        return self._grad_check

    def set_grad_check(self, grad: bool) -> None:
        self._grad_check = grad

    @property
    @abstractmethod
    def input_like(self) -> Dict[str, AbsTensor]:
        pass

    @property
    @abstractmethod
    def output_like(self) -> Dict[str, AbsTensor]:
        pass

    @classmethod
    @abstractmethod
    def from_gir(cls: Type[MT], ir: GraphIR, **kwargs) -> MT:
        pass

    @classmethod
    @abstractmethod
    def load(cls: Type[MT], path: PathLike) -> MT:
        pass

    @abstractmethod
    def dump(self, path: PathLike) -> None:
        pass

    @property
    @abstractmethod
    def native_model(self) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def name_suffix() -> str:
        """Suffix of the model file.

        Returns:
            str: Model suffix such as ".onnx".

        Note:
            Model as a folder can be "".
        """
        pass

    @abstractmethod
    def refine_weights(self) -> None:
        pass

    @abstractmethod
    def make_oracle(self) -> Oracle:
        pass

    @staticmethod
    @abstractmethod
    def operators() -> List[Type[AbsOpBase]]:
        pass

    @property
    def version(self) -> str:
        return "unknown"

    @staticmethod
    def name_prefix() -> str:
        return "model"

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        pass

    @staticmethod
    def add_seed_setter() -> None:
        pass

    def attach_viz(self, ir: GraphIR) -> None:
        if HAS_PYGRAPHVIZ:
            self.dotstring = ir.to_dot()

    def dump_viz(self, path: PathLike) -> None:
        viz_dot(self.dotstring, path)

    @staticmethod
    def skip_dtypes() -> List[DType]:
        return []

    @property
    @abstractmethod
    def import_libs(self) -> List[str]:
        pass

    def emit_def(self, mod_name: str, mod_cls: str) -> str:
        raise NotImplementedError

    def emit_run(self, out_name: str, inp_name: str, mod_name: str) -> str:
        raise NotImplementedError

    def emit_weight(self, mod_name: str, path: Optional[PathLike] = None):
        raise NotImplementedError

    def emit_input(self, inp_name: str, path: Optional[PathLike] = None):
        raise NotImplementedError

    @staticmethod
    def init(name, backend_target=None) -> Type["Model"]:
        if name is None:
            raise ValueError(
                "Model type cannot be None. Use `model.type=[torch|onnx|tensorflow]`."
            )

        if name == "torch":
            from nnsmith.materialize.torch import TorchModelCPU, TorchModelCUDA

            if backend_target == "gpu" or backend_target == "cuda":
                return TorchModelCUDA
            return TorchModelCPU
        elif name == "onnx":
            # device agnoistic
            from nnsmith.materialize.onnx import ONNXModelCPU, ONNXModelCUDA

            if backend_target == "gpu" or backend_target == "cuda":
                return ONNXModelCUDA
            return ONNXModelCPU
        elif name == "tensorflow":
            from nnsmith.materialize.tensorflow import TFModelCPU, TFModelCUDA

            if backend_target == "gpu" or backend_target == "cuda":
                # XLA must align device location of eager mode execution.
                return TFModelCUDA
            return TFModelCPU

        raise ValueError(
            f"Unsupported: ModelType={name} for BackendTarget={backend_target}"
        )


class TestCase:
    def __init__(self, model: Model, oracle: Oracle):
        self.oracle = oracle
        self.model = model

    @staticmethod
    def load(
        model_type: Type[Model], root_folder: str, allow_no_oracle=False
    ) -> "TestCase":
        model_path = os.path.join(
            root_folder, model_type.name_prefix() + model_type.name_suffix()
        )
        model = model_type.load(model_path)

        assert allow_no_oracle or os.path.exists(
            os.path.join(root_folder, Oracle.name())
        ), "Oracle is not found or auto-generated when allow_no_oracle is True."

        oracle = None
        if os.path.exists(os.path.join(root_folder, Oracle.name())):
            oracle_path = os.path.join(root_folder, Oracle.name())
            oracle = Oracle.load(oracle_path)

        return TestCase(model, oracle)

    def dump(self, root_folder: str):
        if self.model and self.model.dotstring:
            self.model.dump_viz(os.path.join(root_folder, "graph.png"))

        self.model.dump(
            os.path.join(
                root_folder, self.model.name_prefix() + self.model.name_suffix()
            )
        )
        self.oracle.dump(os.path.join(root_folder, Oracle.name()))


class Symptom(Enum):
    EXCEPTION = "exception"
    SEGFAULT = "segfault"
    INCONSISTENCY = "inconsistency"
    TIMEOUT = "timeout"
    WORSE_PERF = "worse_perf"


class Stage(Enum):
    COMPILATION = "compilation"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class BugReport(ABC):
    def __init__(
        self,
        testcase: TestCase,
        symptom: Symptom,
        stage: Stage,
        system: str,
        version: str = None,
        version_id: str = None,
        log: str = None,
    ):
        self.testcase = testcase
        self.symptom = symptom
        self.system = system
        self.stage = stage
        self.version = version
        self.version_id = version_id
        self.log = log

    @staticmethod
    def error_msg_name():
        return "err.log"

    def __repr__(self) -> str:
        return f"{self.system} {self.symptom.value} in {self.stage.value}\n{self.log}"

    def dump(self, root_folder: PathLike):
        # create folder if not exists
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        # model*
        # oracle.pkl
        self.testcase.dump(root_folder)
        # err.log
        with open(os.path.join(root_folder, self.error_msg_name()), "w") as f:
            f.write(self.log)
        # meta.json
        with open(os.path.join(root_folder, "meta.json"), "w") as f:
            json.dump(
                {
                    "system": self.system,
                    "symptom": self.symptom.value,
                    "stage": self.stage.value,
                    "version": self.version,
                    "version_id": self.version_id,
                },
                f,
            )

    @staticmethod
    def load(model_type, root_folder: str, allow_partial=False) -> "BugReport":
        symptom = None
        stage = None
        version = None
        version_id = None
        system = None

        assert allow_partial or os.path.exists(
            os.path.join(root_folder, "meta.json")
        ), "meta.json must exist or allow_partial is True where the oracle will be automatically generated"

        if os.path.exists(os.path.join(root_folder, "meta.json")):
            with open(os.path.join(root_folder, "meta.json"), "r") as f:
                meta = json.load(f)

            system = meta["system"]
            symptom = Symptom(meta["symptom"])
            stage = Stage(meta["stage"])
            version = meta["version"]
            version_id = meta["version_id"]

        testcase = TestCase.load(model_type, root_folder, allow_partial)

        log = None
        if os.path.exists(os.path.join(root_folder, BugReport.error_msg_name())):
            with open(os.path.join(root_folder, BugReport.error_msg_name()), "r") as f:
                log = f.read()

        return BugReport(testcase, symptom, stage, system, version, version_id, log)


class Render:
    _IMPORTS = r"${{IMPORTS}}$"
    _DEF = r"${{DEF}}$"
    _MAKE_WEIGHT = r"${{MAKE_WEIGHT}}$"
    _EAGER_RUN = r"${{EAGER_RUN}}$"
    _COMPILE = r"${{COMPILE}}$"
    _MAKE_INPUT = r"${{MAKE_INPUT}}$"
    _COMPILE_RUN = r"${{COMPILE_RUN}}$"
    _CHECK = r"${{CHECK}}$"

    def __init__(
        self,
        template: Optional[str] = None,
        mod_name="m",
        mod_cls="M",
        opt_name="opt",
    ) -> None:
        self.template = (
            f"""
{Render._IMPORTS}

# Model definition
{Render._DEF}

# Initialize weight
{Render._MAKE_WEIGHT}

# Initialize input
{Render._MAKE_INPUT}

# Compile the model
{Render._COMPILE}

# Eager run
{Render._EAGER_RUN}

# Compiled run
{Render._COMPILE_RUN}

# Differential testing
{Render._CHECK}
"""
            if template is None
            else template
        )
        self.mod_name = mod_name
        self.mod_cls = mod_cls
        self.opt_name = opt_name

        self.inp_name = "inp"
        self.eager_result_name = f"{self.mod_name}_out"
        self.compile_result_name = f"{self.opt_name}_out"

        self.imports = ["import numpy as np"]
        self.def_code: Optional[str] = None
        self.weight_code: Optional[str] = None
        self.compile_code: Optional[str] = None
        self.input_code: Optional[str] = None
        self.eager_run_code: Optional[str] = None
        self.compile_run_code: Optional[str] = None
        self.check_code: Optional[str] = None

    def emit_model(self, model: Model):
        for imp in model.import_libs:
            if imp not in self.imports:
                self.imports.append(imp)

        # Define the model like:
        #    class M(nn.Module): ...
        # Initialize an instance:
        #    m = M()
        self.def_code = model.emit_def(mod_name=self.mod_name, mod_cls=self.mod_cls)

        # Compute ${self.mod_name} eagerly over ${self.inp_name}
        # and store the result in ${self.eager_result_name}
        self.eager_run_code = model.emit_run(
            out_name=self.eager_result_name,
            mod_name=self.mod_name,
            inp_name=self.inp_name,
        )

    def emit_weight(self, model: Model, path: Optional[PathLike] = None):
        # Load the model weights from ${path}
        self.weight_code = model.emit_weight(mod_name=self.mod_name, path=path)

    def emit_input(self, model: Model, path: Optional[PathLike] = None):
        # Load the model weights from ${path}
        self.input_code = model.emit_input(inp_name=self.inp_name, path=path)

    def emit_backend(self, backend: BackendFactory):
        for imp in backend.import_libs:
            if imp not in self.imports:
                self.imports.append(imp)

        # Compile the ${self.mod_name} to ${self.opt_name}
        self.compile_code = backend.emit_compile(
            opt_name=self.opt_name, mod_name=self.mod_name, inp_name=self.inp_name
        )
        # Run the compiled ${self.opt_name} over ${self.inp_name}
        # and store the result in ${self.compile_result_name}
        self.compile_run_code = backend.emit_run(
            out_name=self.compile_result_name,
            opt_name=self.opt_name,
            inp_name=self.inp_name,
        )

    def render(self) -> str:
        text = self.template

        def wrap(text, dependencies=None):
            if text is None:
                return "# None"
            if dependencies is not None and not all(dependencies):
                raise ValueError("Render failure: some dependencies are missing")
            return text

        text = text.replace(self._IMPORTS, "\n".join(self.imports))
        text = text.replace(self._DEF, self.def_code)  # Mandatory
        text = text.replace(self._MAKE_WEIGHT, wrap(self.weight_code))
        text = text.replace(self._MAKE_INPUT, wrap(self.input_code))
        # TODO(@ganler): compile optionally depends "make_input" (e.g., torchjit requires trace input)
        text = text.replace(self._COMPILE, wrap(self.compile_code))
        # To run a model eagerly we need the input data (`input_code`) to be available.
        text = text.replace(
            self._EAGER_RUN, wrap(self.eager_run_code, [self.input_code])
        )
        # To run a compiled model we need the model compiled (`compile_code`) and
        # the input data (`input_code`) to be available.
        text = text.replace(
            self._COMPILE_RUN,
            wrap(self.compile_run_code, [self.input_code, self.compile_code]),
        )

        check_text = (
            f"""for i, (l, r) in enumerate(zip({self.eager_result_name}, {self.compile_result_name})):
    np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {{i}}")"""
            if self.eager_run_code and self.compile_run_code
            else None
        )

        text = text.replace(self._CHECK, wrap(check_text))

        return text
