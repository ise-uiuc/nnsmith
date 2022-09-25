from typing import Dict

import iree.compiler.tf
import iree.runtime
import numpy as np
import tensorflow as tf
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import TFModel


class IREEFactory(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False):
        """
        Initialize the IREE backend factory.

        Parameters
        ----------
        target : str, optional
            The compilation target including "cpu" (same as "llvm-cpu"), "vmvx", "vulkan-spirv", by default "cpu"
        optmax : bool, optional
            Release mode or not, by default False
        """
        if target == "cpu":
            target = "llvm-cpu"
        supported_backends = [
            "vmvx",
            "llvm-cpu",
            "vulkan-spirv",
        ]
        assert (
            target in supported_backends
        ), f"Unsupported target {target}. Consider one of {supported_backends}"
        super().__init__(target=target, optmax=optmax)

    @property
    def system_name(self) -> str:
        return "iree"

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        setattr(
            model.net,
            "iree_fn",
            tf.function(input_signature=[model.input_specs])(model.net.call_by_dict),
        )
        # https://iree-python-api.readthedocs.io/en/latest/compiler/tools.html
        vm_flatbuffer = iree.compiler.tf.compile_module(
            model.net,
            target_backends=[self.target],
            exported_names=["iree_fn"],
            optimize=self.optmax,
        )

        compiled_model = iree.runtime.load_vm_flatbuffer(
            vm_flatbuffer,
            backend=self.target,
        )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            outputs = compiled_model.iree_fn(inputs)  # can directly accept np.ndarray
            outputs = {k: np.array(v) for k, v in outputs.items()}
            return outputs

        return closure
