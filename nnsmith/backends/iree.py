import os
import tempfile
from typing import Dict

import iree.compiler.tf
import iree.runtime
import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import TFModel


class IREEFactory(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False, catch_process_crash=True):
        """
        Initialize the IREE backend factory.

        Parameters
        ----------
        target : str, optional
            The compilation target including "cpu" (same as "llvm-cpu"), "cuda", "vmvx", "vulkan-spirv", by default "cpu"
        optmax : bool, optional
            Release mode or not, by default False
        catch_process_crash : bool, optional
            Doing compilation without forking (may crash), by default True
        """
        if target == "cpu":
            target = "llvm-cpu"
        supported_backends = [
            "vmvx",
            "cuda",
            "llvm-cpu",
            "vulkan-spirv",
        ]
        assert (
            target in supported_backends
        ), f"Unsupported target {target}. Consider one of {supported_backends}"
        super().__init__(
            target=target, optmax=optmax, catch_process_crash=catch_process_crash
        )

    @property
    def system_name(self) -> str:
        return "iree"

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        # https://iree-python-api.readthedocs.io/en/latest/compiler/tools.html
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_model_path = os.path.join(tmpdirname, "saved_model")
            tf.saved_model.save(
                model.net,
                saved_model_path,
                signatures=model.net.call_by_dict.get_concrete_function(
                    model.input_specs
                ),
            )
            vm_flatbuffer = iree.compiler.tf.compile_saved_model(
                saved_model_path,
                target_backends=[self.target],
                exported_names=["call_by_dict"],
                optimize=self.optmax,
                extra_args=[
                    # "--iree-mhlo-demote-i64-to-i32=false",
                    # "--iree-flow-demote-i64-to-i32",
                ],
            )

        compiled_model = iree.runtime.load_vm_flatbuffer(
            vm_flatbuffer,
            backend=self.target,
        )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            outputs = compiled_model.call_by_dict(
                inputs
            )  # can directly accept np.ndarray
            outputs = {k: np.array(v) for k, v in outputs.items()}
            return outputs

        return closure
