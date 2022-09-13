import os
import tempfile
from typing import Callable, Dict

import iree.compiler.tf
import iree.runtime
import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import (
    TFModel,
    TFNetCallable,
    np_dict_from_tf,
    tf_dict_from_np,
)


class IREEFactory(BackendFactory):
    def __init__(
        self, device="llvm-cpu", optmax: bool = False, catch_process_crash=True
    ):
        """
        Args:
            device (str, optional):
                'vmvx': CPU,
                'llvm-cpu': CPU,
                'vulkan-spirv': GPU/SwiftShader (requires additional drivers)
                Defaults to 'llvm-cpu'.
            optmax (bool, optional): Whether to apply some default high level optimizations. Defaults to False.
        """
        if device == "cpu":
            device = "llvm-cpu"
        assert device in [
            "vmvx",
            "llvm-cpu",
            "vulkan-spirv",
        ], f"Unsupported device {device}"
        super().__init__(device, optmax, catch_process_crash)

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
                target_backends=[self.device],
                exported_names=["call_by_dict"],
                optimize=self.optmax,
                extra_args=[
                    "--iree-mhlo-demote-i64-to-i32=false",
                    "--iree-flow-demote-i64-to-i32",
                ],
            )

        compiled_model = iree.runtime.load_vm_flatbuffer(
            vm_flatbuffer,
            backend=self.device,
        )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            outputs = compiled_model.call_by_dict(
                inputs
            )  # can directly accept np.ndarray
            outputs = {k: np.array(v) for k, v in outputs.items()}
            return outputs

        return closure
