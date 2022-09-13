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
    def __init__(self, device="cpu", optmax: bool = False, catch_process_crash=True):
        super().__init__(device, optmax, catch_process_crash)

    @property
    def system_name(self) -> str:
        return "iree"

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        backend_choice = "llvm-cpu"  # 'vmvx'

        tmp_path = os.path.join(tempfile.mkdtemp(), "saved_model")
        tf.saved_model.save(
            model.net,
            tmp_path,
            signatures=model.net.call_by_dict.get_concrete_function(model.input_specs),
        )

        vm_flatbuffer = iree.compiler.tf.compile_saved_model(
            tmp_path,
            target_backends=[backend_choice],
            exported_names=["call_by_dict"],
            extra_args=[
                "--iree-mhlo-demote-i64-to-i32=false",
                "--iree-flow-demote-i64-to-i32",
            ],
        )

        compiled_model = iree.runtime.load_vm_flatbuffer(
            vm_flatbuffer,
            backend=backend_choice,
        )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            outputs = compiled_model.call_by_dict(
                inputs
            )  # can directly accept np.ndarray
            outputs = {k: np.array(v) for k, v in outputs.items()}
            return outputs

        return closure
