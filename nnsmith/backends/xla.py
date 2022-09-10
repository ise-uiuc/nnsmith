from os import PathLike
from typing import Callable, Dict

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


class XLAFactory(BackendFactory):
    def __init__(self, device="cpu", optmax: bool = False, catch_process_crash=True):
        super().__init__(device, optmax, catch_process_crash)

    @property
    def system_name(self) -> str:
        return "xla"

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        concrete_net: TFNetCallable = model.concrete_net()
        device: tf.device

        if self.device == "cpu":
            device = tf.device(tf.config.list_logical_devices("CPU")[0].name)
        elif self.device == "cuda":
            device = tf.device(tf.config.list_logical_devices("GPU")[0].name)
        else:
            raise ValueError(f"Unknown device: {self.device}")

        @tf.function(jit_compile=True)
        def compiled_net(**inputs) -> Dict[str, tf.Tensor]:
            return concrete_net(**inputs)

        def closure(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            tf.config.run_functions_eagerly(False)
            with device:
                return np_dict_from_tf(compiled_net(**tf_dict_from_np(inputs)))

        return closure
