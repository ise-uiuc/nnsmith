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
        "xla"

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        concrete_net: TFNetCallable = model.concrete_net()

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            device: tf.device
            # TODO
            if self.device == "cpu":
                device = tf.device("/cpu:0")
            elif self.device == "gpu":
                device = tf.config.experimental.list_physical_devices("GPU")[0]
            tf.config.run_functions_eagerly(False)
            with device:
                return np_dict_from_tf(concrete_net(**tf_dict_from_np(inputs)))

        return closure
