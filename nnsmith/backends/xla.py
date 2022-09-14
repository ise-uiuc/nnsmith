from typing import Dict

import tensorflow as tf  # type: ignore
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import (
    RunTFEagerly,
    TFModel,
    TFNetCallable,
    np_dict_from_tf,
    tf_dict_from_np,
)


class XLAFactory(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False, catch_process_crash=True):
        super().__init__(target, optmax, catch_process_crash)

        if self.target == "cpu":
            self.device = tf.device(tf.config.list_logical_devices("CPU")[0].name)
        elif self.target == "cuda":
            self.device = tf.device(tf.config.list_logical_devices("GPU")[0].name)
        else:
            raise ValueError(
                f"Unknown device: {self.target}. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "xla"

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        concrete_net: TFNetCallable = model.concrete_net()

        @tf.function(jit_compile=True)
        def compiled_net(**inputs) -> Dict[str, tf.Tensor]:
            return concrete_net(**inputs)

        def closure(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            with RunTFEagerly(False), self.device:
                result = np_dict_from_tf(compiled_net(**tf_dict_from_np(inputs)))
            return result

        return closure
