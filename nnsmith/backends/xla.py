from typing import Dict, List

import tensorflow as tf  # type: ignore
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import (
    EagerModeCtx,
    TFModel,
    np_dict_from_tf,
    tf_dict_from_np,
)


class XLA(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = True):
        super().__init__(target, optmax)

        if self.target == "cpu":
            self.device = tf.device(tf.config.list_logical_devices("CPU")[0].name)
        elif self.target == "cuda":
            gpus = tf.config.list_logical_devices("GPU")
            if len(gpus) == 0:
                raise ValueError("No GPU found")
            self.device = tf.device(gpus[0].name)
        else:
            raise ValueError(
                f"Unknown device: {self.target}. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "xla"

    @property
    def version(self) -> str:
        return tf.__version__

    @property
    def import_libs(self) -> List[str]:
        return ["import tensorflow as tf"]

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        with self.device, EagerModeCtx(False):
            compiled = tf.function(jit_compile=True)(model.concrete_net())

        def closure(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            with self.device, EagerModeCtx(False):
                result = np_dict_from_tf(compiled(**tf_dict_from_np(inputs)))
            return result

        return closure
