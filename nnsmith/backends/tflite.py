from os import PathLike
from typing import Dict

import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import TFModel, TFNetCallable


class TFLiteRunner:
    def __init__(self, tfnet_callable: TFNetCallable) -> None:
        self.tfnet_callable = tfnet_callable

    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.tfnet_callable(**input)
        # It can automatically convert input args to np.ndarray, and it outputs np.ndarray.


class TFLiteFactory(BackendFactory):
    """Factory to build TFLite backend.
    Convertion Graph:
        TFModel & one concrete function
                    v  TFLiteConverter with configs
        TFLite model content in bytes -- (f.write) --> model.tflite file
                    v  tf.lite.Interpreter  <-- (f.read) <---
        TFLite Python Callable
    """

    def __init__(self, target, optmax, **kwargs) -> None:
        # https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906
        # TFLite doesn't support NVIDIA GPU.
        assert target != "cuda"
        super().__init__(target, optmax, **kwargs)

    @property
    def system_name(self) -> str:
        return "tflite"

    @dispatch(TFModel)
    def make_backend(
        self,
        model: TFModel,
    ) -> BackendCallable:
        """Create TFLite callable from a concrete function.
        TFModel is required because functions have a weak reference to Variables, which are stored in `tf.Module`.
        Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_concrete_functions
        """
        return self.make_backend_from_content(self.make_content(model))

    def make_backend_from_path(
        self,
        path: PathLike,
    ) -> BackendCallable:
        """Create TFLite callable from path of a TF SavedModel.
        Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_saved_model
        """
        return self.make_backend_from_content(
            self.make_content(path),
        )

    @dispatch(bytes)
    def make_backend_from_content(self, content: bytes) -> BackendCallable:
        # Ref: https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
        interpreter = tf.lite.Interpreter(model_content=content)
        return TFLiteRunner(interpreter.get_signature_runner())

    @dispatch(TFModel)
    def make_content(
        self,
        model: TFModel,
    ) -> bytes:
        """Create TFLite content from a concrete function.
        TFModel is required because functions have a weak reference to Variables, which are stored in `tf.Module`.
        Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_concrete_functions
        """
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            funcs=[model.concrete_net()],
            trackable_obj=model.net,
        )
        return self._tflite_content_from_converter(converter)

    def make_content_with_func(
        self,
        model: TFModel,
        concrete_func: TFNetCallable,
    ) -> bytes:
        """Create TFLite content from a concrete function.
        TFModel is required because functions have a weak reference to Variables, which are stored in `tf.Module`.
        Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_concrete_functions
        """
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            funcs=[concrete_func],
            trackable_obj=model.net,
        )
        return self._tflite_content_from_converter(converter)

    def make_backend_with_func(
        self,
        model: TFModel,
        concrete_func: TFNetCallable,
    ) -> BackendCallable:
        """Create TFLite callable from a concrete function.
        TFModel is required because functions have a weak reference to Variables, which are stored in `tf.Module`.
        Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_concrete_functions
        """
        return self.make_backend_from_content(
            self.make_content_with_func(model, concrete_func)
        )

    @dispatch(str)
    def make_content(
        self,
        path: PathLike,
    ) -> bytes:
        """Create TFLite content from path of a saved model.
        Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_saved_model
        """
        converter = tf.lite.TFLiteConverter.from_saved_model(path)
        return self._tflite_content_from_converter(converter)

    def _tflite_content_from_converter(
        self,
        converter: tf.lite.TFLiteConverter,
    ) -> bytes:
        """Configure TFLite converter and create callable from it.
        converter configuarations: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#attributes_1
        """
        # Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TargetSpec
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        # Ref: https://www.tensorflow.org/api_docs/python/tf/lite/Optimize
        converter.optimizations = {
            tf.lite.Optimize.DEFAULT,
            tf.lite.Optimize.EXPERIMENTAL_SPARSITY,
        }
        # converter.allow_custom_ops = True
        tflite_bytes = converter.convert()
        return tflite_bytes

    def dump_backend(self, path: PathLike, content: bytes) -> None:
        with open(path, "wb") as f:
            f.write(content)

    def load_content(self, path: PathLike) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def load_backend(self, path: PathLike) -> BackendCallable:
        return self.make_backend_from_content(self.load_content(path))
