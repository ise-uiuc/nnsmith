from typing import Any, Callable, Dict, List, Tuple
from multipledispatch import dispatch  # type: ignore
import os
import dill as pickle

import tensorflow as tf  # type: ignore

from nnsmith.graph_gen import Schedule
from nnsmith.materialize.tensorflow.tfnet import TFNet


@dispatch(list)
def randn_from_specs(specs: List[tf.TensorSpec]) -> List[tf.Tensor]:
    return [tf.random.normal(shape=spec.shape, dtype=spec.dtype) for spec in specs]


@dispatch(dict)
def randn_from_specs(specs: Dict[str, tf.TensorSpec]) -> Dict[str, tf.Tensor]:
    return {
        name: tf.random.normal(shape=spec.shape, dtype=spec.dtype)
        for name, spec in specs.items()
    }


def tf_to_tflite_runner(
    saved_dir: str, output_path: str = None
) -> Callable[..., Dict[str, tf.Tensor]]:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    model_content = converter.convert()
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(model_content)
    interpreter = tf.lite.Interpreter(model_content=model_content)
    return interpreter.get_signature_runner()


class TFModel:
    """Wrapper class of TFNet (tf.Module)"""

    def __init__(
        self,
        schedule: Schedule,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.net: TFNet = TFNet(
            schedule=schedule,
            verbose=verbose,
        )

    def save_tfnet(
        self,
        output_dir: str = "saved_tfnet",
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        concrete_net = self.concrete_net(inputs)
        tf.saved_model.save(self.net, output_dir, signatures=concrete_net)
        return concrete_net

    @staticmethod
    def load_tfnet(
        saved_dir: str = "saved_tfnet",
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        return tf.saved_model.load(saved_dir)

    def save(
        self,
        output_dir: str = "saved_tfmodel",
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        os.makedirs(output_dir, exist_ok=True)
        # self.net.schedule.dump(os.path.join(output_dir, "schedule.pkl"))
        # Schedule.dump(self.net.schedule, os.path.join(output_dir, "schedule.pkl"))
        with open(os.path.join(output_dir, "schedule.pkl"), "wb") as f:
            pickle.dump(self.net.schedule, f)
        if inputs is None or isinstance(inputs["i0"], tf.TensorSpec):
            tensor_inputs = self.random_inputs()
        else:
            tensor_inputs = inputs
        outputs_eager_run = self.run_eagerly(tensor_inputs)
        with open(os.path.join(output_dir, "in_out.pkl"), "wb") as f:
            pickle.dump(
                {
                    "inputs": {name: v.numpy() for name, v in tensor_inputs.items()},
                    "outputs": {
                        name: v.numpy() for name, v in outputs_eager_run.items()
                    },
                },
                file=f,
            )
        concrete_net = self.concrete_net(tensor_inputs)
        tf.saved_model.save(
            self.net, os.path.join(output_dir, "tfnet"), signatures=concrete_net
        )
        return concrete_net

    @staticmethod
    def load(
        saved_dir: str = "saved_tfnet", verbose: bool = False
    ) -> Tuple["TFModel", Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        schedule = Schedule.load(os.path.join(saved_dir, "schedule.pkl"))
        model = TFModel(schedule, verbose)
        with open(os.path.join(saved_dir, "in_out.pkl"), "rb") as f:
            in_out = pickle.load(f)
        inputs = {name: tf.convert_to_tensor(v) for name, v in in_out["inputs"].items()}
        outputs = {
            name: tf.convert_to_tensor(v) for name, v in in_out["outputs"].items()
        }
        return model, inputs, outputs

    def random_inputs(self) -> Dict[str, tf.Tensor]:
        return {
            spec.name: tf.random.normal(
                shape=spec.shape,
                dtype=spec.dtype,
                seed=None,
            )
            for spec in self.net.input_specs
        }

    def run_eagerly(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        tf.config.run_functions_eagerly(True)
        return self.net(**inputs)

    def concrete_net(
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        if inputs is None:
            inputs = self.random_inputs()
        return self.net.__call__.get_concrete_function(**inputs)

    @staticmethod
    def assert_eq(x: Dict[str, tf.Tensor], y: Dict[str, tf.Tensor]) -> None:
        for key in x:
            x_v, y_v = x[key], y[key]
            tf.assert_less(
                tf.reduce_max(tf.abs(x_v - y_v)),
                tf.cast(1e-3, dtype=x_v.dtype),
                message=f"Tensors are NOT equal: x[{key}] = {x_v} != {y_v} = y[{key}]",
            )
