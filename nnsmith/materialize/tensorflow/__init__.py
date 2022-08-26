from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
from multipledispatch import dispatch  # type: ignore
import os
import pickle

import tensorflow as tf  # type: ignore


def fix_tensorflow_issues():
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


fix_tensorflow_issues()

from nnsmith.graph_gen import Schedule
from nnsmith.abstract.op import AbsTensor
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.tensorflow.tfnet import TFNet


@dispatch(list)
def randn_from_specs(specs: List[tf.TensorSpec]) -> List[tf.Tensor]:
    return [
        tf.cast(tf.random.normal(shape=spec.shape), dtype=spec.dtype) for spec in specs
    ]


@dispatch(dict)
def randn_from_specs(specs: Dict[str, tf.TensorSpec]) -> Dict[str, tf.Tensor]:
    return {
        name: tf.cast(tf.random.normal(shape=spec.shape), dtype=spec.dtype)
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


def assert_dict_eq_tf(x: Dict[str, tf.Tensor], y: Dict[str, tf.Tensor]) -> None:
    for key in x:
        x_v, y_v = x[key], y[key]
        assert tf.less_equal(
            tf.reduce_max(tf.abs(x_v - y_v)),
            tf.cast(1e-3, dtype=x_v.dtype),
        ), f"Tensors are NOT equal: x[{key}] = {x_v} != {y_v} = y[{key}]"


class TFModel(Model):
    """Wrapper class of TFNet (tf.Module)
    It only stores net. Other information like I/O info are temporarily generated.
    Other values like I/O values should be managed by the user.
    """

    def __init__(
        self,
        schedule: Schedule,
        verbose: bool = False,
    ) -> None:
        """Must provide a schedule to avoid NoneType errors"""
        self.net: TFNet = TFNet(
            schedule=schedule,
            verbose=verbose,
        )

    def set_verbose(self, verbose: bool = True) -> None:
        self.net.verbose = verbose

    @staticmethod
    def from_schedule(self, instructions: Schedule, **kwargs) -> "Model":
        return TFModel(instructions, kwargs["verbose"])

    @property
    def native_model(self) -> TFNet:
        return self.net

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return {
            f"i{i_inp}": self.net.schedule.key2type[key]
            for i_inp, key in enumerate(self.net.schedule.input_keys)
        }

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return {
            f"o{i_out}": self.net.schedule.key2type[key]
            for i_out, key in enumerate(self.net.schedule.leaf_keys)
        }

    @property
    def input_specs(self) -> Dict[str, tf.TensorSpec]:
        ret: Dict[str, tf.TensorSpec] = {}
        for i_inp, key in enumerate(self.net.schedule.input_keys):
            abs_tensor = self.net.schedule.key2type[key]
            ret[f"i{i_inp}"] = tf.TensorSpec(
                abs_tensor.shape, abs_tensor.dtype.tensorflow(), f"i{i_inp}"
            )
        return ret

    @staticmethod
    def name_suffix() -> str:
        return ""

    def make_oracle(
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None
    ) -> Oracle:
        if inputs is None or isinstance(inputs["i0"], tf.TensorSpec):
            input_dict = self.random_inputs()
        else:
            input_dict = inputs
        output_dict = self.run_eagerly(input_dict)

        input_dict = {k: v.numpy() for k, v in input_dict.items()}
        output_dict = {k: v.numpy() for k, v in output_dict.items()}

        return Oracle(input_dict, output_dict)

    def dump(self, path: str = "saved_tfmodel") -> None:
        os.makedirs(path, exist_ok=True)
        # schedule.pkl
        with open(os.path.join(path, TFModel.schedule_pkl_name()), "wb") as f:
            pickle.dump(self.net.schedule, f)
        # tfnet
        concrete_net = self.concrete_net(self.input_specs)
        tf.saved_model.save(
            self.net,
            os.path.join(path, TFModel.tfnet_dir_name()),
            signatures=concrete_net,
        )

    def dump_with_oracle(
        self,
        path: str = "saved_tfmodel",
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> None:
        self.dump(path)
        oracle = self.make_oracle(inputs)
        oracle.dump(os.path.join(path, Oracle.name()))

    def dump_tfnet(
        self,
        path: str = "saved_tfnet",
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        concrete_net = self.concrete_net(inputs)
        tf.saved_model.save(self.net, path, signatures=concrete_net)
        return concrete_net

    @staticmethod
    def load_tfnet(
        saved_dir: str = "saved_tfnet",
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        return tf.saved_model.load(saved_dir)

    @staticmethod
    def load(path: str = "saved_tfmodel") -> "TFModel":
        with open(os.path.join(path, TFModel.schedule_pkl_name()), "rb") as f:
            schedule: Schedule = pickle.load(f)
        model = TFModel(schedule)
        return model

    @staticmethod
    def load_with_oracle(
        path: str = "saved_tfmodel",
    ) -> Tuple["TFModel", Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        model = TFModel.load(path)
        oracle = Oracle.load(os.path.join(path, Oracle.name()))
        input_dict = {name: tf.convert_to_tensor(v) for name, v in oracle.input.items()}
        output_dict = {
            name: tf.convert_to_tensor(v) for name, v in oracle.output.items()
        }
        return model, input_dict, output_dict

    @staticmethod
    def schedule_pkl_name():
        return "schedule.pkl"

    @staticmethod
    def in_out_pkl_name():
        return "in_out.pkl"

    @staticmethod
    def tfnet_dir_name():
        return "tfnet"

    def random_inputs(self) -> Dict[str, tf.Tensor]:
        return {
            spec.name: tf.cast(
                tf.random.normal(
                    shape=spec.shape,
                    seed=None,
                ),
                dtype=spec.dtype,
            )
            for spec in self.input_specs.values()
        }

    def concrete_net(
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None
    ) -> Callable[..., Dict[str, tf.Tensor]]:
        if inputs is None:
            inputs = self.random_inputs()
        return self.net.__call__.get_concrete_function(**inputs)

    def refine_weights(self) -> None:
        raise NotImplementedError()

    def run_eagerly(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        tf.config.run_functions_eagerly(True)
        return self.net(**inputs)
