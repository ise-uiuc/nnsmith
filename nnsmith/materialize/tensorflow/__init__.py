from __future__ import annotations

import os
import pickle
from os import PathLike
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch  # type: ignore


def configure_tensorflow():
    # https://github.com/tensorflow/tensorflow/issues/57359
    # tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


configure_tensorflow()

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.materialize import Model, Oracle, Schedule
from nnsmith.materialize.tensorflow.forward import ALL_TF_OPS
from nnsmith.materialize.tensorflow.tfnet import TFNet, TFNetOutDict

TFNetInputDict = Dict[str, tf.Tensor]
TFNetCallable = Callable[..., TFNetOutDict]


@dispatch(dict)
def randn_from_specs(specs: Dict[str, tf.TensorSpec]) -> Dict[str, tf.Tensor]:
    return {
        name: tf.cast(tf.random.normal(shape=spec.shape), dtype=spec.dtype)
        for name, spec in specs.items()
    }


def np_dict_from_tf(x: Dict[str, tf.Tensor]) -> Dict[str, np.ndarray]:
    return {key: value.numpy() for key, value in x.items()}


def tf_dict_from_np(x: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    return {key: tf.convert_to_tensor(value) for key, value in x.items()}


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
        super().__init__()
        self.net: TFNet = TFNet(
            schedule=schedule,
            verbose=verbose,
        )

    def set_verbose(self, verbose: bool = True) -> None:
        self.net.verbose = verbose

    @staticmethod
    def from_schedule(schedule: Schedule, **kwargs) -> "Model":
        return TFModel(schedule, kwargs.get("verbose", False))

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

        input_dict = np_dict_from_tf(input_dict)
        output_dict = np_dict_from_tf(output_dict)

        return Oracle(input_dict, output_dict)

    def dump(self, path: PathLike = "saved_tfmodel") -> None:
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
        path: PathLike = "saved_tfmodel",
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> None:
        self.dump(path)
        oracle = self.make_oracle(inputs)
        oracle.dump(os.path.join(path, Oracle.name()))

    def dump_tfnet(
        self,
        path: PathLike = "saved_tfnet",
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> TFNetCallable:
        concrete_net = self.concrete_net(inputs)
        tf.saved_model.save(self.net, path, signatures=concrete_net)
        return concrete_net

    @staticmethod
    def load_tfnet(
        saved_dir: str = "saved_tfnet",
    ) -> TFNetCallable:
        return tf.saved_model.load(saved_dir)

    @staticmethod
    def load(path: PathLike = "saved_tfmodel") -> "TFModel":
        with open(os.path.join(path, TFModel.schedule_pkl_name()), "rb") as f:
            schedule: Schedule = pickle.load(f)
        model = TFModel(schedule)
        return model

    @staticmethod
    def load_with_oracle(
        path: PathLike = "saved_tfmodel",
    ) -> Tuple["TFModel", TFNetInputDict, TFNetOutDict]:
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

    def random_inputs(self) -> TFNetInputDict:
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
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] | None = None
    ) -> TFNetCallable:
        if inputs is None:
            inputs = self.input_specs
        return self.net.__call__.get_concrete_function(**inputs)

    def refine_weights(self) -> None:
        pass

    def run_eagerly(self, inputs: TFNetInputDict) -> TFNetOutDict:
        tf.config.run_functions_eagerly(True)
        return self.net(**inputs)

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return list(ALL_TF_OPS)
