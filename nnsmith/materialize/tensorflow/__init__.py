from __future__ import annotations

import logging
import os
import pickle
from abc import ABC
from os import PathLike
from typing import Callable, Dict, List, Type

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").disabled = True

import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch  # type: ignore


def configure_tf_gpu_mem(max_megabytes=None):
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        if isinstance(max_megabytes, int):
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=max_megabytes)]
            )


configure_tf_gpu_mem()

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.materialize import Model, Oracle, Schedule
from nnsmith.materialize.tensorflow.forward import ALL_TF_OPS
from nnsmith.materialize.tensorflow.tfnet import TFNet
from nnsmith.util import register_seed_setter

TFNetCallable = Callable[..., Dict[str, tf.Tensor]]


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


class EagerModeCtx:
    def __init__(self, eagerly: bool) -> None:
        assert isinstance(
            eagerly, bool
        ), f"argument eagerly should not be {eagerly.__class__}. It must be a boolean."
        self.eagerly = eagerly

    def __enter__(self) -> None:
        self.old_mode = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(self.eagerly)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        tf.config.run_functions_eagerly(self.old_mode)


class TFModel(Model, ABC):  # Don't directly instantiate this class
    """Wrapper class of TFNet (tf.Module)
    It only stores net. Other information like I/O info are temporarily generated.
    Other values like I/O values should be managed by the user.
    """

    def __init__(self, schedule: Schedule) -> None:
        """Must provide a schedule to avoid NoneType errors"""
        super().__init__()
        self.schedule = schedule
        self.net: TFNet = TFNet(
            schedule=schedule,
        )

    @classmethod
    def from_schedule(cls, schedule: Schedule, **kwargs) -> "TFModel":
        return cls(schedule)

    @property
    def native_model(self) -> TFNet:
        return self.net

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return {
            f"i{i_inp}": self.schedule.key2type[key]
            for i_inp, key in enumerate(self.schedule.input_keys)
        }

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return {
            f"o{i_out}": self.schedule.key2type[key]
            for i_out, key in enumerate(self.schedule.leaf_keys)
        }

    @property
    def input_specs(self) -> Dict[str, tf.TensorSpec]:
        ret: Dict[str, tf.TensorSpec] = {}
        for i_inp, key in enumerate(self.schedule.input_keys):
            abs_tensor = self.schedule.key2type[key]
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

        return Oracle(input_dict, output_dict, provider="tf[cpu] eager")

    def dump(self, path: PathLike = "saved_tfmodel") -> None:
        os.makedirs(path, exist_ok=True)
        # schedule.pkl
        with open(os.path.join(path, TFModel.schedule_name()), "wb") as f:
            pickle.dump(self.schedule, f)
        # tfnet
        with self.device:
            concrete_net = self.concrete_net(self.input_specs)
            tf.saved_model.save(
                self.net,
                os.path.join(path, TFModel.tfnet_dir_name()),
                signatures=concrete_net,
            )

    def dump_with_oracle(
        self,
        path: PathLike,
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> None:
        self.dump(path)
        oracle = self.make_oracle(inputs)
        oracle.dump(os.path.join(path, Oracle.name()))

    @classmethod
    def load(cls, path: PathLike) -> "TFModel":
        with open(os.path.join(path, cls.schedule_name()), "rb") as f:
            schedule: Schedule = pickle.load(f)
        model = cls(schedule)
        model.net = tf.saved_model.load(os.path.join(path, TFModel.tfnet_dir_name()))
        return model

    @staticmethod
    def schedule_name():
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
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] | None = None
    ) -> TFNetCallable:
        if inputs is None:
            inputs = self.input_specs
        return self.net.__call__.get_concrete_function(**inputs)

    def refine_weights(self) -> None:
        pass

    def run_eagerly(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        with EagerModeCtx(True), self.device:  # disable graph execution
            # TODO some op can only run on GPU (e.g. conv with NCHW)
            results = self.net(**inputs)
        return results

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return list(ALL_TF_OPS)

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("tensorflow", tf.random.set_seed, overwrite=True)


class TFModelCPU(TFModel):
    @property
    def device(self) -> tf.device:
        return tf.device(tf.config.list_logical_devices("CPU")[0].name)


class TFModelGPU(TFModel):
    @property
    def device(self) -> tf.device:
        gpus = tf.config.list_physical_devices("GPU")
        assert gpus, "No GPU available"
        return tf.device(gpus[0].name)
