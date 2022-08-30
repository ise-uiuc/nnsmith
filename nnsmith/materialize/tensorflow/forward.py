from functools import partial
from typing import List, Type

import tensorflow as tf  # type: ignore
from keras import layers
from multipledispatch import dispatch
from tensorflow import keras

from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.tensorflow.dialect import Dense

# core dialect + some future PyTorch-only Operators.
# TF_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["tensorflow"]
TF_REALIZABLE_OPS = [Add, Dense]
ALL_TF_OPS: List[Type[AbsOpBase]] = []

operator_impl = partial(framework_operator_impl, TF_REALIZABLE_OPS, ALL_TF_OPS)


class StopFoldConst(tf.Module):
    def __init__(self, data: tf.Tensor):
        super().__init__()
        self.data = tf.Variable(data, trainable=False)  # default variable is trainable

    def __call__(self, training=None):
        return self.data


"""Implement TensorFlow forward Callables for operator classes"""


@operator_impl(Constant)
def forward_fn(op: Constant):
    data = tf.cast(
        tf.random.normal(op.abs_tensor.shape), op.abs_tensor.dtype.tensorflow()
    )
    return StopFoldConst(data)


@operator_impl(Add)
def forward_fn(op: Add):
    return tf.add


@operator_impl(Dense)
def forward_fn(op: Dense):
    return layers.Dense(
        units=op.ofeat, dtype=op.input_like[0].dtype.tensorflow(), autocast=False
    )
