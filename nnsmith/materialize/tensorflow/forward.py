from multipledispatch import dispatch

import tensorflow as tf  # type: ignore
from tensorflow import keras
from keras import layers

from nnsmith.abstract.op import *


class StopFoldConst(tf.Module):
    def __init__(self, data: tf.Tensor):
        super().__init__()
        self.data = tf.Variable(data, trainable=False)  # default variable is trainable

    def __call__(self, training=None):
        return self.data


"""TensorFlow forward Callables"""


@dispatch(Constant, AbsTensor)
def forward_fn(op: Constant, out_abs_tensor: AbsTensor):
    data = tf.random.normal(op.abs_tensor.shape, dtype=op.abs_tensor.dtype.tensorflow())
    return StopFoldConst(data)


@dispatch(Add, AbsTensor)
def forward_fn(op: Add, out_abs_tensor: AbsTensor):
    return tf.add


@dispatch(Dense, AbsTensor)
def forward_fn(op: Dense, out_abs_tensor: AbsTensor):
    return layers.Dense(
        units=op.ofeat, dtype=out_abs_tensor.dtype.tensorflow(), autocast=False
    )
