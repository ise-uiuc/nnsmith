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
TF_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["tensorflow"]
# TF_REALIZABLE_OPS = [Add, Dense]
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


@operator_impl(ReLU)
def forward_fn(op: ReLU):
    return layers.ReLU()


@operator_impl(GELU)
def forward_fn(op: GELU):
    return keras.activations.gelu


@operator_impl(LeakyReLU)
def forward_fn(op: LeakyReLU):
    return layers.LeakyReLU(alpha=op.negative_slope)


@operator_impl(PReLU)
def forward_fn(op: PReLU):
    return layers.PReLU()


@operator_impl(Sigmoid)
def forward_fn(op: Sigmoid):
    return tf.sigmoid


@operator_impl(Cos)
def forward_fn(op: Cos):
    return tf.cos


@operator_impl(Asin)
def forward_fn(op: Asin):
    return tf.asin


@operator_impl(Acos)
def forward_fn(op: Acos):
    return tf.acos


@operator_impl(Tan)
def forward_fn(op: Tan):
    return tf.tan


@operator_impl(Atan)
def forward_fn(op: Atan):
    return tf.atan


@operator_impl(Abs)
def forward_fn(op: Abs):
    return tf.abs


@operator_impl(Where)
def forward_fn(op: Where):
    return tf.where


@operator_impl(Add)
def forward_fn(op: Add):
    return tf.add


@operator_impl(Sub)
def forward_fn(op: Sub):
    return tf.math.subtract


@operator_impl(Mul)
def forward_fn(op: Mul):
    return tf.multiply


@operator_impl(Div)
def forward_fn(op: Div):
    def _div(up, down):
        if DType.from_tensorflow(up.dtype) in DTYPE_INTS:
            return tf.math.floordiv(up, down)
        else:
            return tf.divide(up, down)

    return _div


# TODO


@operator_impl(Max)
def forward_fn(op: Max):
    return tf.maximum


@operator_impl(Min)
def forward_fn(op: Min):
    return tf.minimum


@operator_impl(Equal)
def forward_fn(op: Equal):
    return tf.equal


@operator_impl(Greater)
def forward_fn(op: Greater):
    return tf.greater


@operator_impl(Less)
def forward_fn(op: Less):
    return tf.less


@operator_impl(And)
def forward_fn(op: And):
    return tf.logical_and


@operator_impl(Or)
def forward_fn(op: Or):
    return tf.logical_or


@operator_impl(Xor)
def forward_fn(op: Xor):
    return tf.math.logical_xor


@operator_impl(Pow)
def forward_fn(op: Pow):
    return tf.pow


@operator_impl(Floor)
def forward_fn(op: Floor):
    return tf.floor


@operator_impl(Ceil)
def forward_fn(op: Ceil):
    return tf.math.ceil


@operator_impl(Clip)
def forward_fn(op: Clip):
    if op.input_like[0].dtype in DTYPE_FLOATS:
        return lambda x: tf.clip_by_value(x, -1.5, 1.5)
    else:
        return lambda x: tf.clip_by_value(x, -1, 1)


@operator_impl(Round)
def forward_fn(op: Round):
    return tf.round


@operator_impl(Sqrt)
def forward_fn(op: Sqrt):
    return tf.sqrt


@operator_impl(Log2)
def forward_fn(op: Log2):
    return tf.experimental.numpy.log2


@operator_impl(Neg)
def forward_fn(op: Neg):
    return tf.negative


@operator_impl(Softmax)
def forward_fn(op: Softmax):
    return layers.Softmax(axis=op.dim)


# @operator_impl(MaxPool2d)
# def forward_fn(op: MaxPool2d):
#     return layers.MaxPool2D(
#         pool_size=(op.kernel_h_size, op.kernel_w_size),
#         strides=op.stride,
#         padding=op.padding,  # TODO https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
#         data_format="channels_first",  # NCHW
#     )


# @operator_impl(AvgPool2d)
# def forward_fn(op: AvgPool2d):
#     return lambda x: tf.nn.avg_pool2d(
#         x,
#         ksize=(op.kernel_h_size, op.kernel_w_size),
#         strides=(op.stride, op.stride),
#         padding=op.padding,
#         data_format="NCHW",
#     )
# # TODO


@operator_impl(Slice)
def forward_fn(op: Slice):
    reg = op.extra_attrs["region"]

    def _slice(x):
        dim_s = x.shape[op.extra_attrs["axis"]]
        start, end = op.start, op.end
        if reg in ["left", "mid"]:
            start -= dim_s
        # actual end would be 0, which is not really 'left'
        if reg == "left" and end < dim_s and end != Slice.INT_MAX:
            end -= dim_s
        s = tuple(
            slice(None, None)
            if i != op.extra_attrs["axis"]
            else slice(start, end, op.step)
            for i in range(op.extra_attrs["ndims"])
        )
        return x[s]

    return _slice


# TODO


# @operator_impl(Pad)
# def forward_fn(op: Pad):
#     if op.extra_attrs["type"] == "constant":
#         # 0 easily cause division by zero...
#         # 1 easily cause false positives (sqrt(1) = 0.99999... != 1 in ORT, so floor(sqrt(1))=0)
#         return lambda x: tf.pad(
#             x, op.padding_list, "CONSTANT", value=0.5
#         )
#     elif op.extra_attrs["type"] == "replicate" or op.extra_attrs["type"] == "reflect":
#         return lambda x: tf.pad(
#             x, op.padding_list, op.extra_attrs["type"]
#         )
#     # TODO no replicate https://www.tensorflow.org/api_docs/python/tf/pad


# TODO expand


@operator_impl(BatchNorm2d)
def forward_fn(op: BatchNorm2d):
    return layers.BatchNormalization(axis=1)  # NCHW


@operator_impl(Conv1d)
def forward_fn(op: Conv1d):
    return layers.Conv1D(
        filters=op.out_channels,
        kernel_size=op.kernel_size,
        strides=op.stride,
        padding=op.padding,
        data_format="channels_first",  # NCHW
        dilation_rate=op.dilation,
    )


@operator_impl(NCHWConv2d)
def forward_fn(op: NCHWConv2d):
    return layers.Conv2D(
        filters=op.out_channels,
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        strides=(op.stride, op.stride),
        padding=op.padding,
        data_format="channels_first",  # NCHW
    )


@operator_impl(Reshape)
def forward_fn(op: Reshape):
    return layers.Reshape(tuple(op.target_shape))


# @operator_impl(Flatten)
# def forward_fn(op: Flatten):
#     return layers.Flatten() # TODO this version doesn't affect batch size


@operator_impl(Flatten)
def forward_fn(op: Flatten):
    def _flatten(x):
        dim = tf.reduce_prod(x.shape)
        return tf.reshape(x, [dim])

    return _flatten


@operator_impl(Transpose)
def forward_fn(op: Transpose):
    def _transpose(x: tf.Tensor):
        dim0, dim1 = op._init_swap_dims(list(x.shape))
        perm = list(range(len(x.shape)))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return tf.transpose(x, perm=perm)

    return _transpose


@operator_impl(Dense)
def forward_fn(op: Dense):
    return layers.Dense(
        units=op.ofeat, dtype=op.input_like[0].dtype.tensorflow(), autocast=False
    )
