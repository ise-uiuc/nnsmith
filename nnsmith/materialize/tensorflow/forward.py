from functools import partial
from typing import List, Type

import tensorflow as tf  # type: ignore
from keras import layers
from tensorflow import keras

from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.tensorflow.dialect import *

# core dialect + some future PyTorch-only Operators.
TF_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["tensorflow"]
# TF_REALIZABLE_OPS = [NHWCConv2dSamePad, NHWCConv2dValidPad] # [Add, Dense, LocalRespNorm]
ALL_TF_OPS: List[Type[AbsOpBase]] = []

operator_impl = partial(framework_operator_impl, TF_REALIZABLE_OPS, ALL_TF_OPS)


"""Implement TensorFlow forward Callables for operator classes"""


@operator_impl(Constant)
def forward_fn(op: Constant):
    dtype = op.abs_tensor.dtype.tensorflow()
    data = tf.cast(tf.random.normal(op.abs_tensor.shape), dtype)
    return lambda: tf.constant(data, dtype=dtype)


@operator_impl(ReLU)
def forward_fn(op: ReLU):
    return layers.ReLU(
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )


@operator_impl(GELU)
def forward_fn(op: GELU):
    return keras.activations.gelu


@operator_impl(LeakyReLU)
def forward_fn(op: LeakyReLU):
    return layers.LeakyReLU(
        alpha=op.negative_slope,
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )


@operator_impl(PReLU)
def forward_fn(op: PReLU):
    return layers.PReLU(
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )


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
    return tf.divide


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
    return layers.Softmax(
        axis=op.dim,
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )


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


@operator_impl(BatchNorm2d)
def forward_fn(op: BatchNorm2d):
    return layers.BatchNormalization(
        axis=1,
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )  # NCHW


@operator_impl(Reshape)
def forward_fn(op: Reshape):
    def _reshape(x):
        return tf.reshape(x, op.target_shape)

    return _reshape


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


@operator_impl(LocalRespNorm)
def forward_fn(op: LocalRespNorm):
    def _lrn(x):
        return tf.raw_ops.LRN(
            input=x,
            depth_radius=op.depth_radius,
            bias=op.extra_attrs["bias"],
            alpha=op.extra_attrs["alpha"],
            beta=op.extra_attrs["beta"],
        )

    return _lrn


@operator_impl(NHWCConv2d)
def forward_fn(op: NHWCConv2d):
    return layers.Conv2D(
        filters=op.out_channels,
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        strides=(op.stride, op.stride),
        data_format="channels_last",
        dilation_rate=(op.dilation_h, op.dilation_w),
        padding=op.extra_attrs["padding"],
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )
