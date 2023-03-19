from functools import partial
from typing import List, Type

import tensorflow as tf  # type: ignore
from keras import layers

from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.tensorflow.dialect import *

# core dialect + some future TensorFlow-only Operators.
TF_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["tensorflow"]
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
    return tf.nn.relu


@operator_impl(GELU)
def forward_fn(op: GELU):
    return tf.nn.gelu


@operator_impl(LeakyReLU)
def forward_fn(op: LeakyReLU):
    return tf.nn.leaky_relu


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
    if op.input_like[0].dtype in DTYPE_GEN_FLOATS:
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
    return lambda x: tf.nn.softmax(
        logits=tf.ensure_shape(x, op.input_like[0].shape),
        axis=op.dim,
    )


@operator_impl(Slice)
def forward_fn(op: Slice):
    reg = op.extra_attrs["region"]
    shape = op.input_like[0].shape
    dim_s = shape[op.extra_attrs["axis"]]
    start, end = op.start, op.end
    if reg in ["left", "mid"]:
        start -= dim_s
    # actual end would be 0, which is not really 'left'
    if reg == "left" and end < dim_s and end != Slice.INT_MAX:
        end -= dim_s
    idx = tuple(
        slice(None, None) if i != op.extra_attrs["axis"] else slice(start, end, op.step)
        for i in range(op.extra_attrs["ndims"])
    )

    return lambda x: x[idx]


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
        aten = op.input_like[0]
        dim0, dim1 = op._init_swap_dims(aten.shape)
        perm = list(range(aten.ndims))
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
    return lambda input, filters: tf.nn.conv2d(
        input=input,
        filters=filters,
        strides=op.stride,
        padding=op.extra_attrs["padding"],
        dilations=(op.dilation_h, op.dilation_w),
    )


@operator_impl(NHWCAtrousConv2d)
def forward_fn(op: NHWCAtrousConv2d):
    return lambda value, filters: tf.nn.atrous_conv2d(
        value=value,
        filters=filters,
        rate=op.rate,
        padding=op.extra_attrs["padding"],
    )


@operator_impl(NHWCDepthwiseConv2d)
def forward_fn(op: NHWCDepthwiseConv2d):
    return lambda input, filter: tf.nn.depthwise_conv2d(
        input=input,
        filter=filter,
        strides=(1, op.stride, op.stride, 1),
        padding=op.extra_attrs["padding"],
        dilations=(op.dilation_h, op.dilation_w),
    )


@operator_impl(NHWCSeparableConv2d)
def forward_fn(op: NHWCSeparableConv2d):
    return lambda input, depthwise_filter, pointwise_filter: tf.nn.separable_conv2d(
        input=input,
        depthwise_filter=depthwise_filter,
        pointwise_filter=pointwise_filter,
        strides=(1, op.stride, op.stride, 1),
        padding=op.extra_attrs["padding"],
        dilations=(op.dilation_h, op.dilation_w),
    )


@operator_impl(NHWCConv2dTranspose)
def forward_fn(op: NHWCConv2dTranspose):
    return lambda input, filters: tf.nn.conv2d_transpose(
        input=input,
        filters=filters,
        output_shape=op.output_like[0].shape,
        strides=op.stride,
        padding=op.extra_attrs["padding"],
    )


@operator_impl(NHWCDepthToSpace)
def forward_fn(op: NHWCDepthToSpace):
    return lambda input: tf.nn.depth_to_space(
        input=input,
        block_size=op.block_size,
    )


@operator_impl(NHWCSpaceToDepth)
def forward_fn(op: NHWCSpaceToDepth):
    return lambda input: tf.nn.space_to_depth(
        input=input,
        block_size=op.block_size,
    )


@operator_impl(Gather)
def forward_fn(op: Gather):
    axis = op.extra_attrs["axis"]
    clip_value_min, clip_value_max = 0, op.input_like[0].shape[axis] - 1

    def _gather(params, indices):
        indices = tf.clip_by_value(indices, clip_value_min, clip_value_max)
        return tf.gather(params, indices, axis=axis)

    return _gather


@operator_impl(Squeeze)
def forward_fn(op: Squeeze):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.squeeze(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.squeeze(x)


@operator_impl(Unsqueeze)
def forward_fn(op: Unsqueeze):
    SanityCheck.true(op.extra_attrs["expand_dim"] != None)
    return lambda x: tf.expand_dims(x, axis=op.extra_attrs["expand_dim"])


@operator_impl(ReduceSum)
def forward_fn(op: ReduceSum):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.reduce_sum(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.reduce_sum(x)


@operator_impl(ReduceMin)
def forward_fn(op: ReduceMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.reduce_min(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.reduce_min(x)


@operator_impl(ReduceMax)
def forward_fn(op: ReduceMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.reduce_max(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.reduce_max(x)


@operator_impl(ReduceMean)
def forward_fn(op: ReduceMean):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.reduce_mean(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.reduce_mean(x)


@operator_impl(ReduceProd)
def forward_fn(op: ReduceProd):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.reduce_prod(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.reduce_prod(x)


@operator_impl(ArgMin)
def forward_fn(op: ArgMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.argmin(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.argmin(x)


@operator_impl(ArgMin)
def forward_fn(op: ArgMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: tf.math.argmax(x, axis=op.extra_attrs["reduce_dim"])
    return lambda x: tf.math.argmax(x)


@operator_impl(Tril)
def forward_fn(op: Tril):
    return lambda x: tf.experimental.numpy.tril(x, k=op.diagonal)


@operator_impl(Triu)
def forward_fn(op: Triu):
    return lambda x: tf.experimental.numpy.triu(x, k=op.diagonal)


@operator_impl(Concat)
def forward_fn(op: Concat):
    axis = op.extra_attrs["axis"]
    return lambda *args: tf.concat(args, axis=axis)


@operator_impl(Cast)
def forward_fn(op: Cast):
    return lambda x: tf.cast(x, dtype=op.extra_attrs["to"].tensorflow())


@operator_impl(TFMatMul)
def forward_fn(op: TFMatMul):
    return tf.matmul


@operator_impl(Reverse)
def forward_fn(op: Reverse):
    return lambda x: tf.reverse(x, axis=op.extra_attrs["axis"])


@operator_impl(Cholesky)
def forward_fn(op: Cholesky):
    return tf.linalg.cholesky


@operator_impl(Eigh)
def forward_fn(op: Eigh):
    return tf.linalg.eigh
