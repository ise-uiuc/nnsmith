import random
from typing import List, Tuple, Union

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DTYPE_GEN_ALL, DTYPE_GEN_FLOATS, DType
from nnsmith.abstract.op import (
    AbsOpBase,
    BcastBinaryOp,
    BinaryOpBase,
    ElementWiseUnaryOp,
    MatMul,
    UnaryOpBase,
    mark_materialize,
    rank_all,
    rank_from,
)
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import ConstraintCheck


@mark_materialize("tensorflow")
class Dense(UnaryOpBase):
    in_dtypes = [(DType.float32,), (DType.float64,)]
    out_dtypes = [(DType.float32,), (DType.float64,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [
            rank_from(2)
        ]  # NOTE: tensorflow Dense layer requires an input with batch as its first axis
        # at least one dim. cannot be zranks_all()
        self.out_ranks = [rank_from(2)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        assert len(input_shapes) == 1, "Linear only takes one input, but got {}".format(
            len(input_shapes)
        )
        return [
            AbsTensor(
                shape=[*input_shapes[0].shape[:-1], self.ofeat],
                dtype=input_shapes[0].dtype,
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        ConstraintCheck.true(input_shapes[0].ndims >= 2)
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]


@mark_materialize("tensorflow")
class SquaredDifference(BcastBinaryOp):
    pass


@mark_materialize("tensorflow")
class LocalRespNorm(ElementWiseUnaryOp):
    # https://www.tensorflow.org/api_docs/python/tf/raw_ops/LRN
    in_dtypes = [(DType.float16,), (DType.float32,)]
    out_dtypes = [(DType.float16,), (DType.float32,)]

    def __init__(
        self,
        depth_radius: Union[int, z3.ExprRef],
    ):
        super().__init__()
        self.depth_radius = depth_radius
        self.extra_attrs["bias"] = random.uniform(0.01, 100)
        self.extra_attrs["alpha"] = random.uniform(0.01, 100)
        # cuDNN requires beta >= 0.01
        self.extra_attrs["beta"] = random.uniform(0.011, 1)

        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        SanityCheck.eq(len(input_shapes), 1)
        input_shape = input_shapes[0]
        cons = []
        # cuDNN requires depth_radius in [1, 7]
        cons.append(nnsmith_ge(self.depth_radius, 1))
        cons.append(nnsmith_le(self.depth_radius, 7))
        cons.append(
            nnsmith_le(
                self.depth_radius, nnsmith_div(nnsmith_sub(input_shape.shape[3], 1), 2)
            )
        )  # depth_radius <= (input_shape[3] - 1) / 2
        return cons


class NHWCConv2dBase(BinaryOpBase):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
        padding: str,
    ):
        """See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d"""
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        SanityCheck.true(padding in ["VALID", "SAME"])
        self.extra_attrs["padding"] = padding
        self.inp_ranks = [(4,), (4,)]
        self.out_ranks = [(4,)]

    def __str__(self) -> str:
        return (
            self.name()
            + f" (stride={self.stride}, "
            + f"dilation={(self.dilation_h,self.dilation_w)})"
        )

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        # https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        ni, hi, wi, _ = input_shapes[0].shape
        kh, kw, _, _ = input_shapes[1].shape
        no = ni
        co = self.out_channels
        mimic_kh = kh + (self.dilation_h - 1) * (kh - 1)
        mimic_kw = kw + (self.dilation_w - 1) * (kw - 1)
        if self.extra_attrs["padding"] == "VALID":
            ho = nnsmith_div(hi - mimic_kh, self.stride) + 1
            wo = nnsmith_div(wi - mimic_kw, self.stride) + 1
        elif self.extra_attrs["padding"] == "SAME":
            ho = nnsmith_div(hi + self.stride - 1, self.stride)
            wo = nnsmith_div(wi + self.stride - 1, self.stride)
        return [AbsTensor(shape=[no, ho, wo, co], dtype=input_shapes[0].dtype)]

    def requires(self, input_shapes):
        """
        https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
        https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
        https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d
        """
        _, hi, wi, ci = input_shapes[0].shape
        """
        Filter
        * In conv2d/atrous_conv2d: [kernel_h, kernel_w, in_channels, out_channels]
        * In depthwise/separable_conv2d: [kernel_h, kernel_w, in_channels, channel_multiplier]
        """
        kh, kw, ci_2, co_or_cm = input_shapes[1].shape
        mimic_kh = kh + (self.dilation_h - 1) * (kh - 1)
        mimic_kw = kw + (self.dilation_w - 1) * (kw - 1)

        cons = [
            nnsmith_eq(ci, ci_2),
            nnsmith_ge(self.out_channels, 1),
            nnsmith_ge(self.dilation_h, 1),
            nnsmith_ge(self.dilation_w, 1),
            nnsmith_ge(mimic_kh, 1),
            nnsmith_ge(mimic_kw, 1),
            nnsmith_ge(self.stride, 1),
        ]

        if self.extra_attrs["padding"] == "VALID":
            cons.append(nnsmith_le(mimic_kh, hi))
            cons.append(nnsmith_le(mimic_kw, wi))

        # The following constraint is from TensorFlow tracing:
        # `strides > 1` not supported in conjunction with `dilation_rate > 1`
        cons.append(  # 0 == (stride - 1) * (max(dh, dw) - 1)
            nnsmith_or(
                nnsmith_eq(self.stride, 1),
                nnsmith_and(
                    nnsmith_eq(self.dilation_h, 1),
                    nnsmith_eq(self.dilation_w, 1),
                ),
            )
        )
        return cons

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype), (4, out_abs_tensor[0].dtype)]


class NHWCConv2d(NHWCConv2dBase):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
        padding: str,
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            padding,
        )

    def requires(self, input_shapes):
        cons = super().requires(input_shapes)
        cons.append(nnsmith_eq(self.out_channels, input_shapes[1].shape[3]))
        return cons


@mark_materialize("tensorflow")
class NHWCConv2dValidPad(NHWCConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            "VALID",
        )


@mark_materialize("tensorflow")
class NHWCConv2dSamePad(NHWCConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            "SAME",
        )


class NHWCAtrousConv2d(NHWCConv2dBase):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        rate: Union[int, z3.ExprRef],
        padding: str,
    ):
        super().__init__(
            out_channels,
            1,
            rate,
            rate,
            padding,
        )
        self.rate = rate

    def __str__(self) -> str:
        return self.name() + f"(rate={self.rate})"

    def requires(self, input_shapes):
        cons = super().requires(input_shapes)
        cons.append(nnsmith_eq(self.out_channels, input_shapes[1].shape[3]))
        cons.append(
            nnsmith_le(self.rate, 16)
        )  # dirty hack: otherwise rate is often assigned with huge values.
        return cons


@mark_materialize("tensorflow")
class NHWCAtrousConv2dSamePad(NHWCAtrousConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        rate: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            rate,
            "SAME",
        )


@mark_materialize("tensorflow")
class NHWCAtrousConv2dValidPad(NHWCAtrousConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        rate: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            rate,
            "VALID",
        )


class NHWCDepthwiseConv2d(NHWCConv2dBase):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
        padding: str,
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            padding,
        )

    def requires(self, input_shapes):
        cons = super().requires(input_shapes)
        _, _, ci, cm = input_shapes[1].shape
        cons.append(nnsmith_eq(self.out_channels, ci * cm))
        return cons


@mark_materialize("tensorflow")
class NHWCDepthwiseConv2dValidPad(NHWCDepthwiseConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            "VALID",
        )


@mark_materialize("tensorflow")
class NHWCDepthwiseConv2dSamePad(NHWCDepthwiseConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            "SAME",
        )


class NHWCSeparableConv2d(NHWCConv2dBase):
    in_dtypes = [
        (DType.float16, DType.float16, DType.float16),
        (DType.float32, DType.float32, DType.float32),
        (DType.float64, DType.float64, DType.float64),
    ]
    out_dtypes = [(DType.float16,), (DType.float32,), (DType.float64,)]

    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
        padding: str,
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            padding,
        )
        self.inp_ranks = [(4,), (4,), (4,)]

    def requires(self, input_shapes):
        cons = super().requires(input_shapes)
        _, _, ci, cm = input_shapes[1].shape
        one1, one2, ci_mul_cm, co = input_shapes[2].shape
        cons.append(nnsmith_eq(one1, 1))
        cons.append(nnsmith_eq(one2, 1))
        cons.append(nnsmith_eq(ci_mul_cm, ci * cm))
        cons.append(nnsmith_eq(self.out_channels, co))
        return cons

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (4, out_abs_tensor[0].dtype),
            (4, out_abs_tensor[0].dtype),
            (4, out_abs_tensor[0].dtype),
        ]


@mark_materialize("tensorflow")
class NHWCSeparableConv2dValidPad(NHWCSeparableConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            "VALID",
        )


@mark_materialize("tensorflow")
class NHWCSeparableConv2dSamePad(NHWCSeparableConv2d):
    def __init__(
        self,
        out_channels: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            out_channels,
            stride,
            dilation_h,
            dilation_w,
            "SAME",
        )


class NHWCConv2dTranspose(BinaryOpBase):
    def __init__(
        self,
        stride: Union[int, z3.ExprRef],
        padding: str,
    ):
        super().__init__()
        self.stride = stride
        SanityCheck.true(padding in ["VALID", "SAME"])
        self.extra_attrs["padding"] = padding
        self.inp_ranks = [(4,), (4,)]
        self.out_ranks = [(4,)]

    def __str__(self) -> str:
        return self.name() + f" (strides={self.stride})"

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        ni, hi, wi, _ = input_shapes[0].shape
        kh, kw, co, _ = input_shapes[1].shape
        no = ni

        if self.extra_attrs["padding"] == "SAME":
            ho = hi * self.stride
            wo = wi * self.stride
        else:
            ho = (hi - 1) * self.stride + kh
            wo = (wi - 1) * self.stride + kw

        return [AbsTensor(shape=(no, ho, wo, co), dtype=input_shapes[0].dtype)]

    def requires(self, input_shapes):
        _, _, _, ci = input_shapes[0].shape
        kh, kw, _, ci_2 = input_shapes[1].shape

        cons = [
            nnsmith_eq(ci, ci_2),
            nnsmith_ge(kh, 1),
            nnsmith_ge(kw, 1),
            nnsmith_ge(self.stride, 1),
        ]

        return cons

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype), (4, out_abs_tensor[0].dtype)]


@mark_materialize("tensorflow")
class NHWCConv2dTransposeSamePad(NHWCConv2dTranspose):
    def __init__(
        self,
        stride: Union[int, z3.ExprRef],
    ):
        super().__init__(
            stride,
            "SAME",
        )


@mark_materialize("tensorflow")
class NHWCConv2dTransposeValidPad(NHWCConv2dTranspose):
    def __init__(
        self,
        stride: Union[int, z3.ExprRef],
    ):
        super().__init__(
            stride,
            "VALID",
        )


@mark_materialize("tensorflow")
class NHWCDepthToSpace(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def __str__(self) -> str:
        return self.name() + f" (block_size={self.block_size})"

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        ni, hi, wi, ci = input_shapes[0].shape
        no = ni
        ho = hi * self.block_size
        wo = wi * self.block_size
        co = nnsmith_div(ci, self.block_size * self.block_size)
        return [AbsTensor(shape=[no, ho, wo, co], dtype=input_shapes[0].dtype)]

    def requires(self, input_shapes: List[AbsTensor]):
        _, _, _, ci = input_shapes[0].shape
        return [
            nnsmith_ge(self.block_size, 2),
            nnsmith_eq(nnsmith_mod(ci, self.block_size * self.block_size), 0),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype)]


@mark_materialize("tensorflow")
class NHWCSpaceToDepth(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def __str__(self):
        return self.name() + f" (block_size={self.block_size})"

    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:
        ni, hi, wi, ci = itensors[0].shape
        no = ni
        ho = nnsmith_div(hi, self.block_size)
        wo = nnsmith_div(wi, self.block_size)
        co = ci * self.block_size * self.block_size
        return [AbsTensor(shape=(no, ho, wo, co), dtype=itensors[0].dtype)]

    def requires(self, itensors: List[AbsTensor]):
        _, hi, wi, _ = itensors[0].shape
        return [
            nnsmith_ge(self.block_size, 2),
            nnsmith_eq(nnsmith_mod(hi, self.block_size), 0),
            nnsmith_eq(nnsmith_mod(wi, self.block_size), 0),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype)]


@mark_materialize("tensorflow")
class Gather(BinaryOpBase):
    in_dtypes = [(i, j) for i in DTYPE_GEN_ALL for j in [DType.int32, DType.int64]]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(1), rank_all()]

    def __str__(self) -> str:
        return (
            self.name()
            + f'(axis={self.extra_attrs["axis"] if "axis" in self.extra_attrs else None})'
        )

    def _init_axis(self, input_shapes: List[Union[int, z3.ExprRef]]):
        if "axis" not in self.extra_attrs:
            self.extra_attrs["axis"] = random.randint(0, len(input_shapes) - 1)
        return self.extra_attrs["axis"]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        axis = self._init_axis(input_shapes[0].shape)
        p_shape, i_shape = [*input_shapes[0].shape], [*input_shapes[1].shape]
        o_shape = p_shape[:axis] + i_shape + p_shape[axis + 1 :]
        return [AbsTensor(shape=o_shape, dtype=input_shapes[0].dtype)]

    def requires(self, input_shapes: List[AbsTensor]):
        axis = self._init_axis(input_shapes[0].shape)
        return [nnsmith_ge(input_shapes[0].shape[axis], 1)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        total_rank = out_abs_tensor[0].ndims
        p_rank = random.randint(1, total_rank + 1)
        return [
            (p_rank, out_abs_tensor[0].dtype),
            (total_rank + 1 - p_rank, DType.int32),
        ]


@mark_materialize("tensorflow")
class TFMatMul(MatMul):
    def __init__(self):
        super().__init__()
        self.inp_ranks = [(2, 3), (2, 3)]
        self.out_ranks = [(2, 3)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        if out_abs_tensor[0].ndims == 2:
            return [
                (2, out_abs_tensor[0].dtype),
                (2, out_abs_tensor[0].dtype),
            ]
        # at least one of them is 3
        ranks = [3, random.choice([2, 3])]
        random.shuffle(ranks)
        return [
            (ranks[0], out_abs_tensor[0].dtype),
            (ranks[1], out_abs_tensor[0].dtype),
        ]


@mark_materialize("tensorflow")
class Reverse(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(1)]
        self.out_ranks = [rank_from(1)]

    def _init_axis(self, input_shape: List[Union[int, z3.ExprRef]]):
        # axis is a list of integers
        # |axis| <= rank
        if "axis" not in self.extra_attrs:
            axis = []
            for i in range(len(input_shape)):
                if random.random() < 0.5:  # prob
                    axis.append(i)
            # TODO(@ganler): tflite crashes when axis is empty
            # remove this when tf fixes https://github.com/tensorflow/tensorflow/issues/62679
            axis = axis or [0]
            self.extra_attrs["axis"] = axis
        ConstraintCheck.le(len(self.extra_attrs["axis"]), len(input_shape))
        if self.extra_attrs["axis"]:
            ConstraintCheck.lt(max(self.extra_attrs["axis"]), len(input_shape))
        return self.extra_attrs["axis"]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        _ = self._init_axis(input_shapes[0].shape)
        return input_shapes

    def requires(self, input_shapes):
        _ = self._init_axis(input_shapes[0].shape)
        return super().requires(input_shapes)

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]


@mark_materialize("tensorflow")
class Cholesky(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS + [DType.complex64, DType.complex128]]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS + [DType.complex64, DType.complex128]]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(2)]
        self.out_ranks = [rank_from(2)]

    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:
        return itensors

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        ConstraintCheck.ge(itensors[0].ndims, 2)
        # last two dimensions must be equal
        return [nnsmith_eq(itensors[0].shape[-1], itensors[0].shape[-2])]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]


@mark_materialize("tensorflow")
class Eigh(AbsOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS + [DType.complex64, DType.complex128]]
    out_dtypes = [
        (i, i) for i in DTYPE_GEN_FLOATS + [DType.complex64, DType.complex128]
    ]
    orank_relation = [
        None,
        lambda x: x + 1,
    ]  # the 2nd output has one more rank than the 1st

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(2)]
        self.out_ranks = [rank_from(1), rank_from(2)]

    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:
        # e ~ [..., N], v ~ [..., N, N]
        return [AbsTensor(itensors[0].shape[:-1], dtype=itensors[0].dtype), itensors[0]]

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        ConstraintCheck.ge(itensors[0].ndims, 2)
        # last two dimensions must be equal
        return [nnsmith_eq(itensors[0].shape[-1], itensors[0].shape[-2])]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[1].ndims, out_abs_tensor[1].dtype),
        ]
