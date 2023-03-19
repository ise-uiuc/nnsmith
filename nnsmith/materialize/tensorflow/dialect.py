import random
from typing import List, Tuple, Union

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DTYPE_GEN_ALL, DTYPE_GEN_FLOATS, DType
from nnsmith.abstract.op import (
    AbsOpBase,
    BcastBinaryOp,
    ElementWiseUnaryOp,
    MatMul,
    UnaryOpBase,
    mark_materialize,
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


class NHWCConv2d(UnaryOpBase):
    in_dtypes = [(DType.float16,), (DType.float32,), (DType.float64,)]
    out_dtypes = [(DType.float16,), (DType.float32,), (DType.float64,)]

    def __init__(
        self,
        in_channels: Union[int, z3.ExprRef],
        out_channels: Union[int, z3.ExprRef],
        kernel_h_size: Union[int, z3.ExprRef],
        kernel_w_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
        padding: str,
    ):
        """See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        SanityCheck.true(padding in ["valid", "same"])
        self.extra_attrs["padding"] = padding
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def __str__(self) -> str:
        return (
            self.name()
            + f" (kernel={(self.kernel_h_size,self.kernel_w_size)}, "
            + f"stride={self.stride}, "
            + f"dilation={(self.dilation_h,self.dilation_w)}, "
            + f"ochannels={self.out_channels})"
        )

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        # https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        ni, hi, wi, _ = input_shapes[0].shape
        no = ni
        co = self.out_channels
        mimic_kh = self.kernel_h_size + (self.dilation_h - 1) * (self.kernel_h_size - 1)
        mimic_kw = self.kernel_w_size + (self.dilation_w - 1) * (self.kernel_w_size - 1)
        if self.extra_attrs["padding"] == "valid":
            ho = nnsmith_div(hi - mimic_kh, self.stride) + 1
            wo = nnsmith_div(wi - mimic_kw, self.stride) + 1
        elif self.extra_attrs["padding"] == "same":
            ho = nnsmith_div(hi + self.stride - 1, self.stride)
            wo = nnsmith_div(wi + self.stride - 1, self.stride)
        return [AbsTensor(shape=[no, ho, wo, co], dtype=input_shapes[0].dtype)]

    def requires(self, input_shapes):
        _, hi, wi, ci = input_shapes[0].shape
        mimic_kh = self.kernel_h_size + (self.dilation_h - 1) * (self.kernel_h_size - 1)
        mimic_kw = self.kernel_w_size + (self.dilation_w - 1) * (self.kernel_w_size - 1)

        cons = [
            nnsmith_eq(self.in_channels, ci),
            nnsmith_ge(self.out_channels, 1),
            nnsmith_ge(self.dilation_h, 1),
            nnsmith_ge(self.dilation_w, 1),
            nnsmith_ge(mimic_kh, 1),
            nnsmith_ge(mimic_kw, 1),
            nnsmith_ge(self.stride, 1),
        ]

        if self.extra_attrs["padding"] == "valid":
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

    def n_floats(self, input_shapes):
        # FIXME: maybe need to take dilation into account?
        padding = 0
        padded_data = AbsTensor(input_shapes[0].shape, dtype=input_shapes[0].dtype)
        padded_data.shape[2] = nnsmith_add(
            padded_data.shape[2], nnsmith_mul(2, padding)
        )
        padded_data.shape[3] = nnsmith_add(
            padded_data.shape[3], nnsmith_mul(2, padding)
        )
        w = AbsTensor(
            [
                self.out_channels,
                self.in_channels,
                self.kernel_h_size,
                self.kernel_w_size,
            ],
            dtype=input_shapes[0].dtype,
        )
        outs = super().n_floats(input_shapes)
        return nnsmith_add(nnsmith_add(w.nelement(), padded_data.nelement()), outs)

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype)]


@mark_materialize("tensorflow")
class NHWCConv2dValidPad(NHWCConv2d):
    def __init__(
        self,
        in_channels: Union[int, z3.ExprRef],
        out_channels: Union[int, z3.ExprRef],
        kernel_h_size: Union[int, z3.ExprRef],
        kernel_w_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_h_size,
            kernel_w_size,
            stride,
            dilation_h,
            dilation_w,
            "valid",
        )


@mark_materialize("tensorflow")
class NHWCConv2dSamePad(NHWCConv2d):
    def __init__(
        self,
        in_channels: Union[int, z3.ExprRef],
        out_channels: Union[int, z3.ExprRef],
        kernel_h_size: Union[int, z3.ExprRef],
        kernel_w_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_h_size,
            kernel_w_size,
            stride,
            dilation_h,
            dilation_w,
            "same",
        )


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
