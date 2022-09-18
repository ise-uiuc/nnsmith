import random
from typing import List, Tuple, Union

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import (
    FLOPS_LIM,
    Z3_CONS_FLOPS,
    BcastBinaryOp,
    ElementWiseUnaryOp,
    UnaryOpBase,
    int_from,
    mark_materialize,
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
            int_from(2)
        ]  # NOTE: tensorflow Dense layer requires an input with batch as its first axis
        # at least one dim. cannot be zranks_all()
        self.out_ranks = [int_from(2)]

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
        # limit FLOPS
        if Z3_CONS_FLOPS:
            cons.append(nnsmith_le(self.flops(input_shapes), FLOPS_LIM))
        return cons

    def flops(self, input_shapes):
        return nnsmith_mul(
            nnsmith_mul(
                nnsmith_mul(
                    self.type_transfer(input_shapes)[0].nelement(), self.in_channels
                ),
                self.kernel_h_size,
            ),
            self.kernel_w_size,
        )

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
