from functools import partial
from typing import Type

import torch

from nnsmith.abstract.dtype import DTYPE_INTS
from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.torch.dialect import Flatten, Linear, TorchReduceSum

# Implementation of operators.

# core dialect + some future PyTorch-only Operators.
TORCH_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["torch"]
ALL_TORCH_OPS: List[Type[AbsOpBase]] = []

operator_impl = partial(framework_operator_impl, TORCH_REALIZABLE_OPS, ALL_TORCH_OPS)

# forward_fn:  forward


@operator_impl(Constant)
def forward_fn(op: Constant):
    data = torch.randn(op.abs_tensor.shape).to(op.abs_tensor.dtype.torch())
    return lambda: torch.nn.parameter.Parameter(
        data, requires_grad=data.is_floating_point()
    )


@operator_impl(ReLU)
def forward_fn(op: ReLU):
    return torch.nn.ReLU()


@operator_impl(GELU)
def forward_fn(op: GELU):
    return torch.nn.GELU()


@operator_impl(LeakyReLU)
def forward_fn(op: LeakyReLU):
    return torch.nn.LeakyReLU(op.negative_slope)


@operator_impl(PReLU)
def forward_fn(op: PReLU):
    return torch.nn.PReLU()


@operator_impl(Sigmoid)
def forward_fn(op: Sigmoid):
    return torch.nn.Sigmoid()


@operator_impl(Sin)
def forward_fn(op: Sin):
    return torch.sin


@operator_impl(Cos)
def forward_fn(op: Cos):
    return torch.cos


@operator_impl(Asin)
def forward_fn(op: Asin):
    return torch.asin


@operator_impl(Acos)
def forward_fn(op: Acos):
    return torch.acos


@operator_impl(Tan)
def forward_fn(op: Tan):
    return torch.tan


@operator_impl(Atan)
def forward_fn(op: Atan):
    return torch.atan


# Abs
@operator_impl(Abs)
def forward_fn(op: Abs):
    return torch.abs


@operator_impl(Where)
def forward_fn(op: Where):
    return torch.where


@operator_impl(Add)
def forward_fn(op: Add):
    return torch.add


@operator_impl(Sub)
def forward_fn(op: Sub):
    return torch.sub


@operator_impl(Mul)
def forward_fn(op: Mul):
    return torch.mul


@operator_impl(Div)
def forward_fn(op: Div):
    return lambda up, down: torch.div(
        up,
        down,
        rounding_mode="floor" if DType.from_torch(up.dtype) in DTYPE_INTS else None,
    )


@operator_impl(Max)
def forward_fn(op: Max):
    return torch.max


@operator_impl(Min)
def forward_fn(op: Min):
    return torch.min


@operator_impl(Equal)
def forward_fn(op: Equal):
    return torch.eq


@operator_impl(Greater)
def forward_fn(op: Greater):
    return torch.gt


@operator_impl(Less)
def forward_fn(op: Less):
    return torch.lt


@operator_impl(And)
def forward_fn(op: And):
    return torch.logical_and


@operator_impl(Or)
def forward_fn(op: Or):
    return torch.logical_or


@operator_impl(Xor)
def forward_fn(op: Xor):
    return torch.logical_xor


@operator_impl(Pow)
def forward_fn(op: Pow):
    return torch.pow


# Floor
@operator_impl(Floor)
def forward_fn(op: Floor):
    return torch.floor


# Ceil
@operator_impl(Ceil)
def forward_fn(op: Ceil):
    return torch.ceil


@operator_impl(Clip)
def forward_fn(op: Clip):
    if op.input_like[0].dtype in DTYPE_FLOATS:
        return lambda x: torch.clip(x, -1.5, 1.5)
    else:
        return lambda x: torch.clip(x, -1, 1)


@operator_impl(Round)
def forward_fn(op: Round):
    return torch.round


@operator_impl(Sqrt)
def forward_fn(op: Sqrt):
    return torch.sqrt


@operator_impl(Log2)
def forward_fn(op: Log2):
    return torch.log2


@operator_impl(Neg)
def forward_fn(op: Neg):
    return torch.neg


@operator_impl(Softmax)
def forward_fn(op: Softmax):
    return torch.nn.Softmax(dim=op.dim)


@operator_impl(MaxPool2d)
def forward_fn(op: MaxPool2d):
    return torch.nn.MaxPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@operator_impl(AvgPool2d)
def forward_fn(op: AvgPool2d):
    return torch.nn.AvgPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@operator_impl(Slice)
def forward_fn(op: Slice):
    reg = op.extra_attrs["region"]

    def _func(x):
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

    return _func


@operator_impl(Pad)
def forward_fn(op: Pad):
    if op.extra_attrs["type"] == "constant":
        # 0 easily cause division by zero...
        # 1 easily cause false positives (sqrt(1) = 0.99999... != 1 in ORT, so floor(sqrt(1))=0)
        return lambda x: torch.nn.functional.pad(
            x, op.padding_list, "constant", value=0.5
        )
    elif op.extra_attrs["type"] == "replicate" or op.extra_attrs["type"] == "reflect":
        return lambda x: torch.nn.functional.pad(
            x, op.padding_list, op.extra_attrs["type"]
        )


@operator_impl(Expand)
def forward_fn(op: Expand):
    return lambda x: x.expand(*op.type_transfer([AbsTensor.from_torch(x)])[0].shape)


@operator_impl(BatchNorm2d)
def forward_fn(op: BatchNorm2d):
    return torch.nn.BatchNorm2d(num_features=op.nfeat)


@operator_impl(Conv1d)
def forward_fn(op: Conv1d):
    return torch.nn.Conv1d(
        in_channels=op.in_channels,
        out_channels=op.out_channels,
        kernel_size=op.kernel_size,
        stride=op.stride,
        padding=op.padding,
        dilation=op.dilation,
    )


@operator_impl(NCHWConv2d)
def forward_fn(op: NCHWConv2d):
    return torch.nn.Conv2d(
        op.in_channels,
        op.out_channels,
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
        dilation=(op.dilation_h, op.dilation_w),
    )


@operator_impl(Reshape)
def forward_fn(op: Reshape):
    return lambda x: x.reshape(*op.target_shape)


@operator_impl(Flatten)
def forward_fn(op: Flatten):
    return lambda x: x.flatten()


@operator_impl(Transpose)
def forward_fn(op: Transpose):
    def f(x: torch.Tensor):
        dim0, dim1 = op._init_swap_dims(list(x.shape))
        return x.transpose(dim0, dim1)

    return f


# NearestInterp
@operator_impl(NearestInterp)
def forward_fn(op: NearestInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="nearest")


# LinearInterp
@operator_impl(LinearInterp)
def forward_fn(op: LinearInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="linear")


# BilinearInterp
@operator_impl(BilinearInterp)
def forward_fn(op: BilinearInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="bilinear")


@operator_impl(BicubicInterp)
def forward_fn(op: BicubicInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="bicubic")


# TrilinearInterp
@operator_impl(TrilinearInterp)
def forward_fn(op: TrilinearInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="trilinear")


@operator_impl(Squeeze)
def forward_fn(op: Squeeze):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.squeeze(op.extra_attrs["reduce_dim"])
    else:
        return lambda x: x.squeeze()


@operator_impl(TorchReduceSum)
def forward_fn(op: TorchReduceSum):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.sum(op.extra_attrs["reduce_dim"])
    return lambda x: x.sum()


# ReduceMin
@operator_impl(ReduceMin)
def forward_fn(op: ReduceMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.min(op.extra_attrs["reduce_dim"]).values
    return lambda x: x.min()


# ReduceMax
@operator_impl(ReduceMax)
def forward_fn(op: ReduceMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.max(op.extra_attrs["reduce_dim"]).values
    return lambda x: x.max()


# ReduceMean
@operator_impl(ReduceMean)
def forward_fn(op: ReduceMean):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.mean(op.extra_attrs["reduce_dim"])
    return lambda x: x.mean()


# ArgMin
@operator_impl(ArgMin)
def forward_fn(op: ArgMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.argmin(op.extra_attrs["reduce_dim"])
    return lambda x: x.argmin()


# ArgMax
@operator_impl(ArgMax)
def forward_fn(op: ArgMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.argmax(op.extra_attrs["reduce_dim"])
    return lambda x: x.argmax()


# Tril
@operator_impl(Tril)
def forward_fn(op: Tril):
    return lambda x: x.tril(op.diagonal)


# Triu
@operator_impl(Triu)
def forward_fn(op: Triu):
    return lambda x: x.triu(op.diagonal)


# Linear
@operator_impl(Linear)
def forward_fn(op: Linear):
    return torch.nn.Linear(in_features=op.ifeat, out_features=op.ofeat)


@operator_impl(Concat)
def forward_fn(op: Concat):
    axis = op.extra_attrs["axis"]
    return lambda *args: torch.cat(args, dim=axis)


@operator_impl(Cast)
def forward_fn(op: Cast):
    return lambda x: x.to(dtype=op.extra_attrs["to"].torch())


@operator_impl(MatMul)
def forward_fn(op: MatMul):
    return torch.matmul
