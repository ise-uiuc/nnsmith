from multipledispatch import dispatch

import torch

from nnsmith.abstract.op import *

# Implementation of operators.
class Dummy:  # For simple syntactic checking;
    pass


# forward_fn:  forward
@dispatch(Dummy)
def forward_fn(op):
    pass


class StopFoldConst(torch.nn.Module):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.dtype = data.dtype
        self.param = torch.nn.parameter.Parameter(
            data, requires_grad=data.is_floating_point()
        )

    @torch.no_grad()
    def forward(self):
        return self.param.to(dtype=self.dtype)


@dispatch(Constant)
def forward_fn(op: Constant):
    data = torch.randn(op.abs_tensor.shape).to(op.abs_tensor.dtype.torch())
    return StopFoldConst(data)


@dispatch(ReLU)
def forward_fn(op: ReLU):
    return torch.nn.ReLU()


@dispatch(GELU)
def forward_fn(op: GELU):
    return torch.nn.GELU()


@dispatch(LeakyReLU)
def forward_fn(op: LeakyReLU):
    return torch.nn.LeakyReLU(op.negative_slope)


@dispatch(PReLU)
def forward_fn(op: PReLU):
    return torch.nn.PReLU()


@dispatch(Sigmoid)
def forward_fn(op: Sigmoid):
    return torch.nn.Sigmoid()


@dispatch(Sin)
def forward_fn(op: Sin):
    return torch.sin


@dispatch(Cos)
def forward_fn(op: Cos):
    return torch.cos


@dispatch(Asin)
def forward_fn(op: Asin):
    return torch.asin


@dispatch(Acos)
def forward_fn(op: Acos):
    return torch.acos


@dispatch(Tan)
def forward_fn(op: Tan):
    return torch.tan


@dispatch(Atan)
def forward_fn(op: Atan):
    return torch.atan


# Abs
@dispatch(Abs)
def forward_fn(op: Abs):
    return torch.abs


@dispatch(Where)
def forward_fn(op: Where):
    return torch.where


@dispatch(Add)
def forward_fn(op: Add):
    return torch.add


@dispatch(Sub)
def forward_fn(op: Sub):
    return torch.sub


@dispatch(Mul)
def forward_fn(op: Mul):
    return torch.mul


@dispatch(Div)
def forward_fn(op: Div):
    return lambda up, down: torch.div(
        up,
        down,
        rounding_mode="floor" if DType.from_torch(up.dtype) in DTYPE_INTS else None,
    )


@dispatch(Max)
def forward_fn(op: Max):
    return torch.max


@dispatch(Min)
def forward_fn(op: Min):
    return torch.min


@dispatch(Equal)
def forward_fn(op: Equal):
    return torch.eq


@dispatch(Greater)
def forward_fn(op: Greater):
    return torch.gt


@dispatch(Less)
def forward_fn(op: Less):
    return torch.lt


@dispatch(And)
def forward_fn(op: And):
    return torch.logical_and


@dispatch(Or)
def forward_fn(op: Or):
    return torch.logical_or


@dispatch(Xor)
def forward_fn(op: Xor):
    return torch.logical_xor


@dispatch(Pow)
def forward_fn(op: Pow):
    return torch.pow


# Floor
@dispatch(Floor)
def forward_fn(op: Floor):
    return torch.floor


# Ceil
@dispatch(Ceil)
def forward_fn(op: Ceil):
    return torch.ceil


@dispatch(Clip)
def forward_fn(op: Clip):
    if op.input_like[0].dtype in DTYPE_FLOATS:
        return lambda x: torch.clip(x, -1.5, 1.5)
    else:
        return lambda x: torch.clip(x, -1, 1)


@dispatch(Round)
def forward_fn(op: Round):
    return torch.round


@dispatch(Sqrt)
def forward_fn(op: Sqrt):
    return torch.sqrt


@dispatch(Log2)
def forward_fn(op: Log2):
    return torch.log2


@dispatch(Neg)
def forward_fn(op: Neg):
    return torch.neg


@dispatch(Softmax)
def forward_fn(op: Softmax):
    return torch.nn.Softmax(dim=op.dim)


@dispatch(MaxPool2d)
def forward_fn(op: MaxPool2d):
    return torch.nn.MaxPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@dispatch(AvgPool2d)
def forward_fn(op: AvgPool2d):
    return torch.nn.AvgPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@dispatch(Slice)
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


@dispatch(Pad)
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


@dispatch(Expand)
def forward_fn(op: Expand):
    return lambda x: x.expand(*op.type_transfer([AbsTensor.from_torch(x)])[0].shape)


@dispatch(BatchNorm2d)
def forward_fn(op: BatchNorm2d):
    return torch.nn.BatchNorm2d(num_features=op.nfeat)


@dispatch(Conv1d)
def forward_fn(op: Conv1d):
    return torch.nn.Conv1d(
        in_channels=op.in_channels,
        out_channels=op.out_channels,
        kernel_size=op.kernel_size,
        stride=op.stride,
        padding=op.padding,
        dilation=op.dilation,
    )


@dispatch(NCHWConv2d)
def forward_fn(op: NCHWConv2d):
    return torch.nn.Conv2d(
        op.in_channels,
        op.out_channels,
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@dispatch(Reshape)
def forward_fn(op: Reshape):
    return lambda x: x.reshape(*op.target_shape)


@dispatch(Flatten)
def forward_fn(op: Flatten):
    return lambda x: x.flatten()


@dispatch(Transpose)
def forward_fn(op: Transpose):
    def f(x: torch.Tensor):
        dim0, dim1 = op._init_swap_dims(list(x.shape))
        return x.transpose(dim0, dim1)

    return f


# NearestInterp
@dispatch(NearestInterp)
def forward_fn(op: NearestInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="nearest")


# LinearInterp
@dispatch(LinearInterp)
def forward_fn(op: LinearInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="linear")


# BilinearInterp
@dispatch(BilinearInterp)
def forward_fn(op: BilinearInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="bilinear")


@dispatch(BicubicInterp)
def forward_fn(op: BicubicInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="bicubic")


# TrilinearInterp
@dispatch(TrilinearInterp)
def forward_fn(op: TrilinearInterp):
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="trilinear")


@dispatch(Squeeze)
def forward_fn(op: Squeeze):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.squeeze(op.extra_attrs["reduce_dim"])
    else:
        return lambda x: x.squeeze()


@dispatch(ReduceSum)
def forward_fn(op: ReduceSum):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.sum(op.extra_attrs["reduce_dim"])
    return lambda x: x.sum()


# ReduceMin
@dispatch(ReduceMin)
def forward_fn(op: ReduceMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.min(op.extra_attrs["reduce_dim"]).values
    return lambda x: x.min()


# ReduceMax
@dispatch(ReduceMax)
def forward_fn(op: ReduceMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.max(op.extra_attrs["reduce_dim"]).values
    return lambda x: x.max()


# ReduceMean
@dispatch(ReduceMean)
def forward_fn(op: ReduceMean):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.mean(op.extra_attrs["reduce_dim"])
    return lambda x: x.mean()


# ArgMin
@dispatch(ArgMin)
def forward_fn(op: ArgMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.argmin(op.extra_attrs["reduce_dim"])
    return lambda x: x.argmin()


# ArgMax
@dispatch(ArgMax)
def forward_fn(op: ArgMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: x.argmax(op.extra_attrs["reduce_dim"])
    return lambda x: x.argmax()


# Tril
@dispatch(Tril)
def forward_fn(op: Tril):
    return lambda x: x.tril(op.diagonal)


# Triu
@dispatch(Triu)
def forward_fn(op: Triu):
    return lambda x: x.triu(op.diagonal)


# Linear
@dispatch(Linear)
def forward_fn(op: Linear):
    return torch.nn.Linear(in_features=op.ifeat, out_features=op.ofeat)


@dispatch(Concat)
def forward_fn(op: Concat):
    axis = op.extra_attrs["axis"]
    return lambda *args: torch.cat(args, dim=axis)


@dispatch(Cast)
def forward_fn(op: Cast):
    return lambda x: x.to(dtype=op.extra_attrs["to"].torch())


# Gemm
@dispatch(Gemm)
def forward_fn(op: Gemm):
    extra_attrs = op._set_or_get_extra_attrs()
    return lambda *args: torch.addmm(
        *args, beta=extra_attrs["beta"], alpha=extra_attrs["alpha"]
    )
