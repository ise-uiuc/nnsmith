import torch
from multipledispatch import dispatch

from nnsmith.abstract.op import *

# used proxy gradient functions
SLOPE = 0.1


class PGTruncFuncBase(torch.autograd.Function):
    # incomplete class
    @staticmethod
    def backward(ctx, grad_output):
        # let f' = x
        return grad_output


class PGFloorFunc(PGTruncFuncBase):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)


class PGRoundFunc(PGTruncFuncBase):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)


class PGCeilFunc(PGTruncFuncBase):
    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input)


class PGReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # let f' = l * x in bound
        # let f' = x out bound
        grad_input = grad_output.clone()
        return torch.where(input > 0, grad_input, grad_input * SLOPE)


class PGClipFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.save_for_backward(input)
        ctx.clip_min = min
        ctx.clip_max = max
        return torch.clip(input, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # let f' = l * x in bound
        # let f' = x out bound
        grad_input = grad_output.clone()
        return (
            torch.where(
                (input > ctx.clip_max).logical_or(input < ctx.clip_min),
                grad_input * SLOPE,
                grad_input,
            ),
            None,
            None,
        )


# Modules.


class PGCeil(torch.nn.Module):
    def __init__(self):
        super(PGCeil, self).__init__()

    def forward(self, x):
        return PGCeilFunc.apply(x)


class PGFloor(torch.nn.Module):
    def __init__(self):
        super(PGFloor, self).__init__()

    def forward(self, x):
        return PGFloorFunc.apply(x)


class PGRound(torch.nn.Module):
    def __init__(self):
        super(PGRound, self).__init__()

    def forward(self, x):
        return PGRoundFunc.apply(x)


class PGReLU(torch.nn.Module):
    def __init__(self):
        super(PGReLU, self).__init__()

    def forward(self, x):
        return PGReLUFunc.apply(x)


class PGClip(torch.nn.Module):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        super(PGClip, self).__init__()

    def forward(self, x):
        return PGClipFunc.apply(x, self.min, self.max)


# proxy_fn:    proxy


@dispatch(ReLU)
def proxy_fn(op: ReLU):
    return PGReLU()


@dispatch(Ceil)
def proxy_fn(op: Ceil):
    return PGCeil()


# PGFloor
@dispatch(Floor)
def proxy_fn(op: Floor):
    return PGFloor()


# PGClip
@dispatch(Clip)
def proxy_fn(op: Clip):
    if op.input_like[0].dtype in DTYPE_FLOATS:
        return PGClip(-1.5, 1.5)
    else:
        return PGClip(-1, 1)


# Round
@dispatch(Round)
def proxy_fn(op: Round):
    return PGRound()
