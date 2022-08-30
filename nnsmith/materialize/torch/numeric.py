import torch
from multipledispatch import dispatch

from nnsmith.abstract.op import *


@torch.jit.ignore
def numeric_valid(outputs) -> bool:
    with torch.no_grad():
        return all([torch.isfinite(out).all() for out in outputs])


# generalized loss fn
def smoothed_relu(x):
    return torch.relu(x)


def loss_ge_zero(x):
    return smoothed_relu(-x)


def loss_le_zero(x):
    return smoothed_relu(x)


def loss_lt_zero(x):
    return loss_le(x, -1e-10)


def loss_gt_zero(x):
    return loss_ge(x, 1e-10)


def loss_ge(x, y):
    return loss_ge_zero(x - y)


def loss_le(x, y):
    return loss_le_zero(x - y)


def loss_gt(x, y):
    return loss_gt_zero(x - y)


def loss_lt(x, y):
    return loss_lt_zero(x - y)


# loss_fn:     backward


@dispatch(Div)
def loss_fn(op: Div):
    return lambda op, y: loss_gt_zero(torch.abs(y))


@dispatch(Pow)
def loss_fn(op: Pow):
    def torch_loss(a, b):
        # a >= 0 && b*log(a) <= 20
        l0 = loss_gt_zero(a)
        if torch.any(l0 > 0):
            return ("l0", l0)
        l1 = loss_le(
            b * torch.log(torch.maximum(a, torch.tensor(1e-40, dtype=a.dtype))), 40
        )
        return ("l1", l1)

    return torch_loss


@dispatch(Acos)
def loss_fn(op: Acos):
    return lambda x: loss_le(x.abs(), 1)


@dispatch(Sqrt)
def loss_fn(op: Sqrt):
    return lambda x: loss_ge(x, 0)


@dispatch(Asin)
def loss_fn(op: Asin):
    return lambda x: loss_le(x.abs(), 1)


@dispatch(Log2)
def loss_fn(op: Log2):
    return lambda x: loss_gt_zero(x)
