import torch
import os


def smoothed_relu(x):
    lf = os.getenv('NNSMITH_LOSS', 'v1')
    if lf == 'v1':
        return torch.nn.ReLU()(x)
    elif lf == 'v2':
        mask = x < 0
        a = torch.exp(torch.minimum(x, torch.zeros_like(x))) - 1
        return torch.where(mask, a, x)
    else:
        raise ValueError('Unknown loss function: {}'.format(lf))


def loss_ge_zero(x):
    return smoothed_relu(-x)


def loss_le_zero(x):
    return smoothed_relu(x)


loss_lt_zero = loss_le_zero
loss_gt_zero = loss_ge_zero


def loss_ge(x, y):
    return loss_ge_zero(x - y)


def loss_le(x, y):
    return loss_le_zero(x - y)


def loss_gt(x, y):
    return loss_gt_zero(x - y)


def loss_lt(x, y):
    return loss_lt_zero(x - y)
