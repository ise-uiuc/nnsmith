import torch


def smoothed_relu(x):
    # return torch.where(x < 0, torch.exp(x) - 1, x)
    return torch.nn.LeakyReLU(0.1)(x)


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
