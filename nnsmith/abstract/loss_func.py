import torch


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
