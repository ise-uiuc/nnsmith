from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import torch
import z3
# Recommended resources: https://theory.stanford.edu/~nikolaj/programmingz3.html
# Another plausible tool (Interval Analysis): https://simon-rohou.fr/research/tubex-lib/doc/toctree.html
# Please follow the PyTorch API conventions: https://pytorch.org/docs/stable/nn.html

# There are 3 types of constraints at this point:
# 1. Shape variables must be greater than 0;
# 2. Shape variable must avoid devision by 0;
# 3. Extra constraints introduced by individual operators;


class ShapeVar:
    def __init__(self, shape: List[Union[int, z3.ArithRef]]):
        self.shape = shape

    def __repr__(self):
        return str(self.shape)

    def gt_zero(self):
        return [s > 0 for s in self.shape if isinstance(s, z3.ArithRef)]

    def torch(self):
        # NOTE: Only for concrete shapes.
        return torch.Size(self.shape)


class AbsOpBase(ABC):
    @abstractmethod
    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        raise NotImplementedError

    def extra_constraints(self):
        return []


class Input(AbsOpBase):
    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        return [input_shapes[0]]


class ReLU(AbsOpBase):
    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        return [input_shapes[0]]


class Add(AbsOpBase):
    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 2
        assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
        return [input_shapes[0]]


class Expand(AbsOpBase):
    def __init__(self, target_shape: List[int]):
        """See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        """
        self.target_shape = target_shape

    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        shape_var = ShapeVar([])
        for i, v in enumerate(self.target_shape):
            if v == -1:
                auto_dim = i - (len(self.target_shape) -
                                len(input_shapes[0].shape))
                assert auto_dim >= 0
                shape_var.shape.append(input_shapes[0].shape[auto_dim])
            else:
                shape_var.shape.append(v)
        return [shape_var]


class NCHWConv2d(AbsOpBase):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int):
        """See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        assert len(input_shapes[0].shape) == 4  # NCHW please.
        assert input_shapes[0].shape[1] == self.in_channels
        shape_var = ShapeVar([])
        # Batch dim: just copy
        shape_var.shape.append(input_shapes[0].shape[0])
        shape_var.shape.append(self.out_channels)        # Output channels
        shape_var.shape.append(
            (input_shapes[0].shape[2] - self.kernel_size[0] + 2 * self.padding) // self.stride + 1)
        shape_var.shape.append(
            (input_shapes[0].shape[3] - self.kernel_size[0] + 2 * self.padding) // self.stride + 1)
        return [shape_var]


class Reshape(AbsOpBase):
    def __init__(self, target_shape: List[int]):
        """See https://pytorch.org/docs/stable/generated/torch.reshape.html
        """
        self.target_shape = target_shape

    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        if -1 not in self.target_shape:
            return [ShapeVar(self.target_shape)]
        # else
        shape_var = ShapeVar(self.target_shape)
        auto_dim = -1
        accum = 1
        for i, v in enumerate(self.target_shape):
            if v == -1:
                if auto_dim != -1:
                    raise ValueError(
                        "Only one auto-dim is allowed! "
                        "See https://pytorch.org/docs/stable/generated/torch.reshape.html")
                auto_dim = i
            else:
                accum *= v
        # print(input_shapes[0].shape)
        # print(accum, reduce(lambda x, y: x * y, input_shapes[0].shape))
        # First see if there's any symbols in the expression
        symbol_indices = []
        for v in input_shapes[0].shape:
            if isinstance(v, z3.ArithRef):
                symbol_indices.append(i)
        if len(symbol_indices) == 0:
            shape_var.shape[auto_dim] = reduce(
                lambda x, y: x * y, input_shapes[0].shape) // accum
        else:
            shape_var.shape[auto_dim] = reduce(
                lambda x, y: x * y, input_shapes[0].shape) / accum

        return [shape_var]


class Transpose(AbsOpBase):
    def __init__(self, dim0: int, dim1: int):
        """See https://pytorch.org/docs/stable/generated/torch.transpose.html
        """
        self.dim0 = dim0
        self.dim1 = dim1

    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        assert len(input_shapes[0].shape) >= max(self.dim0, self.dim1) + 1
        shape_var = input_shapes[0]
        shape_var.shape[self.dim0], shape_var.shape[self.dim1] = shape_var.shape[self.dim1], shape_var.shape[self.dim0]
        return [shape_var]


if __name__ == '__main__':
    # Test shape functions

    # ReLU
    lhs = torch.relu(torch.randn(1, 1, 1, 1)).shape
    rhs = torch.Size(ReLU().shape_function([ShapeVar([1, 1, 1, 1])])[0].shape)
    assert lhs == rhs, f"{lhs} != {rhs}"

    # Add
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(2, 3, 4, 5)
    c = a + b
    assert c.shape == torch.Size(Add().shape_function(
        [ShapeVar([2, 3, 4, 5]), ShapeVar([2, 3, 4, 5])])[0].shape)

    # Expand
    a = torch.randn(4, 5)
    assert a.expand(2, 3, 4, 5).shape == torch.Size(Expand(
        [2, 3, 4, 5]).shape_function([ShapeVar([4, 5])])[0].shape)
    lhs = a.expand(1, -1, 5).shape
    rhs = Expand([1, -1, 5]).shape_function([ShapeVar([4, 5])])[0].torch()
    assert lhs == rhs, f"{lhs} != {rhs}"

    # NCHWConv2d
    a = torch.randn(2, 3, 24, 24)
    assert torch.conv2d(a, torch.randn(3, 3, 3, 3), stride=1, padding=1).shape == NCHWConv2d(
        3, 3, (3, 3), 1, 1).shape_function([ShapeVar([2, 3, 24, 24])])[0].torch()

    # Reshape
    a = torch.randn(2, 3, 4, 5)
    assert a.reshape(1, -1, 5).shape == Reshape([1, -1, 5]).shape_function(
        [ShapeVar([2, 3, 4, 5])])[0].torch()
    print(Reshape(
        [1, -1, 5]).shape_function([ShapeVar([2, z3.Int('x'), 4, 5])])[0])

    # Transpose
    assert a.transpose(0, 3).shape == Transpose(0, 3).shape_function(
        [ShapeVar([2, 3, 4, 5])])[0].torch()
