from abc import ABC, abstractmethod
from functools import reduce
from sys import path
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

# TODO: Make operator's parameters symbolic.
# FIXME: Z3 solving is way slower than numerical computing. Try to use exceptions to reject invalid inputs;


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

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ArithRef) for s in self.shape)


def check_shape_fn(func):
    def wrapper_check_shape_fn(self, input_shapes):
        assert len(input_shapes) == len(self.inp_dims), "Requires {} inputs, but got {}".format(
            len(self.inp_dims), len(input_shapes))
        res = func(self, input_shapes)
        assert len(res) == len(self.out_dims), "Requires {} outputs, but got {}".format(
            len(self.out_dims), len(res))
        return res
    return wrapper_check_shape_fn


def check_require_fn(func):
    def wrapper_check_require_fn(self, input_shapes):
        assert len(input_shapes) == len(self.inp_dims), "Requires {} inputs, but got {}".format(
            len(self.inp_dims), len(input_shapes))
        return func(self, input_shapes)
    return wrapper_check_require_fn


class AbsOpBase(ABC):
    # `[3, 3]` this means this op requires 2 inputs. Where the 1st one has 2 dimensions, and the 2nd one has 3 dimensions.
    # `-1` means arbitrary dimantions.
    inp_dims = []
    # NOTE: the concrete values of out_dims are not useful. Just make sure the length is correct.
    out_dims = []

    @abstractmethod  # Overload me!
    # Exception means rejection.
    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        raise NotImplementedError

    @check_shape_fn  # Public API.
    def shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        return self._shape_function(input_shapes)

    # Overload me!
    # Extra constraints for the input tensors.
    # Exception means rejection.
    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return []

    @check_require_fn  # Public API.
    def requires(self, input_shapes):
        return self._requires(input_shapes)


class UnaryOpBase(AbsOpBase):
    out_dims = [-1]


class BinaryOpBase(AbsOpBase):
    out_dims = [-1]


class ElementWiseUnaryOp(UnaryOpBase):
    inp_dims = [-1]
    out_dims = [-1]

    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        return [input_shapes[0]]


class Input(ElementWiseUnaryOp):
    pass


class ReLU(ElementWiseUnaryOp):
    pass


class LeakyReLU(ElementWiseUnaryOp):
    """See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """

    def __init__(self, negative_slope=0.01) -> None:
        self.negative_slope = negative_slope


class PReLU(ElementWiseUnaryOp):
    pass


class Sigmoid(ElementWiseUnaryOp):
    pass


class Sin(ElementWiseUnaryOp):
    pass


class Cos(ElementWiseUnaryOp):
    pass


class Asin(ElementWiseUnaryOp):
    pass


class Acos(ElementWiseUnaryOp):
    pass


class Tan(ElementWiseUnaryOp):
    pass


class Atan(ElementWiseUnaryOp):
    pass


class Abs(ElementWiseUnaryOp):
    pass


class Ceil(ElementWiseUnaryOp):
    pass


class Clip(ElementWiseUnaryOp):
    pass


class Round(ElementWiseUnaryOp):
    pass


class Sqrt(ElementWiseUnaryOp):
    pass


class Log(ElementWiseUnaryOp):
    def __init__(self, base):
        if not isinstance(base, z3.ArithRef):
            assert base > 0
        self.base = base


class Not(ElementWiseUnaryOp):
    pass


class Add(BinaryOpBase):
    inp_dims = [-1, -1]

    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
        return [input_shapes[0]]

    def _requires(self, input_shapes):
        assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
        ret = []
        for l, r in zip(input_shapes[0].shape, input_shapes[1].shape):
            if isinstance(l, z3.ArithRef) or isinstance(r, z3.ArithRef):
                ret.append(l == r)
            else:
                assert l == r
        return ret


class Expand(UnaryOpBase):
    inp_dims = [-1]

    def __init__(self, target_shape: List[int]):
        """See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        """
        self.target_shape = target_shape

    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
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


class NCHWConv2d(UnaryOpBase):
    inp_dims = [4]  # NCHW

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

    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes[0].shape) == 4  # NCHW please.
        # not symbolic
        if not isinstance(self.in_channels, z3.ArithRef) and not isinstance(input_shapes[0].shape[1], z3.ArithRef):
            assert input_shapes[0].shape[1] == self.in_channels

        is_symbolic_inp = input_shapes[0].constains_symbol() or isinstance(
            self.kernel_size, z3.ArithRef) or isinstance(self.stride, z3.ArithRef) or isinstance(self.padding, z3.ArithRef)

        shape_var = ShapeVar([])
        # Batch dim: just copy
        shape_var.shape.append(input_shapes[0].shape[0])
        shape_var.shape.append(self.out_channels)        # Output channels
        if not is_symbolic_inp:
            shape_var.shape.append(
                (input_shapes[0].shape[2] - self.kernel_size[0] + 2 * self.padding) // self.stride + 1)
            shape_var.shape.append(
                (input_shapes[0].shape[3] - self.kernel_size[0] + 2 * self.padding) // self.stride + 1)
        else:
            shape_var.shape.append(
                (input_shapes[0].shape[2] - self.kernel_size[0] + 2 * self.padding) / self.stride + 1)
            shape_var.shape.append(
                (input_shapes[0].shape[3] - self.kernel_size[0] + 2 * self.padding) / self.stride + 1)
        return [shape_var]


class Reshape(UnaryOpBase):
    inp_dims = [-1]

    def __init__(self, target_shape: List[int]):
        """See https://pytorch.org/docs/stable/generated/torch.reshape.html
        """
        self.target_shape = target_shape

    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
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

    def _requires(self, input_shapes):
        # If your target shape is concrete, then your output shape's total pixels must be the same as the input shape's.
        if -1 not in self.target_shape:
            total_pixels = reduce(lambda x, y: x * y, self.target_shape)
            return [total_pixels == reduce(lambda x, y: x * y, input_shapes[0].shape)]
        else:
            # If you use auto mode (specifying -1 for some dimensions), then the total number of input pixels must be exactly divisible by that of the output shape.
            minimul_pixels = reduce(
                lambda x, y: x * y, [v for v in self.target_shape if v != -1])
            return [minimul_pixels % reduce(lambda x, y: x * y, input_shapes[0].shape) == 0]


class Transpose(UnaryOpBase):
    inp_dims = [-1]

    def __init__(self, dim0: int, dim1: int):
        """See https://pytorch.org/docs/stable/generated/torch.transpose.html
        """
        self.dim0 = dim0
        self.dim1 = dim1

    def _shape_function(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
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
    print(NCHWConv2d(
        3, 3, (3, 3), 1, 1).shape_function([ShapeVar([2, *z3.Ints('c h w')])])[0])

    # Reshape
    a = torch.randn(2, 3, 4, 5)
    assert a.reshape(1, -1, 5).shape == Reshape([1, -1, 5]).shape_function(
        [ShapeVar([2, 3, 4, 5])])[0].torch()
    print(Reshape(
        [1, -1, 5]).shape_function([ShapeVar([2, z3.Int('x'), 4, 5])])[0])

    # Transpose
    assert a.transpose(0, 3).shape == Transpose(0, 3).shape_function(
        [ShapeVar([2, 3, 4, 5])])[0].torch()
