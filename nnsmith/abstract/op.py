from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union, Callable
from inspect import signature
import random

# Import z3 ahead of torch (See https://github.com/Z3Prover/z3/issues/5656)
import z3
import torch
# Recommended resources: https://theory.stanford.edu/~nikolaj/programmingz3.html
# Another plausible tool (Interval Analysis): https://simon-rohou.fr/research/tubex-lib/doc/toctree.html
# Please follow the PyTorch API conventions: https://pytorch.org/docs/stable/nn.html

# There are following types of constraints at this point:
# 1. Shape variables must be greater than 0;
# 2. Shape variables must avoid devision by 0;
# 3. Intra-input shape constraints; e.g., add(x, y) where x.shape() must be equal to y.shape();
# 4. Extra constraints introduced by individual operators;

# FIXME: Z3 solving is way slower than numerical computing. Try to use exceptions to reject invalid inputs;
# TODO: add interval analysis for shape dimension size;


class ShapeVar:
    def __init__(self, shape: List[Union[int, z3.ArithRef]]):
        self.shape = list(shape)

    def __repr__(self):
        return str(self.shape)

    def gt_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ArithRef):
                ret.append(s > 0)
            else:
                assert s > 0
        return ret

    def torch(self):
        # NOTE: Only for concrete shapes.
        return torch.Size(self.shape)

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ArithRef) for s in self.shape)

    @staticmethod
    def from_torch(torch_shape):
        return ShapeVar([torch_shape[i] for i in range(len(torch_shape))])


def check_shape_fn(func):
    def wrapper_check_shape_fn(self, input_shapes):
        assert self.inp_dims, "Empty input dimensions in {}".format(
            self.__class__.__name__)
        assert self.out_dims, "Empty output dimensions in {}".format(
            self.__class__.__name__)
        assert len(input_shapes) == len(self.inp_dims), "{} requires {} inputs, but got {}".format(
            self.__class__.__name__,
            len(self.inp_dims), len(input_shapes))
        res = func(self, input_shapes)
        assert len(res) == len(self.out_dims), "{} requires {} outputs, but got {}".format(
            self.__class__.__name__,
            len(self.out_dims), len(res))
        return res
    return wrapper_check_shape_fn


def check_require_fn(func):
    def wrapper_check_require_fn(self, input_shapes):
        assert len(input_shapes) == len(self.inp_dims), "{} requires {} inputs, but got {}".format(
            self.__class__.__name__,
            len(self.inp_dims), len(input_shapes))
        return func(self, input_shapes)
    return wrapper_check_require_fn


class AbsOpBase(ABC):
    def __init__(self):
        # `[3, 3]` this means this op requires 2 inputs. Where the 1st one has 2 dimensions, and the 2nd one has 3 dimensions.
        # `-1` means arbitrary dimantions; NOTE: but should be concretized during execution.
        # All symbols of correponding operator must be the constructor's parameters.
        self.inp_dims = []
        # NOTE: the concrete values of out_dims are not useful. Just make sure the length is correct.
        # NOTE: the output shape of input dimensions should be concretized during the execution.
        self.out_dims = []
        # Require the input dimension sizes to be equivalent.
        self.same_inp_dims = False
        # NOTE: the input of operator constructors are all Union[int, z3.ArithRef].
        self.extra_attrs = {}

    @abstractmethod  # Overload me!
    # Exception means rejection.
    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        raise NotImplementedError

    @check_shape_fn  # Public API.
    def shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        return self._shape_fn(input_shapes)

    # Overload me!
    # Extra constraints for the input tensors.
    # Exception means rejection.
    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return []

    @abstractmethod
    def torch(self) -> Callable[..., torch.Tensor]:
        raise NotImplementedError

    @check_require_fn  # Public API.
    def requires(self, input_shapes):
        return self._requires(input_shapes)

    def __repr__(self) -> str:
        return self.__class__.__name__


def concretize(op: AbsOpBase, model: z3.ModelRef) -> AbsOpBase:
    construct_param_dict = signature(op.__init__).parameters
    values = []
    symbolic_idx = []
    for idx, key in enumerate(construct_param_dict):
        param = getattr(op, key)
        values.append(param)
        if isinstance(param, z3.ArithRef):
            symbolic_idx.append(idx)
    for idx in symbolic_idx:
        values[idx] = model.eval(values[idx]).as_long()

    concrete_op = op.__class__(*values)
    concrete_op.inp_dims = op.inp_dims
    concrete_op.out_dims = op.out_dims
    concrete_op.same_inp_dims = op.same_inp_dims
    concrete_op.extra_attrs = op.extra_attrs

    return concrete_op


class UnaryOpBase(AbsOpBase):
    def __init__(self):
        super().__init__()
        self.out_dims = [-1]


class BinaryOpBase(AbsOpBase):
    def __init__(self):
        super().__init__()
        self.out_dims = [-1]


class ElementWiseUnaryOp(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.inp_dims = [-1]
        self.out_dims = [-1]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        return [input_shapes[0]]


class Input(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return lambda x: x


class ReLU(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.nn.ReLU()


class LeakyReLU(ElementWiseUnaryOp):
    def __init__(self):
        """See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        """
        super().__init__()
        self.negative_slope = 0.01

    def torch(self):
        return torch.nn.LeakyReLU(self.negative_slope)


class PReLU(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.nn.PReLU()


class Sigmoid(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.sigmoid


class Sin(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.sin


class Cos(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.cos


class Asin(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.asin


class Acos(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.acos


class Tan(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.tan


class Atan(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.atan


class Abs(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.abs


class Ceil(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.ceil


class Clip(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()
        self.min = -1
        self.max = 1

    def torch(self):
        return lambda x: torch.clip(x, self.min, self.max)


class Round(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.round


class Sqrt(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.sqrt


class Log2(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.log2


class Neg(ElementWiseUnaryOp):
    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.neg


class Add(BinaryOpBase):
    def __init__(self):
        super().__init__()
        self.inp_dims = [-1, -1]
        self.same_inp_dims = True

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
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

    def torch(self):
        return torch.add


class Expand(UnaryOpBase, ABC):
    # expand_dim cannot be symbolic. So just expand it.
    def __init__(self, expand_last_dim: int, expand_n: Union[int, z3.ArithRef]):
        """See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        """
        super().__init__()
        self.inp_dims = [-1]
        self.expand_last_dim = expand_last_dim
        self.expand_n = expand_n

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        if self.expand_last_dim <= len(input_shapes[0].shape):
            input_shapes[0].shape[-self.expand_last_dim] = self.expand_n
            return input_shapes
        else:  # expand it;
            # for example. we have:
            #       input shape [u, v]
            #       expand_last_dim <- 4
            #       return [expand_n, 1, u, v] where `1` is padded.
            return [ShapeVar([self.expand_n, *([1] * (self.expand_last_dim - len(input_shapes[0].shape) - 1)), *input_shapes[0].shape])]

    def _requires(self, input_shapes):
        assert self.expand_last_dim > 0

        input_shape = input_shapes[0].shape
        if isinstance(self.expand_n, z3.ArithRef):
            if self.expand_last_dim <= len(input_shape):  # index valid
                # TODO(JK) unittest for expand an original dim with size>1
                return [z3.Or(
                    z3.And(input_shape[-self.expand_last_dim]
                           == 1, self.expand_n >= 1),
                    z3.And(input_shape[-self.expand_last_dim]
                           == self.expand_n),
                    self.expand_n == -1),
                ]
        else:
            # It is also valid to expand to 0. But just too tricky...
            assert self.expand_n >= 1
        return []

    def torch(self):
        return lambda x: x.expand(*self.shape_fn([ShapeVar.from_torch(x.shape)])[0].shape)


class ExpandLast1(Expand):
    def __init__(self, expand_n: Union[int, z3.ArithRef]):
        super().__init__(expand_last_dim=1, expand_n=expand_n)


class ExpandLast2(Expand):
    def __init__(self, expand_n: Union[int, z3.ArithRef]):
        super().__init__(expand_last_dim=2, expand_n=expand_n)


class ExpandLast3(Expand):
    def __init__(self, expand_n: Union[int, z3.ArithRef]):
        super().__init__(expand_last_dim=3, expand_n=expand_n)


class ExpandLast4(Expand):
    def __init__(self, expand_n: Union[int, z3.ArithRef]):
        super().__init__(expand_last_dim=4, expand_n=expand_n)


class NCHWConv2d(UnaryOpBase):

    def __init__(self,
                 in_channels: Union[int, z3.ArithRef],
                 out_channels: Union[int, z3.ArithRef],
                 kernel_h_size: Union[int, z3.ArithRef],
                 kernel_w_size: Union[int, z3.ArithRef],
                 stride: Union[int, z3.ArithRef],
                 padding: Union[int, z3.ArithRef]):
        """See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.padding = padding

        self.inp_dims = [4]  # NCHW
        self.out_dims = [4]  # NCHW

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        # not symbolic
        if not isinstance(self.in_channels, z3.ArithRef) and not isinstance(input_shapes[0].shape[1], z3.ArithRef):
            assert input_shapes[0].shape[1] == self.in_channels

        is_symbolic_inp = input_shapes[0].constains_symbol() or isinstance(self.kernel_w_size, z3.ArithRef) or isinstance(
            self.kernel_h_size, z3.ArithRef) or isinstance(self.stride, z3.ArithRef) or isinstance(self.padding, z3.ArithRef)

        shape_var = ShapeVar([])
        # Batch dim: just copy
        shape_var.shape.append(input_shapes[0].shape[0])
        shape_var.shape.append(self.out_channels)        # Output channels
        if not is_symbolic_inp:
            shape_var.shape.append(
                (input_shapes[0].shape[2] - self.kernel_h_size + 2 * self.padding) // self.stride + 1)
            shape_var.shape.append(
                (input_shapes[0].shape[3] - self.kernel_w_size + 2 * self.padding) // self.stride + 1)
        else:
            shape_var.shape.append(
                (input_shapes[0].shape[2] - self.kernel_h_size + 2 * self.padding) / self.stride + 1)
            shape_var.shape.append(
                (input_shapes[0].shape[3] - self.kernel_w_size + 2 * self.padding) / self.stride + 1)
        return [shape_var]

    def _requires(self, input_shapes):
        cons = []
        ret = []
        # TODO: Use eager mode for debugging.
        cons.append(self.in_channels == input_shapes[0].shape[1])
        cons.append(self.out_channels >= 1)
        cons.append(self.kernel_h_size >= 1)
        cons.append(self.kernel_w_size >= 1)
        # TODO(JK): fix the dialation case for the kernel size constraints.
        cons.append(self.kernel_h_size < input_shapes[0].shape[2])
        cons.append(self.kernel_w_size < input_shapes[0].shape[3])
        cons.append(self.stride >= 1)
        cons.append(self.padding >= 0)
        for c in cons:
            if isinstance(c, z3.ExprRef):
                ret.append(c)
            else:
                assert c
        return ret

    def torch(self):
        return torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_h_size, self.kernel_w_size), stride=self.stride,
                               padding=self.padding)


class Reshape(UnaryOpBase, ABC):
    def __init__(self):
        super().__init__()
        self.inp_dims = [-1]
        self.target_shape: List[Union[int, z3.ArithRef]]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
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

    def torch(self):
        return lambda x: x.reshape(*self.target_shape)


# Expand 6 times.
class Reshape1D(Reshape):
    # Inputs are target shape.
    def __init__(self, dim0: Union[int, z3.ArithRef]):
        super().__init__()
        self.dim0 = dim0
        self.target_shape = [dim0]
        self.out_dims = [1]


class Reshape2D(Reshape):
    def __init__(self, dim0: Union[int, z3.ArithRef], dim1: Union[int, z3.ArithRef]):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.target_shape = [dim0, dim1]
        self.out_dims = [2]


class Reshape3D(Reshape):
    def __init__(self, dim0: Union[int, z3.ArithRef], dim1: Union[int, z3.ArithRef], dim2: Union[int, z3.ArithRef]):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.target_shape = [dim0, dim1, dim2]
        self.out_dims = [3]


class Reshape4D(Reshape):
    def __init__(self, dim0: Union[int, z3.ArithRef], dim1: Union[int, z3.ArithRef], dim2: Union[int, z3.ArithRef],
                 dim3: Union[int, z3.ArithRef]):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.target_shape = [dim0, dim1, dim2, dim3]
        self.out_dims = [4]


class Reshape5D(Reshape):
    def __init__(self, dim0: Union[int, z3.ArithRef], dim1: Union[int, z3.ArithRef], dim2: Union[int, z3.ArithRef],
                 dim3: Union[int, z3.ArithRef], dim4: Union[int, z3.ArithRef]):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.dim4 = dim4
        self.target_shape = [dim0, dim1, dim2, dim3, dim4]
        self.out_dims = [5]


class Reshape6D(Reshape):
    def __init__(self, dim0: Union[int, z3.ArithRef], dim1: Union[int, z3.ArithRef], dim2: Union[int, z3.ArithRef],
                 dim3: Union[int, z3.ArithRef], dim4: Union[int, z3.ArithRef], dim5: Union[int, z3.ArithRef]):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.dim4 = dim4
        self.dim5 = dim5
        self.target_shape = [dim0, dim1, dim2, dim3, dim4, dim5]
        self.out_dims = [6]


class Transpose(UnaryOpBase, ABC):
    def __init__(self):
        """See https://pytorch.org/docs/stable/generated/torch.transpose.html
        """
        super().__init__()
        self.inp_dims = [-1]

    def _init_swap_dims(self, input_shape: List[Union[int, z3.ArithRef]]):
        assert len(input_shape) >= 1
        if 'dim0' not in self.extra_attrs or 'dim1' not in self.extra_attrs:
            max_dim = len(input_shape) - 1
            self.extra_attrs['dim0'] = random.randint(0, max_dim)
            self.extra_attrs['dim1'] = random.randint(0, max_dim)
        return self.extra_attrs['dim0'], self.extra_attrs['dim1']

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        dim0, dim1 = self._init_swap_dims(input_shapes[0].shape)
        shape_var = input_shapes[0]
        shape_var.shape[dim0], shape_var.shape[dim1] = shape_var.shape[dim1], shape_var.shape[dim0]
        return [shape_var]

    def _requires(self, input_shapes):
        dim0, dim1 = self._init_swap_dims(input_shapes[0].shape)
        assert len(input_shapes[0].shape) >= max(dim0, dim1) + 1
        return []

    def torch(self):
        def f(x: torch.Tensor):
            dim0, dim1 = self._init_swap_dims([x.shape])
            return x.transpose(dim0, dim1)
        return f


def _glob_leaf_op_classes():
    ret = []

    def _glob_leaf_op_classes_rec(cls):
        nonlocal ret
        for c in cls.__subclasses__():
            if c.__subclasses__():
                _glob_leaf_op_classes_rec(c)
            elif c is not Input:
                ret.append(c)
    _glob_leaf_op_classes_rec(AbsOpBase)
    return ret


ALL_OP_TYPES = _glob_leaf_op_classes()

if __name__ == '__main__':
    # Test shape functions
    print(len(ALL_OP_TYPES), 'operators supported:')
    print(ALL_OP_TYPES)

    # ReLU
    lhs = torch.relu(torch.randn(1, 1, 1, 1)).shape
    rhs = torch.Size(ReLU().shape_fn([ShapeVar([1, 1, 1, 1])])[0].shape)
    assert lhs == rhs, f"{lhs} != {rhs}"

    # Add
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(2, 3, 4, 5)
    c = a + b
    assert c.shape == torch.Size(Add().shape_fn(
        [ShapeVar([2, 3, 4, 5]), ShapeVar([2, 3, 4, 5])])[0].shape)

    # Expand
    source_shape = (4, 1)
    a = torch.randn(source_shape)
    abs_op = ExpandLast4(expand_n=2)
    assert a.expand(2, 1, *source_shape).shape == torch.Size(
        abs_op.shape_fn([ShapeVar(source_shape)])[0].shape)

    abs_op = ExpandLast1(expand_n=2)
    rhs = torch.Size(abs_op.shape_fn(
        [ShapeVar(list(source_shape))])[0].shape)
    lhs = a.expand(4, 2).shape
    assert lhs == rhs, f"{lhs} != {rhs}"

    # NCHWConv2d
    source_shape = (2, 3, 24, 24)
    a = torch.randn(*source_shape)
    out = torch.conv2d(a, torch.randn(3, 3, 3, 4), stride=1, padding=1)
    assert out.shape == NCHWConv2d(
        3, 3, 3, 4, 1, 1).shape_fn([ShapeVar(source_shape)])[0].torch()
    print(NCHWConv2d(
        3, 3, 3, 4, 1, 1).shape_fn([ShapeVar([2, *z3.Ints('c h w')])])[0])

    # Reshape
    source_shape = (2, 3, 4)
    target_shape = (1, 2, 3, 2, 2)
    a = torch.randn(*source_shape)
    assert a.reshape(*target_shape).shape == Reshape5D(*target_shape).shape_fn(
        [ShapeVar(source_shape)])[0].torch()

    # Dirty fix for z3 bug by wrapping the context using seprated functions.
    def test_reshape_symbol():  # See https://github.com/Z3Prover/z3/issues/989
        s = z3.Solver()
        v = z3.Ints('a b c d e')
        abs_op = Reshape5D(*v)
        cons = abs_op.requires([ShapeVar(source_shape)])
        for c in cons:
            s.add(c)
        for c in abs_op.shape_fn([ShapeVar(source_shape)])[0].gt_zero():
            s.add(c)
        assert s.check() == z3.sat
        print(s.model())
    test_reshape_symbol()

    # Test `concrete` function.
    p0, p1, p2, p3, p4, p5 = z3.Ints('p0 p1 p2 p3 p4 p5')
    op = NCHWConv2d(p0, p1, p2, p3, p4, p5)
    s = z3.Solver()
    shape = ShapeVar([1, 3, 224, 224])
    for c in op.requires([shape]):
        s.add(c)
    for c in op.shape_fn([shape])[0].gt_zero():
        s.add(c)
    assert s.check() == z3.sat
    model = s.model()
    concrete_op = concretize(op, model)
    assert concrete_op.in_channels == model[p0].as_long()
    assert concrete_op.out_channels == model[p1].as_long()
    assert concrete_op.kernel_h_size == model[p2].as_long()
    assert concrete_op.kernel_w_size == model[p3].as_long()
    assert concrete_op.stride == model[p4].as_long()
    assert concrete_op.padding == model[p5].as_long()
