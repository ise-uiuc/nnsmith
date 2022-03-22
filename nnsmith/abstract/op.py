from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import fnmatch
from functools import reduce
import functools
import os
from typing import List, Tuple, Union, Callable, Type
from inspect import signature
import random
import itertools
import warnings

# Import z3 ahead of torch (See https://github.com/Z3Prover/z3/issues/5656)
import z3
import torch

from nnsmith.error import SanityCheck, ConstraintCheck
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

ARITH_MAX_WIDTH: int = 64
_INFERRED = False
_DEV = torch.device("cpu")
FLOPS_LIM = os.getenv("NNSMITH_FLOPS_LIM", None)
if FLOPS_LIM == 'on':  # use predefined value
    FLOPS_LIM = 2**22
elif FLOPS_LIM is None:
    pass
else:
    FLOPS_LIM = float(FLOPS_LIM)


def _op_set_use_cuda(use_cuda):
    global _DEV
    _DEV = torch.device('cuda' if use_cuda else 'cpu')


__MIN_RANK__ = 0
__MAX_RANK__ = 5


def int_from(start):
    return tuple(range(start, __MAX_RANK__ + 1))


def int_range(start, end):
    return tuple(range(start, end + 1))


def int_until(end):
    return tuple(range(__MIN_RANK__, end + 1))


def int_all():
    return tuple(range(__MIN_RANK__, __MAX_RANK__ + 1))


def align_bvs(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef], carry=False, mult=False):
    left_is_arith = isinstance(left, (int, float, z3.ArithRef))
    right_is_arith = isinstance(right, (int, float, z3.ArithRef))
    # If both values are of arithmetic type, we do not need to do anything.
    if left_is_arith and right_is_arith:
        return (left, right)
    # We assume that the width of an arithmetic type is ARITH_MAX_WIDTH.
    if left_is_arith:
        if isinstance(left, int):
            left_size = min(ARITH_MAX_WIDTH, left.bit_length())
        else:
            left_size = ARITH_MAX_WIDTH
    elif isinstance(left, z3.BitVecRef):
        left_size = left.size()
    else:
        raise RuntimeError(
            f"Unsupported alignment value {left} of type {type(left)}")
    # We assume that the width of an arithmetic type is ARITH_MAX_WIDTH.
    if right_is_arith:
        if isinstance(right, int):
            right_size = min(ARITH_MAX_WIDTH, right.bit_length())
        else:
            right_size = ARITH_MAX_WIDTH
    elif isinstance(right, z3.BitVecRef):
        right_size = right.size()
    else:
        raise RuntimeError(
            f"Unsupported alignment value {right} of type {type(right)}")
    # Extend the bitvector that is smaller with the necessary amount of zeroes.
    SanityCheck.true(not (
        carry and mult), "Carry and multiplication extension are mutually exclusive")
    SanityCheck.le(left_size, ARITH_MAX_WIDTH,
                   f"Bitvector sizes must not exceed {ARITH_MAX_WIDTH} bits.")
    SanityCheck.le(right_size, ARITH_MAX_WIDTH,
                   f"Bitvector sizes must not exceed {ARITH_MAX_WIDTH} bits.")
    diff = left_size - right_size
    if left_is_arith:
        if diff > 0:
            right = z3.Concat(z3.BitVecVal(0, diff), right)
        if isinstance(left, z3.IntNumRef):
            left = left.as_long()
        return z3.BitVecVal(left, right.size()), z3.simplify(right)
    if right_is_arith:
        if diff < 0:
            left = z3.Concat(z3.BitVecVal(0, abs(diff)), left)
        if isinstance(left, z3.IntNumRef):
            left = left.as_long()
        return left, z3.BitVecVal(right, left.size())
    if diff < 0:
        left = z3.Concat(z3.BitVecVal(0, abs(diff)), left)
    elif diff > 0:
        right = z3.Concat(z3.BitVecVal(0, diff), right)

    if carry and max(left_size, right_size) < ARITH_MAX_WIDTH:
        left = z3.Concat(z3.BitVecVal(0, 1), left)
        right = z3.Concat(z3.BitVecVal(0, 1), right)
    if mult:
        max_val = right.size() + left.size()
        if max_val >= ARITH_MAX_WIDTH:
            return (left, right)
        else:
            max_val = ARITH_MAX_WIDTH - max_val
        left = z3.Concat(z3.BitVecVal(0, max_val), left)
        right = z3.Concat(z3.BitVecVal(0, max_val), right)
    return (left, right)


def nnsmith_mul(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right, mult=True)
    return left * right


def nnsmith_add(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right, carry=True)
    return left + right


def nnsmith_sub(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    return left - right


def nnsmith_eq(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    return left == right


def nnsmith_neq(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    return left != right


def nnsmith_ge(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.UGE(left, right)
    return left >= right


def nnsmith_gt(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.UGT(left, right)
    return left > right


def nnsmith_le(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.ULE(left, right)
    return left <= right


def nnsmith_lt(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.ULT(left, right)
    return left < right


def nnsmith_div(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.UDiv(left, right)
    return left / right


def nnsmith_mod(left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.URem(left, right)
    return left % right


class DType(Enum):
    # float16 = 'float16'
    float32 = torch.float32
    float64 = torch.float64
    # int8 = 'int8'
    # int16 = 'int16'
    int32 = torch.int32
    int64 = torch.int64
    bool = torch.bool
    # complex64 = 'complex64'
    # complex128 = 'complex128'

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        s = super().__str__()
        assert s.startswith('DType.'), s
        return s[len('DType.'):]

    @staticmethod
    def is_float(dtype):
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)

        return dtype in [DType.float32, DType.float64]

    @staticmethod
    def from_str(s):
        return {
            'float32': DType.float32,
            'float64': DType.float64,
            'int32': DType.int32,
            'int64': DType.int64,
            'bool': DType.bool,
        }[s]

    @staticmethod
    def torch(s):
        if not isinstance(s, str):
            s = str(s)
        return {
            'float32': torch.float32,
            'float64': torch.float64,
            'int32': torch.int32,
            'int64': torch.int64,
            'bool': torch.bool,
        }[s]


DTypeComb = Tuple[DType, ...]

DTYPE_ALL = list(DType.__members__.values())
DTYPE_NON_BOOLS = [dtype for dtype in DTYPE_ALL if dtype != DType.bool]
DTYPE_FLOATS = [DType.float32, DType.float64]
DTYPE_INTS = [DType.int32, DType.int64]


class ShapeVar:
    def __init__(self, shape: List[Union[int, z3.ExprRef]], dtype: Union[DType, torch.dtype]):
        self.shape = list(shape)
        self.dtype = DType(dtype)

    def __repr__(self):
        return f'ShapeVar(shape={str(self.shape)}, dtype={self.dtype.value})'

    def gt_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_gt(s, 0))
            else:
                ConstraintCheck.gt(s, 0)
        return ret

    def eq(self, other):
        SanityCheck.eq(self.ndims, other.ndims)
        ret = []
        for i in range(self.ndims):
            if isinstance(self.shape[i], z3.ExprRef) or isinstance(other.shape[i], z3.ExprRef):
                ret.append(nnsmith_eq(self.shape[i], other.shape[i]))
            else:
                ConstraintCheck.gt(self.shape[i], other.shape[i])
        return ret

    def torch(self):
        # NOTE: Only for concrete shapes.
        return torch.Size(self.shape)

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ExprRef) for s in self.shape)

    def nelement(self):
        if len(self.shape) == 0:  # Scalar
            return 1
        return reduce(lambda x, y: nnsmith_mul(x, y), self.shape, 1)

    def deepcopy(self):
        return ShapeVar(shape=list(self.shape), dtype=self.dtype)

    @staticmethod
    def from_torch(torch_tensor):
        return ShapeVar(list(torch_tensor.shape), torch_tensor.dtype)

    @property
    def ndims(self):
        return len(self.shape)


def check_shape_fn(func):
    def wrapper_check_shape_fn(self, input_shapes):
        SanityCheck.true(self.out_ranks, "Empty output dimensions in {}".format(
            self.__class__.__name__))
        SanityCheck.eq(len(input_shapes), len(self.inp_ranks), "{} requires {} inputs, but got {}".format(
            self.__class__.__name__,
            len(self.inp_ranks), len(input_shapes)))
        res = func(self, [s.deepcopy() for s in input_shapes])
        SanityCheck.eq(len(res), len(self.out_ranks), "{} requires {} outputs, but got {}".format(
            self.__class__.__name__,
            len(self.out_ranks), len(res)))
        return res
    return wrapper_check_shape_fn


def check_require_fn(func):
    def wrapper_check_require_fn(self, input_shapes: List[ShapeVar]):
        if not _INFERRED:
            auto_infer_in_dtypes()
        SanityCheck.eq(len(input_shapes), len(self.inp_ranks), "{} requires {} inputs, but got {}".format(
            self.__class__.__name__,
            len(self.inp_ranks), len(input_shapes)))
        return func(self, [s.deepcopy() for s in input_shapes])
    return wrapper_check_require_fn


def _prepend_to(x, max_dim):
    return [1 for i in range(max_dim - len(x))] + x


def z3_bcast(x: Union[int, z3.ExprRef], y: Union[int, z3.ExprRef], *args: Union[int, z3.ExprRef]):
    x, y = align_bvs(x, y)
    return z3.simplify(z3.If(nnsmith_eq(y, 1), x, y)) if len(args) == 0 else z3_bcast(z3_bcast(x, y), *args)


def broadcast_shapes(*shapes: List[Union[z3.ExprRef, int]]) -> List[Union[z3.ExprRef, int]]:
    """this function does not check the validity of broadcast. Please always pair it with broadcast_cons"""
    SanityCheck.gt(len(shapes), 0)
    if len(shapes) == 1:
        return shapes[0]
    max_dim = max(map(lambda x: len(x), shapes))
    max_shape = [None] * (max_dim)
    for j in range(max_dim):
        i = -j - 1
        args_dim_sz = [_prepend_to(x, max_dim)[i] for x in shapes]
        if any(isinstance(s, z3.ExprRef) for s in args_dim_sz):
            max_shape[i] = z3.simplify(z3_bcast(*args_dim_sz))
        else:
            max_shape[i] = max(*args_dim_sz)
    return max_shape


def broadcast_cons(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.ExprRef]:
    tgt_shape = broadcast_shapes(*shapes)
    cons = []
    max_dim = len(tgt_shape)
    for j in range(max_dim):
        i = -j - 1
        if isinstance(tgt_shape[i], z3.ExprRef):
            axis_cons = []
            for x in shapes:
                if len(x) > j:
                    axis_cons.append(
                        z3.Or(nnsmith_eq(x[i], tgt_shape[i]), nnsmith_eq(x[i], 1)))
            axis_cons = z3.simplify(z3.And(*axis_cons))
            cons.append(axis_cons)
        else:
            args_dim_sz = [_prepend_to(x, max_dim)[i] for x in shapes]
            valid = all(nnsmith_eq(s, tgt_shape[i]) or nnsmith_eq(
                s, 1) for s in args_dim_sz)
            # TODO(JK): enable this after fixing issue #2
            # assert valid, "Invalid broadcast shapes {}. Specific dim sizes: {}".format(shapes, args_dim_sz)
            cons.append(z3.BoolVal(valid))
    return cons


def broadcast_cons_binary(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.ExprRef]:
    SanityCheck.eq(len(shapes), 2)
    tgt_shape = broadcast_shapes(*shapes)
    cons = []
    max_dim = len(tgt_shape)
    lhs, rhs = shapes
    lhs = _prepend_to(lhs, max_dim)
    rhs = _prepend_to(rhs, max_dim)
    for j in range(max_dim):
        i = -j - 1
        if isinstance(tgt_shape[i], z3.ExprRef):
            cons.append(z3.simplify(
                z3.Or(nnsmith_eq(lhs[i], 1), nnsmith_eq(rhs[i], 1), nnsmith_eq(lhs[i], rhs[i]))))
        else:
            valid = nnsmith_eq(lhs[i], 1) or nnsmith_eq(
                rhs[i], 1) or nnsmith_eq(lhs[i], rhs[i])
            # TODO(JK): enable this after fixing issue #2
            # assert valid, "Invalid broadcast shapes lhs={}, rhs={}".format(lhs, rhs)
            cons.append(z3.BoolVal(valid))
    return cons


def broadcast_to_cons(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.ExprRef]:
    """Unidirectional broadcast. Last input is the target shape.

    Examples of valid unidirectional broadcast:
    [1, 2, 3] -> [0, 1, 2, 3]
    [1] -> [3]

    Examples of invalid unidirectional broadcast:
    [0, 1, 2, 3] -> [1, 2, 3]
    [3] -> [1]

    Logic: for each dim: src_dim == tgt_dim or src_dim == 1
    """
    srcs, tgt = shapes[:-1], shapes[-1]
    cons = []
    max_dim = len(tgt)
    for src in srcs:
        ConstraintCheck.true(len(src) <= max_dim)
        src = _prepend_to(src, max_dim)
        for i in range(max_dim):
            if isinstance(tgt[i], z3.ExprRef) or isinstance(src[i], z3.ExprRef):
                cons.append(z3.simplify(
                    z3.Or(nnsmith_eq(src[i], 1), nnsmith_eq(src[i], tgt[i]))))
            else:
                valid = nnsmith_eq(src[i], 1) or nnsmith_eq(src[i], tgt[i])
                # TODO(JK): enable this after fixing issue #2
                # assert valid, "Invalid broadcast shapes lhs={}, rhs={}".format(lhs, rhs)
                cons.append(z3.BoolVal(valid))
    return cons


class AbsOpBase(ABC):
    # number of parameters; None means it's fixed that can be inferred through `signature`.
    num_var_param = None
    # whether this op is broadcastable or not
    bcastable = False
    # input dtypes: enumerates all possible input dtype combinations. Size of the list is the number of combinations.
    # Each element is a tuple of allowed input dtypes. NOTE: len(list) can >= the # of inputs, for handling ops with arbitrary arity.
    # For example, [(DType.float32, DType.float32), (DType.float64, DType.float64), (DType.int32, DType.int32)] means that
    # this op can accept one of float32xfloat32, float64xfloat64, and int32xint32 as input dtypes.
    in_dtypes: List[DTypeComb] = None  # Overwrite me!
    out_dtypes: List[DTypeComb] = None
    # whether to disable the op during graph generation
    _skip = False

    def __init__(self):
        # `[3, 3]` this means this op requires 2 inputs. Where the 1st one has 2 dimensions, and the 2nd one has 3 dimensions.
        # `-1` means arbitrary dimantions; NOTE: but should be concretized during execution.
        # All symbols of correponding operator must be the constructor's parameters.
        # [ <inp0>(support_dim0, support_dim1, ...), <inp1>(...), ... ]
        self.inp_ranks = []
        # NOTE: the concrete values of out_ranks are not useful. Just make sure the length is correct.
        # NOTE: the output shape of input dimensions should be concretized during the execution.
        self.out_ranks = []
        # Require the input dimension sizes to be equivalent.
        self.same_inp_dims = False
        # NOTE: the input of operator constructors are all Union[int, z3.ExprRef].
        self.extra_attrs = {}

    @classmethod
    def get_num_var_param(cls):
        if cls.num_var_param is None:
            return len(signature(cls.__init__).parameters) - 1
        return random.choice(cls.num_var_param)

    @abstractmethod  # Overload me!
    # Exception means rejection.
    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        raise NotImplementedError

    @check_shape_fn  # Public API.
    def shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        self.last_outs = self._shape_fn(input_shapes)
        return self.last_outs

    # Overload me!
    # Extra constraints for the input tensors.
    # Exception means rejection.
    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return []

    @abstractmethod
    def torch(self) -> Callable[..., torch.Tensor]:
        raise NotImplementedError

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        raise NotImplementedError

    @check_require_fn  # Public API.
    def requires(self, input_shapes):
        return self._requires(input_shapes)

    def n_floats(self, input_shapes: List[ShapeVar]) -> z3.ExprRef:
        return reduce(nnsmith_add, [i.nelement() for i in self.last_outs])

    def flops(self, input_shapes):
        return 0

    def __repr__(self) -> str:
        return self.__class__.__name__


def concretize(op: AbsOpBase, model: z3.ModelRef) -> AbsOpBase:
    if isinstance(op, Constant) or isinstance(op, Input):
        ret_op = deepcopy(op)
        values = []

        for idx, s in enumerate(op.shape_var.shape):
            if isinstance(s, z3.ExprRef):
                ret_op.shape_var.shape[idx] = model.eval(s).as_long()

        return ret_op

    # Non-inp / const types.
    construct_param_dict = signature(op.__init__).parameters
    values = []
    symbolic_idx = []

    if op.num_var_param is not None:
        # input is a variable list.
        key = list(construct_param_dict.keys())[0]
        values = list(getattr(op, key))
        symbolic_idx = list(range(len(values)))
    else:
        for idx, key in enumerate(construct_param_dict):
            param = getattr(op, key)
            values.append(param)
            if isinstance(param, z3.ExprRef):
                symbolic_idx.append(idx)

    for idx in symbolic_idx:
        values[idx] = model.eval(values[idx]).as_long()

    concrete_op = op.__class__(*values)
    concrete_op.inp_ranks = op.inp_ranks
    concrete_op.out_ranks = op.out_ranks
    concrete_op.same_inp_dims = op.same_inp_dims
    concrete_op.extra_attrs = op.extra_attrs

    return concrete_op


class UnaryOpBase(AbsOpBase):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()
        self.out_ranks = [int_all()]


class BinaryOpBase(AbsOpBase):
    in_dtypes = [(i, i) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()
        self.out_ranks = [int_all()]


class TernaryOpBase(AbsOpBase):
    in_dtypes = [(i, i, i) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()
        self.out_ranks = [int_all()]


class ElementWiseUnaryOp(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_all()]
        self.out_ranks = [int_all()]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [
            (out_shape_var[0].ndims, out_shape_var[0].dtype),
        ]

# class ElementWiseBinaryOp(BinaryOpBase):
#     def __init__(self):
#         super().__init__()
#         self.inp_ranks = [-1, -1]
#         self.same_inp_dims = True

#     def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
#         assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
#         return [input_shapes[0]]

#     def _requires(self, input_shapes):
#         assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
#         ret = []
#         for l, r in zip(input_shapes[0].shape, input_shapes[1].shape):
#             if isinstance(l, z3.ExprRef) or isinstance(r, z3.ExprRef):
#                 ret.append(nnsmith_eq(l, r))
#             else:
#                 assert l == r
#         return ret


class BcastBinaryOp(BinaryOpBase):
    bcastable = True
    # by default, output dtype is the same as the first input dtype
    _bcast_out_dtypes = None

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_all(), int_all()]
        self.same_inp_dims = False
        self.bcastable = True

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        tgt_shape = broadcast_shapes(*(ish.shape for ish in input_shapes))
        dtype = input_shapes[0].dtype if self._bcast_out_dtypes is None else self._bcast_out_dtypes[0]
        return [ShapeVar(tgt_shape, dtype)]

    def _requires(self, input_shapes):
        return broadcast_cons_binary(*(ish.shape for ish in input_shapes))

    # FIXME: should be more flexible but need some constraints.
    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [
            (out_shape_var[0].ndims, out_shape_var[0].dtype),
            (out_shape_var[0].ndims, out_shape_var[0].dtype),
        ]


class BcastBinaryOp1(BcastBinaryOp):  # +-*/ max min
    in_dtypes = [(i, i) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    _bcast_out_dtypes = None


class BcastBinaryOp2(BcastBinaryOp):  # > < =
    in_dtypes = [(i, i) for i in DTYPE_ALL]
    out_dtypes = [(DType.bool,)]
    _bcast_out_dtypes = [DType.bool]


class BcastBinaryOp3(BcastBinaryOp):  # logical and or xor
    in_dtypes = [(DType.bool, DType.bool)]
    out_dtypes = [(DType.bool,)]
    _bcast_out_dtypes = [DType.bool]


class Where(TernaryOpBase):
    bcastable = True
    in_dtypes = [(DType.bool, i, i) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_all(), int_all(), int_all()]
        self.same_inp_dims = False
        self.same_inp_dtypes = True
        self.bcastable = True

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        # assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
        tgt_shape = broadcast_shapes(*(ish.shape for ish in input_shapes))
        dtype = input_shapes[1].dtype
        return [ShapeVar(tgt_shape, dtype)]

    def _requires(self, input_shapes):
        return broadcast_cons(*(ish.shape for ish in input_shapes)) \
            + [z3.BoolVal(input_shapes[1].dtype == input_shapes[2].dtype)]

    def torch(self):
        return torch.where

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [
            (out_shape_var[0].ndims, DType.bool),
            (out_shape_var[0].ndims, out_shape_var[0].dtype),
            (out_shape_var[0].ndims, out_shape_var[0].dtype),
        ]


# bcast binary ops from https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
# TODO bitwise_and/or/xor?
Add = type('Add', (BcastBinaryOp1,), {'torch': lambda self: torch.add})
Sub = type('Sub', (BcastBinaryOp1,), {'torch': lambda self: torch.sub})
Mul = type('Mul', (BcastBinaryOp1,), {'torch': lambda self: torch.mul})
# NOTE(JK): didn't find multi-input version of Max and Min in torch, so assume binary ops
Max = type('Max', (BcastBinaryOp1,), {'torch': lambda self: torch.max})
Min = type('Min', (BcastBinaryOp1,), {'torch': lambda self: torch.min})

Equal = type('Equal', (BcastBinaryOp2,), {'torch': lambda self: torch.eq})
Greater = type('Greater', (BcastBinaryOp2,), {'torch': lambda self: torch.gt})
Less = type('Less', (BcastBinaryOp2,), {'torch': lambda self: torch.lt})

And = type('And', (BcastBinaryOp3,), {'torch': lambda self: torch.logical_and})
Or = type('Or', (BcastBinaryOp3,), {'torch': lambda self: torch.logical_or})
Xor = type('Xor', (BcastBinaryOp3,), {'torch': lambda self: torch.logical_xor})

# TODO: support exactly what onnx spec says (e.g., int support in the rhs)
# lhs_dtypes = (DType.int32, DType.int64, DType.float32, DType.float64)
# rhs_dtypes = (DType.int32, DType.int64, DType.float32, DType.float64)
# Pow.in_dtypes = itertools.product(lhs_dtypes, rhs_dtypes)

# NOTE(JK): For Mean and Sum there is no corresponding torch op, so we ignore them
# Sum = type('Sum', (BcastBinaryOp,), {'torch': lambda self: torch.sum})
# Mean = type('Mean', (BcastBinaryOp,), {'torch': lambda self: torch.mean})


class StopFoldConst(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.dtype = data.dtype
        self.param = torch.nn.parameter.Parameter(data, requires_grad=False)

    @torch.no_grad()
    def forward(self):
        return self.param.to(dtype=self.dtype, device=_DEV)


class Input(AbsOpBase):
    in_dtypes = [()]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self, dim: int):
        super().__init__()
        self.inp_ranks = []
        self.out_ranks = [(dim,)]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        SanityCheck.eq(len(input_shapes), 0)
        return [self.shape_var]

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        SanityCheck.eq(len(input_shapes), 0)
        return []

    def torch(self) -> Callable[..., torch.Tensor]:
        raise NotImplementedError()


class Constant(AbsOpBase):
    in_dtypes = [()]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __str__(self) -> str:
        return super().__str__() + ' ' + str(self.extra_attrs)

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.inp_ranks = []
        self.out_ranks = [(dim,)]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        SanityCheck.eq(len(input_shapes), 0)
        return [self.shape_var]

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        SanityCheck.eq(len(input_shapes), 0)
        return []

    def torch(self) -> Callable[..., torch.Tensor]:
        data = torch.randn(self.shape_var.shape, device=_DEV).to(
            self.shape_var.dtype.value)
        return StopFoldConst(data)


class Placeholder:
    def __init__(self, out_shape: ShapeVar):
        self.out_shape = out_shape
        self.inp_ranks = []
        self.out_ranks = [(out_shape.ndims,)]

    def __repr__(self):
        return f'Placeholder({self.out_shape})'

    def to_const(self):
        const_node = Constant(self.out_shape.ndims)
        const_node.shape_var = self.out_shape
        return const_node

    def to_input(self):
        input_node = Input(self.out_shape.ndims)
        input_node.shape_var = self.out_shape
        return input_node


class LegacyConstant0D(Constant):
    def __init__(self):
        super().__init__(0)
        # TODO more dtypes

    @property
    def shape_var(self):
        return ShapeVar([], dtype=self.extra_attrs['dtype'])


class LegacyConstant1D(Constant):
    def __init__(self, dim0: Union[int, z3.ExprRef]):
        super().__init__(1)
        self.dim0 = dim0

    @property
    def shape_var(self):
        return ShapeVar([self.dim0], dtype=self.extra_attrs['dtype'])


class LegacyConstant2D(Constant):
    def __init__(self, dim0: Union[int, z3.ExprRef], dim1: Union[int, z3.ExprRef]):
        super().__init__(2)
        self.dim0 = dim0
        self.dim1 = dim1

    @property
    def shape_var(self):
        return ShapeVar(
            [self.dim0, self.dim1], dtype=self.extra_attrs['dtype'])


class LegacyConstant3D(Constant):
    def __init__(self, dim0: Union[int, z3.ExprRef], dim1: Union[int, z3.ExprRef], dim2: Union[int, z3.ExprRef]):
        super().__init__(3)
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2

    @property
    def shape_var(self):
        return ShapeVar(
            [self.dim0, self.dim1, self.dim2], dtype=self.extra_attrs['dtype'])


class LegacyConstant4D(Constant):
    def __init__(self, dim0: Union[int, z3.ExprRef], dim1: Union[int, z3.ExprRef], dim2: Union[int, z3.ExprRef], dim3: Union[int, z3.ExprRef]):
        super().__init__(4)
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

    @property
    def shape_var(self):
        return ShapeVar(
            [self.dim0, self.dim1, self.dim2, self.dim3], dtype=self.extra_attrs['dtype'])


# FIXME: Div will cause fuzzing crash.
Div = type('Div', (BcastBinaryOp1,), {
    'torch': lambda self:
        lambda x, y: torch.div(x, y, rounding_mode='floor' if DType(
            x.dtype) in DTYPE_INTS else None),
    'torch_loss': lambda self, _, x: torch.where(x.abs() < 1e-3, x.abs(), torch.zeros_like(x))})


class Pow(BcastBinaryOp):
    in_dtypes = [(i, i) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def torch(self):
        return torch.pow

    def torch_loss(self, a, b):
        return (a - 1).abs() + torch.where(b > 28., b.abs(), torch.zeros_like(b))
        # Another complicated proposal but not working:
        # See: https://en.cppreference.com/w/c/numeric/math/pow
        # Inf:
        #   a > 1, b is too big => b should be smaller.
        #   a = 0, b < 0 => a should be bigger.
        # Nan: a < 0, 0 < b < 1 => either a should be positive or |b| should be bigger.
        # res = torch.pow(a, b)
        # return torch.where(
        #     torch.isinf(res),
        #     torch.where(b > 32.,
        #                 b,
        #                 torch.where(a == 0., a, )),
        #     torch.where(torch.isnan(res),
        #                 torch.where(a > 0, torch.zeros_like(a), a.abs()),
        #                 torch.zeros_like(a)))


class ReLU(ElementWiseUnaryOp):
    # FIXME(JK): ints are somehow not supported in onnxruntime, which we use to gen inputs.
    # Make it include ints once we use other backends other than onnxruntime.
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.nn.ReLU()


class GELU(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.nn.GELU()


class LeakyReLU(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        """See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        """
        super().__init__()
        self.negative_slope = 0.01

    def torch(self):
        return torch.nn.LeakyReLU(self.negative_slope)


class PReLU(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.nn.PReLU(device=_DEV)


class Sigmoid(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.sigmoid


class TrigonometricOp(ElementWiseUnaryOp):
    pass


class Sin(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.sin


class Cos(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.cos


class Asin(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.asin

    def torch_loss(self, x):
        return torch.where(x.abs() > 1, x.abs() - 1, torch.zeros_like(x))


class Acos(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.acos

    def torch_loss(self, x):
        return torch.where(x.abs() > 1, x.abs(), torch.zeros_like(x))


class Tan(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.tan


class Atan(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.atan


class Abs(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.abs


class Ceil(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.ceil


class Clip(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def __init__(self):
        super().__init__()
        self.min = -1
        self.max = 1

    def torch(self):
        return lambda x: torch.clip(x, self.min, self.max)


class Round(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.round


class Sqrt(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.sqrt

    def torch_loss(self, x):
        # return torch.max(torch.tensor(0.), x) - 0.
        return torch.where(x <= 0, 1. - x, torch.zeros_like(x))


class Log2(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.log2

    def torch_loss(self, x):
        return torch.where(x <= 0, 1. - x, torch.zeros_like(x))


class Neg(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def __init__(self):
        super().__init__()

    def torch(self):
        return torch.neg


class Softmax(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self, dim: Union[int, z3.ExprRef]):
        super().__init__()
        self.dim = dim
        self.inp_ranks = [int_from(1)]
        self.out_ranks = [int_from(1)]

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return [
            nnsmith_lt(self.dim, input_shapes[0].ndims),
            nnsmith_ge(self.dim, 0)]

    def torch(self) -> Callable[..., torch.Tensor]:
        return torch.nn.Softmax(dim=self.dim)


class Pool2d(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self,
                 kernel_h_size: Union[int, z3.ExprRef],
                 kernel_w_size: Union[int, z3.ExprRef],
                 stride: Union[int, z3.ExprRef],
                 padding: Union[int, z3.ExprRef]):
        super().__init__()
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.padding = padding

        self.inp_ranks = [(4,)]  # NCHW
        self.out_ranks = [(4,)]  # NCHW

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        is_symbolic_inp = input_shapes[0].constains_symbol() or isinstance(self.kernel_w_size, z3.ExprRef) or isinstance(
            self.kernel_h_size, z3.ExprRef) or isinstance(self.stride, z3.ExprRef) or isinstance(self.padding, z3.ExprRef)

        shape_var = ShapeVar([], dtype=input_shapes[0].dtype)
        # Batch dim: just copy
        shape_var.shape.append(input_shapes[0].shape[0])
        # Output channels
        shape_var.shape.append(input_shapes[0].shape[1])
        if not is_symbolic_inp:
            shape_var.shape.append(
                (input_shapes[0].shape[2] - self.kernel_h_size + 2 * self.padding) // self.stride + 1)
            shape_var.shape.append(
                (input_shapes[0].shape[3] - self.kernel_w_size + 2 * self.padding) // self.stride + 1)
        else:
            shape_var.shape.append(
                (nnsmith_div(nnsmith_add(nnsmith_sub(input_shapes[0].shape[2], self.kernel_h_size), 2 * self.padding), self.stride) + 1))
            shape_var.shape.append(
                (nnsmith_div(nnsmith_add(nnsmith_sub(input_shapes[0].shape[3], self.kernel_w_size), 2 * self.padding), self.stride) + 1))
        return [shape_var]

    def _requires(self, input_shapes):
        cons = []
        ret = []
        cons.append(nnsmith_ge(self.kernel_h_size, 1))
        cons.append(nnsmith_ge(self.kernel_w_size, 1))
        cons.append(nnsmith_le(self.kernel_h_size,
                    nnsmith_add(input_shapes[0].shape[2], 2 * self.padding)))
        cons.append(nnsmith_le(self.kernel_w_size,
                    nnsmith_add(input_shapes[0].shape[3], 2 * self.padding)))
        cons.append(nnsmith_ge(self.stride, 1))
        cons.append(nnsmith_ge(self.padding, 0))
        # not too extream to avoid torch exporter issue
        cons.append(nnsmith_le(self.padding, 255))
        cons.append(nnsmith_le(
            self.padding, nnsmith_div(self.kernel_h_size, 2)))
        cons.append(nnsmith_le(
            self.padding, nnsmith_div(self.kernel_w_size, 2)))
        # limit FLOPS
        if FLOPS_LIM is not None:
            cons.append(nnsmith_le(self.flops(input_shapes), FLOPS_LIM))
        for c in cons:
            if isinstance(c, z3.ExprRef):
                ret.append(c)
            else:
                ConstraintCheck.true(c)
        return ret

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(4, out_shape_var[0].dtype)]


class MaxPool2d(Pool2d):
    def torch(self) -> Callable[..., torch.Tensor]:
        return torch.nn.MaxPool2d(kernel_size=(self.kernel_h_size, self.kernel_w_size), stride=self.stride, padding=self.padding)


class AvgPool2d(Pool2d):
    def torch(self) -> Callable[..., torch.Tensor]:
        return torch.nn.AvgPool2d(kernel_size=(self.kernel_h_size, self.kernel_w_size), stride=self.stride, padding=self.padding)


def _pad_num_var_param(rstart=1, max=None):
    r = rstart  # rank
    ret = []
    while r <= __MAX_RANK__:
        h = r * 2
        if max is not None and h > max:
            break
        ret.append(h)
        r += 1
    return ret


class Pad(UnaryOpBase):
    num_var_param = _pad_num_var_param()
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self, padding_list, pad_t):
        super().__init__()
        self.padding_list = padding_list
        self.extra_attrs['type'] = pad_t
        self.inp_ranks = [int_from(len(padding_list) // 2)]
        self.out_ranks = [int_from(len(padding_list) // 2)]
        assert len(
            self.padding_list) % 2 == 0, f'padding_list must be even, got {self.padding_list}'

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        pad = self.padding_list
        isv = input_shapes[0].shape
        cons = []
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            # When using negative padding, neither side should erase more than the original size
            cons.append(nnsmith_ge(nnsmith_add(pad[i * 2], isv[j]), 0))
            cons.append(nnsmith_ge(nnsmith_add(
                pad[i * 2 + 1], isv[j]), 0))
            cons.append(nnsmith_gt(nnsmith_add(
                pad[i * 2 + 1], nnsmith_add(pad[i * 2], isv[j])), 0))
        return cons

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        isv = input_shapes[0].shape
        pad = self.padding_list
        s = list(isv)
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            s[j] = nnsmith_add(nnsmith_add(
                s[j], pad[i * 2]), pad[i * 2 + 1])
        return [ShapeVar(s, input_shapes[0].dtype)]

    def torch(self) -> Callable[..., torch.Tensor]:
        if self.extra_attrs['type'] == 'constant':
            return lambda x: torch.nn.functional.pad(x, self.padding_list, 'constant', value=0)
        elif self.extra_attrs['type'] == 'replicate' or self.extra_attrs['type'] == 'reflect':
            return lambda x: torch.nn.functional.pad(x, self.padding_list, self.extra_attrs['type'])

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(out_shape_var[0].ndims, out_shape_var[0].dtype)]


class ConstPad(Pad):
    def __init__(self, *padding_list):
        super().__init__(padding_list, 'constant')


class ReplicatePad(Pad):
    num_var_param = _pad_num_var_param(2, max=6)

    def __init__(self, *padding_list):
        super().__init__(padding_list, 'replicate')
        self.inp_ranks = [int_range(len(padding_list) // 2 + 1, 4)]
        self.out_ranks = [int_range(len(padding_list) // 2 + 1, 4)]


class ReflectPad(Pad):
    num_var_param = _pad_num_var_param(2, max=6)

    def __init__(self, *padding_list):
        super().__init__(padding_list, 'reflect')
        self.inp_ranks = [int_range(len(padding_list) // 2 + 1, 4)]
        self.out_ranks = [int_range(len(padding_list) // 2 + 1, 4)]

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        cons = super()._requires(input_shapes)
        pad = self.padding_list
        isv = input_shapes[0].shape
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            # per torch's complaint: Padding size should be less than the corresponding input dimension
            cons.append(nnsmith_lt(pad[i * 2], isv[j]))
            cons.append(nnsmith_lt(pad[i * 2 + 1], isv[j]))
        return cons


class Expand(UnaryOpBase, ABC):
    in_dtypes = [(i,) for i in DTYPE_ALL]
    out_dtypes = [(i,) for i in DTYPE_ALL]
    # expand_dim cannot be symbolic. So just expand it.

    def __init__(self, expand_last_dim: int, expand_n: Union[int, z3.ExprRef]):
        """See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        """
        super().__init__()
        self.inp_ranks = [int_all()]
        SanityCheck.ge(expand_last_dim, 1)
        self.expand_last_dim = expand_last_dim
        self.expand_n = expand_n

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        if self.expand_last_dim <= len(input_shapes[0].shape):
            # NOTE: Werid, deepcopy is useless here.
            shape = ShapeVar(shape=[*input_shapes[0].shape],
                             dtype=input_shapes[0].dtype)
            shape.shape[-self.expand_last_dim] = self.expand_n
            return [shape]
        else:  # expand it;
            # for example. we have:
            #       input shape [u, v]
            #       expand_last_dim <- 4
            #       return [expand_n, 1, u, v] where `1` is padded.
            dtype = input_shapes[0].dtype
            return [ShapeVar([self.expand_n, *([1] * (self.expand_last_dim - len(input_shapes[0].shape) - 1)), *input_shapes[0].shape], dtype)]

    def _requires(self, input_shapes):
        SanityCheck.ge(self.expand_last_dim, 1)

        input_shape = input_shapes[0].shape
        if isinstance(self.expand_n, z3.ExprRef):
            if self.expand_last_dim <= len(input_shape):  # index valid
                cons = [
                    nnsmith_eq(
                        input_shape[-self.expand_last_dim], 1),
                    nnsmith_ge(self.expand_n, 1)]
                return cons
            return [nnsmith_ge(self.expand_n, 1)]
        else:
            # It is also valid to expand to 0. But just too tricky...
            ConstraintCheck.ge(self.expand_n, 1)
            if self.expand_last_dim <= len(input_shape):
                ConstraintCheck.true(input_shape[-self.expand_last_dim] ==
                                     1 or input_shape[-self.expand_last_dim] == self.expand_n)
        return []

    def torch(self):
        return lambda x: x.expand(*self._shape_fn([ShapeVar.from_torch(x)])[0].shape)

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        inp_rank = self.expand_last_dim if out_shape_var[
            0].ndims < self.expand_last_dim else out_shape_var[0].ndims
        ConstraintCheck.ge(out_shape_var[0].ndims, self.expand_last_dim)
        return [
            (inp_rank, out_shape_var[0].dtype)
        ]


class ExpandLast1(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=1, expand_n=expand_n)


class ExpandLast2(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=2, expand_n=expand_n)


class ExpandLast3(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=3, expand_n=expand_n)


class ExpandLast4(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=4, expand_n=expand_n)


class BatchNorm2d(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, nfeat):
        super().__init__()
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]
        self.nfeat = nfeat

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(4, DType.float32)]

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return [nnsmith_eq(self.nfeat, input_shapes[0].shape[1])]

    def torch(self) -> Callable[..., torch.Tensor]:
        return torch.nn.BatchNorm2d(num_features=self.nfeat)


class NCHWConv2d(UnaryOpBase):
    # FIXME: torch exporter does not support float64, may miss bugs
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self,
                 in_channels: Union[int, z3.ExprRef],
                 out_channels: Union[int, z3.ExprRef],
                 kernel_h_size: Union[int, z3.ExprRef],
                 kernel_w_size: Union[int, z3.ExprRef],
                 stride: Union[int, z3.ExprRef],
                 padding: Union[int, z3.ExprRef]):
        """See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.padding = padding

        self.inp_ranks = [(4,)]  # NC(H,)W
        self.out_ranks = [(4,)]  # NC(H,)W

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        # not symbolic
        if not isinstance(self.in_channels, z3.ExprRef) and not isinstance(input_shapes[0].shape[1], z3.ExprRef):
            ConstraintCheck.eq(input_shapes[0].shape[1], self.in_channels)

        is_symbolic_inp = input_shapes[0].constains_symbol() or isinstance(self.kernel_w_size, z3.ExprRef) or isinstance(
            self.kernel_h_size, z3.ExprRef) or isinstance(self.stride, z3.ExprRef) or isinstance(self.padding, z3.ExprRef)

        shape_var = ShapeVar(
            [input_shapes[0].shape[0], self.out_channels], dtype=input_shapes[0].dtype)
        if not is_symbolic_inp:
            shape_var.shape.append(
                (input_shapes[0].shape[2] - self.kernel_h_size + 2 * self.padding) // self.stride + 1)
            shape_var.shape.append(
                (input_shapes[0].shape[3] - self.kernel_w_size + 2 * self.padding) // self.stride + 1)
        else:
            shape_var.shape.append(
                (nnsmith_div(nnsmith_add(nnsmith_sub(input_shapes[0].shape[2], self.kernel_h_size), 2 * self.padding), self.stride) + 1))
            shape_var.shape.append(
                (nnsmith_div(nnsmith_add(nnsmith_sub(input_shapes[0].shape[3], self.kernel_w_size), 2 * self.padding), self.stride) + 1))
        return [shape_var]

    def _requires(self, input_shapes):
        cons = []
        ret = []
        # TODO: Use eager mode for debugging.
        cons.append(nnsmith_eq(self.in_channels, input_shapes[0].shape[1]))
        cons.append(nnsmith_ge(self.out_channels, 1))
        cons.append(nnsmith_ge(self.kernel_h_size, 1))
        cons.append(nnsmith_ge(self.kernel_w_size, 1))
        # TODO(JK): fix the dialation case for the kernel size constraints.
        cons.append(nnsmith_le(self.kernel_h_size,
                    nnsmith_add(input_shapes[0].shape[2], 2 * self.padding)))
        cons.append(nnsmith_le(self.kernel_w_size,
                    nnsmith_add(input_shapes[0].shape[3], 2 * self.padding)))
        cons.append(nnsmith_ge(self.stride, 1))
        cons.append(nnsmith_ge(self.padding, 0))
        # not too extream to avoid torch exporter issue
        cons.append(nnsmith_le(self.padding, 255))
        # limit FLOPS
        if FLOPS_LIM is not None:
            cons.append(nnsmith_le(self.flops(input_shapes), FLOPS_LIM))
        for c in cons:
            if isinstance(c, z3.ExprRef):
                ret.append(c)
            else:
                ConstraintCheck.true(c)
        return ret

    def torch(self):
        return torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_h_size, self.kernel_w_size), stride=self.stride,
                               padding=self.padding, device=_DEV)

    def flops(self, input_shapes):
        w = ShapeVar([self.out_channels, self.in_channels, self.kernel_h_size,
                     self.kernel_w_size], dtype=input_shapes[0].dtype)
        return nnsmith_mul(self._shape_fn(input_shapes)[0].nelement(), w.nelement())

    def n_floats(self, input_shapes):
        padded_data = ShapeVar(
            input_shapes[0].shape, dtype=input_shapes[0].dtype)
        padded_data.shape[2] = nnsmith_add(
            padded_data.shape[2], nnsmith_mul(2, self.padding))
        padded_data.shape[3] = nnsmith_add(
            padded_data.shape[3], nnsmith_mul(2, self.padding))
        w = ShapeVar([self.out_channels, self.in_channels, self.kernel_h_size,
                     self.kernel_w_size], dtype=input_shapes[0].dtype)
        outs = super().n_floats(input_shapes)
        return nnsmith_add(nnsmith_add(w.nelement(), padded_data.nelement()), outs)

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(4, out_shape_var[0].dtype)]


class ReshapeBase(UnaryOpBase):
    num_var_param = int_range(1, 4)
    in_dtypes = [(i,) for i in DTYPE_ALL]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self, *target_shape):
        super().__init__()
        self.inp_ranks = [int_range(1, 4)]
        self.out_ranks = [(len(target_shape), )]
        self.target_shape: List[Union[int, z3.ExprRef]] = target_shape

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        __MAX_SOLVE_SYMBOL__ = 8
        # otherwise OOM.
        ConstraintCheck.le(input_shapes[0].ndims +
                           len(self.target_shape), __MAX_SOLVE_SYMBOL__)

        if -1 not in self.target_shape:
            return [ShapeVar(self.target_shape, dtype=input_shapes[0].dtype)]
        # else
        shape_var = ShapeVar(self.target_shape, dtype=input_shapes[0].dtype)
        auto_dim = -1
        accum = 1
        for i, v in enumerate(self.target_shape):
            # TODO: What to do about bitvectors here?
            if v == -1:
                SanityCheck.eq(auto_dim, -1)
                auto_dim = i
            else:
                accum = nnsmith_mul(accum, v)

        # First see if there's any symbols in the expression
        symbol_indices = [
            v for v in input_shapes[0].shape if isinstance(v, z3.ExprRef)]
        if len(symbol_indices) == 0:
            shape_var.shape[auto_dim] = reduce(
                lambda x, y: x * y, input_shapes[0].shape, 1) // accum
        else:
            shape_var.shape[auto_dim] = nnsmith_div(reduce(
                lambda x, y: nnsmith_mul(x, y), input_shapes[0].shape, 1), accum)

        return [shape_var]

    def _requires(self, input_shapes):
        # TODO: How to handle -1 with input shapes?
        # If your target shape is concrete, then your output shape's total pixels must be the same as the input shape's.
        if -1 not in self.target_shape:
            total_pixels = reduce(
                lambda x, y: nnsmith_mul(x, y), self.target_shape, 1)
            cons = [nnsmith_eq(total_pixels, reduce(
                lambda x, y: nnsmith_mul(x, y), input_shapes[0].shape, 1))]
            for s in self.target_shape:
                cons.append(nnsmith_ge(s, 1))
            if os.getenv('NNSMITH_CONS_RESHAPE', 'on') != 'off':
                # should not be too extreme!
                __DIM_LIMIT__ = 4096
                lim = __DIM_LIMIT__
                for s in self.target_shape[::-1]:
                    cons.append(nnsmith_le(s, lim))
                    lim //= 2
                    lim = max(lim, 1)
            return cons
        else:
            # If you use auto mode (specifying -1 for some dimensions), then the total number of input pixels must be exactly divisible by that of the output shape.
            minimul_pixels = reduce(
                lambda x, y: nnsmith_mul(x, y), [v for v in self.target_shape if v != -1], 1)
            return [nnsmith_eq(nnsmith_mod(reduce(lambda x, y: nnsmith_mul(x, y), input_shapes[0].shape, 1), minimul_pixels), 0)]

    def torch(self):
        return lambda x: x.reshape(*self.target_shape)

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(-1, out_shape_var[0].dtype)]


class Reshape(ReshapeBase):
    pass


class Flatten(ReshapeBase):
    num_var_param = None
    # Inputs are target shape.

    def __init__(self, dim0: Union[int, z3.ExprRef]):
        super().__init__(1, dim0)
        self.dim0 = dim0

    def torch(self):
        # See https://github.com/pytorch/pytorch/issues/74142
        return lambda x: x.flatten().unsqueeze(0)


class Transpose(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self):
        """See https://pytorch.org/docs/stable/generated/torch.transpose.html
        """
        super().__init__()
        self.inp_ranks = [int_from(2)]
        self.out_ranks = [int_from(2)]

    def _init_swap_dims(self, input_shape: List[Union[int, z3.ExprRef]]):
        ConstraintCheck.ge(len(input_shape), 2)
        self.inp_ranks = [len(input_shape)]
        if 'dim0' not in self.extra_attrs or 'dim1' not in self.extra_attrs:
            max_dim = len(input_shape) - 1
            self.extra_attrs['dim0'] = random.randint(0, max_dim)
            self.extra_attrs['dim1'] = (random.randint(
                1, max_dim) + self.extra_attrs['dim0']) % (1 + max_dim)
        return self.extra_attrs['dim0'], self.extra_attrs['dim1']

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        dim0, dim1 = self._init_swap_dims(input_shapes[0].shape)
        shape = list(input_shapes[0].shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return [ShapeVar(shape, input_shapes[0].dtype)]

    def _requires(self, input_shapes):
        dim0, dim1 = self._init_swap_dims(input_shapes[0].shape)
        SanityCheck.ge(len(input_shapes[0].shape), max(
            dim0, dim1) + 1, f'dim={len(input_shapes[0].shape)}.transpose({dim0},{dim1})')
        return []

    def torch(self):
        def f(x: torch.Tensor):
            dim0, dim1 = self._init_swap_dims(list(x.shape))
            return x.transpose(dim0, dim1)
        return f

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(out_shape_var[0].ndims, out_shape_var[0].dtype)]

# Sum, Min, Max, Mean, ArgMin, ArgMax, Squeeze, Size


class InterpBase(UnaryOpBase):
    num_var_param = int_range(1, 3)

    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self, *size):
        super().__init__()
        self.size = size
        self.inp_ranks = [(len(size) + 2,)]
        self.out_ranks = [(len(size) + 2,)]

    def _requires(self, input_shapes: List[ShapeVar]):
        return [nnsmith_gt(v, 0) for v in self.size]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        shape = list(input_shapes[0].shape)
        for i in range(len(self.size)):
            shape[-(1 + i)] = self.size[-(1 + i)]
        return [ShapeVar(shape, input_shapes[0].dtype)]

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(out_shape_var[0].ndims, out_shape_var[0].dtype)]


class NearestInterp(InterpBase):
    def torch(self) -> Callable[..., torch.Tensor]:
        return lambda x: torch.nn.functional.interpolate(x, size=self.size, mode='nearest')


class LinearInterp(InterpBase):
    num_var_param = [1]

    def torch(self) -> Callable[..., torch.Tensor]:
        return lambda x: torch.nn.functional.interpolate(x, size=self.size, mode='linear')


class BilinearInterp(InterpBase):
    num_var_param = [2]

    def torch(self) -> Callable[..., torch.Tensor]:
        return lambda x: torch.nn.functional.interpolate(x, size=self.size, mode='bilinear')


class BicubicInterp(InterpBase):
    num_var_param = [2]

    def torch(self) -> Callable[..., torch.Tensor]:
        return lambda x: torch.nn.functional.interpolate(x, size=self.size, mode='bicubic')


class TrilinearInterp(InterpBase):
    num_var_param = [3]

    def torch(self) -> Callable[..., torch.Tensor]:
        return lambda x: torch.nn.functional.interpolate(x, size=self.size, mode='trilinear')


class ReduceBase(UnaryOpBase, ABC):
    _reduce_out_dtype = None  # None means same as input dtype

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_from(1)]
        self.out_ranks = [int_range(0, __MAX_RANK__ - 1)]

    def __str__(self) -> str:
        return super().__str__() + f'(dim={self.extra_attrs["reduce_dim"] if "reduce_dim" in self.extra_attrs else None})'

    def _init_reduce_dim(self, input_shape: List[Union[int, z3.ExprRef]]):
        if 'reduce_dim' not in self.extra_attrs:
            self.extra_attrs['reduce_dim'] = random.randint(
                0, max(0, len(input_shape) - 1))
        return self.extra_attrs['reduce_dim']

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        svar_list = []
        for i, v in enumerate(input_shapes[0].shape):
            if i != self._init_reduce_dim(input_shapes[0].shape):
                svar_list.append(v)
        return [ShapeVar(svar_list, input_shapes[0].dtype if self._reduce_out_dtype is None else self._reduce_out_dtype)]

    def _requires(self, input_shapes: List[ShapeVar]):
        reduce_dim = self._init_reduce_dim(input_shapes[0].shape)
        return []

    def _get_irank(self, orank):
        if orank == 0:
            return random.randint(0, 1)
        return orank + 1

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(self._get_irank(out_shape_var[0].ndims), out_shape_var[0].dtype)]


class Squeeze(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def _requires(self, input_shapes):
        reduce_dim = self._init_reduce_dim(input_shapes[0].shape)
        return [nnsmith_eq(input_shapes[0].shape[reduce_dim], 1)]

    def torch(self):
        return lambda x: x.squeeze(self.extra_attrs['reduce_dim'])


class ReduceSum(ReduceBase):
    # pytorch exporter doesn't support int32
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS if i != DType.int32]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS if i != DType.int32]

    def torch(self):
        return lambda x: x.sum(self.extra_attrs['reduce_dim'])


class ReduceMin(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def torch(self):
        return lambda x: x.min(self.extra_attrs['reduce_dim']).values


class ReduceMax(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def torch(self):
        return lambda x: x.max(self.extra_attrs['reduce_dim']).values


class ReduceMean(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def torch(self):
        return lambda x: x.mean(self.extra_attrs['reduce_dim'])


class ArgMin(ReduceBase):
    # FIXME(JK): ints are somehow not supported in onnxruntime, which we use to gen inputs.
    # Make it include ints once we use other backends other than onnxruntime.
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(DType.int64,)]
    _reduce_out_dtype = DType.int64

    def torch(self):
        return lambda x: x.argmin(self.extra_attrs['reduce_dim'])

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(self._get_irank(out_shape_var[0].ndims), random.choice(self.in_dtypes)[0])]


class ArgMax(ReduceBase):
    # FIXME(JK): ints are somehow not supported in onnxruntime, which we use to gen inputs.
    # Make it include ints once we use other backends other than onnxruntime.
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(DType.int64,)]
    _reduce_out_dtype = DType.int64

    def torch(self):
        return lambda x: x.argmax(self.extra_attrs['reduce_dim'])

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(self._get_irank(out_shape_var[0].ndims), random.choice(self.in_dtypes)[0])]


class Linear(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [int_from(1)]
        # at least one dim. cannot be zranks_all()
        self.out_ranks = [int_from(1)]

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        return [ShapeVar(shape=[*input_shapes[0].shape[:-1], self.ofeat], dtype=DType.float32)]

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat)
        ]

    def torch(self) -> Callable[..., torch.Tensor]:
        return torch.nn.Linear(in_features=self.ifeat, out_features=self.ofeat)

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(out_shape_var[0].ndims, DType.float32)]


def partialclass(cls, name, *args, **kwds) -> Type[AbsOpBase]:
    return type(name, (cls,),
                {'__init__': functools.partialmethod(cls.__init__, *args, **kwds)})


class Concat(AbsOpBase):
    MAX_ARITY = 5
    MAX_RANK = 5
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __str__(self) -> str:
        return 'Concat ' + str(self.extra_attrs)

    def __init__(self, arity):
        super().__init__()
        SanityCheck.le(arity, Concat.MAX_ARITY)
        self.arity = arity
        self.inp_ranks = [(int_from(1))] * arity
        self.out_ranks = [(int_from(1))]
        self.same_inp_dims = True

    def _init_concat_axis(self, input_shapes: List[ShapeVar]) -> int:
        if 'axis' not in self.extra_attrs:
            self.extra_attrs['axis'] = random.randint(
                0, input_shapes[0].ndims - 1)
        return self.extra_attrs['axis']

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        ndims = input_shapes[0].ndims
        SanityCheck.gt(ndims, self._init_concat_axis(input_shapes))
        for s in input_shapes:
            SanityCheck.eq(s.ndims, ndims)
        cons = []
        for d in range(ndims):
            if d != self._init_concat_axis(input_shapes):
                cons.extend(nnsmith_eq(s.shape[d], input_shapes[0].shape[d])
                            for s in input_shapes)
        return cons

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        SanityCheck.true(input_shapes[0].ndims > 0)
        axis = self._init_concat_axis(input_shapes)
        os = ShapeVar(input_shapes[0].shape, input_shapes[0].dtype)
        os.shape[axis] = reduce(
            nnsmith_add, [s.shape[axis] for s in input_shapes])
        return [os]

    def torch(self):
        axis = self.extra_attrs['axis']
        return lambda *args: torch.cat(args, dim=axis)

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [(out_shape_var[0].ndims, out_shape_var[0].dtype) for _ in range(self.arity)]


# the semantic of `in_dtypes` is not possible dtypes in "max rank". but simply in "rank". don't mess up the definition.
class Concat1(Concat):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(1)


class Concat2(Concat):
    in_dtypes = [(i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(2)


class Concat3(Concat):
    in_dtypes = [(i, i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(3)


class Concat4(Concat):
    in_dtypes = [(i, i, i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(4)


class Concat5(Concat):
    in_dtypes = [(i, i, i, i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(5)


class Cast(ElementWiseUnaryOp, ABC):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self, dtype):
        super().__init__()
        self.inp_ranks = [int_all()]
        self.out_ranks = [int_all()]
        self.extra_attrs = {'to': dtype}

    def __str__(self) -> str:
        return 'Cast ' + str(self.extra_attrs)

    def _requires(self, input_shapes: List[ShapeVar]) -> List[z3.ExprRef]:
        return []

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        assert len(input_shapes) == 1
        return [ShapeVar(input_shapes[0].shape, self.extra_attrs['to'])]

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        return [
            (out_shape_var[0].ndims, self.extra_attrs['to'])
        ]

    def torch(self):
        return lambda x: x.to(dtype=self.extra_attrs['to'].value)


class CastF32(Cast):
    out_dtypes = [(DType.float32,)]

    def __init__(self):
        super().__init__(DType.float32)


class CastF64(Cast):
    out_dtypes = [(DType.float64,)]

    def __init__(self):
        super().__init__(DType.float64)


class CastI32(Cast):
    out_dtypes = [(DType.int32,)]

    def __init__(self):
        super().__init__(DType.int32)


class CastI64(Cast):
    out_dtypes = [(DType.int64,)]

    def __init__(self):
        super().__init__(DType.int64)


class CastBool(Cast):
    out_dtypes = [(DType.bool,)]

    def __init__(self):
        super().__init__(DType.bool)


class Gemm(TernaryOpBase):
    # https://pytorch.org/docs/stable/generated/torch.addmm.html?highlight=addmm#torch.addmm
    in_dtypes = [(i, i, i) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_until(2), (2,), (2,)]
        self.out_ranks = [(2,)]

    def _set_or_get_extra_attrs(self, dtype=None):
        if 'alpha' not in self.extra_attrs:
            assert dtype is not None, 'dtype must be specified at the first time of this call'
            alpha = random.uniform(-2, 2)
            beta = random.uniform(-2, 2)
            if dtype in DTYPE_INTS:
                beta, alpha = int(beta), int(alpha)
            self.extra_attrs['alpha'] = alpha
            self.extra_attrs['beta'] = beta
        return self.extra_attrs

    def _requires(self, input_shapes: List[ShapeVar]):
        ConstraintCheck.true(input_shapes[0].ndims <= 2)
        out_shape = self.shape_fn(input_shapes)[0]
        cons = broadcast_to_cons(input_shapes[0].shape, out_shape.shape)

        # matmul constraint
        mat1, mat2 = input_shapes[1], input_shapes[2]
        cons.append(mat1.shape[1] == mat2.shape[0])
        self._set_or_get_extra_attrs(input_shapes[0].dtype.value)
        if FLOPS_LIM is not None:
            cons.append(nnsmith_le(self.flops(input_shapes), FLOPS_LIM))
        return cons

    def _shape_fn(self, input_shapes: List[ShapeVar]) -> List[ShapeVar]:
        mat1, mat2 = input_shapes[1], input_shapes[2]
        return [ShapeVar([mat1.shape[0], mat2.shape[1]], input_shapes[0].dtype)]

    def torch(self):
        extra_attrs = self._set_or_get_extra_attrs()
        return lambda *args: torch.addmm(*args, beta=extra_attrs['beta'], alpha=extra_attrs['alpha'])

    def flops(self, input_shapes):
        mat1, mat2 = input_shapes[1], input_shapes[2]
        return mat1.shape[0] * mat1.shape[1] * mat2.shape[1]

    def deduct_inp_ranks_and_dtype(self, out_shape_var: List[ShapeVar]) -> List[Tuple[int, DType]]:
        dtype = out_shape_var[0].dtype
        return [
            (-1, dtype),
            (2, dtype),
            (2, dtype)]


def _glob_leaf_op_classes() -> List[Type[AbsOpBase]]:
    ret = []

    def _glob_leaf_op_classes_rec(cls):
        nonlocal ret
        if cls is Input or cls is Constant:
            return
        for c in cls.__subclasses__():
            if c.__subclasses__():
                _glob_leaf_op_classes_rec(c)
            else:
                ret.append(c)
    _glob_leaf_op_classes_rec(AbsOpBase)
    return ret


def _glob_nonleaf_op_classes() -> List[Type[AbsOpBase]]:
    ret = []

    def _glob_nonleaf_op_classes_rec(cls):
        nonlocal ret
        if cls is Input or cls is Constant:
            return
        for c in cls.__subclasses__():
            if c.__subclasses__():
                _glob_nonleaf_op_classes_rec(c)
                ret.append(c)
    _glob_nonleaf_op_classes_rec(AbsOpBase)
    return ret


ALL_NON_LEAF_OP_TYPES = _glob_nonleaf_op_classes()
ALL_OP_TYPES = _glob_leaf_op_classes()
ALL_OP_STR2TYPE = {c.__name__: c for c in ALL_OP_TYPES}
EXPANDED_OP_V0 = [Constant, Cast]
EXPANDED_OP_V1 = [Concat, Constant, Expand, Reshape, ArgMax,
                  ArgMin, ReduceMax, ReduceMin, ReduceMean, Squeeze,
                  ReduceSum, TrigonometricOp]
EXPANDED_OP = EXPANDED_OP_V1  # points to latest version


def config_skip_op(skip_config):
    SKIP_FOR_BKEND = {
        'trt': [
            # unsupported
            'Xor',
            'Equal:bool,bool',
            'Gemm:int32,int32,int32',
            # 'Acos:float64', 'Asin:float64', 'Atan:float64', 'Ceil:float64',
            # 'Cos:float64', 'Sin:float64', 'Tan:float64', 'GELU:float64', 'LeakyReLU:float64',
            # 'Abs:int64', 'Abs:int32',
            # # buggy, see https://github.com/NVIDIA/TensorRT/issues/1781
            # 'Less', 'Greater', 'Equal',
            # buggy
            'LegacyConstant*',
        ],
        'tvm': [],
        'ort': [],
        'xla': [],
        'tch': [],
        'dummy': [],
    }
    print('skip config:', skip_config)
    skip_config = skip_config.split(',')
    skip = []
    for op in skip_config:
        if op.startswith('backend:'):
            skip.extend(SKIP_FOR_BKEND[op[len('backend:'):]])
        else:
            skip.append(op)
    for op_name_pattern in skip:
        skip_comb = None
        if op_name_pattern.find(':') != -1:
            op_name_pattern, skip_comb = op_name_pattern.split(':')
            skip_comb = skip_comb.split(',')
        op_name_pattern = op_name_pattern.lower()
        for op_name in fnmatch.filter(map(lambda x: x.__name__.lower(), ALL_OP_TYPES), op_name_pattern):
            op_id = [i.__name__.lower() for i in ALL_OP_TYPES].index(op_name)
            op = ALL_OP_TYPES[op_id]
            msg = ['skip op:', op_name]
            if skip_comb is not None:  # only skip some dtype combinations
                skip_comb = tuple(map(DType.from_str, skip_comb))
                msg += ['skip dtype combination:', skip_comb]
                assert skip_comb in op.in_dtypes, 'combination {} not found in op({}).in_dtypes: {}'.format(
                    skip_comb, op_name, op.in_dtypes)
                op.in_dtypes.remove(skip_comb)
            else:  # skip entire op
                msg += ['skip entire']
                op._skip = True
            print(*msg)


def _check_comb(comb: DTypeComb, op: AbsOpBase):
    inps = []
    for dtype, ndims in zip(comb, op.inp_ranks):
        ndim = min(ndims)
        # TODO use symbolic solver
        inps.append(torch.empty([2] * ndims, dtype=dtype.value))
    try:
        _ = op.torch()(*inps)
    except Exception as e:
        return False
    return True


def auto_infer_in_dtypes(verbose=False):
    global _INFERRED
    if _INFERRED:
        return
    _INFERRED = True
    _WHITE_LIST = (Input, Expand, NCHWConv2d, Reshape)

    def create_op(op_t: Type[AbsOpBase]):
        construct_param_dict = signature(op_t.__init__).parameters
        values = []
        for key, val in construct_param_dict.items():
            if key == 'self':
                continue
            values.append((key, 1))  # TODO consider type hints?
        return op_t(**dict(values))

    for op_t in ALL_OP_TYPES:
        if issubclass(op_t, _WHITE_LIST):
            continue
        if op_t.in_dtypes is not None:
            continue
        if verbose:
            print(f'Try auto inferring input dtype spec for `{op_t.__name__}`')
        valid_combs = None
        op = create_op(op_t)
        in_dtype_combs: List[DTypeComb] = itertools.product(
            DTYPE_ALL, repeat=len(op.inp_ranks))
        valid_combs = [
            comb for comb in in_dtype_combs if _check_comb(comb, op)]
        if len(valid_combs) == 0:
            raise RuntimeError(
                f'No valid input dtype combination found for `{op_t.__name__}`')

        if verbose:
            print('infered result:', valid_combs)
        if op_t.in_dtypes is not None:
            # we disable type promotion for bcast binary ops so the difference is fine
            if verbose and valid_combs != op_t.in_dtypes and not issubclass(op_t, (BcastBinaryOp1, BcastBinaryOp2, BcastBinaryOp3)):
                warnings.warn('Inferred result for `{}` different from given one.\nInferred={}\n, given={}'.format(
                    op_t.__name__, valid_combs, op_t.in_dtypes))
        else:
            op_t.in_dtypes = valid_combs


if __name__ == '__main__':
    # Test shape functions
    print(len(ALL_OP_TYPES), 'operators supported:')
    print(ALL_OP_STR2TYPE.keys())
    print('Non leaf ops: ', ALL_NON_LEAF_OP_TYPES)
    assert Reshape in ALL_OP_TYPES
    auto_infer_in_dtypes()

    # ReLU
    lhs = torch.relu(torch.randn(1, 1, 1, 1)).shape
    rhs = torch.Size(ReLU().shape_fn(
        [ShapeVar([1, 1, 1, 1], DType.float32)])[0].shape)
    assert lhs == rhs, f"{lhs} != {rhs}"

    # Add
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(2, 3, 4, 5)
    c = a + b
    assert c.shape == torch.Size(Add().shape_fn(
        [ShapeVar([2, 3, 4, 5], DType.float32), ShapeVar([2, 3, 4, 5], DType.float32)])[0].shape)

    # Expand
    source_shape = (4, 1)
    a = torch.randn(source_shape)
    abs_op = ExpandLast4(expand_n=2)
    assert a.expand(2, 1, *source_shape).shape == torch.Size(
        abs_op.shape_fn([ShapeVar(source_shape, DType.float32)])[0].shape)

    abs_op = ExpandLast1(expand_n=2)
    rhs = torch.Size(abs_op.shape_fn(
        [ShapeVar(list(source_shape), DType.float32)])[0].shape)
    lhs = a.expand(4, 2).shape
    assert lhs == rhs, f"{lhs} != {rhs}"

    # NCHWConv2d
    source_shape = (2, 3, 24, 24)
    a = torch.randn(*source_shape)
    out = torch.conv2d(a, torch.randn(3, 3, 3, 4), stride=1, padding=1)
    assert out.shape == NCHWConv2d(
        3, 3, 3, 4, 1, 1).shape_fn([ShapeVar(source_shape, DType.float32)])[0].torch()
    print(NCHWConv2d(
        3, 3, 3, 4, 1, 1).shape_fn([ShapeVar([2, *z3.Ints('c h w')], DType.float32)])[0])

    # Reshape
    source_shape = (2, 3, 4)
    target_shape = (1, 2, 3, 2, 2)
    a = torch.randn(*source_shape)
    assert a.reshape(*target_shape).shape == Reshape(*target_shape).shape_fn(
        [ShapeVar(source_shape, DType.float32)])[0].torch()

    # Dirty fix for z3 bug by wrapping the context using seprated functions.
    def test_reshape_symbol():  # See https://github.com/Z3Prover/z3/issues/989
        s = z3.Solver()
        v = z3.Ints('a b c d e')
        abs_op = Reshape(*v)
        cons = abs_op.requires([ShapeVar(source_shape, DType.float32)])
        for c in cons:
            s.add(c)
        for c in abs_op.shape_fn([ShapeVar(source_shape, DType.float32)])[0].gt_zero():
            s.add(c)
        assert s.check() == z3.sat
        print(s.model())
    test_reshape_symbol()

    # Test `concrete` function.
    p0, p1, p2, p3, p4, p5 = z3.Ints('p0 p1 p2 p3 p4 p5')
    op = NCHWConv2d(p0, p1, p2, p3, p4, p5)
    s = z3.Solver()
    shape = ShapeVar([1, 3, 224, 224], DType.float32)
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

    # Test `concrete` function.
    p0, p1, p2, p3 = z3.Ints('p0 p1 p2 p3')
    op = AvgPool2d(p0, p1, p2, p3)
    s = z3.Solver()
    shape = ShapeVar([1, 3, 224, 224], DType.float32)
    for c in op.requires([shape]):
        s.add(c)
    for c in op.shape_fn([shape])[0].gt_zero():
        s.add(c)
    assert s.check() == z3.sat
    model = s.model()
    concrete_op = concretize(op, model)
    assert concrete_op.kernel_h_size == model[p0].as_long()
    assert concrete_op.kernel_w_size == model[p1].as_long()
    assert concrete_op.stride == model[p2].as_long()
    assert concrete_op.padding == model[p3].as_long()
