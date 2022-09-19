import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from inspect import signature
from typing import Dict, List, Optional, Tuple, Type, Union

import z3

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DTYPE_ALL, DTYPE_FLOATS, DTYPE_NON_BOOLS, DType
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import ConstraintCheck, SanityCheck

# There are following types of constraints at this point:
# 1. Shape variables must be greater than 0;
# 2. Shape variables must avoid devision by 0;
# 3. Intra-input shape constraints; e.g., add(x, y) where x.shape() must be equal to y.shape();
# 4. Extra constraints introduced by individual operators;

FLOPS_LIM = os.getenv("NNSMITH_FLOPS_LIM", "auto")
if FLOPS_LIM == "auto":  # use predefined value
    FLOPS_LIM = 2**30
elif FLOPS_LIM == "off":
    FLOPS_LIM = None
else:
    FLOPS_LIM = float(FLOPS_LIM)

# control wheter to model FLOPS in z3 too. If not, we will check it after model is concretized.
Z3_CONS_FLOPS = os.getenv("NNSMITH_Z3_CONS_FLOPS", "on")
assert Z3_CONS_FLOPS in [
    "on",
    "off",
], "NNSMITH_Z3_CONS_FLOPS must be either 'on' or 'off'"
Z3_CONS_FLOPS = Z3_CONS_FLOPS == "on"


__MIN_RANK__ = 0
__MAX_RANK__ = 5

FULL_OPERATOR_SETS: Dict[str, List[Type["AbsOpBase"]]] = dict()

_PRAGMA_ONCE_CORE_OP = True  # have to use this dirty hack to avoid namespace violation.


class mark_abstract:
    def __init__(self, dialect):
        assert (
            dialect != "core" or _PRAGMA_ONCE_CORE_OP
        ), "`core` is exclusive to nnsmith.abstract.op. Please use other dilect names"
        self.dialect = dialect

    def __call__(self, op_type: Type["AbsOpBase"]) -> Type["AbsOpBase"]:
        op_type.dialect = self.dialect
        return op_type


class mark_materialize:
    def __init__(self, dialect: str):
        self.dialect = dialect

    def __call__(self, op_type: Type["AbsOpBase"]) -> Type["AbsOpBase"]:
        op_list = FULL_OPERATOR_SETS.setdefault(self.dialect, [])

        if op_type not in op_list:
            op_list.append(op_type)
            op_list.sort(key=lambda x: x.__name__)
            op_type = mark_abstract(self.dialect)(op_type)

        return op_type


def int_from(start):
    return tuple(range(start, __MAX_RANK__ + 1))


def int_range(start, end):
    return tuple(range(start, end + 1))


def int_until(end):
    return tuple(range(__MIN_RANK__, end + 1))


def int_all():
    return tuple(range(__MIN_RANK__, __MAX_RANK__ + 1))


def check_shape_fn(func):
    def wrapper_check_shape_fn(self, input_shapes):
        SanityCheck.true(
            self.out_ranks,
            "Empty output dimensions in {}".format(self.__class__.__name__),
        )
        SanityCheck.eq(
            len(input_shapes),
            len(self.inp_ranks),
            "{} requires {} inputs, but got {}".format(
                self.__class__.__name__, len(self.inp_ranks), len(input_shapes)
            ),
        )
        res = func(self, [s.deepcopy() for s in input_shapes])
        SanityCheck.eq(
            len(res),
            len(self.out_ranks),
            "{} requires {} outputs, but got {}".format(
                self.__class__.__name__, len(self.out_ranks), len(res)
            ),
        )
        return res

    return wrapper_check_shape_fn


def check_require_fn(func):
    def wrapper_check_require_fn(self, input_shapes: List[AbsTensor]):
        SanityCheck.eq(
            len(input_shapes),
            len(self.inp_ranks),
            "{} requires {} inputs, but got {}".format(
                self.__class__.__name__, len(self.inp_ranks), len(input_shapes)
            ),
        )
        return func(self, [s.deepcopy() for s in input_shapes])

    return wrapper_check_require_fn


def _prepend_to(x, max_dim):
    return [1 for i in range(max_dim - len(x))] + x


def z3_bcast(
    x: Union[int, z3.ExprRef], y: Union[int, z3.ExprRef], *args: Union[int, z3.ExprRef]
):
    x, y = align_bvs(x, y)
    return (
        z3.simplify(z3.If(nnsmith_eq(y, 1), x, y))
        if len(args) == 0
        else z3_bcast(z3_bcast(x, y), *args)
    )


def broadcast_shapes(
    *shapes: List[Union[z3.ExprRef, int]]
) -> List[Union[z3.ExprRef, int]]:
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


def broadcast_cons(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.BoolRef]:
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
                        z3.Or(nnsmith_eq(x[i], tgt_shape[i]), nnsmith_eq(x[i], 1))
                    )
            axis_cons = z3.simplify(z3.And(*axis_cons))
            cons.append(axis_cons)
        else:
            args_dim_sz = [_prepend_to(x, max_dim)[i] for x in shapes]
            valid = all(
                nnsmith_eq(s, tgt_shape[i]) or nnsmith_eq(s, 1) for s in args_dim_sz
            )
            # TODO(JK): enable this after fixing issue #2
            # assert valid, "Invalid broadcast shapes {}. Specific dim sizes: {}".format(shapes, args_dim_sz)
            cons.append(z3.BoolVal(valid))
    return cons


def broadcast_cons_binary(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.BoolRef]:
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
            cons.append(
                z3.simplify(
                    z3.Or(
                        nnsmith_eq(lhs[i], 1),
                        nnsmith_eq(rhs[i], 1),
                        nnsmith_eq(lhs[i], rhs[i]),
                    )
                )
            )
        else:
            valid = (
                nnsmith_eq(lhs[i], 1)
                or nnsmith_eq(rhs[i], 1)
                or nnsmith_eq(lhs[i], rhs[i])
            )
            # TODO(JK): enable this after fixing issue #2
            # assert valid, "Invalid broadcast shapes lhs={}, rhs={}".format(lhs, rhs)
            cons.append(z3.BoolVal(valid))
    return cons


def broadcast_to_cons(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.BoolRef]:
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
                cons.append(
                    z3.simplify(
                        z3.Or(nnsmith_eq(src[i], 1), nnsmith_eq(src[i], tgt[i]))
                    )
                )
            else:
                valid = nnsmith_eq(src[i], 1) or nnsmith_eq(src[i], tgt[i])
                # TODO(JK): enable this after fixing issue #2
                # assert valid, "Invalid broadcast shapes lhs={}, rhs={}".format(lhs, rhs)
                cons.append(z3.BoolVal(valid))
    return cons


@mark_abstract("core")
class AbsOpBase(ABC):
    # number of parameters; None means it's fixed that can be inferred through `signature`.
    num_var_param = None
    # whether this op is broadcastable or not
    bcastable = False
    # input dtypes: enumerates all possible input dtype combinations. Size of the list is the number of combinations.
    # Each element is a tuple of allowed input dtypes. NOTE: len(list) can >= the # of inputs, for handling ops with arbitrary arity.
    # For example, [(DType.float32, DType.float32), (DType.float64, DType.float64), (DType.int32, DType.int32)] means that
    # this op can accept one of float32xfloat32, float64xfloat64, and int32xint32 as input dtypes.
    in_dtypes: List[Tuple[DType, ...]] = None  # Overwrite me!
    out_dtypes: List[Tuple[DType, ...]] = None
    # whether to disable the op during graph generation
    _skip = False

    dialect = None

    def __init__(self):
        assert self.dialect, "Set dialect with `mark_dialect` or `mark_realize`"

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
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        raise NotImplementedError

    @check_shape_fn  # Public API.
    def checked_type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        self.last_outs = self.type_transfer(input_shapes)
        return self.last_outs

    # Overload me!
    # Extra constraints for the input tensors.
    # Exception means rejection.
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return []

    @abstractmethod
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        raise NotImplementedError

    @check_require_fn  # Public API.
    def checked_requires(self, input_shapes):
        return self.requires(input_shapes)

    def n_floats(self, input_shapes: List[AbsTensor]) -> z3.ExprRef:
        return reduce(nnsmith_add, [i.nelement() for i in self.last_outs])

    def flops(self, input_shapes):
        return 0

    def __repr__(self) -> str:
        return self.__class__.__name__

    @classmethod
    def __str__(cls) -> str:
        return cls.name()

    @classmethod
    def name(cls) -> str:
        if hasattr(cls, "dialect"):
            return cls.dialect + "." + cls.__name__.split(".")[-1]
        return cls.__name__.split(".")[-1]


def concretize_op(op: AbsOpBase, model: Optional[z3.ModelRef]) -> AbsOpBase:
    if isinstance(op, Constant) or isinstance(op, Input):
        ret_op = deepcopy(op)
        values = []

        for idx, s in enumerate(op.abs_tensor.shape):
            if isinstance(s, z3.ExprRef):
                ret_op.abs_tensor.shape[idx] = model.eval(s).as_long()

        return ret_op

    # Non-inp / const types.
    construct_param_dict = signature(op.__init__).parameters
    values = []
    symbolic_idx = []

    if op.num_var_param is not None:
        # input is a variable list.
        key = list(construct_param_dict.keys())[0]
        values = list(getattr(op, key))
        symbolic_idx = [
            i for i in range(len(values)) if isinstance(values[i], z3.ExprRef)
        ]
    else:
        for idx, key in enumerate(construct_param_dict):
            param = getattr(op, key)
            values.append(param)
            if isinstance(param, z3.ExprRef):
                symbolic_idx.append(idx)

    for idx in symbolic_idx:
        values[idx] = model.eval(values[idx]).as_long()

    concrete_op = type(op)(*values)
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

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]


def bcast_rand_ndims(num_svars, target_ndims):
    res = [random.randint(0, target_ndims) for _ in range(num_svars)]
    res[random.randint(0, num_svars - 1)] = target_ndims
    return res


class BcastBinaryOp(BinaryOpBase):
    bcastable = True
    # by default, output dtype is the same as the first input dtype
    _bcast_out_dtypes = None

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_all(), int_all()]
        self.same_inp_dims = False
        self.bcastable = True

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        tgt_shape = broadcast_shapes(*(ish.shape for ish in input_shapes))
        dtype = (
            input_shapes[0].dtype
            if self._bcast_out_dtypes is None
            else self._bcast_out_dtypes[0]
        )
        return [AbsTensor(tgt_shape, dtype)]

    def requires(self, input_shapes):
        return broadcast_cons_binary(*(ish.shape for ish in input_shapes))

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        x, y = bcast_rand_ndims(2, out_abs_tensor[0].ndims)
        return [
            (x, out_abs_tensor[0].dtype),
            (y, out_abs_tensor[0].dtype),
        ]


class BcastBinaryOp1(BcastBinaryOp):  # +-*/ max min
    in_dtypes = [(i, i) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    _bcast_out_dtypes = None


class Comparator(BcastBinaryOp):  # > < =
    in_dtypes = [(i, i) for i in DTYPE_ALL]
    out_dtypes = [(DType.bool,)]
    _bcast_out_dtypes = [DType.bool]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        x, y = bcast_rand_ndims(2, out_abs_tensor[0].ndims)
        in_dtypes = random.choice(self.in_dtypes)
        return [
            (x, in_dtypes[0]),
            (y, in_dtypes[1]),
        ]


class Logical(BcastBinaryOp):  # logical and or xor
    in_dtypes = [(DType.bool, DType.bool)]
    out_dtypes = [(DType.bool,)]
    _bcast_out_dtypes = [DType.bool]


@mark_materialize("core")
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

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        # assert len(input_shapes[0].shape) == len(input_shapes[1].shape)
        tgt_shape = broadcast_shapes(*(ish.shape for ish in input_shapes))
        dtype = input_shapes[1].dtype
        return [AbsTensor(tgt_shape, dtype)]

    def requires(self, input_shapes):
        return broadcast_cons(*(ish.shape for ish in input_shapes)) + [
            z3.BoolVal(input_shapes[1].dtype == input_shapes[2].dtype)
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        x, y, z = bcast_rand_ndims(3, out_abs_tensor[0].ndims)
        return [
            (x, DType.bool),
            (y, out_abs_tensor[0].dtype),
            (z, out_abs_tensor[0].dtype),
        ]


# bcast binary ops from https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
# TODO bitwise_and/or/xor?
Add = mark_materialize("core")(
    type(
        "Add",
        (BcastBinaryOp1,),
        {"__module__": __name__},
    )
)
Sub = mark_materialize("core")(
    type(
        "Sub",
        (BcastBinaryOp1,),
        {"__module__": __name__},
    )
)
Mul = mark_materialize("core")(
    type(
        "Mul",
        (BcastBinaryOp1,),
        {"__module__": __name__},
    )
)
# NOTE(JK): didn't find multi-input version of Max and Min in torch, so assume binary ops
Max = mark_materialize("core")(
    type(
        "Max",
        (BcastBinaryOp1,),
        {"__module__": __name__},
    )
)
Min = mark_materialize("core")(
    type(
        "Min",
        (BcastBinaryOp1,),
        {"__module__": __name__},
    )
)

Equal = mark_materialize("core")(type("Equal", (Comparator,), {"__module__": __name__}))
Greater = mark_materialize("core")(
    type(
        "Greater",
        (Comparator,),
        {"__module__": __name__},
    )
)
Less = mark_materialize("core")(type("Less", (Comparator,), {"__module__": __name__}))
And = mark_materialize("core")(
    type(
        "And",
        (Logical,),
        {"__module__": __name__},
    )
)
Or = mark_materialize("core")(
    type(
        "Or",
        (Logical,),
        {"__module__": __name__},
    )
)
Xor = mark_materialize("core")(
    type(
        "Xor",
        (Logical,),
        {"__module__": __name__},
    )
)


class Input(AbsOpBase):
    in_dtypes = [()]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self, dim: int):
        super().__init__()
        self.inp_ranks = []
        self.out_ranks = [(dim,)]
        self.abs_tensor: AbsTensor = None

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 0)
        return [self.abs_tensor]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        SanityCheck.eq(len(input_shapes), 0)
        return []

    def __str__(self):
        return "Input"

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        pass


class Constant(AbsOpBase):
    in_dtypes = [()]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __str__(self) -> str:
        return self.name() + " " + str(self.extra_attrs).replace(":", "=")

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.inp_ranks = []
        self.out_ranks = [(dim,)]
        self.abs_tensor: AbsTensor = None

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 0)
        return [self.abs_tensor]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        SanityCheck.eq(len(input_shapes), 0)
        return []

    def __str__(self):
        return "Constant"

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        pass


class Placeholder:
    def __init__(self, out_shape: AbsTensor):
        self.out_shape = out_shape
        self.inp_ranks = []
        self.out_ranks = [(out_shape.ndims,)]

    def __repr__(self):
        return f"Placeholder({self.out_shape})"

    def to_const(self):
        const_node = Constant(self.out_shape.ndims)
        const_node.abs_tensor = self.out_shape
        return const_node

    def to_input(self):
        input_node = Input(self.out_shape.ndims)
        input_node.abs_tensor = self.out_shape
        return input_node

    def __str__(self):
        return "Placeholder"


# FIXME: Div will cause fuzzing crash. No integer to avoid division by zero.
@mark_materialize("core")
class Div(BcastBinaryOp):
    in_dtypes = [(i, i) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Pow(BcastBinaryOp):
    in_dtypes = [(i, i) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class GELU(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class LeakyReLU(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self):
        """See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html"""
        super().__init__()
        self.negative_slope = 0.01


@mark_materialize("core")
class PReLU(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]


@mark_materialize("core")
class Sigmoid(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


class TrigonometricOp(ElementWiseUnaryOp):
    pass


@mark_materialize("core")
class Sin(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Cos(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Asin(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Acos(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Tan(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Atan(TrigonometricOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Abs(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]


@mark_materialize("core")
class ReLU(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Ceil(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Floor(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Clip(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]


@mark_materialize("core")
class Round(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Sqrt(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Log2(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class Neg(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]


@mark_materialize("core")
class Softmax(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self, dim: Union[int, z3.ExprRef]):
        super().__init__()
        self.dim = dim
        self.inp_ranks = [int_from(1)]
        self.out_ranks = [int_from(1)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [nnsmith_lt(self.dim, input_shapes[0].ndims), nnsmith_ge(self.dim, 0)]


class Pool2d(UnaryOpBase):
    # TODO: distinguish stride_h and stride_w
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(
        self,
        kernel_h_size: Union[int, z3.ExprRef],
        kernel_w_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        padding: Union[int, z3.ExprRef],
    ):
        super().__init__()
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.padding = padding

        self.inp_ranks = [(4,)]  # NCHW
        self.out_ranks = [(4,)]  # NCHW

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:

        abs_tensor = AbsTensor([], dtype=input_shapes[0].dtype)
        # Batch dim: just copy
        abs_tensor.shape.append(input_shapes[0].shape[0])
        # Output channels
        abs_tensor.shape.append(input_shapes[0].shape[1])
        abs_tensor.shape.append(
            (
                nnsmith_div(
                    nnsmith_add(
                        nnsmith_sub(input_shapes[0].shape[2], self.kernel_h_size),
                        2 * self.padding,
                    ),
                    self.stride,
                )
                + 1
            )
        )
        abs_tensor.shape.append(
            (
                nnsmith_div(
                    nnsmith_add(
                        nnsmith_sub(input_shapes[0].shape[3], self.kernel_w_size),
                        2 * self.padding,
                    ),
                    self.stride,
                )
                + 1
            )
        )
        return [abs_tensor]

    def requires(self, input_shapes):
        cons = []
        ret = []
        cons.append(nnsmith_ge(self.kernel_h_size, 1))
        cons.append(nnsmith_ge(self.kernel_w_size, 1))
        cons.append(
            nnsmith_le(
                self.kernel_h_size,
                nnsmith_add(input_shapes[0].shape[2], 2 * self.padding),
            )
        )
        cons.append(
            nnsmith_le(
                self.kernel_w_size,
                nnsmith_add(input_shapes[0].shape[3], 2 * self.padding),
            )
        )
        cons.append(nnsmith_ge(self.stride, 1))
        cons.append(nnsmith_ge(self.padding, 0))
        # not too extream to avoid torch exporter issue
        cons.append(nnsmith_le(self.padding, 255))
        cons.append(nnsmith_le(self.padding, nnsmith_div(self.kernel_h_size, 2)))
        cons.append(nnsmith_le(self.padding, nnsmith_div(self.kernel_w_size, 2)))

        # limit FLOPS
        if Z3_CONS_FLOPS:
            cons.append(nnsmith_le(self.flops(input_shapes), FLOPS_LIM))
        for c in cons:
            ret.append(c)
        return ret

    def flops(self, input_shapes):
        return nnsmith_mul(
            nnsmith_mul(
                self.checked_type_transfer(input_shapes)[0].nelement(),
                self.kernel_h_size,
            ),
            self.kernel_w_size,
        )

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype)]


@mark_materialize("core")
class MaxPool2d(Pool2d):
    pass


@mark_materialize("core")
class AvgPool2d(Pool2d):
    pass


@mark_materialize("core")
class Slice(UnaryOpBase):
    # pytorch slice always exported as a stack of single-dim slices, so only model sinlge-dim slice here
    # pytorch slice only supports forward slicing, so only model forward slicing here
    in_dtypes = [(i,) for i in DTYPE_ALL]
    INT_MAX = 2**63 - 1
    INT_MIN = -(2**63)

    def __init__(self, start, end, step):
        super().__init__()
        self.inp_ranks = [int_from(1)]
        self.out_ranks = [int_from(1)]
        self.start = start
        self.end = end
        self.step = step

    def __str__(self) -> str:
        if "axis" in self.extra_attrs:
            tail = {
                "axis": self.extra_attrs["axis"],
                "region": self.extra_attrs["region"],
            }
        else:
            tail = {}
        if isinstance(self.start, int):
            tail["start"] = self.start
        if isinstance(self.end, int):
            tail["end"] = self.end
        if isinstance(self.step, int):
            tail["step"] = self.step
        return self.name() + " " + str(tail).replace(":", "=")

    def _get_attrs(self, ndims):
        ConstraintCheck.true(ndims > 0)
        if "axis" not in self.extra_attrs:
            self.extra_attrs["ndims"] = ndims
            self.extra_attrs["axis"] = random.randint(0, ndims - 1)
            # specifying the region of the start and end pointer.
            # start \in [0, dim_s-1] if region=='right' else [-dim_s, -1]
            # end \in [-dim_s, -1] if region=='left' else [0, dim_s]
            self.extra_attrs["region"] = random.choice(["left", "mid", "right"])
            if random.uniform(0, 1) < 0.1:
                # torch exporter does not support start=INT_MIN
                # if random.uniform(0, 1) < 0.5:
                #     # because pytorch only supports forward slicing,
                #     # start cannot be INT_MAX, otherwise it slices empty tensor
                #     self.start = self.INT_MIN
                # else:
                self.end = self.INT_MAX
        return self.extra_attrs["axis"]

    def requires(self, input_shapes: List[AbsTensor]):
        inp = input_shapes[0]
        axis = self._get_attrs(inp.ndims)
        reg = self.extra_attrs["region"]
        cons = []
        dim_s = inp.shape[axis]
        # range for start
        l, r = (0, nnsmith_sub(dim_s, 1))
        # range for end
        ll, rr = (0, dim_s)
        assert not isinstance(self.start, int)
        cons.append(
            z3.And(  # start \in [l, r]
                nnsmith_ge(self.start, l), nnsmith_le(self.start, r)
            )
        )
        if not isinstance(self.end, int):
            cons.append(
                z3.And(  # end \in [ll, rr]
                    nnsmith_ge(self.end, ll), nnsmith_le(self.end, rr)
                )
            )
            cons.append(nnsmith_gt(self.end, self.start))
        else:
            assert self.end == self.INT_MAX

        cons.append(nnsmith_ge(self.step, 1))  # forward slicing only
        cons.append(nnsmith_le(self.step, dim_s))
        return cons

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        inp = input_shapes[0]
        axis = self._get_attrs(inp.ndims)
        s = list(inp.shape)
        end = self.end
        if self.end == Slice.INT_MAX:
            end = inp.shape[axis]
        s[axis] = nnsmith_div(
            nnsmith_add(nnsmith_sub(end, self.start), nnsmith_sub(self.step, 1)),
            self.step,
        )
        return [AbsTensor(s, input_shapes[0].dtype)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]


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

    def __str__(self) -> str:
        return f"{self.name()} (padding={list(self.padding_list)})"

    def __init__(self, padding_list, pad_t):
        super().__init__()
        self.padding_list = padding_list
        self.extra_attrs["type"] = pad_t
        self.inp_ranks = [int_from(len(padding_list) // 2)]
        self.out_ranks = [int_from(len(padding_list) // 2)]
        assert (
            len(self.padding_list) % 2 == 0
        ), f"padding_list must be even, got {self.padding_list}"

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        pad = self.padding_list
        isv = input_shapes[0].shape
        cons = []
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            # When using negative padding, neither side should erase more than the original size
            cons.append(nnsmith_gt(nnsmith_add(pad[i * 2], isv[j]), 0))
            cons.append(nnsmith_gt(nnsmith_add(pad[i * 2 + 1], isv[j]), 0))
            cons.append(
                nnsmith_gt(
                    nnsmith_add(pad[i * 2 + 1], nnsmith_add(pad[i * 2], isv[j])), 0
                )
            )
        return cons

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        isv = input_shapes[0].shape
        pad = self.padding_list
        s = list(isv)
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            s[j] = nnsmith_add(nnsmith_add(s[j], pad[i * 2]), pad[i * 2 + 1])
        return [AbsTensor(s, input_shapes[0].dtype)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]


@mark_materialize("core")
class ConstPad(Pad):
    def __init__(self, *padding_list):
        super().__init__(padding_list, "constant")


@mark_materialize("core")
class ReplicatePad(Pad):
    num_var_param = _pad_num_var_param(2, max=6)

    def __init__(self, *padding_list):
        super().__init__(padding_list, "replicate")
        self.inp_ranks = [int_range(len(padding_list) // 2 + 1, 4)]
        self.out_ranks = [int_range(len(padding_list) // 2 + 1, 4)]


@mark_materialize("core")
class ReflectPad(Pad):
    num_var_param = _pad_num_var_param(2, max=6)

    def __init__(self, *padding_list):
        super().__init__(padding_list, "reflect")
        self.inp_ranks = [int_range(len(padding_list) // 2 + 1, 4)]
        self.out_ranks = [int_range(len(padding_list) // 2 + 1, 4)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        cons = super().requires(input_shapes)
        pad = self.padding_list
        isv = input_shapes[0].shape
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            # per torch's complaint: Padding size should be less than the corresponding input dimension
            cons.append(nnsmith_lt(pad[i * 2], isv[j]))
            cons.append(nnsmith_lt(pad[i * 2 + 1], isv[j]))
            # same sign to avoid ORT bugs
            cons.append(nnsmith_ge(pad[i * 2] * pad[i * 2 + 1], 0))
        return cons


class Expand(UnaryOpBase, ABC):
    in_dtypes = [(i,) for i in DTYPE_ALL]
    out_dtypes = [(i,) for i in DTYPE_ALL]
    # expand_dim cannot be symbolic. So just expand it.

    def __init__(self, expand_last_dim: int, expand_n: Union[int, z3.ExprRef]):
        """See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html"""
        super().__init__()
        self.inp_ranks = [int_all()]
        SanityCheck.ge(expand_last_dim, 1)
        self.expand_last_dim = expand_last_dim
        self.expand_n = expand_n

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        if self.expand_last_dim <= len(input_shapes[0].shape):
            # NOTE: Werid, deepcopy is useless here.
            shape = AbsTensor(
                shape=[*input_shapes[0].shape], dtype=input_shapes[0].dtype
            )
            shape.shape[-self.expand_last_dim] = self.expand_n
            return [shape]
        else:  # expand it;
            # for example. we have:
            #       input shape [u, v]
            #       expand_last_dim <- 4
            #       return [expand_n, 1, u, v] where `1` is padded.
            dtype = input_shapes[0].dtype
            return [
                AbsTensor(
                    [
                        self.expand_n,
                        *(
                            [1]
                            * (self.expand_last_dim - len(input_shapes[0].shape) - 1)
                        ),
                        *input_shapes[0].shape,
                    ],
                    dtype,
                )
            ]

    def requires(self, input_shapes):
        SanityCheck.ge(self.expand_last_dim, 1)

        input_shape = input_shapes[0].shape
        if self.expand_last_dim <= len(input_shape):  # index valid
            cons = [
                nnsmith_eq(input_shape[-self.expand_last_dim], 1),
                nnsmith_ge(self.expand_n, 1),
            ]
            return cons
        return [nnsmith_ge(self.expand_n, 1)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        inp_rank = (
            self.expand_last_dim
            if out_abs_tensor[0].ndims < self.expand_last_dim
            else out_abs_tensor[0].ndims
        )
        ConstraintCheck.ge(out_abs_tensor[0].ndims, self.expand_last_dim)
        return [(inp_rank, out_abs_tensor[0].dtype)]


@mark_materialize("core")
class ExpandLast1(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=1, expand_n=expand_n)


@mark_materialize("core")
class ExpandLast2(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=2, expand_n=expand_n)


@mark_materialize("core")
class ExpandLast3(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=3, expand_n=expand_n)


@mark_materialize("core")
class ExpandLast4(Expand):
    def __init__(self, expand_n: Union[int, z3.ExprRef]):
        super().__init__(expand_last_dim=4, expand_n=expand_n)


@mark_materialize("core")
class BatchNorm2d(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, nfeat):
        super().__init__()
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]
        self.nfeat = nfeat

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, DType.float32)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_eq(self.nfeat, input_shapes[0].shape[1]),
            nnsmith_ge(input_shapes[0].shape[0], 2),
        ]  # batch size = 1 -> fail training.


@mark_materialize("core")
class Conv1d(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(
        self,
        in_channels: Union[int, z3.ExprRef],
        out_channels: Union[int, z3.ExprRef],
        kernel_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        padding: Union[int, z3.ExprRef],
        dilation: Union[int, z3.ExprRef],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.inp_ranks = [(3,)]  # NCL
        self.out_ranks = [(3,)]  # NCL

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        abs_tensor = AbsTensor(
            [input_shapes[0].shape[0], self.out_channels], dtype=input_shapes[0].dtype
        )
        mimic_k = self.kernel_size + (self.dilation - 1) * (self.kernel_size - 1)
        abs_tensor.shape.append(
            (
                nnsmith_div(
                    nnsmith_add(
                        nnsmith_sub(input_shapes[0].shape[2], mimic_k), 2 * self.padding
                    ),
                    self.stride,
                )
                + 1
            )
        )

        return [abs_tensor]

    def requires(self, input_shapes):
        # FIXME: Handling flops.
        cons = []
        cons.append(nnsmith_eq(self.in_channels, input_shapes[0].shape[1]))
        cons.append(nnsmith_ge(self.out_channels, 1))
        cons.append(nnsmith_ge(self.dilation, 1))
        mimic_k = self.kernel_size + (self.dilation - 1) * (self.kernel_size - 1)
        cons.append(nnsmith_ge(mimic_k, 1))
        cons.append(nnsmith_ge(self.stride, 1))
        cons.append(nnsmith_ge(self.padding, 0))
        cons.append(
            nnsmith_le(mimic_k, nnsmith_add(input_shapes[0].shape[2], 2 * self.padding))
        )
        # not too extream to avoid torch exporter issue
        cons.append(nnsmith_le(self.padding, 255))
        return cons

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(3, out_abs_tensor[0].dtype)]

    def __repr__(self) -> str:
        repr = f"Conv1d({self.in_channels}, {self.out_channels}, k={self.kernel_size}"
        if not isinstance(self.stride, int) or self.stride != 1:
            repr += f", s={self.stride}"
        if not isinstance(self.padding, int) or self.padding != 0:
            repr += f", p={self.padding}"
        if not isinstance(self.dilation, int) or self.dilation != 1:
            repr += f", d={self.dilation}"
        repr += ")"
        return repr


@mark_materialize("core")
class NCHWConv2d(UnaryOpBase):
    # FIXME: torch exporter does not support float64, may miss bugs
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(
        self,
        in_channels: Union[int, z3.ExprRef],
        out_channels: Union[int, z3.ExprRef],
        kernel_h_size: Union[int, z3.ExprRef],
        kernel_w_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        padding: Union[int, z3.ExprRef],
        dilation_h: Union[int, z3.ExprRef],
        dilation_w: Union[int, z3.ExprRef],
    ):
        """See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.padding = padding
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.inp_ranks = [(4,)]  # NC(H,)W
        self.out_ranks = [(4,)]  # NC(H,)W

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        abs_tensor = AbsTensor(
            [input_shapes[0].shape[0], self.out_channels], dtype=input_shapes[0].dtype
        )

        mimic_kh = self.kernel_h_size + (self.dilation_h - 1) * (self.kernel_h_size - 1)
        mimic_kw = self.kernel_w_size + (self.dilation_w - 1) * (self.kernel_w_size - 1)

        abs_tensor.shape.append(
            (
                nnsmith_div(
                    nnsmith_add(
                        nnsmith_sub(input_shapes[0].shape[2], mimic_kh),
                        2 * self.padding,
                    ),
                    self.stride,
                )
                + 1
            )
        )
        abs_tensor.shape.append(
            (
                nnsmith_div(
                    nnsmith_add(
                        nnsmith_sub(input_shapes[0].shape[3], mimic_kw),
                        2 * self.padding,
                    ),
                    self.stride,
                )
                + 1
            )
        )
        return [abs_tensor]

    def requires(self, input_shapes):
        cons = []
        # TODO: Use eager mode for debugging.
        cons.append(nnsmith_eq(self.in_channels, input_shapes[0].shape[1]))
        cons.append(nnsmith_ge(self.out_channels, 1))
        cons.append(nnsmith_ge(self.dilation_h, 1))
        cons.append(nnsmith_ge(self.dilation_w, 1))
        mimic_kh = self.kernel_h_size + (self.dilation_h - 1) * (self.kernel_h_size - 1)
        mimic_kw = self.kernel_w_size + (self.dilation_w - 1) * (self.kernel_w_size - 1)
        cons.append(nnsmith_ge(mimic_kh, 1))
        cons.append(nnsmith_ge(mimic_kw, 1))
        cons.append(nnsmith_ge(self.stride, 1))
        cons.append(nnsmith_ge(self.padding, 0))
        cons.append(
            nnsmith_le(
                mimic_kh, nnsmith_add(input_shapes[0].shape[2], 2 * self.padding)
            )
        )
        cons.append(
            nnsmith_le(
                mimic_kw, nnsmith_add(input_shapes[0].shape[3], 2 * self.padding)
            )
        )
        # not too extream to avoid torch exporter issue
        cons.append(nnsmith_le(self.padding, 255))
        # limit FLOPS
        if Z3_CONS_FLOPS:
            cons.append(nnsmith_le(self.flops(input_shapes), FLOPS_LIM))
        return cons

    def flops(self, input_shapes):
        w = AbsTensor(
            [
                self.out_channels,
                self.in_channels,
                self.kernel_h_size,
                self.kernel_w_size,
            ],
            dtype=input_shapes[0].dtype,
        )
        return nnsmith_mul(
            nnsmith_mul(
                nnsmith_mul(
                    self.type_transfer(input_shapes)[0].nelement(), self.in_channels
                ),
                self.kernel_h_size,
            ),
            self.kernel_w_size,
        )

    def n_floats(self, input_shapes):
        # FIXME: maybe need to take dilation into account?
        padded_data = AbsTensor(input_shapes[0].shape, dtype=input_shapes[0].dtype)
        padded_data.shape[2] = nnsmith_add(
            padded_data.shape[2], nnsmith_mul(2, self.padding)
        )
        padded_data.shape[3] = nnsmith_add(
            padded_data.shape[3], nnsmith_mul(2, self.padding)
        )
        w = AbsTensor(
            [
                self.out_channels,
                self.in_channels,
                self.kernel_h_size,
                self.kernel_w_size,
            ],
            dtype=input_shapes[0].dtype,
        )
        outs = super().n_floats(input_shapes)
        return nnsmith_add(nnsmith_add(w.nelement(), padded_data.nelement()), outs)

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype)]

    def __repr__(self) -> str:
        repr = f"Conv2d({self.in_channels}, {self.out_channels}, k=({self.kernel_h_size},{self.kernel_w_size})"
        if not isinstance(self.stride, int) or self.stride != 1:
            repr += f", s={self.stride}"
        if not isinstance(self.padding, int) or self.padding != 0:
            repr += f", p={self.padding}"
        if (
            not isinstance(self.dilation_h, int)
            or self.dilation_h != 1
            or self.dilation_w != 1
        ):
            repr += f", d=({self.dilation_h}, {self.dilation_w})"
        repr += ")"
        return repr


def random_group(n, k):
    xs = sorted([random.randint(0, n - k) for _ in range(k - 1)])
    xs = [0] + xs + [n - k]
    ret = []
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(k):
        st = xs[i] + i
        ed = xs[i + 1] + i + 1
        assert st < ed, (xs, st, ed)
        assert ed <= n, (st, ed, n)
        assert st >= 0, (st, ed, n)
        ret.append([perm[j] for j in range(st, ed)])
    return ret


@mark_materialize("core")
class Reshape(UnaryOpBase):
    num_var_param = int_range(1, 4)
    in_dtypes = [(i,) for i in DTYPE_ALL]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self, *target_shape):
        super().__init__()
        self.inp_ranks = [int_range(1, 4)]
        self.out_ranks = [(len(target_shape),)]
        self.target_shape: List[Union[int, z3.ExprRef]] = list(target_shape)

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        __MAX_SOLVE_SYMBOL__ = 8
        # otherwise OOM.
        ConstraintCheck.le(
            input_shapes[0].ndims + len(self.target_shape), __MAX_SOLVE_SYMBOL__
        )

        if -1 not in self.target_shape:
            return [AbsTensor(self.target_shape, dtype=input_shapes[0].dtype)]
        # else
        abs_tensor = AbsTensor(self.target_shape, dtype=input_shapes[0].dtype)
        auto_dim = -1
        accum = 1
        for i, v in enumerate(self.target_shape):
            # TODO: What to do about bitvectors here?
            if v == -1:
                SanityCheck.eq(auto_dim, -1)
                auto_dim = i
            else:
                accum = nnsmith_mul(accum, v)

        abs_tensor.shape[auto_dim] = nnsmith_div(
            reduce(lambda x, y: nnsmith_mul(x, y), input_shapes[0].shape, 1), accum
        )

        return [abs_tensor]

    def requires(self, input_shapes):
        ret = []

        inp = input_shapes[0]
        src_len, dst_len = inp.ndims, len(self.target_shape)
        if src_len == 0:
            src_len = 1  # special handling for scalar
        if dst_len == 0:
            dst_len = 1  # special handling for scalar
        gres_config = os.getenv("NNSMITH_GRES", "4")
        if gres_config == "5":
            ng = 1
        elif gres_config == "3":
            ng = min(src_len, dst_len)
        elif gres_config == "4":
            ub = min(src_len, dst_len)
            ng = random.choices(
                range(1, ub + 1), k=1, weights=[2**i for i in range(ub)]
            )[0]
        else:
            raise ValueError(f"NNSMITH_GRES={gres_config} is not recognized")
        src_group = random_group(src_len, ng)
        dst_group = random_group(dst_len, ng)
        self.ng = ng
        self.src_group = src_group
        self.dst_group = dst_group
        assert len(src_group) == len(dst_group) == ng, (src_group, dst_group)

        # group constraints
        src_vars = inp.shape
        dst_vars = self.target_shape
        if len(src_vars) == 0:
            src_vars = [1]  # special handling for scalar
        if len(dst_vars) == 0:
            dst_vars = [1]  # special handling for scalar
        cons_group = []
        for gid in range(ng):
            src_idx = src_group[gid]
            dst_idx = dst_group[gid]
            src_prod = reduce(nnsmith_mul, [src_vars[i] for i in src_idx], 1)
            dst_prod = reduce(nnsmith_mul, [dst_vars[i] for i in dst_idx], 1)
            cons_group.append(nnsmith_eq(src_prod, dst_prod))

        ret.extend(cons_group)
        if os.getenv("NNSMITH_CONS_RESHAPE", "off") != "off":
            # should not be too extreme!
            __DIM_LIMIT__ = 4096
            lim = __DIM_LIMIT__
            for s in self.target_shape[::-1]:
                ret.append(nnsmith_le(s, lim))
                lim //= 2
                lim = max(lim, 1)
        assert -1 not in self.target_shape
        return ret

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(-1, out_abs_tensor[0].dtype)]


@mark_materialize("core")
class Transpose(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self):
        """See https://pytorch.org/docs/stable/generated/torch.transpose.html"""
        super().__init__()
        self.inp_ranks = [int_from(2)]
        self.out_ranks = [int_from(2)]

    def _init_swap_dims(self, input_shape: List[Union[int, z3.ExprRef]]):
        ConstraintCheck.ge(len(input_shape), 2)
        self.inp_ranks = [len(input_shape)]
        if "dim0" not in self.extra_attrs or "dim1" not in self.extra_attrs:
            max_dim = len(input_shape) - 1
            self.extra_attrs["dim0"] = random.randint(0, max_dim)
            self.extra_attrs["dim1"] = (
                random.randint(1, max_dim) + self.extra_attrs["dim0"]
            ) % (1 + max_dim)
        return self.extra_attrs["dim0"], self.extra_attrs["dim1"]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        dim0, dim1 = self._init_swap_dims(input_shapes[0].shape)
        shape = list(input_shapes[0].shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return [AbsTensor(shape, input_shapes[0].dtype)]

    def requires(self, input_shapes):
        dim0, dim1 = self._init_swap_dims(input_shapes[0].shape)
        SanityCheck.ge(
            len(input_shapes[0].shape),
            max(dim0, dim1) + 1,
            f"dim={len(input_shapes[0].shape)}.transpose({dim0},{dim1})",
        )
        return []

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]


# Sum, Min, Max, Mean, ArgMin, ArgMax, Squeeze, Size


class InterpBase(UnaryOpBase):
    num_var_param = int_range(1, 3)

    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]

    def __init__(self, *size):
        super().__init__()
        self.size = list(size)
        self.inp_ranks = [(len(size) + 2,)]
        self.out_ranks = [(len(size) + 2,)]

    def requires(self, input_shapes: List[AbsTensor]):
        return [nnsmith_gt(v, 0) for v in self.size]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        shape = list(input_shapes[0].shape)
        for i in range(len(self.size)):
            shape[-(1 + i)] = self.size[-(1 + i)]
        return [AbsTensor(shape, input_shapes[0].dtype)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]


@mark_materialize("core")
class NearestInterp(InterpBase):
    pass


@mark_materialize("core")
class LinearInterp(InterpBase):
    num_var_param = [1]


@mark_materialize("core")
class BilinearInterp(InterpBase):
    num_var_param = [2]


@mark_materialize("core")
class BicubicInterp(InterpBase):
    num_var_param = [2]


@mark_materialize("core")
class TrilinearInterp(InterpBase):
    num_var_param = [3]


class ReduceBase(UnaryOpBase, ABC):
    _reduce_out_dtype = None  # None means same as input dtype

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_from(1)]  # TVM bug ~ crash on scalar.min()
        self.out_ranks = [int_range(0, __MAX_RANK__ - 1)]

    def __str__(self) -> str:
        return (
            self.name()
            + f'(dim={self.extra_attrs["reduce_dim"] if "reduce_dim" in self.extra_attrs else None})'
        )

    def _init_reduce_dim(self, input_shape: List[Union[int, z3.ExprRef]]):
        if "reduce_dim" not in self.extra_attrs:
            if len(input_shape) == 0:
                self.extra_attrs["reduce_dim"] = None
            else:
                self.extra_attrs["reduce_dim"] = random.randint(0, len(input_shape) - 1)
        return self.extra_attrs["reduce_dim"]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        svar_list = []
        for i, v in enumerate(input_shapes[0].shape):
            if i != self._init_reduce_dim(input_shapes[0].shape):
                svar_list.append(v)
        return [
            AbsTensor(
                svar_list,
                input_shapes[0].dtype
                if self._reduce_out_dtype is None
                else self._reduce_out_dtype,
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]):
        self._init_reduce_dim(input_shapes[0].shape)
        return []

    def _get_irank(self, orank):
        return orank + 1

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(self._get_irank(out_abs_tensor[0].ndims), out_abs_tensor[0].dtype)]


@mark_materialize("core")
class Squeeze(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def requires(self, input_shapes):
        reduce_dim = self._init_reduce_dim(input_shapes[0].shape)
        if reduce_dim is None:
            return []
        return [nnsmith_eq(input_shapes[0].shape[reduce_dim], 1)]


@mark_materialize("core")
class ReduceSum(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]


@mark_materialize("core")
class ReduceMin(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]


@mark_materialize("core")
class ReduceMax(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]


@mark_materialize("core")
class ReduceMean(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_FLOATS]


@mark_materialize("core")
class ArgMin(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(DType.int64,)]
    _reduce_out_dtype = DType.int64

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (self._get_irank(out_abs_tensor[0].ndims), random.choice(self.in_dtypes)[0])
        ]


@mark_materialize("core")
class ArgMax(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(DType.int64,)]
    _reduce_out_dtype = DType.int64

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (self._get_irank(out_abs_tensor[0].ndims), random.choice(self.in_dtypes)[0])
        ]


class TriBase(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_ALL]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self, diagonal: Union[int, z3.ExprRef]):
        super().__init__()
        self.diagonal = diagonal
        # tril is only for 2-D matrix
        self.inp_ranks = [(2,)]
        self.out_ranks = [(2,)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(2, out_abs_tensor[0].dtype)]


@mark_materialize("core")
class Tril(TriBase):
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        ConstraintCheck.true(input_shapes[0].ndims == 2)
        nrow = input_shapes[0].shape[0]
        ncol = input_shapes[0].shape[1]
        return [z3.And(self.diagonal >= -nrow, (ncol - 1) >= self.diagonal)]


@mark_materialize("core")
class Triu(TriBase):
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        ConstraintCheck.true(input_shapes[0].ndims == 2)
        nrow = input_shapes[0].shape[0]
        ncol = input_shapes[0].shape[1]
        return [z3.And(self.diagonal >= -(nrow - 1), ncol >= self.diagonal)]


class Concat(AbsOpBase):
    MAX_ARITY = 5
    MAX_RANK = 5
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __str__(self) -> str:
        return "Concat " + str(self.extra_attrs).replace(":", "=")

    def __init__(self, arity):
        super().__init__()
        SanityCheck.le(arity, Concat.MAX_ARITY)
        self.arity = arity
        self.inp_ranks = [(int_from(1))] * arity
        self.out_ranks = [(int_from(1))]
        self.same_inp_dims = True

    def _init_concat_axis(self, input_shapes: List[AbsTensor]) -> int:
        if "axis" not in self.extra_attrs:
            self.extra_attrs["axis"] = random.randint(0, input_shapes[0].ndims - 1)
        return self.extra_attrs["axis"]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        ndims = input_shapes[0].ndims
        SanityCheck.gt(ndims, self._init_concat_axis(input_shapes))
        for s in input_shapes:
            SanityCheck.eq(s.ndims, ndims)
        cons = []
        for d in range(ndims):
            if d != self._init_concat_axis(input_shapes):
                cons.extend(
                    nnsmith_eq(s.shape[d], input_shapes[0].shape[d])
                    for s in input_shapes
                )
        return cons

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.true(input_shapes[0].ndims > 0)
        axis = self._init_concat_axis(input_shapes)
        os = AbsTensor(input_shapes[0].shape, input_shapes[0].dtype)
        os.shape[axis] = reduce(nnsmith_add, [s.shape[axis] for s in input_shapes])
        return [os]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)
            for _ in range(self.arity)
        ]


# the semantic of `in_dtypes` is not possible dtypes in "max rank". but simply in "rank". don't mess up the definition.
@mark_materialize("core")
class Concat1(Concat):
    in_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(1)


@mark_materialize("core")
class Concat2(Concat):
    in_dtypes = [(i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(2)


@mark_materialize("core")
class Concat3(Concat):
    in_dtypes = [(i, i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(3)


@mark_materialize("core")
class Concat4(Concat):
    in_dtypes = [(i, i, i, i) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__(4)


@mark_materialize("core")
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
        self.extra_attrs = {"to": dtype}

    def __str__(self) -> str:
        return "Cast " + str(self.extra_attrs).replace(":", "=")

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return []

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        assert len(input_shapes) == 1
        return [AbsTensor(input_shapes[0].shape, self.extra_attrs["to"])]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, self.extra_attrs["to"])]


@mark_materialize("core")
class CastF32(Cast):
    out_dtypes = [(DType.float32,)]

    def __init__(self):
        super().__init__(DType.float32)


@mark_materialize("core")
class CastF64(Cast):
    out_dtypes = [(DType.float64,)]

    def __init__(self):
        super().__init__(DType.float64)


@mark_materialize("core")
class CastI32(Cast):
    out_dtypes = [(DType.int32,)]

    def __init__(self):
        super().__init__(DType.int32)


@mark_materialize("core")
class CastI64(Cast):
    out_dtypes = [(DType.int64,)]

    def __init__(self):
        super().__init__(DType.int64)


@mark_materialize("core")
class CastBool(Cast):
    out_dtypes = [(DType.bool,)]

    def __init__(self):
        super().__init__(DType.bool)


@mark_materialize("core")
class MatMul(BinaryOpBase):
    in_dtypes = [(i, i) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def __init__(self):
        super().__init__()
        # Consider at most 3D tensors (batched mm)
        self.inp_ranks = [int_range(1, 3), int_range(1, 3)]
        self.out_ranks = [int_until(3)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        # https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul
        # shape: [*batches(?), *rc (row and col)]
        lhs = input_shapes[0].shape
        rhs = input_shapes[1].shape

        lrc = lhs[-2:]
        rrc = rhs[-2:]
        orc = [*lrc[:-1], *rrc[1:]]

        lbatch = lhs[: -len(lrc)]
        rbatch = rhs[: -len(rrc)]
        batches = []
        if len(lbatch) > len(rbatch):
            batches = lbatch[: len(lbatch) - len(rbatch)]
            for x, y in zip(lbatch[len(batches) :], rbatch):
                batches.append(nnsmith_max(x, y))
        else:
            batches = rbatch[: len(rbatch) - len(lbatch)]
            for x, y in zip(lbatch, rbatch[len(batches) :]):
                batches.append(nnsmith_max(x, y))

        return [AbsTensor([*batches, *orc], input_shapes[0].dtype)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        cons = []

        lhs = input_shapes[0].shape
        rhs = input_shapes[1].shape

        lrc = lhs[-2:]
        rrc = rhs[-2:]

        # CHECK: l.cols = r.rows
        cons.append(lrc[-1] == rrc[0])

        # CHECK: batch dim broadcastable
        lbatch = lhs[: -len(lrc)]
        rbatch = rhs[: -len(rrc)]
        common_tail = min(len(lbatch), len(rbatch))
        for x, y in zip(lbatch[-common_tail:], rbatch[-common_tail:]):
            cons.append(nnsmith_or(x == y, nnsmith_or(x == 1, y == 1)))

        return cons

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # rank(a) = batch_rank(a) + rc_rank(a)
        # rank(b) = batch_rank(b) + rc_rank(b)
        # out_rank = max(br_a, br_b) + (rcr_a + rcr_b) - 2
        # 1 <= rcr_a or rcr_b <= min(2, out_rank + 2)

        # br_a = ranks[0], rcr_a = ranks[1]
        # br_b = ranks[2], rcr_b = ranks[3]
        ranks = [0, 1, 0, 1]

        def check_sat():
            return (
                out_abs_tensor[0].ndims
                == max(ranks[0], ranks[2]) + (ranks[1] + ranks[3]) - 2
            )

        while not check_sat():
            inc_candidates = []
            if ranks[1] < 2:
                inc_candidates.append(1)
            else:
                inc_candidates.append(0)

            if ranks[3] < 2:
                inc_candidates.append(3)
            else:
                inc_candidates.append(2)
            choice = random.choice(inc_candidates)
            ranks[choice] += 1

        return [
            (ranks[0] + ranks[1], out_abs_tensor[0].dtype),
            (ranks[2] + ranks[3], out_abs_tensor[0].dtype),
        ]


_PRAGMA_ONCE_CORE_OP = False
