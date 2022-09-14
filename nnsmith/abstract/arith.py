from typing import Union

import z3

from nnsmith.error import SanityCheck

ARITH_MAX_WIDTH: int = 64


def align_bvs(
    left: Union[float, int, z3.ExprRef],
    right: Union[float, int, z3.ExprRef],
    carry=False,
    mult=False,
):
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
        raise RuntimeError(f"Unsupported alignment value {left} of type {type(left)}")
    # We assume that the width of an arithmetic type is ARITH_MAX_WIDTH.
    if right_is_arith:
        if isinstance(right, int):
            right_size = min(ARITH_MAX_WIDTH, right.bit_length())
        else:
            right_size = ARITH_MAX_WIDTH
    elif isinstance(right, z3.BitVecRef):
        right_size = right.size()
    else:
        raise RuntimeError(f"Unsupported alignment value {right} of type {type(right)}")
    # Extend the bitvector that is smaller with the necessary amount of zeroes.
    SanityCheck.true(
        not (carry and mult),
        "Carry and multiplication extension are mutually exclusive",
    )
    SanityCheck.le(
        left_size,
        ARITH_MAX_WIDTH,
        f"Bitvector sizes must not exceed {ARITH_MAX_WIDTH} bits.",
    )
    SanityCheck.le(
        right_size,
        ARITH_MAX_WIDTH,
        f"Bitvector sizes must not exceed {ARITH_MAX_WIDTH} bits.",
    )
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


def nnsmith_mul(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right, mult=True)
    return left * right


def nnsmith_add(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right, carry=True)
    return left + right


def nnsmith_sub(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    return left - right


def nnsmith_eq(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    return left == right


def nnsmith_neq(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    return left != right


def nnsmith_ge(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.UGE(left, right)
    return left >= right


def nnsmith_gt(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.UGT(left, right)
    return left > right


def nnsmith_le(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.ULE(left, right)
    return left <= right


def nnsmith_lt(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.ULT(left, right)
    return left < right


def nnsmith_div(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.UDiv(left, right)
    if isinstance(left, int) and isinstance(right, int):
        return left // right
    return left / right


def nnsmith_mod(
    left: Union[float, int, z3.ExprRef], right: Union[float, int, z3.ExprRef]
):
    left, right = align_bvs(left, right)
    if isinstance(left, z3.BitVecRef) or isinstance(right, z3.BitVecRef):
        return z3.URem(left, right)
    return left % right


def nnsmith_min(left, right):
    if isinstance(left, int) and isinstance(right, int):
        return min(left, right)
    left, right = align_bvs(left, right)
    return z3.If(nnsmith_le(left, right), left, right)


def nnsmith_max(left, right):
    if isinstance(left, int) and isinstance(right, int):
        return max(left, right)
    left, right = align_bvs(left, right)
    return z3.If(nnsmith_ge(left, right), left, right)


def nnsmith_and(left, right):
    if isinstance(left, bool) and isinstance(right, bool):
        return left and right
    return z3.And(left, right)


def nnsmith_or(left, right):
    if isinstance(left, bool) and isinstance(right, bool):
        return left or right
    return z3.Or(left, right)


def nnsmith_not(expr):
    if isinstance(expr, bool):
        return not expr
    return z3.Not(expr)
