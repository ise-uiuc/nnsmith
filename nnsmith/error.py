# Also, there's a timeout error which is managed by subprocess module.
class ModeledError(Exception):
    """
    This is a base class for all modeled errors.
    """
    pass


class CrashError(ModeledError):
    pass


class IncorrectResult(ModeledError):
    pass


class NaNError(ModeledError):
    pass


class PerfDegradation(ModeledError):
    pass


class RuntimeFailure(ModeledError):
    pass

# Timeout...


class MaybeDeadLoop(ModeledError):
    pass


class NNSmithInternalError(Exception):
    pass


def assert_eq(lhs, rhs, msg=""):
    if lhs != rhs:
        raise NNSmithInternalError(
            f'Failed asertion :: {msg} | {lhs} != {rhs}')


def assert_gt(lhs, rhs, msg=""):
    if lhs <= rhs:
        raise NNSmithInternalError(
            f'Failed asertion :: {msg} | {lhs} <= {rhs}')


def assert_ge(lhs, rhs, msg=""):
    if lhs < rhs:
        raise NNSmithInternalError(f'Failed asertion :: {msg} | {lhs} < {rhs}')


def assert_lt(lhs, rhs, msg=""):
    if lhs >= rhs:
        raise NNSmithInternalError(
            f'Failed asertion :: {msg} | {lhs} >= {rhs}')


def assert_le(lhs, rhs, msg=""):
    if lhs > rhs:
        raise NNSmithInternalError(f'Failed asertion :: {msg} | {lhs} > {rhs}')


def assert_none(obj, msg=""):
    if obj is not None:
        raise NNSmithInternalError(
            f'Failed asertion :: {msg} | expr is not None')


def assert_not_none(obj, msg=""):
    if obj is None:
        raise NNSmithInternalError(f'Failed asertion :: {msg} | expr is None')


def assert_true(cond, msg=""):
    if not cond:
        raise NNSmithInternalError(
            f'Failed asertion :: {msg} | condition is not True')


def assert_false(cond, msg=""):
    if cond:
        raise NNSmithInternalError(
            f'Failed asertion :: {msg} | condition is not False')
