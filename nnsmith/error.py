# Also, there's a timeout error which is managed by subprocess module.
from abc import ABC


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
    """Fatal unexpected internal errors in NNSmith that should shut down the program immediately."""
    pass


class ConstraintError(Exception):
    """Expected possible constarint unsat error used in shape transfer function."""
    pass


class AbsCheck(ABC):
    _EXCEPTION_TYPE = None

    @classmethod
    def eq(cls, lhs, rhs, msg=""):
        if lhs != rhs:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | {lhs} != {rhs}')

    @classmethod
    def gt(cls, lhs, rhs, msg=""):
        if lhs <= rhs:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | {lhs} <= {rhs}')

    @classmethod
    def ge(cls, lhs, rhs, msg=""):
        if lhs < rhs:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | {lhs} < {rhs}')

    @classmethod
    def lt(cls, lhs, rhs, msg=""):
        if lhs >= rhs:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | {lhs} >= {rhs}')

    @classmethod
    def le(cls, lhs, rhs, msg=""):
        if lhs > rhs:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | {lhs} > {rhs}')

    @classmethod
    def none(cls, obj, msg=""):
        if obj is not None:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | expr is not None')

    @classmethod
    def not_none(cls, obj, msg=""):
        if obj is None:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | expr is None')

    @classmethod
    def true(cls, cond, msg=""):
        if not cond:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | condition is not True')

    @classmethod
    def false(cls, cond, msg=""):
        if cond:
            raise cls._EXCEPTION_TYPE(
                f'Failed asertion :: {msg} | condition is not False')


class SanityCheck(AbsCheck):
    _EXCEPTION_TYPE = NNSmithInternalError


class ConstraintCheck(AbsCheck):
    _EXCEPTION_TYPE = ConstraintError
