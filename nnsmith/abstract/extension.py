import functools
import types
from typing import List, Type

from nnsmith.abstract.op import AbsOpBase
from nnsmith.abstract.tensor import AbsTensor

TABLE_UPDATE_REQUIRES = {}


def change_requires(
    system_name: str, op_types: List[Type[AbsOpBase]]
) -> List[Type[AbsOpBase]]:
    for optype in op_types:
        if optype.name() in TABLE_UPDATE_REQUIRES[system_name]:
            optype.requires = TABLE_UPDATE_REQUIRES[system_name][optype.name()]
    return op_types


def copy_requires(f):
    g = types.FunctionType(
        f.__code__, f.__globals__, name=f.__name__, closure=f.__closure__
    )
    return functools.update_wrapper(g, f)


class rewrite_requires:
    def __init__(self, tag: str, opname: str):
        self.tag = tag
        self.opname = opname

    def __call__(self, f):
        TABLE_UPDATE_REQUIRES.setdefault(self.tag, {}).setdefault(self.opname, f)
        return f


class patch_requires:
    def __init__(self, tag: str, opname: str):
        self.tag = tag
        self.opname = opname
        self.prev_fn = None

    def __call__(self, f):
        def patch_with_prev(op: AbsOpBase, itensors: List[AbsTensor]):
            if self.prev_fn is None:
                self.prev_fn = copy_requires(op.requires)
            return f(op, itensors) + self.prev_fn(op, itensors)

        TABLE_UPDATE_REQUIRES.setdefault(self.tag, {}).setdefault(
            self.opname, patch_with_prev
        )
        return patch_with_prev
