from functools import reduce
from typing import List, Union

import z3

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DType
from nnsmith.error import ConstraintCheck, SanityCheck


class AbsTensor:
    def __init__(self, shape: List[Union[int, z3.ExprRef]], dtype: DType):
        assert isinstance(
            shape, (list, tuple)
        ), f"Shape must be a list/tuple, but got {shape}"
        self.shape = list(shape)
        self.dtype = DType(dtype)

    def __repr__(self):
        return f"AbsTensor<{self.dtype.short()}>{str(self.shape)}"

    def __eq__(self, other):
        return self.shape == other.shape and self.dtype == other.dtype

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
            if isinstance(self.shape[i], z3.ExprRef) or isinstance(
                other.shape[i], z3.ExprRef
            ):
                ret.append(nnsmith_eq(self.shape[i], other.shape[i]))
            else:
                ConstraintCheck.eq(self.shape[i], other.shape[i])
        return ret

    def torch(self):
        import torch

        return torch.Size(self.shape)

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ExprRef) for s in self.shape)

    def nelement(self):
        if len(self.shape) == 0:  # Scalar
            return 1
        return reduce(lambda x, y: nnsmith_mul(x, y), self.shape, 1)

    def deepcopy(self):
        return AbsTensor(shape=list(self.shape), dtype=self.dtype)

    @staticmethod
    def from_torch(torch_tensor):
        return AbsTensor(list(torch_tensor.shape), DType.from_torch(torch_tensor.dtype))

    @property
    def ndims(self):
        return len(self.shape)
