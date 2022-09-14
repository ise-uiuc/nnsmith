from math import prod
from random import randint
from typing import List, Tuple, Union

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DTYPE_ALL, DTYPE_NON_BOOLS, DType
from nnsmith.abstract.op import ReduceBase, UnaryOpBase, int_from, mark_materialize
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import ConstraintCheck


@mark_materialize("torch")
class Linear(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [int_from(1)]
        self.out_ranks = [int_from(1)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        assert len(input_shapes) == 1, "Linear only takes one input, but got {}".format(
            len(input_shapes)
        )
        return [
            AbsTensor(
                shape=[*input_shapes[0].shape[:-1], self.ofeat], dtype=DType.float32
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        ConstraintCheck.true(input_shapes[0].ndims >= 1)
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, DType.float32)]


@mark_materialize("torch")
class Flatten(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_ALL]
    out_dtypes = [(i,) for i in DTYPE_ALL]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [int_from(1)]
        self.out_ranks = [(1,)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        inp = input_shapes[0]
        return [
            AbsTensor(
                shape=[prod(inp.shape)],
                dtype=inp.dtype,
            )
        ]

    def requires(self, input_shapes):
        return []

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(randint(0, 4), out_abs_tensor[0].dtype)]


@mark_materialize("torch")
class TorchReduceSum(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_NON_BOOLS]
    out_dtypes = [(i,) for i in DTYPE_NON_BOOLS]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        output = super().type_transfer(input_shapes)
        # This is a PyTorch trick...
        if input_shapes[0].dtype == DType.int32:
            output[0].dtype = DType.int64
        return output
