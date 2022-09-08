from typing import List, Tuple, Union

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import (
    BcastBinaryOp,
    ElementWiseUnaryOp,
    UnaryOpBase,
    int_from,
    mark_materialize,
)
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import ConstraintCheck


@mark_materialize("tensorflow")
class Dense(UnaryOpBase):
    in_dtypes = [(DType.float32,), (DType.float64,)]
    out_dtypes = [(DType.float32,), (DType.float64,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [
            int_from(2)
        ]  # NOTE: tensorflow Dense layer requires an input with batch as its first axis
        # at least one dim. cannot be zranks_all()
        self.out_ranks = [int_from(2)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        assert len(input_shapes) == 1, "Linear only takes one input, but got {}".format(
            len(input_shapes)
        )
        return [
            AbsTensor(
                shape=[*input_shapes[0].shape[:-1], self.ofeat],
                dtype=input_shapes[0].dtype,
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        ConstraintCheck.true(input_shapes[0].ndims >= 2)
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]


@mark_materialize("tensorflow")
class SquaredDifference(BcastBinaryOp):
    pass


@mark_materialize("tensorflow")
class LocalRespNorm(ElementWiseUnaryOp):
    # https://www.tensorflow.org/api_docs/python/tf/raw_ops/LRN
    in_dtypes = [(DType.float16,), (DType.float32,)]
    out_dtypes = [(DType.float16,), (DType.float32,)]

    def __init__(
        self,
        depth_radius: Union[int, z3.ExprRef],
        bias: Union[float, z3.ExprRef],
        alpha: Union[float, z3.ExprRef],
        inv_beta: Union[float, z3.ExprRef],
    ):
        super().__init__()
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.inv_beta = inv_beta

        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        SanityCheck.eq(len(input_shapes), 1)
        input_shape = input_shapes[0]
        cons = []
        # cuDNN requires depth_radius in [1, 7]
        cons.append(nnsmith_ge(self.depth_radius, 1))
        cons.append(nnsmith_le(self.depth_radius, 7))
        cons.append(
            nnsmith_le(
                self.depth_radius, nnsmith_div(nnsmith_sub(input_shape.shape[3], 1), 2)
            )
        )  # depth_radius <= (input_shape[3] - 1) / 2
        cons.append(nnsmith_gt(self.bias, 0))
        cons.append(nnsmith_gt(self.alpha, 0))
        # cuDNN requires beta >= 0.01
        cons.append(nnsmith_ge(self.inv_beta, 1))
        cons.append(nnsmith_le(self.inv_beta, 100))
        return cons
