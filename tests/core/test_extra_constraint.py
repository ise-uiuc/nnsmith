import pytest

from nnsmith.abstract.arith import nnsmith_lt
from nnsmith.abstract.extension import activate_ext, patch_requires
from nnsmith.abstract.op import Constant, Input, NCHWConv2d, ReLU
from nnsmith.graph_gen import model_gen


def test_only_conv_relu():
    gen = model_gen(
        opset=[ReLU, NCHWConv2d],
        max_nodes=5,
        rank_choices=(4,),
        dtype_choices=("float32",),
    )

    ir = gen.make_concrete()

    for inst in ir.insts:
        assert type(inst.iexpr.op) in [ReLU, Input, Constant, NCHWConv2d]


def test_constrain_conv_ksize():
    @patch_requires("global", "core.NCHWConv2d")
    def limit_conv2d(self, _):
        # let the kernels to be > 3
        return [nnsmith_lt(3, self.kernel_h_size), nnsmith_lt(3, self.kernel_w_size)]

    opset = [ReLU, NCHWConv2d]
    activate_ext(opset)
    gen = model_gen(
        opset=opset,
        max_nodes=5,
        rank_choices=(4,),
        dtype_choices=("float32",),
    )

    ir = gen.make_concrete()
    for inst in ir.insts:
        assert type(inst.iexpr.op) in [ReLU, Input, Constant, NCHWConv2d]
        if isinstance(inst.iexpr.op, NCHWConv2d):
            assert inst.iexpr.op.kernel_h_size > 3
            assert inst.iexpr.op.kernel_w_size > 3
