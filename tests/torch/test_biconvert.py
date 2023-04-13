import pytest
import torch

from nnsmith.materialize.torch.parse import parse
from nnsmith.materialize.torch.symbolnet import FxTracing, SymbolNet


def test_biconvert():
    class MyModel(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)

        def forward(self, i0, i1):
            v0 = i0 + 3.14 + i1[0, 0]
            v1 = self.linear(v0)
            v1_0, v1_1 = torch.split(v1, [1, 3], dim=-1)
            v2 = torch.mul(input=v1_0, other=v1_1)
            v3 = torch.cat([v2, v2], dim=-1)
            v4 = v3.flatten()
            return v4

    model = MyModel()
    i0 = torch.rand(2, 3)
    i1 = torch.rand(1, 2)
    ir = parse(model, i0, i1)
    assert (
        ir.pretty().strip()
        == """\
v0_0 = Input() 	# inst id: 0
v1_0 = core.ConcreteOp<operator.getitem>(v0_0) 	# inst id: 1
v2_0 = Input() 	# inst id: 2
v3_0 = core.ConcreteOp<operator.add>(v2_0) 	# inst id: 3
v4_0 = core.ConcreteOp<operator.add>(v3_0, v1_0) 	# inst id: 4
v5_0 = core.ConcreteOp<torch.nn.Linear(in_features=3, out_features=4, bias=True)>(v4_0) 	# inst id: 5
v6_0, v6_1 = core.ConcreteOp<torch.functional.split>(v5_0) 	# inst id: 6
v7_0 = core.ConcreteOp<torch.mul>(v6_0, v6_1) 	# inst id: 7
v8_0 = core.ConcreteOp<torch.cat>(v7_0, v7_0) 	# inst id: 8
v9_0 = core.ConcreteOp<torch.Tensor.flatten>(v8_0) 	# inst id: 9"""
    )

    ir.remove_unused(ir.insts[-1])  # mutate: remove the last flatten op.

    net = SymbolNet(ir)
    with FxTracing():
        traced = torch.fx.symbolic_trace(net)
        assert (
            traced.code.strip()
            == R"""def forward(self, *args):
    _args = args
    getitem = _args[0]
    getitem_1 = _args[1];  _args = None
    getitem_2 = getitem[(0, 0)];  getitem = None
    add = getitem_1 + 3.14;  getitem_1 = None
    add_1 = add + getitem_2;  add = getitem_2 = None
    m5 = self.m5(add_1);  add_1 = None
    split = torch.functional.split(m5, [1, 3], dim = -1);  m5 = None
    getitem_3 = split[0]
    getitem_4 = split[0]
    getitem_5 = split[1]
    getitem_6 = split[1];  split = None
    mul = torch.mul(input = getitem_3, other = getitem_5);  getitem_3 = getitem_5 = None
    cat = torch.cat([mul, mul], dim = -1);  mul = None
    return (cat,)"""
        )
