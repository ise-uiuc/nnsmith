import pytest
import torch

from nnsmith.materialize.torch.parse import parse
from nnsmith.materialize.torch.symbolnet import FxTracing, SymbolNet


def test_biconvert(tmp_path):
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
    print(f"eager: {model(i0, i1)}")
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
            str(traced.graph).strip()
            == """\
graph():
    %_args : [#users=2] = placeholder[target=*args]
    %getitem : [#users=1] = call_function[target=operator.getitem](args = (%_args, 0), kwargs = {})
    %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%_args, 1), kwargs = {})
    %getitem_2 : [#users=1] = call_function[target=operator.getitem](args = (%getitem, (0, 0)), kwargs = {})
    %add : [#users=1] = call_function[target=operator.add](args = (%getitem_1, 3.14), kwargs = {})
    %add_1 : [#users=1] = call_function[target=operator.add](args = (%add, %getitem_2), kwargs = {})
    %mlist_0 : [#users=1] = call_module[target=mlist.0](args = (%add_1,), kwargs = {})
    %split : [#users=4] = call_function[target=torch.functional.split](args = (%mlist_0, [1, 3]), kwargs = {dim: -1})
    %getitem_3 : [#users=1] = call_function[target=operator.getitem](args = (%split, 0), kwargs = {})
    %getitem_4 : [#users=0] = call_function[target=operator.getitem](args = (%split, 0), kwargs = {})
    %getitem_5 : [#users=1] = call_function[target=operator.getitem](args = (%split, 1), kwargs = {})
    %getitem_6 : [#users=0] = call_function[target=operator.getitem](args = (%split, 1), kwargs = {})
    %mul : [#users=1] = call_function[target=torch.mul](args = (), kwargs = {input: %getitem_3, other: %getitem_5})
    %cat : [#users=1] = call_function[target=torch.cat](args = ([%mul, %mul],), kwargs = {dim: -1})
    return (cat,)"""
        )
        print(traced.code)
        traced.to_folder(tmp_path)


"""graph.print_tabular():
opcode         name         target                                                  args                   kwargs
-------------  -----------  ------------------------------------------------------  ---------------------  ----------------------------------------
placeholder    arg0         arg0                                                    ()                     {}
placeholder    arg1         arg1                                                    ()                     {}
call_function  add          <built-in function add>                                 (arg0, 3.14)           {}
call_function  getitem      <built-in function getitem>                             (arg1, (0, 0))         {}
call_function  add_1        <built-in function add>                                 (add, getitem)         {}
call_module    self_linear  self_linear                                             (add_1,)               {}
call_function  split        <function split at 0x7fd4e8a464d0>                      (self_linear, [1, 3])  {'dim': -1}
call_function  getitem_1    <built-in function getitem>                             (split, 0)             {}
call_function  getitem_2    <built-in function getitem>                             (split, 1)             {}
call_function  mul          <built-in method mul of type object at 0x7fd56cac2400>  ()                     {'input': getitem_1, 'other': getitem_2}
call_function  cat          <built-in method cat of type object at 0x7fd56cac2400>  ([mul, mul],)          {'dim': -1}
call_method    flatten      flatten                                                 (cat,)                 {}
output         output       output                                                  ([flatten],)           {}
"""
