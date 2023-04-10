import operator
from typing import Any, Dict, List, Tuple, Union, cast

import torch
import torch._dynamo as dynamo
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.fx.passes.shape_prop import ShapeProp

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import ConcreteOp, Input
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR, InstExpr
from nnsmith.materialize.torch.forward import forward_fn


class PropInterpreter(ShapeProp):
    def run_node(self, n: fx.node.Node) -> Any:
        result = super().run_node(n)
        n.meta["res"] = result
        return result


def parse(model: nn.Module, *example_args: List[torch.Tensor]) -> GraphIR:
    gm: fx.GraphModule = dynamo.export(model, *example_args)[0]
    # store shape info on nodes
    sp = PropInterpreter(gm)
    sp.run(*example_args)

    def load_args(args: Union[List, Dict[str, Any]]) -> Union[List, Dict[str, Any]]:
        """
        Map nodes to their outputs while keeping structures and other values the same.
        """
        return torch.fx.graph.map_arg(args, lambda n: n.meta["res"])

    named_modules = dict(gm.named_modules())
    ir = GraphIR()
    name2retvals: Dict[str, List[str]] = {}
    for i_node, node in enumerate(gm.graph.nodes):
        node = cast(fx.node.Node, node)
        if node.op == "placeholder":
            iexpr = InstExpr(Input(dim=len(node.meta["res"].shape)), [])
        else:
            args_flatten, args_treespec = pytree.tree_flatten(node.args)
            kwargs_flatten, kwargs_treespec = pytree.tree_flatten(node.kwargs)
            input_nodes = [
                a
                for a in (args_flatten + kwargs_flatten)
                if isinstance(a, fx.node.Node)
            ]
            input_valstrs = list(map(lambda n: name2retvals[n.name][0], input_nodes))
            input_like = list(
                map(
                    lambda ts: AbsTensor(
                        shape=ts.shape, dtype=DType.from_torch(ts.dtype)
                    ),
                    pytree.tree_flatten(
                        list(map(lambda n: n.meta["res"], input_nodes))
                    )[0],
                )
            )
            output_like = list(
                map(
                    lambda ts: AbsTensor(
                        shape=ts.shape, dtype=DType.from_torch(ts.dtype)
                    ),
                    pytree.tree_flatten(node.meta["res"])[0],
                )
            )
            nodes2empty = (
                lambda n: ConcreteOp.empty if isinstance(n, fx.node.Node) else n
            )
            args_wo_nodes = pytree.tree_map(nodes2empty, node.args)
            kwargs_wo_nodes = pytree.tree_map(nodes2empty, node.kwargs)
            if node.op == "call_function":
                if (
                    node.target is operator.getitem
                    and isinstance(node.args[0], fx.node.Node)
                    and not isinstance(node.args[0].meta["res"], torch.Tensor)
                ):
                    name2retvals[node.name] = [
                        name2retvals[node.args[0].name][node.args[1]]
                    ]
                    continue
                else:
                    target_str = node._pretty_print_target(node.target)
            elif node.op == "call_method":
                target_str = f"torch.Tensor.{node.target}"
            elif node.op == "call_module":
                target = named_modules[node.target]
                target_str = repr(target)
                if target.__module__.startswith("torch.nn.modules"):
                    target_str = f"torch.nn.{target_str}"
            elif node.op == "get_attr":
                raise NotImplementedError(f"{node.op = }, {node.name = }")
            elif node.op == "output":
                continue
            else:
                raise ValueError(f"Unexpected {node.op = }")

            iexpr = InstExpr(
                ConcreteOp(
                    target_str, args_wo_nodes, kwargs_wo_nodes, input_like, output_like
                ),
                input_valstrs,
            )

        name2retvals[node.name] = ir.add_inst(iexpr).retvals()
    # end for
    return ir


if __name__ == "__main__":

    class MyModel(nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, i0):
            v0 = i0 + 3.14 + i0[0, 0]
            v1 = self.linear(v0)
            v1_0, v1_1 = torch.split(v1, [1, 3], dim=-1)
            v2 = torch.mul(input=v1_0, other=v1_1)
            v3 = torch.cat([v2, v2], dim=-1)
            v4 = v3.flatten()
            return v4

    model = MyModel()
    i0 = torch.rand(2, 3)
    i1 = 4.3
    print(f"eager: {model(i0)}")

    ir = parse(model, i0)
    print(ir.pretty())

    ir.remove_unused(ir.insts[-1])  # remove the last flatten op.

    from nnsmith.materialize.torch.symbolnet import FxTracing, SymbolNet

    net = SymbolNet(ir)
    with FxTracing():
        traced = torch.fx.symbolic_trace(net)
        print(traced.graph)
        print(traced.code)
        traced.to_folder("gened")

"""
opcode         name         target                                                  args                   kwargs
-------------  -----------  ------------------------------------------------------  ---------------------  ----------------------------------------
placeholder    arg0         arg0                                                    ()                     {}
call_function  add          <built-in function add>                                 (arg0, 3.14)           {}
call_function  getitem      <built-in function getitem>                             (arg0, (0, 0))         {}
call_function  add_1        <built-in function add>                                 (add, getitem)         {}
call_module    self_linear  self_linear                                             (add_1,)               {}
call_function  split        <function split at 0x7f44c3b02440>                      (self_linear, [1, 3])  {'dim': -1}
call_function  getitem_1    <built-in function getitem>                             (split, 0)             {}
call_function  getitem_2    <built-in function getitem>                             (split, 1)             {}
call_function  mul          <built-in method mul of type object at 0x7f4547b83400>  ()                     {'input': getitem_1, 'other': getitem_2}
call_function  cat          <built-in method cat of type object at 0x7f4547b83400>  ([mul, mul],)          {'dim': -1}
call_method    flatten      flatten                                                 (cat,)                 {}
output         output       output                                                  ([flatten],)           {}
"""
