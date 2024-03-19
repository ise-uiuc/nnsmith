import operator
from typing import Any, Dict, List, cast

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.fx.passes.shape_prop import ShapeProp

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import ConcreteOp, Input
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR, InstExpr


class PropInterpreter(ShapeProp):
    def run_node(self, n: fx.node.Node) -> Any:
        result = super().run_node(n)
        n.meta["res"] = result
        return result


def parse(model: nn.Module, *example_args: List[torch.Tensor]) -> GraphIR:
    gm: fx.GraphModule = fx.symbolic_trace(model)
    # store shape info on nodes
    sp = PropInterpreter(gm)
    sp.run(*example_args)

    named_modules = dict(gm.named_modules())
    ir = GraphIR()
    name2retvals: Dict[str, List[str]] = {}
    for node in gm.graph.nodes:
        node = cast(fx.node.Node, node)
        if node.op == "placeholder":
            input_node = Input(dim=len(node.meta["res"].shape))
            input_node.abs_tensor = AbsTensor(
                shape=list(node.meta["res"].shape),
                dtype=DType.from_torch(node.meta["res"].dtype),
            )
            iexpr = InstExpr(input_node, [])
        else:
            args_flatten, _ = pytree.tree_flatten(node.args)
            kwargs_flatten, _ = pytree.tree_flatten(node.kwargs)
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
            nodes2empty = lambda n: (
                ConcreteOp.empty if isinstance(n, fx.node.Node) else n
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
