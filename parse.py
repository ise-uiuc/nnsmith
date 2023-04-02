import inspect
import operator
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import torch
import torch._dynamo as dynamo
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.fx.passes.shape_prop import ShapeProp

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsOpBase, Placeholder
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR, InstExpr, InstIR


class PropInterpreter(ShapeProp):
    def run_node(self, n: fx.node.Node) -> Any:
        result = super().run_node(n)
        n.meta["res"] = result
        return result


class ConcreteOp(AbsOpBase):
    def __init__(
        self,
        target: Callable,
        args_flatten: List[Any],
        args_tensor_indices: List[int],
        args_treespec: pytree.TreeSpec,
        kwargs_flatten: List[Any],
        kwargs_tensor_indices: List[int],
        kwargs_treespec: pytree.TreeSpec,
        outputs: List[Any],
        target_name: str = None,
    ) -> None:
        # super().__init__()
        self.target = target
        self.target_name = (target_name if target_name else str(target)).strip("<>")

        self.args_flatten = args_flatten
        self.kwargs_flatten = kwargs_flatten
        self.args_treespec = args_treespec
        self.kwargs_treespec = kwargs_treespec

        ts_args_fully_flatten = pytree.tree_flatten(
            [args_flatten[i] for i in args_tensor_indices]
        )[0]
        ts_kwargs_fully_flatten = pytree.tree_flatten(
            [kwargs_flatten[i] for i in kwargs_tensor_indices]
        )[0]

        self.bind_input_like(
            [
                AbsTensor(shape=t.shape, dtype=DType.from_torch(t.dtype))
                for t in ts_args_fully_flatten
            ]
            + [
                AbsTensor(shape=t.shape, dtype=DType.from_torch(t.dtype))
                for t in ts_kwargs_fully_flatten
            ]
        )
        # outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        outputs = pytree.tree_flatten(outputs)[0]
        self.bind_output_like(
            [
                AbsTensor(shape=o.shape, dtype=DType.from_torch(o.dtype))
                for o in outputs  # if isinstance(o, torch.Tensor)
            ]
        )
        # self.in_dtypes = [(
        #     i.dtype for i in self._input_like
        # )]
        # self.out_dtypes = [(
        #     o.dtype for o in self._output_like
        # )]
        # self.inp_ranks = [
        #     (i.ndims,) for i in self._input_like
        # ]
        # self.out_ranks = [
        #     (o.ndims,) for o in self._output_like
        # ]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        return self._output_like

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(o.ndims, o.dtype) for o in self._input_like]

    def n_input(self):
        return len(self._input_like)

    def n_output(self):
        return len(self._output_like)

    def __str__(self):
        return f"{super().__str__()}<{self.target_name}>"


def parse(model: nn.Module, *example_args: List[torch.Tensor]) -> GraphIR:
    dynamo.reset()
    gm: fx.GraphModule = dynamo.export(model, *example_args)[0]
    # store shape info on nodes
    sp = PropInterpreter(gm)
    sp.run(*example_args)

    def load_args(
        args: Union[List[Any], Dict[str, Any]]
    ) -> Union[List[Any], Dict[str, Any]]:
        """
        map nodes to their outputs while keeping structures and other values the same
        """
        return torch.fx.graph.map_arg(args, lambda n: n.meta["res"])

    named_modules = dict(gm.named_modules())
    ir = GraphIR()
    name_2_retvals: Dict[str, List[str]] = {}
    for i_node, node in enumerate(gm.graph.nodes):
        node = cast(fx.node.Node, node)
        print(f"{i_node = }, {node = }", flush=True)
        if node.op == "placeholder":
            iexpr = InstExpr(
                Placeholder(
                    ttype=AbsTensor(
                        shape=list(node.meta["tensor_meta"].shape),
                        dtype=DType.from_torch(node.meta["tensor_meta"].dtype),
                    )
                ),
                [],
            )
        else:
            args_flatten, args_treespec = pytree.tree_flatten(node.args)
            kwargs_flatten, kwargs_treespec = pytree.tree_flatten(node.kwargs)
            args_nodes_indices = [
                i for i, a in enumerate(args_flatten) if isinstance(a, fx.node.Node)
            ]
            kwargs_nodes_indices = [
                i for i, a in enumerate(kwargs_flatten) if isinstance(a, fx.node.Node)
            ]
            args_flatten_loaded = load_args(args_flatten)
            kwargs_flatten_loaded = load_args(kwargs_flatten)
            input_valstrs = [
                name_2_retvals[args_flatten[i].name][0] for i in args_nodes_indices
            ] + [
                name_2_retvals[kwargs_flatten[i].name][0] for i in kwargs_nodes_indices
            ]
            if node.op == "call_function":
                target = node.target
                if (
                    target is operator.getitem
                    and isinstance(node.args[0], fx.node.Node)
                    and not isinstance(node.args[0].meta["res"], torch.Tensor)
                ):
                    name_2_retvals[node.name] = [
                        name_2_retvals[node.args[0].name][node.args[1]]
                    ]
                    continue
            elif node.op == "call_method":
                target = getattr(torch.Tensor, node.target)
            elif node.op == "call_module":
                target = named_modules[node.target]
            elif node.op == "get_attr":
                raise NotImplementedError(f"{node.name}")
            elif node.op == "output":
                continue
            else:
                raise ValueError(f"Unexpected {node.op = }")

            try:
                target_name = node._pretty_print_target(target)
            except:
                target_name = None
            iexpr = InstExpr(
                ConcreteOp(
                    target,
                    args_flatten_loaded,
                    args_nodes_indices,
                    args_treespec,
                    kwargs_flatten_loaded,
                    kwargs_nodes_indices,
                    kwargs_treespec,
                    node.meta["res"],
                    target_name=target_name,
                ),
                input_valstrs,
            )

        name_2_retvals[node.name] = ir.add_inst(iexpr).retvals()

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
