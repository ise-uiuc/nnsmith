import z3  # Always import z3 first to avoid incompatibility issue.
# See https://github.com/Z3Prover/z3/issues/5656
import networkx as nx

from typing import Dict, Tuple, List
from inspect import signature
import random
import time
import os

from nnsmith.abstract.op import *


class RequiredDimNotFound(Exception):
    pass


class SimpleGenerator:
    def __init__(self, init_dim_size=4, skip=[], viz=False):
        # TODO: all operator types.
        self.op_candidates = [op for op in ALL_OP_TYPES if op not in skip]
        self.solver = z3.Solver()

        # Node -> op: AbsOpBase
        # Edge -> shape_idx:-> self.alive_shapes
        self.abstract_graph = nx.MultiDiGraph()

        # <op idx, shape variable, output operand idx>
        self.alive_shapes: List[Tuple[int, ShapeVar, int]] = []
        # dim size -> list[shape idx -> output_tensor_pool]
        self.dim2shape_idx: Dict[int, List[int]] = {}
        self.viz_cnt = 0
        self.is_viz = viz

        input_node = Input()
        input_node.inp_dims = input_node.out_dims = [init_dim_size]
        input_tensor_shape = ShapeVar(
            shape=[z3.Int('i%s' % k) for k in range(init_dim_size)])
        self.insert_node(input_node, [input_tensor_shape], ishape_indices=[])
        for c in input_tensor_shape.gt_zero():
            self.solver.add(c)
        self.input_node = input_node  # TODO: multiple input/output.

    def extra_exit_check(self) -> bool:
        """
        Returns:
            bool: add more checks to determine whether to exit the generation.
        """
        return False

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        init_time = time.time()
        while time.time() - init_time < max_gen_millisec / 1000 and len(
                self.abstract_graph.nodes) < max_node_size:
            if self.extra_exit_check():
                break
            node_t = self.pick_next_op_type()
            self.try_insert_node_type(node_t)

    def shape_idx_to_op_idx(self, shape_idx: int) -> int:
        return self.alive_shapes[shape_idx][0]

    def check_sat(self) -> bool:
        return self.solver.check() == z3.sat

    def pick_next_op_type(self):
        return random.choice(self.op_candidates)

    def insert_node(self, node: AbsOpBase, oshapes: List[ShapeVar], ishape_indices: List[int]):
        new_node_idx = len(self.abstract_graph.nodes)
        for i, shape_var in enumerate(oshapes):
            if node.out_dims[i] == -1:
                node.out_dims[i] = len(shape_var.shape)
            else:
                assert node.out_dims[i] == len(shape_var.shape), "{}'s dimension size is not {} in {}".format(
                    shape_var.shape, node.out_dims[i], node.__class__.__name__)
            shape_idx = len(self.alive_shapes)
            self.alive_shapes.append((new_node_idx, shape_var, i))
            self.dim2shape_idx.setdefault(
                len(shape_var.shape), []).append(shape_idx)

        self.abstract_graph.add_node(
            new_node_idx, op=node, label=str(node))

        for in_operand_idx, idx in enumerate(ishape_indices):
            old_node_idx, _, out_operand_idx = self.alive_shapes[idx]
            self.abstract_graph.add_edge(old_node_idx, new_node_idx, shape_idx=idx, operand_idx=(
                out_operand_idx, in_operand_idx), label=f'{out_operand_idx}-{in_operand_idx}: {self.alive_shapes[idx][1]}')

        if self.is_viz:
            self.viz()

    def try_insert_node_type(self, node_t, max_shape_var_pick_time=5) -> bool:
        op_param_n = signature(node_t).parameters
        op_id = len(self.abstract_graph.nodes)
        op_params = [z3.Int('op%s_%s' % (op_id, k))
                     for k in range(len(op_param_n))]

        op: AbsOpBase = node_t(*op_params)

        n_inp = len(op.inp_dims)
        same_input_dims = op.same_inp_dims

        dim_spec_list = []

        if same_input_dims:  # find `n_inp` under the same input shapes.
            final_dim = -1
            for dim in op.inp_dims:
                if dim != -1:
                    if final_dim == -1:
                        final_dim = dim
                    else:
                        assert final_dim == dim
            if final_dim == -1:
                final_dim = random.choice(list(self.dim2shape_idx.keys()))
            dim_spec_list = [final_dim] * n_inp
        else:  # inputs have different dimension sizes.
            dim_spec_list = op.inp_dims

        try:
            for _ in range(max_shape_var_pick_time):
                ishape_indices = self.pick_shape_var_idx(dim_spec_list)
                if self.try_insert_node(op, ishape_indices):
                    return True
        except RequiredDimNotFound:
            return False

        return False

    def pick_shape_var_idx(self, dim_size_list: List[int]) -> List[int]:
        """Randomly pick indices to shape variables from the output pool.

        Args:
            dim_size_list (List[int]): required dimension sizes of the shape variables.

        Returns:
            List[int]: indices to applicable shape variables.
        """
        shape_var_candidates = []
        for dim_size in dim_size_list:
            if dim_size == -1:  # Arbitrary dimension size.
                shape_var_candidates.append(
                    random.randint(0, len(self.alive_shapes) - 1))
            elif dim_size in self.dim2shape_idx:
                shape_var_candidates.append(
                    random.choice(self.dim2shape_idx[dim_size]))
            else:
                raise RequiredDimNotFound(
                    'Cannot find a shape variable with dimension size %s.' % dim_size)
        return shape_var_candidates

    def try_insert_node(self, node: AbsOpBase, ishape_indices: List[int]) -> bool:
        input_shapes = [self.alive_shapes[idx][1] for idx in ishape_indices]
        constraints0 = node.requires(input_shapes)
        self.solver.push()
        for c in constraints0:
            self.solver.add(c)
        if not self.check_sat():
            self.solver.pop()
            return False

        output_shapes = node.shape_fn(input_shapes)

        self.solver.push()
        for shape in output_shapes:
            for c in shape.gt_zero():
                self.solver.add(c)

        if not self.check_sat():
            self.solver.pop()
            return False

        self.insert_node(node, output_shapes, ishape_indices)
        return True

    def viz(self, filename: str = None):
        if filename is None:
            filename = f'step{self.viz_cnt}.png'
        G = self.abstract_graph
        nx.drawing.nx_pydot.write_dot(G, 'graph.dot')
        os.system(f'dot -Tpng graph.dot > {filename}')
        self.viz_cnt += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nodes', type=int, default=10)
    parser.add_argument('--dim_size', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=10000)
    parser.add_argument('--viz', type=bool, default=False)
    args = parser.parse_args()

    strt_time = time.time()
    gen = SimpleGenerator(init_dim_size=args.dim_size, viz=args.viz)
    gen.abstract_gen(max_node_size=args.max_nodes,
                     max_gen_millisec=args.timeout)
    print(f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')
    solution = gen.solver.model()
    print(f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
    print(solution)

    gen.viz('final_graph.png')

    # Draw with NetworkX
    # import matplotlib.pyplot as plt
    # import pygraphviz as pgv

    # fig_size = max(8, args.max_nodes)
    # plt.figure(figsize=(fig_size, fig_size * 1.2))

    # pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')

    # nx.draw(G, pos, node_size=fig_size * 500)
    # node_labels = nx.get_node_attributes(G, 'label')
    # nx.draw_networkx_labels(G, pos, labels=node_labels)
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # plt.savefig("graph_nx.png")
