import z3  # Always import z3 first to avoid incompatibility issue.
# See https://github.com/Z3Prover/z3/issues/5656
import networkx as nx
import torch
from torch import nn

from typing import Dict, Tuple, List
from inspect import signature
import random
import time
import os

from nnsmith.abstract.op import *
from nnsmith.export import torch2onnx
import copy


class RequiredDimNotFound(Exception):
    pass


class SymbolNet(nn.Module):
    def __init__(self, graph: nx.MultiDiGraph, model: z3.ModelRef):
        super(SymbolNet, self).__init__()
        self.tensors = []  # 1) edges; 2) leaf nodes; 3) input -> 0;
        self.ref_cnt = []  # ref cnt -> tensors; erased on 0;
        self.instructions = []  # <Func, <input idx>, <output idx>>
        self.n_output = 0
        # keep track of layers and weights so that the tracing can work properly
        self.mlist = nn.ModuleList()
        # NOTE: All leaf nodes are output tensors.

        tmp_op_output_map = {}  # node id -> output idx in tensors;
        for node_id in nx.topological_sort(graph):
            n_inp = graph.nodes[node_id]['nin']
            n_out = graph.nodes[node_id]['nout']

            tmp_op_output_map[node_id] = len(self.tensors)
            for _ in range(n_out):
                self.tensors.append(None)
                self.ref_cnt.append(0)

            input_idx = [None] * n_inp
            output_idx = [None] * n_out
            op = concretize(graph.nodes[node_id]['op'], model)

            # Glob inputs
            for from_node, _, (out_idx, in_idx) in graph.in_edges(node_id, data='operand_idx'):
                required = tmp_op_output_map[from_node] + out_idx
                input_idx[in_idx] = required
                self.ref_cnt[required] += 1

            # Glob outputs
            out_edges = graph.out_edges(node_id, data='operand_idx')
            if len(out_edges) == 0:  # leaf node
                # create fake output indices
                output_idx = list(range(
                    tmp_op_output_map[node_id], tmp_op_output_map[node_id] + n_out))
                for out_idx in output_idx:
                    self.ref_cnt[out_idx] += 1
                    self.n_output += 1
            else:
                for _, _, (out_idx, in_idx) in out_edges:
                    output_idx[out_idx] = tmp_op_output_map[node_id] + out_idx

            if len(input_idx) != 0:
                cur_op = op.torch()
                if isinstance(cur_op, nn.Module):
                    self.mlist.append(cur_op)
                self.instructions.append((cur_op, input_idx, output_idx))
            else:  # Should be input node
                assert type(op) is Input

    # TODO: Support multiple inputs.

    @torch.no_grad()
    def forward(self, x):
        local_ref_cnt = self.ref_cnt.copy()
        self.tensors[0] = x
        for inst, inps, outs in self.instructions:
            # print()
            # print(inst, inps, outs)
            outputs = inst(*[self.tensors[idx] for idx in inps])
            if not isinstance(outputs, list):
                outputs = [outputs]
            for idx in inps:
                local_ref_cnt[idx] -= 1
                if local_ref_cnt[idx] == 0:
                    self.tensors[idx] = None
            for idx, output in list(zip(outs, outputs)):
                assert self.tensors[idx] is None, 'tensor[{}] is not None.'.format(
                    idx)
                if local_ref_cnt[idx] > 0:  # Will be used.
                    self.tensors[idx] = output
        return tuple(t for t in self.tensors if t is not None)

    # TODO: Support multiple & dynamic inputs
    def set_input_spec(self, input_shape):
        self.plausible_input_shape = self.input_spec = {'i0': input_shape}


class SimpleGenerator:
    def __init__(self, init_dim_size=4, skip=[], viz=False, min_size_rng=[10, 100], seed=None, verbose=False):
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)

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

        # FIXME: Apply concolic execution when concretizing the symbols.
        # The batch size should not have a big min size (avoid unnecessary computation);
        for i in range(1, len(input_tensor_shape.shape)):
            self.solver.add(input_tensor_shape.shape[i] > random.randint(
                min_size_rng[0], min_size_rng[1]))
        self.input_shape = input_tensor_shape  # TODO: multiple input/output.

    def concretize_input_shape(self, model):
        shape = []
        for s in self.input_shape.shape:
            if isinstance(s, z3.ArithRef):
                shape.append(model.eval(s).as_long())
            else:
                shape.append(s)
        return shape

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
        if self.verbose:
            print('checking...')
        res = self.solver.check() == z3.sat
        if self.verbose:
            print('done')
        return res

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
            new_node_idx, op=node, nin=len(ishape_indices), nout=len(oshapes), label=str(node))

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
        if self.verbose:
            print(node, constraints0)
            self.viz('currentgraph.png')
        self.solver.push()
        for c in constraints0:
            self.solver.add(c)
        if not self.check_sat():
            self.solver.pop()
            return False

        # make a copy
        output_shapes = node.shape_fn(copy.deepcopy(input_shapes))

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
    parser.add_argument('--output_path', type=str, default='output.onnx')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    strt_time = time.time()
    gen = SimpleGenerator(init_dim_size=args.dim_size,
                          viz=args.viz, seed=args.seed, verbose=args.verbose)
    gen.abstract_gen(max_node_size=args.max_nodes,
                     max_gen_millisec=args.timeout)
    print(f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')
    solution = gen.solver.model()
    print(f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
    print(solution)

    gen.viz(args.output_path + '.png')

    input_shape = gen.concretize_input_shape(solution)
    print(f'Input shape: {input_shape}')

    net = SymbolNet(gen.abstract_graph, solution)
    net.eval()
    net.set_input_spec(input_shape)
    torch2onnx(model=net, filename=args.output_path)

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
