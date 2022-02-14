from collections import Counter, defaultdict
import math
import textwrap
from sklearn.utils import shuffle
import z3  # Always import z3 first to avoid incompatibility issue.
# See https://github.com/Z3Prover/z3/issues/5656
import networkx as nx
from summary import ParamShapeSummary
import torch
from torch import nn
import numpy as np
import uuid

import pickle
import cloudpickle
from typing import Dict, NamedTuple, Tuple, List, Optional
from inspect import signature
import traceback
import random
import time
import os
import copy

from nnsmith.error import NNSmithInternalError, SanityCheck, ConstraintCheck, ConstraintError
from nnsmith.export import torch2onnx
from nnsmith.abstract.op import *


class RequiredDimNotFound(Exception):
    pass


ALIVE_SHAPE_TYPE = List[Tuple[int, ShapeVar, int]]


InputInfo = NamedTuple(
    'InputInfo', [('op', Input), ('oid', int), ('node_id', int), ('input_name', str)])


class SymbolNet(nn.Module):
    def __init__(self, graph: nx.MultiDiGraph, model: z3.ModelRef, verbose=False, alive_shapes: ALIVE_SHAPE_TYPE = None,
                 record_intermediate=False, use_gradient=False):
        super(SymbolNet, self).__init__()
        self.verbose = verbose
        self.tensors = []  # 1) edges; 2) leaf nodes; 3) input -> 0;
        self.ref_cnt = []  # ref cnt -> tensors; erased on 0;
        self.instructions = []  # <Func, <input idx>, <output idx>>
        self.n_output = 0
        self.inp_id_cnt = 0

        # keep track of layers and weights so that the tracing can work properly
        self.mlist = nn.ModuleList()
        self.graph = graph
        self.concrete_graph = graph.copy()
        # NOTE: All leaf nodes are output tensors.
        self.alive_shapes = alive_shapes
        if alive_shapes is None:
            warnings.warn(
                "Please supply `alive_shapes` if possible. This will be used to check dtype correctness.")
        # whether or not to register intermediate tensors as output tensors. Useful (at least) for checking nan
        self.record_intermediate = record_intermediate

        self.input_info: List[InputInfo] = []

        tmp_op_output_map = {}  # node id -> output idx in tensors;
        shape_vars = {}
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
            self.concrete_graph.nodes[node_id]['op'] = op

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

            if not isinstance(op, Input):
                cur_op = op.torch()
                if isinstance(cur_op, nn.Module):
                    self.mlist.append(cur_op)
                self.instructions.append(
                    (cur_op, input_idx, output_idx, op, node_id))
            else:  # Should be input node
                SanityCheck.true(type(op) is Input, 'type(op) should be Input')
                SanityCheck.eq(len(output_idx), 1)
                op.idx = self.inp_id_cnt
                self.inp_id_cnt += 1
                self.input_info.append(
                    InputInfo(op=op, oid=output_idx[0], node_id=node_id, input_name=f'i{op.idx}'))

            # concretize shapevars
            ishape_indices = self.graph.nodes[node_id]['ishape_indices']
            shape_indices = self.graph.nodes[node_id]['shape_indices']
            for shape_idx in shape_indices:
                shape = self.alive_shapes[shape_idx][1].shape
                dtype = self.alive_shapes[shape_idx][1].dtype
                shape = [model.eval(i).as_long() if isinstance(
                    i, z3.ExprRef) else i for i in shape]
                assert shape_idx not in shape_vars
                shape_vars[shape_idx] = ShapeVar(shape, dtype)
            self.concrete_graph.nodes[node_id]['in_svs'] = [
                shape_vars[i] for i in ishape_indices]
            self.concrete_graph.nodes[node_id]['out_svs'] = [
                shape_vars[i] for i in shape_indices]

        if self.verbose:
            print('input_info=', self.input_info)
        self.input_spec = {
            f'i{ii.op.idx}': ii.op.shape_var.shape for ii in self.input_info}
        self.plausible_input_shape = {
            f'i{ii.op.idx}': ii.op.shape_var for ii in self.input_info}
        self.first_run = True
        self.hacked = {}  # make forward deterministic

        self.use_gradient = use_gradient
        if use_gradient:
            self.enable_training()
        self.check_intermediate_numeric = False
        self.invalid_found_last = None

    def to_picklable(self):
        self.alive_shapes = None
        del self.graph

    def backward(self):
        if self.loss is not None:
            self.optimizer.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1e-1)
            self.optimizer.step()

    def training_reset(self):
        self.loss = None
        self.stop_updating_loss = False

    def stop_training(self):
        self.use_gradient = False
        self.loss = None

    def enable_training(self, extra_trainable=[]):
        self.use_gradient = True
        to_train = []
        for t in extra_trainable:
            to_train.append(t)
        for t in self.parameters():
            to_train.append(t)
        self.optimizer = torch.optim.Adam(to_train, lr=5e-2)
        self.training_reset()

    def _check_out_dtype(self, outputs, node_id, op):
        if self.alive_shapes is None:
            return
        msg_head = f'In dtype checking for {op} (#{node_id}): '
        shape_indices = self.graph.nodes[node_id]['shape_indices']
        SanityCheck.eq(len(outputs), len(shape_indices), msg_head +
                       f'{len(outputs)} != {len(shape_indices)}')
        for out, shape_idx in zip(outputs, shape_indices):
            SanityCheck.eq(out.dtype, self.alive_shapes[shape_idx][1].dtype.value, msg_head +
                           f'torch dtype ({out.dtype}) != symbolic dtype ({self.alive_shapes[shape_idx][1].dtype.value})')

    def get_random_inps(self) -> List[torch.Tensor]:
        return [torch.rand(ii.op.shape_var.shape) for ii in self.input_info]

    def rand_input_gen(self, max_iter=10, use_cuda=False) -> Optional[List[torch.Tensor]]:
        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        sat_inputs = None

        n_step = max_iter
        interval = 1 / n_step
        for v in np.linspace(-1, 1, n_step):
            inputs = [v + torch.rand(ii.op.shape_var.shape)
                      * interval for ii in self.input_info]

            if use_cuda:
                inputs = [inp.cuda() for inp in inputs]
                self = self.cuda()

            self.forward(*inputs)

            if not self.invalid_found_last:
                sat_inputs = inputs
                break

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def grad_input_gen(self, max_iter=10, init_tensors=None, use_cuda=False) -> Optional[List[torch.Tensor]]:
        if init_tensors is None:
            inputs = [torch.nn.parameter.Parameter(torch.rand(ii.op.shape_var.shape))
                      for ii in self.input_info]
        else:
            inputs = [torch.nn.parameter.Parameter(
                tensor.data) for tensor in init_tensors]
        self.enable_training(extra_trainable=inputs)

        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        if use_cuda:
            inputs = [inp.cuda() for inp in inputs]
            self = self.cuda()

        sat_inputs = None
        for _ in range(max_iter):
            self.training_reset()

            try:
                _ = self(*inputs)
            except ConstraintError as _:
                break

            if self.invalid_found_last:  # need_to_train
                self.backward()
            else:
                sat_inputs = [v.data for v in inputs]
                break

        self.stop_training()
        if sat_inputs is None:
            print('[grad] no valid range found!!!')

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def forward(self, *args, **kwargs):
        # required: input_info, tensors, ref_cnt, instructions, hacked, first_run verbose # alive_shapes, graph
        xs = [None] * len(self.input_info)
        for i in range(len(args)):
            xs[i] = args[i]
        for ii in self.input_info:
            if ii.input_name in kwargs:
                xs[ii.op.idx] = kwargs[ii.input_name]
        assert all(x is not None for x in xs), xs
        local_ref_cnt = self.ref_cnt.copy()
        self.tensors = [None for _ in self.tensors]
        self.invalid_found_last = False

        for ii in self.input_info:
            self.tensors[ii.oid] = xs[ii.op.idx]

        for inst, inps, outs, op, node_id in self.instructions:
            input_tensors = [self.tensors[idx] for idx in inps]
            if isinstance(op, Div):
                if not self.first_run:
                    cond = self.hacked[node_id]
                else:
                    cond = (input_tensors[1] == 0).any()
                if cond:
                    input_tensors[1] = torch.clip(
                        input_tensors[1], torch.ones(size=[1], dtype=input_tensors[1].dtype).to(input_tensors[1].device))
                self.hacked[node_id] = cond
            if self.verbose:
                print(
                    f'executing instruction op={op}, node_id={node_id}, inps={inps}, outs={outs}')
                print('input_tensors=')
                for i in input_tensors:
                    print(f'  (shape={i.shape} dtype={i.dtype})')
            outputs = inst(*input_tensors)
            if not isinstance(outputs, list):
                outputs = [outputs]
            self._check_out_dtype(outputs, node_id, op)

            if self.check_intermediate_numeric or (self.use_gradient and not self.stop_updating_loss):
                with torch.no_grad():
                    invalid_mask = [torch.isnan(out).any() or torch.isinf(
                        out).any() for out in outputs]

                self.invalid_found_last |= any(invalid_mask)
                if self.invalid_found_last and (self.use_gradient and not self.stop_updating_loss):
                    print(
                        f'Detected NaN or Inf in outputs ~ {op} ~ id {node_id}.')
                    if self.verbose:
                        for inp_i, inp in enumerate(input_tensors):
                            print(
                                f'[inp]@{inp_i} :: {inp.min().data:.5f} ~ {inp.max().data:.5f}')

                    ConstraintCheck.true(hasattr(
                        op, 'torch_loss'), f'op={op} has no `torch_loss` but produces NaN or INF!')
                    vul_op_loss = op.torch_loss(*input_tensors)

                    if self.verbose:
                        print(
                            f'vulnerable op loss :: {vul_op_loss.min().data:.5f} ~ {vul_op_loss.max().data:.5f}')
                    if self.loss is None:
                        self.loss = vul_op_loss.mean()
                    else:
                        self.loss += vul_op_loss.mean()
                    self.stop_updating_loss = True
                    return outputs

            if self.verbose:
                print('outputs=', ','.join(
                    f'(shape={i.shape} dtype={i.dtype})' for i in outputs))
            for idx in inps:
                local_ref_cnt[idx] -= 1
                if local_ref_cnt[idx] == 0 and not self.record_intermediate:
                    self.tensors[idx] = None
            for idx, output in list(zip(outs, outputs)):
                SanityCheck.none(self.tensors[idx], 'tensor[{}] is not None.'.format(
                    idx))
                if local_ref_cnt[idx] > 0:  # Will be used.
                    self.tensors[idx] = output
        self.first_run = False
        return tuple(t for t in self.tensors if t is not None)


class SimpleGenerator:

    def __init__(self, min_dims=[1, 3, 48, 48], skip=[Input], viz_sbs=False, megabyte_lim=6 * 1024, seed=None, verbose=False, use_bitvec=False,
                 viz_verbose=False):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.verbose = verbose
        self.viz_verbose = viz_verbose
        auto_infer_in_dtypes(self.verbose)

        self.op_candidates = [
            op for op in ALL_OP_TYPES if op not in skip and not op._skip]
        self.solver = z3.Solver()
        # 4 bytes per float (assume we use float32)
        self.limit_float = 1024**2 * megabyte_lim / 4

        # Node -> op: AbsOpBase
        # Edge -> shape_idx:-> self.alive_shapes
        self.abstract_graph = nx.MultiDiGraph()
        self.picklable_graph = nx.MultiDiGraph()

        # <op idx, shape variable, output operand idx>
        self.alive_shapes: ALIVE_SHAPE_TYPE = []
        # dim size -> list[shape idx -> output_tensor_pool]
        self.dim2shape_idx: Dict[int, List[int]] = {}
        self.viz_cnt = 0
        self.is_viz_sbs = viz_sbs

        self.use_bitvec = use_bitvec
        # self.input_shape = self.insert_input_node(min_dims)
        self.min_dims = min_dims
        self.n_floats = 0
        self.n_inps = 0
        self.monotonic_placeholder_id = 0
        self.monotonic_nx_node_idx = 0
        self.reusable_placeholder_nx_indices = []
        self.last_soln = None
        self.wts = None

        # <op idx>
        self.placeholders: List[int] = []
        init_placeholder = self.create_placeholder(len(min_dims))
        self.forward_insert_node(init_placeholder, [], oshapes=[
                                 init_placeholder.out_shape])

    def create_placeholder(self, dim, dtype=None):
        shapevar = ShapeVar(
            shape=[self.new_sym('var%s_%s' % (
                self.monotonic_placeholder_id, k)) for k in range(dim)],
            dtype=dtype if dtype is not None else random.choice(DTYPE_ALL))
        self.monotonic_placeholder_id += 1
        return Placeholder(shapevar)

    def new_sym(self, name):
        if self.use_bitvec:
            return z3.BitVec(name, 8)
        else:
            return z3.Int(name)

    @ abstractmethod
    def insert_input_node(self, min_dims, shape=None, dtype=DType.float32) -> ShapeVar:
        raise NotImplementedError

    @ abstractmethod
    def try_insert_node(self, node: AbsOpBase, ishape_indices: List[int]) -> bool:
        raise NotImplementedError

    @ abstractmethod
    def try_occupy_placeholder(self, node: AbsOpBase, placeholder_indices: List[int]) -> bool:
        raise NotImplementedError

    @ abstractmethod
    def get_symbol_solutions(self) -> List:
        raise NotImplementedError

    def extra_exit_check(self) -> bool:
        """
        Returns:
            bool: add more checks to determine whether to exit the generation.
        """
        return False

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        z3.set_param(
            "smt.phase_selection",
            5,
            "smt.arith.random_initial_value",
            True,
            "sat.phase",
            "random",
            "timeout",
            max_gen_millisec // 3,
            "memory_max_size",
            16 * 1024,  # MB
        )
        init_time = time.time()
        while time.time() - init_time < max_gen_millisec / 1000 and len(
                self.abstract_graph.nodes) < max_node_size:
            if self.extra_exit_check():
                break
            node_t = self.pick_next_op_type()
            self.try_insert_node_type(node_t)
        if len(self.abstract_graph.nodes) != max_node_size:
            print(
                f'[WARNING]: graph size: {len(self.abstract_graph.nodes)} != expected size: {max_node_size}')
        # init graph placeholders
        shuffled_placeholder = self.placeholders
        self.abstract_graph.nodes[shuffled_placeholder[0]
                                  ]['op'] = self.abstract_graph.nodes[shuffled_placeholder[0]]['op'].to_input()
        for holder_idx in shuffled_placeholder[1:]:
            if random.randint(0, 1):
                self.abstract_graph.nodes[holder_idx]['op'] = self.abstract_graph.nodes[holder_idx]['op'].to_const(
                )
            else:
                self.abstract_graph.nodes[holder_idx]['op'] = self.abstract_graph.nodes[holder_idx]['op'].to_input(
                )

    def shape_idx_to_op_idx(self, shape_idx: int) -> int:
        return self.alive_shapes[shape_idx][0]

    def check_sat(self, *assumptions):
        start = time.time()
        cres = self.solver.check(*assumptions)

        checking_time = int((time.time() - start) * 1000)
        if checking_time > 3000 and self.cur_node:  # 3s
            warnings.warn(
                f'[WARNING] check {self.cur_node} {checking_time} ms')

        if self.verbose:
            print(cres, '<-- checking time:', checking_time, 'ms')

            if cres == z3.unsat:
                print(f'Unsat core: {self.solver.unsat_core()}')
        if cres == z3.sat:
            self.last_soln = self.solver.model()
        return cres

    def compute_wts(self):
        self.wts = [1] * len(self.op_candidates)
        normalize_op_t = [Constant, Cast]
        op_t_idx = {}
        for i in range(len(self.op_candidates)):
            for op_t in normalize_op_t:
                if issubclass(self.op_candidates[i], op_t):
                    op_t_idx[op_t] = op_t_idx.get(op_t, []) + [i]

        for idx in op_t_idx.values():
            for i in idx:
                self.wts[i] = 1.0 / len(idx)

    def pick_next_op_type(self):
        if self.wts is None:
            self.compute_wts()
        return random.choices(self.op_candidates, k=1, weights=self.wts)[0]

    def forward_insert_node(self, node: AbsOpBase, ishape_indices: List[int], oshapes: List[ShapeVar] = None, force_shape_indices=None) -> int:
        if oshapes is None:
            input_shapes = [self.alive_shapes[idx][1]
                            for idx in ishape_indices]
            oshapes = node.shape_fn(copy.deepcopy(input_shapes))

        succ_nid = self.get_new_node_id()
        if isinstance(node, Placeholder):
            self.placeholders.append(succ_nid)

        shape_indices = []
        if force_shape_indices is None:
            for i, shape_var in enumerate(oshapes):
                if node.out_ranks[i] == -1:
                    node.out_ranks[i] = len(shape_var.shape)
                else:
                    SanityCheck.eq(node.out_ranks[i], len(shape_var.shape), "{}'s dimension size is not {} in {}".format(
                        shape_var.shape, node.out_ranks[i], node.__class__.__name__))
                shape_idx = len(self.alive_shapes)
                shape_indices.append(shape_idx)
                self.alive_shapes.append((succ_nid, shape_var, i))
                self.dim2shape_idx.setdefault(
                    len(shape_var.shape), []).append(shape_idx)
        else:
            shape_indices = force_shape_indices

        # NOTE: because of backward insertion, we may not be able to limit the symbol size as there will be some
        # trivially equivalent symbols which harms the readability. (e.g., relations like `a = b` is not known).
        self.abstract_graph.add_node(
            succ_nid, op=node,
            nin=len(ishape_indices),
            nout=len(oshapes),
            shape_indices=shape_indices,
            ishape_indices=ishape_indices,
            label=textwrap.fill(
                f'#{succ_nid} ~ {node}' if not self.viz_verbose else '', width=30))

        for in_operand_idx, idx in enumerate(ishape_indices):
            pred_nid, svar, out_operand_idx = self.alive_shapes[idx]
            self.abstract_graph.add_edge(pred_nid, succ_nid, key=str(uuid.uuid1()), shape_idx=idx, operand_idx=(
                out_operand_idx, in_operand_idx), label=f'{out_operand_idx}-{in_operand_idx}: <{svar.dtype}>{svar.shape}' if not self.viz_verbose else '')

        if self.is_viz_sbs:
            self.viz()

        return succ_nid

    def get_new_node_id(self):
        if self.reusable_placeholder_nx_indices:
            return self.reusable_placeholder_nx_indices.pop()
        ret = self.monotonic_nx_node_idx
        self.monotonic_nx_node_idx += 1
        return ret

    def id2nxnode(self, id):
        return self.abstract_graph.nodes[id]

    def backward_insert_node(self, node, input_nodes: List[Union[int, Placeholder]], occupied_idx):
        # self.placeholder idx -> nx graph node idx
        occ_holder_idx_nx = [self.placeholders[i] for i in occupied_idx]

        ishape_indices = []
        for input_node in input_nodes:
            # Insert Placeholder in `input_nodes`
            if isinstance(input_node, Placeholder):
                nid = self.get_new_node_id()
                shape_idx = len(self.alive_shapes)
                self.alive_shapes.append((nid, input_node.out_shape, 0))
                self.dim2shape_idx.setdefault(
                    input_node.out_shape.ndims, []
                ).append(shape_idx)
                self.abstract_graph.add_node(
                    nid,
                    op=input_node,
                    nin=0,
                    nout=1,
                    ishape_indices=[],
                    shape_indices=[shape_idx],
                    label=textwrap.fill(
                        f'#{nid} ~ {input_node}' if not self.viz_verbose else '', width=30),
                )
                ishape_indices.append(shape_idx)
                self.placeholders.append(nid)
            else:
                ishape_indices.append(input_node)

        # Insert node
        op_nx_idx = self.forward_insert_node(
            node,
            ishape_indices,
            [self.alive_shapes[self.id2nxnode(nx_nid)['shape_indices'][0]][1]
                for nx_nid in occ_holder_idx_nx],
            force_shape_indices=[self.id2nxnode(nx_nid)['shape_indices'][0] for nx_nid in occ_holder_idx_nx])

        # Insert edges and remove placeholders
        for i, nx_idx in enumerate(occ_holder_idx_nx):
            for (src, dst, key) in list(self.abstract_graph.edges(nx_idx, keys=True)):
                # multi-graph
                edge_info = self.abstract_graph.get_edge_data(
                    src, dst, key=key)
                _, svar, out_operand_idx = self.alive_shapes[edge_info['shape_idx']]
                out_operand_idx = i
                in_operand_idx = edge_info['operand_idx'][1]
                self.abstract_graph.add_edge(
                    op_nx_idx,
                    dst,
                    key=str(uuid.uuid1()),
                    shape_idx=edge_info['shape_idx'],  # reuse old alive shape
                    operand_idx=(out_operand_idx, in_operand_idx),
                    label=f'{out_operand_idx}-{in_operand_idx}: <{svar.dtype}>{svar.shape}' if not self.viz_verbose else ''
                )
                self.alive_shapes[edge_info['shape_idx']] = (
                    op_nx_idx, *self.alive_shapes[edge_info['shape_idx']][1:])
                self.abstract_graph.remove_edge(src, dst, key=key)

            # remove placeholders
            self.abstract_graph.remove_node(nx_idx)
            self.reusable_placeholder_nx_indices.append(nx_idx)
            self.placeholders.remove(nx_idx)

        if self.is_viz_sbs:
            self.viz()

    def try_insert_node_type_at(self, node_t, ishape_indices: List[int]) -> bool:
        if self.verbose:
            print(f'Inserting node #{len(self.abstract_graph.nodes)}: '
                  f'trying to insert node type {node_t.__name__}')
        if issubclass(node_t, Input):
            try:
                self.insert_input_node(self.min_dims)
            # TODO: check the exception type (ideally only z3 check_failure), don't drop internal errors
            except:
                return False
            return True
        op_param_n = signature(node_t).parameters
        op_id = len(self.abstract_graph.nodes)
        op_params = [self.new_sym('op%s_%s' % (op_id, k))
                     for k in range(len(op_param_n))]

        op: AbsOpBase = node_t(*op_params)

        try:
            if self.try_insert_node(op, ishape_indices):
                return True
        except RequiredDimNotFound:
            if self.verbose:
                traceback.print_exc()
            return False
        except ConstraintError:
            if self.verbose:
                traceback.print_exc()
            return False

        return False

    def try_forward_insert(self, op: AbsOpBase):
        n_inp = len(op.inp_ranks)
        dim_spec_list = []

        if op.same_inp_dims:  # find `n_inp` under the same input shapes.
            final_dim = -1
            for dim in op.inp_ranks:
                if dim != -1:
                    if final_dim == -1:
                        final_dim = dim
                    else:
                        SanityCheck.eq(final_dim, dim)
            if final_dim == -1:
                final_dim = random.choice(list(self.dim2shape_idx.keys()))
            dim_spec_list = [final_dim] * n_inp
        else:  # inputs have different dimension sizes.
            dim_spec_list = op.inp_ranks

        ishape_indices = self.pick_shape_var_idx(
            type(op), dim_spec_list, op.in_dtypes, candidate_shapes=[s[1] for s in self.alive_shapes])

        if self.try_insert_node(op, ishape_indices):
            return True

        return False

    def try_backward_insert(self, op: AbsOpBase):
        # we know that: Y = op(X)
        # S1 - select Y: Y must be a placeholder; (this also means the graph must start w/ a placeholder)
        placeholder_indices = self.pick_shape_var_idx(
            type(op), op.out_ranks, op.out_dtypes, candidate_shapes=[self.id2nxnode(idx)['op'].out_shape for idx in self.placeholders])
        
        print(type(op))
        print([self.id2nxnode(self.placeholders[idx])['op'].out_shape for idx in placeholder_indices])

        if self.try_occupy_placeholder(op, placeholder_indices):
            return True

        return False

    def try_insert_node_type(self, node_t, max_shape_var_pick_time=3) -> bool:
        if self.verbose:
            print(f'Inserting node #{len(self.abstract_graph.nodes)}: '
                  f'trying to insert node type {node_t.__name__}')

        op_param_n = signature(node_t).parameters
        op_id = len(self.abstract_graph.nodes)
        op_params = [self.new_sym('op%s_%s' % (op_id, k))
                     for k in range(len(op_param_n))]

        op: AbsOpBase = node_t(*op_params)

        try:
            for _ in range(max_shape_var_pick_time):
                if random.randint(0, 1):
                    if self.try_forward_insert(op):
                        return True
                else:
                    if self.try_backward_insert(op):
                        return True
        except RequiredDimNotFound:
            if self.verbose:
                traceback.print_exc()
            return False
        except ConstraintError:
            if self.verbose:
                traceback.print_exc()
            return False

        return False

    def filter_shapes(self, ndim, dtype, candidate_shapes: List[ShapeVar]):
        cans = range(len(candidate_shapes))

        cans = list(filter(  # filter with ndim
            lambda sid: candidate_shapes[sid].ndims == ndim or ndim == -1, cans))
        if len(cans) == 0:
            raise RequiredDimNotFound(
                'Cannot find a shape variable with #dimensions %s.' % ndim)

        if dtype is not None:
            cans = list(filter(  # filter with dtype
                lambda sid: candidate_shapes[sid].dtype == dtype, cans))
            if len(cans) == 0:
                raise RequiredDimNotFound(
                    'Cannot find a shape variable with #dimensions %s and dtype %s.' % (ndim, dtype))

        return cans

    def pick_shape(self, node_t, candidates):
        return random.choice(candidates)

    def pick_shape_var_idx(self, node_t, ndim_list: List[int], dtype_combs: List[DTypeComb], candidate_shapes: List[ShapeVar]) -> List[int]:
        """Randomly pick indices to shape variables from the output pool.

        Args:
            ndim_list (List[int]): required dimension sizes of the shape variables.

        Returns:
            List[int]: indices to applicable shape variables.
        """

        shape_var_candidates = []
        if self.verbose:
            print('dtype_combs:', dtype_combs)

        all_can_dtypes = []
        for i, ndim in enumerate(ndim_list):
            all_can_dtypes.extend([candidate_shapes[i].dtype for i in self.filter_shapes(
                ndim=ndim, dtype=None, candidate_shapes=candidate_shapes)])
        # only use dtypes currently available after ndim filtering
        dtype_combs = [comb for comb in dtype_combs if all(
            i in all_can_dtypes for i in comb)]
        if len(dtype_combs) == 0:
            raise RequiredDimNotFound('Op %s: Cannot find a shape variable with dim_spec %s and dtype combinations %s.' % (
                node_t, ndim_list, dtype_combs))
        dtype_comb = random.choice(dtype_combs)
        for i, ndim in enumerate(ndim_list):
            candidates = self.filter_shapes(
                ndim=ndim, dtype=dtype_comb[i], candidate_shapes=candidate_shapes)
            shape_var_candidates.append(
                self.pick_shape(node_t, candidates))

        return shape_var_candidates

    def viz(self, filename: str = None):
        if filename is None:
            filename = f'step{self.viz_cnt}.png'
        G = self.abstract_graph
        nx.drawing.nx_pydot.write_dot(G, 'graph.dot')
        os.system(f'dot -Tpng graph.dot > {filename}')
        self.viz_cnt += 1


class PureSymbolGen(SimpleGenerator):
    def insert_input_node(self, min_dims, dtype=DType.float32, constrain_min=True) -> ShapeVar:
        input_tensor_shape = ShapeVar(
            shape=[self.new_sym('i%s_s%s' % (self.n_inps, k)) for k in range(len(min_dims))], dtype=dtype)
        input_node = Input(self.n_inps, dtype, *input_tensor_shape.shape)

        self.forward_insert_node(input_node, [], oshapes=[input_tensor_shape])
        for c in input_tensor_shape.gt_zero():
            self.solver.add(c)

        if not self.use_bitvec and constrain_min:  # bit vector is randomizable
            # The batch size should not have a big min size (avoid unnecessary computation);
            # FIXME: input constraints will make SMT solving costly.
            for i in range(len(input_tensor_shape.shape)):
                self.solver.add(input_tensor_shape.shape[i] >= min_dims[i])
        check_res = self.check_sat()
        # FIXME sometimes the constraints are too complicated to return stable result.
        SanityCheck.eq(check_res, z3.sat,
                       msg=f'Constraints not sat but {check_res}.')
        self.n_floats = nnsmith_add(
            self.n_floats, input_tensor_shape.nelement())
        self.n_inps += 1
        return input_tensor_shape

    # subclasses may override this
    def extra_constraints(self, node: AbsOpBase, ishape_indices: List[int]):
        return []

    def try_insert_node(self, node: AbsOpBase, ishape_indices: List[int]) -> bool:
        input_shapes = [self.alive_shapes[idx][1] for idx in ishape_indices]
        constraints = node.requires(input_shapes)

        if self.verbose:
            print('---> Trying to solve: ', node, constraints)
            print('---> total constraints: \n',
                  '\n'.join(sorted(map(str, set(self.solver.assertions())))))
            # self.viz('currentgraph.png')

        # make a copy
        output_shapes = node.shape_fn(copy.deepcopy(input_shapes))
        tmp_n_floats = nnsmith_add(self.n_floats, node.n_floats(input_shapes))

        for shape in output_shapes:
            for c in shape.gt_zero():
                constraints.append(c)

        self.cur_node = node
        constraints.extend(self.extra_constraints(node, ishape_indices))
        check_res = self.check_sat(
            *constraints, nnsmith_le(tmp_n_floats, self.limit_float))
        if check_res == z3.unknown:  # Timeout thing.
            self.on_timeout(node, ishape_indices)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.solver.add(c)
        self.n_floats = tmp_n_floats

        self.forward_insert_node(node, ishape_indices, output_shapes)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, occ_holder_indices: List[int]) -> bool:
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        to_occupy = [self.id2nxnode(self.placeholders[i])['op']
                     for i in occ_holder_indices]

        occupied_holder_shapes = [holder.out_shape for holder in to_occupy]

        # S2.2: try to reuse some existing outputs;
        # TODO: allow reuse existing alive shapes
        # n_inps = len(node.inp_ranks)
        # max_try = 2
        # n_reuse = n_inps - 1
        # while n_reuse > 0 and max_try > 0:
        #     # TODO...
        #     max_try -= 1
        #     n_reuse -= 1

        # S2.2: reusing outputs failed. as a fallback, promote all free vars to placeholders.
        new_inp_placeholders = []
        for dim in node.deduct_inp_ranks([s.ndims for s in occupied_holder_shapes]):
            new_inp_placeholders.append(self.create_placeholder(
                dim if dim != -1 else random.randint(0, 4)))

        input_shapes = [p.out_shape for p in new_inp_placeholders]
        constraints = node.requires(input_shapes)
        output_shapes = node.shape_fn(copy.deepcopy(input_shapes))

        for i, shape in enumerate(output_shapes):
            constraints.extend(shape.eq(occupied_holder_shapes[i]))
            constraints.extend(shape.gt_zero())

        self.cur_node = node
        # TODO: not considering extra constraints for now.
        # TODO: consider nfloats.
        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.solver.add(c)

        self.backward_insert_node(
            node, new_inp_placeholders, occ_holder_indices)

        return True

    def on_timeout(self, node: AbsOpBase, ishape_indices: List[int]):
        pass

    def get_symbol_solutions(self) -> List:
        SanityCheck.not_none(self.last_soln)
        return self.last_soln
        # res = self.solver.check()
        # assert res == z3.sat, res
        # return self.solver.model()


class GenerationTable:
    # Hyper-parameters
    _MAX_CONF = 4.0
    _BASE_VAL = 1.0
    _MIN_CONF = 0.1
    _INIT_VAL = 2.0

    def distribute_wts(self):
        wts = [1] * len(ALL_OP_TYPES)
        normalize_op_t = [Constant, Cast]
        op_t_idx = {}
        for i in range(len(ALL_OP_TYPES)):
            for op_t in normalize_op_t:
                if issubclass(ALL_OP_TYPES[i], op_t):
                    op_t_idx[op_t] = op_t_idx.get(op_t, []) + [i]

        for idx in op_t_idx.values():
            for i in idx:
                wts[i] = 1.0 / len(idx)

        for i in range(len(ALL_OP_TYPES)):
            ii = self.row_mapper(ALL_OP_TYPES[i])
            jj = self.col_mapper(ALL_OP_TYPES[i])
            self.np_table[ii] *= wts[i]
            self.np_table[jj] *= wts[i]

    def __init__(self):
        self.np_table = np.ones((len(ALL_OP_TYPES), len(
            ALL_OP_TYPES) - 1)) * self._INIT_VAL  # do not count Input
        self.distribute_wts()
        # Close those impossible connections.
        for src_t in ALL_OP_TYPES:
            for tar_t in ALL_OP_TYPES:
                if tar_t is Input:
                    continue

                inp_dims = tar_t(
                    *[None for _ in signature(tar_t).parameters]).inp_ranks
                out_dims = src_t(
                    *[None for _ in signature(src_t).parameters]).out_ranks

                if -1 in inp_dims or -1 in out_dims or set(inp_dims).intersection(out_dims):
                    continue

                self.np_table[self.row_mapper(
                    src_t)][self.col_mapper(tar_t)] = 0.

    def row_mapper(self, t):
        if isinstance(t, int):
            return t
        return ALL_OP_TYPES.index(t)

    def col_mapper(self, t):
        if isinstance(t, int):
            return t
        return ALL_OP_TYPES.index(t) - 1

    def on_new_cov(self, src_t, tar_t):
        if self.row_mapper(src_t) == 0:  # Ignore input node.
            return
        val = self.np_table[self.row_mapper(src_t)][self.col_mapper(tar_t)]
        self.np_table[self.row_mapper(src_t)][self.col_mapper(
            tar_t)] = min(self._MAX_CONF, max(self._BASE_VAL, val * 1.1))

    def on_no_cov(self, src_t, tar_t):
        if self.row_mapper(src_t) == 0:
            return
        self.np_table[self.row_mapper(
            src_t)][self.col_mapper(tar_t)] = self._BASE_VAL  # reset.

    def on_unsolvable(self, src_t, tar_t):
        if self.row_mapper(src_t) == 0:
            return
        val = self.np_table[self.row_mapper(src_t)][self.col_mapper(tar_t)]
        self.np_table[self.row_mapper(src_t)][self.col_mapper(
            tar_t)] = max(self._MIN_CONF, min(self._BASE_VAL, val * 0.9))

    def lookup(self, src_t, tar_t):
        return self.np_table[self.row_mapper(src_t)][self.col_mapper(tar_t)]

    def __len__(self):
        return len(self.np_table)

    def __getitem__(self, t):
        return self.np_table[self.row_mapper(t)]


class CoverageTableGen(PureSymbolGen):
    def __init__(self, table: GenerationTable, state, **kwargs):
        self.table = table
        SanityCheck.true('unsolvable' in state, 'unsolvable not in state')
        self.state = state
        super(CoverageTableGen, self).__init__(**kwargs)

    def pick_alive_shape(self, node_t, candidates):
        # node_t target node...
        # candidates -> outputs of input nodes...
        successor_probability = self.table.np_table.transpose()[
            self.table.col_mapper(node_t)]
        candidate_ops = [type(self.abstract_graph.nodes[self.alive_shapes[alive_shape_idx][0]]['op'])
                         for alive_shape_idx in candidates]
        candidate_indices = [self.table.row_mapper(op) for op in candidate_ops]
        candidate_conf = successor_probability[candidate_indices]
        if candidate_conf.sum() == 0:
            raise NNSmithInternalError(
                f'Candidate prob is zero -- candidates: {[op.__name__ for op in candidate_ops]} -> {node_t}')
        return np.random.choice(candidates, p=candidate_conf / candidate_conf.sum())

    def pick_next_op_type(self):
        probability_vector = self.table.np_table.sum(axis=1)
        return np.random.choice(ALL_OP_TYPES, p=probability_vector / probability_vector.sum())

    def on_timeout(self, node: AbsOpBase, ishape_indices: List[int]):
        # node -> ishape_indices :: on_unsolvable
        for idx in ishape_indices:
            self.state['unsolvable'].append(
                (type(node).__name__, type(self.id2nxnode(self.alive_shapes[idx][0])['op']).__name__))


class Bin:
    def __init__(self, lb, ub, scale='linear', base=None):
        self.lb = lb
        self.ub = ub
        assert scale in ['linear', 'log']
        self.scale = scale
        self.base = base

    def to_linear(self, x):
        if self.scale == 'log':
            x = math.pow(self.base, x)
        return int(x)

    def sample(self):
        x = random.uniform(self.lb, self.ub)
        return self.to_linear(x)

    def sample_range(self):
        if self.ub == None:  # one-sided
            return self.to_linear(self.lb), None
        lb = self.sample()
        ub = self.sample()
        if lb > ub:
            lb, ub = ub, lb
        if lb == ub:
            ub = lb + 1
        return lb, ub


class GuidedGen(PureSymbolGen):
    def __init__(self, summaries=None, scale='log', base=2, default_bins=8, **kwargs):
        super(GuidedGen, self).__init__(**kwargs)

        self.base = 2
        self.param_config = {
            'NCHWConv2d': {
                'kernel_h_size': [Bin(i, i + 1, scale=scale, base=base) for i in range(8)],
                'kernel_w_size': [Bin(i, i + 1, scale=scale, base=base) for i in range(8)],
                'stride': [Bin(i, i + 1, scale=scale, base=base) for i in range(8)],
                'padding': [Bin(i, i + 1, scale=scale, base=base) for i in range(8)] + [Bin(0, 1)],
                'in_channels': [Bin(i, i + 1, scale=scale, base=base) for i in range(8)] +
                [Bin(8, None, scale=scale, base=base)],
                'out_channels': [],  # skip
            },
        }
        self.default_config = defaultdict(
            lambda: [Bin(i, i + 1, scale=scale, base=base) for i in range(default_bins)])

    def range_constrain(self, param, lb, ub):
        ret = []
        if lb is not None:
            ret.append(nnsmith_ge(param, lb))
        if ub is not None:
            ret.append(nnsmith_lt(param, ub))
        return ret

    def extra_constraints(self, node: AbsOpBase, ishape_indices: List[int]):
        ret = []
        construct_param_dict = signature(node.__init__).parameters
        config = self.param_config.get(
            node.__class__.__name__, self.default_config)

        # if len(construct_param_dict) > 0:
        #     print('Op {} constraint:'.format(node))
        for idx, key in enumerate(construct_param_dict):
            # pc = counter['param_' + key]  # type: Counter
            param = getattr(node, key)
            # bin_id = min(pc.keys(), key=lambda k: pc)
            bins = config[key]
            if len(bins) == 0:
                continue
            bin_id = random.randint(0, len(bins) - 1)
            lb, ub = bins[bin_id].sample_range()
            # print('\t{} <= {} < {}'.format(lb, key, ub))
            ret.extend(self.range_constrain(param, lb, ub))
        return ret


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nodes', type=int, default=5)
    parser.add_argument('--min_dims', type=list, default=[1, 3, 48, 48])
    parser.add_argument('--timeout', type=int, default=50000)
    parser.add_argument('--viz_sbs', action='store_true',
                        help='visualize the step by step')
    parser.add_argument('--output_path', type=str, default='output.onnx')
    parser.add_argument('--input_gen', type=str, default='v3')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_bitvec', action='store_true')
    parser.add_argument('--viz_graph', action='store_true')
    parser.add_argument('--mode', default='random')
    return parser.parse_args()


def random_model_gen(
        min_dims=[1, 3, 48, 48],
        viz_sbs=False,
        max_nodes=5,
        seed=None,
        use_bitvec=False,
        timeout=50000,
        verbose=False,
        mode='random',
        **kwargs):
    if verbose:
        strt_time = time.time()

    GenCls = {
        'random': PureSymbolGen,
        'guided': GuidedGen,
    }[mode]
    gen = GenCls(min_dims=min_dims,
                 viz_sbs=viz_sbs, seed=seed, verbose=verbose, use_bitvec=use_bitvec, **kwargs)
    gen.abstract_gen(max_node_size=max_nodes,
                     max_gen_millisec=timeout)
    if verbose:
        print(
            f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')

    solution = gen.get_symbol_solutions()
    if verbose:
        print(
            f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
        print(solution)

    return gen, solution


def table_model_gen(
        table,
        state,
        min_dims=[1, 3, 48, 48],
        viz_sbs=False,
        max_nodes=5,
        seed=None,
        use_bitvec=False,
        timeout=50000,
        verbose=False):
    if verbose:
        strt_time = time.time()

    gen = CoverageTableGen(table=table, state=state, min_dims=min_dims,
                           viz_sbs=viz_sbs, seed=seed, verbose=verbose, use_bitvec=use_bitvec)

    gen.abstract_gen(max_node_size=max_nodes,
                     max_gen_millisec=timeout)
    if verbose:
        print(
            f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')

    solution = gen.get_symbol_solutions()
    if verbose:
        print(
            f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
        print(solution)

    return gen, solution


if __name__ == '__main__':
    args = parse_args()

    strt_time = time.time()

    seed = args.seed
    if seed is None:
        # If we have not selected a seed, choose random one.
        seed = random.getrandbits(32)
    print(f"Using seed {seed}")
    torch.manual_seed(seed)

    gen, solution = random_model_gen(min_dims=args.min_dims, seed=seed, viz_sbs=args.viz_sbs, max_nodes=args.max_nodes,
                                     use_bitvec=args.use_bitvec, timeout=args.timeout, verbose=args.verbose, mode=args.mode)

    if args.verbose or args.viz_graph:
        gen.viz(args.output_path + '.png')

    net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose,
                    alive_shapes=gen.alive_shapes)

    # turn this on so that nan in the intermediate tensors can be detected too
    input_st = time.time()

    sat_inputs = None
    if args.input_gen == 'v3':
        with torch.no_grad():
            net.eval()
            sat_inputs = net.rand_input_gen()
            infer_succ = sat_inputs is not None
    elif args.input_gen == 'grad':
        try:
            sat_inputs = net.grad_input_gen()
        except RuntimeError as e:
            if 'does not have a grad_fn' in str(e):
                # means some op are not differentiable.
                pass
            else:
                raise e

    ed_time = time.time()

    if sat_inputs is not None:
        torch2onnx(net, args.output_path, verbose=args.verbose)

    stats = {
        'gen_succ': True,
        'infer_succ': infer_succ,
        'elapsed_time': ed_time - strt_time,
        'gen_model_time': input_st - strt_time,
        'infer_domain_time': ed_time - input_st,
        'sat_inputs': sat_inputs,
        'seed': seed,
    }
    pickle.dump(stats, open(args.output_path + '-stats.pkl', 'wb'))

    net.to_picklable()
    cloudpickle.dump(net, open(args.output_path +
                     '-net.pkl', 'wb'), protocol=4)
