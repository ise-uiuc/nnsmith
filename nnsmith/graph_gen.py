from collections import Counter, defaultdict
import math
import textwrap
import z3  # Always import z3 first to avoid incompatibility issue.
# See https://github.com/Z3Prover/z3/issues/5656
import networkx as nx
from summary import ParamShapeSummary
import torch
from torch import nn
import numpy as np

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


NNSMITH_LIMNF_V = os.getenv('NNSMITH_LIMNF_V', '0')
assert NNSMITH_LIMNF_V in ['0', '1']
NNSMITH_BV_SIZE = os.getenv('NNSMITH_BV_SIZE', '8')


class RequiredDimNotFound(Exception):
    pass


ALIVE_SHAPE_TYPE = List[Tuple[int, ShapeVar, int]]


InputInfo = NamedTuple(
    'InputInfo', [('op', Input), ('oid', int), ('node_id', int), ('input_name', str)])

__MB_LIM__ = 6 * 1024


class SymbolNet(nn.Module):
    def __init__(self, graph: nx.MultiDiGraph, model: z3.ModelRef, verbose=False, alive_shapes: ALIVE_SHAPE_TYPE = None,
                 record_intermediate=False, use_gradient=False, megabyte_lim=__MB_LIM__):
        super(SymbolNet, self).__init__()
        self.megabyte_lim = megabyte_lim
        self.verbose = verbose
        self.tensors = []  # 1) edges; 2) leaf nodes; 3) input -> 0;
        self.ref_cnt = []  # ref cnt -> tensors; erased on 0;
        self.instructions = []  # <Func, <input idx>, <output idx>>
        self.n_output = 0

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
        n_floats, flops = 0, 0
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
            # ensure n_floats and flops within limit
            tmp_inp = [shape_vars[i] for i in ishape_indices]
            op.shape_fn(tmp_inp)
            n_floats += op.n_floats(tmp_inp)
            assert n_floats * 8 <= megabyte_lim * 1024 * \
                1024, f'Current number of elements ({n_floats/1024/1024}m) exceeded memory limit ({megabyte_lim} MB)'
            if FLOPS_LIM is not None:
                assert op.flops(
                    tmp_inp) < FLOPS_LIM, f'Current number of flops ({op.flops(tmp_inp)}m) exceeded limit ({FLOPS_LIM} m)'

        if self.verbose:
            print('input_info=', self.input_info)
        self.input_spec = {
            f'i{ii.op.idx}': ii.op.shape for ii in self.input_info}
        self.plausible_input_shape = {f'i{ii.op.idx}': ShapeVar(
            ii.op.shape, dtype=ii.op.dtype) for ii in self.input_info}
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
        return [torch.rand(ii.op.shape) for ii in self.input_info]

    def rand_input_gen(self, max_iter=10, use_cuda=False) -> Optional[List[torch.Tensor]]:
        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        sat_inputs = None

        n_step = max_iter
        interval = 1 / n_step
        dev = torch.device('cuda' if use_cuda else 'cpu')
        for v in np.linspace(-1, 1, n_step):
            inputs = [v + torch.rand(ii.op.shape, device=dev)
                      * interval for ii in self.input_info]

            if use_cuda:
                self = self.cuda()

            self.forward(*inputs)

            if not self.invalid_found_last:
                sat_inputs = inputs
                break

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def grad_input_gen(self, max_iter=10, init_tensors=None, use_cuda=False) -> Optional[List[torch.Tensor]]:
        dev = torch.device('cuda' if use_cuda else 'cpu')
        if init_tensors is None:
            inputs = [torch.nn.parameter.Parameter(torch.rand(ii.op.shape, device=dev))
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
                        input_tensors[1], torch.ones(size=[1], dtype=input_tensors[1].dtype, device=input_tensors[1].device))
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

    def __init__(self, min_dims=[1, 3, 48, 48], skip=[Input], viz_sbs=False, megabyte_lim=__MB_LIM__, seed=None, verbose=False, use_bitvec=False,
                 viz_verbose=False, merge_op_v=None, limnf=True):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        self.verbose = verbose
        self.viz_verbose = viz_verbose
        auto_infer_in_dtypes(self.verbose)

        self.op_candidates = [
            op for op in ALL_OP_TYPES if op not in skip and not op._skip]
        if os.getenv('NNSMITH_DEBUG_CONV_ONLY', None) is not None:  # for debugging only
            self.op_candidates = [NCHWConv2d]
        if use_bitvec:
            self.solver = z3.SolverFor("QF_UFBV")
        else:
            self.solver = z3.Solver()

        # 4 bytes per float (assume we use float64)
        self.limit_float = 1024**2 * megabyte_lim // 8

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
        self.last_soln = None
        self.wts = None
        self.merge_op_v = merge_op_v or 'v0'  # v0 as default version
        self.limnf = limnf
        self.n_floats_cons = []

    def new_sym(self, name, bv_size=None):
        if self.use_bitvec:
            bv_size = bv_size or NNSMITH_BV_SIZE
            if isinstance(bv_size, str) and bv_size.startswith('random'):
                bv_size = random.randint(1, int(bv_size[len('random'):]))
            elif isinstance(bv_size, str):
                bv_size = int(bv_size)
            zero_size = ARITH_MAX_WIDTH - bv_size
            return z3.ZeroExt(zero_size, z3.BitVec(name, bv_size))
        else:
            return z3.Int(name)

    def new_syms(self, names):
        if self.use_bitvec:
            # bv_size = random.randint(6, 8)
            bv_sizes = list(map(len, random_group(
                int(os.getenv("NNSMITH_BITS", 30)), len(names))))
            assert len(bv_sizes) == len(names)
            return [self.new_sym(name, bvsize) for name, bvsize in zip(names, bv_sizes)]
        else:
            return [self.new_sym(name) for name in names]

    @ abstractmethod
    def insert_input_node(self, min_dims, shape=None, dtype=DType.float32) -> ShapeVar:
        raise NotImplementedError

    @ abstractmethod
    def try_insert_node(self, node: AbsOpBase, ishape_indices: List[int]) -> bool:
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

    # def concretize_input_shape(self, model):
    #     shape = []
    #     for s in self.input_shape.shape:
    #         if isinstance(s, z3.ExprRef):
    #             shape.append(model.eval(s, model_completion=True).as_long())
    #         else:
    #             shape.append(s)
    #     return shape

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
            50 * 1024,  # MB
        )
        num_inputs = max(
            1, int((max_node_size + 9) // 10 + random.gauss(0, 1)))
        for _ in range(num_inputs):
            self.insert_input_node(self.min_dims)
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
        # self.fix_graph_dependency()

    def shape_idx_to_op_idx(self, shape_idx: int) -> int:
        return self.alive_shapes[shape_idx][0]

    def check_arith_ref(self, var):
        SanityCheck.true(isinstance(
            var, (z3.BitVecRef, z3.BoolRef, bool)), f"{type(var)}not supported.")
        if not isinstance(var, bool):
            for child in var.children():
                self.check_arith_ref(child)

    def check_sat(self, *assumptions):
        start = time.time()
        if self.use_bitvec:
            for assump in assumptions:
                self.check_arith_ref(assump)
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
        normalize_op_t = {'latest': EXPANDED_OP, 'v1': EXPANDED_OP_V1,
                          'v0': EXPANDED_OP_V0}[self.merge_op_v]
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

    def insert_node(self, node: AbsOpBase, ishape_indices: List[int], oshapes: List[ShapeVar] = None):
        if oshapes is None:
            input_shapes = [self.alive_shapes[idx][1]
                            for idx in ishape_indices]
            oshapes = node.shape_fn(copy.deepcopy(input_shapes))

        new_node_idx = len(self.abstract_graph.nodes)
        shape_idx_st = len(self.alive_shapes)
        shape_indices = []
        for i, shape_var in enumerate(oshapes):
            if node.out_dims[i] == -1:
                node.out_dims[i] = len(shape_var.shape)
            else:
                SanityCheck.eq(node.out_dims[i], len(shape_var.shape), "{}'s dimension size is not {} in {}".format(
                    shape_var.shape, node.out_dims[i], node.__class__.__name__))
            shape_idx = len(self.alive_shapes)
            shape_indices.append(shape_idx)
            self.alive_shapes.append((new_node_idx, shape_var, i))
            self.dim2shape_idx.setdefault(
                len(shape_var.shape), []).append(shape_idx)
        shape_idx_ed = len(self.alive_shapes)

        self.abstract_graph.add_node(
            new_node_idx, op=node, nin=len(ishape_indices), nout=len(oshapes),
            label=textwrap.fill(
                (f'#{new_node_idx}, [{shape_idx_st},{shape_idx_ed}), ' if not self.viz_verbose else '') +
                f'{node}', width=30), shape_indices=shape_indices, ishape_indices=ishape_indices)

        for in_operand_idx, idx in enumerate(ishape_indices):
            old_node_idx, svar, out_operand_idx = self.alive_shapes[idx]
            self.abstract_graph.add_edge(old_node_idx, new_node_idx, shape_idx=idx, operand_idx=(
                out_operand_idx, in_operand_idx), label=f'{out_operand_idx}-{in_operand_idx}: {svar}' if not self.viz_verbose else '')

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

    def try_insert_node_type(self, node_t, max_shape_var_pick_time=3) -> bool:
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
                        SanityCheck.eq(final_dim, dim)
            if final_dim == -1:
                final_dim = random.choice(list(self.dim2shape_idx.keys()))
            dim_spec_list = [final_dim] * n_inp
        else:  # inputs have different dimension sizes.
            dim_spec_list = op.inp_dims

        try:
            for _ in range(max_shape_var_pick_time):
                ishape_indices = self.pick_shape_var_idx(
                    node_t, dim_spec_list, op.in_dtypes)
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

    def filter_alive_shapes(self, ndim, dtype):
        cans = range(len(self.alive_shapes))

        cans = list(filter(  # filter with ndim
            lambda sid: self.alive_shapes[sid][1].ndims == ndim or ndim == -1, cans))
        if len(cans) == 0:
            raise RequiredDimNotFound(
                'Cannot find a shape variable with #dimensions %s.' % ndim)

        if dtype is not None:
            cans = list(filter(  # filter with dtype
                lambda sid: self.alive_shapes[sid][1].dtype == dtype, cans))
            if len(cans) == 0:
                raise RequiredDimNotFound(
                    'Cannot find a shape variable with #dimensions %s and dtype %s.' % (ndim, dtype))

        return cans

    def pick_alive_shape(self, node_t, candidates):
        return random.choice(candidates)

    def pick_shape_var_idx(self, node_t, ndim_list: List[int], dtype_combs: List[DTypeComb]) -> List[int]:
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
            all_can_dtypes.extend([self.alive_shapes[i][1].dtype for i in self.filter_alive_shapes(
                ndim=ndim, dtype=None)])
        # only use dtypes currently available after ndim filtering
        dtype_combs = [comb for comb in dtype_combs if all(
            i in all_can_dtypes for i in comb)]
        if len(dtype_combs) == 0:
            raise RequiredDimNotFound('Op %s: Cannot find a shape variable with dim_spec %s and dtype combinations %s.' % (
                node_t, ndim_list, dtype_combs))
        dtype_comb = random.choice(dtype_combs)
        for i, ndim in enumerate(ndim_list):
            candidates = self.filter_alive_shapes(
                ndim=ndim, dtype=dtype_comb[i])
            shape_var_candidates.append(
                self.pick_alive_shape(node_t, candidates))

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
        syms = self.new_syms(['i%s_s%s' % (self.n_inps, k)
                             for k in range(len(min_dims))])
        input_tensor_shape = ShapeVar(shape=syms, dtype=dtype)
        input_node = Input(self.n_inps, dtype, *input_tensor_shape.shape)

        self.insert_node(input_node, [], oshapes=[input_tensor_shape])
        for c in input_tensor_shape.gt_zero():
            self.solver.add(c)

        if not self.use_bitvec and constrain_min:  # bit vector is randomizable
            # The batch size should not have a big min size (avoid unnecessary computation);
            # FIXME: input constraints will make SMT solving costly.
            for i in range(len(input_tensor_shape.shape)):
                self.solver.add(nnsmith_ge(
                    input_tensor_shape.shape[i], min_dims[i]))
        check_res = self.check_sat()
        # FIXME sometimes the constraints are too complicated to return stable result.
        SanityCheck.eq(check_res, z3.sat,
                       msg=f'Constraints not sat but {check_res}.')
        if self.limnf:
            if NNSMITH_LIMNF_V == '0':
                self.n_floats = nnsmith_add(
                    self.n_floats, input_tensor_shape.nelement())
            elif NNSMITH_LIMNF_V == '1':
                self.n_floats_cons.append(nnsmith_le(
                    input_tensor_shape.nelement(), self.limit_float // 16))
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
        if self.limnf:
            if NNSMITH_LIMNF_V == '0':
                tmp_n_floats = nnsmith_add(
                    self.n_floats, node.n_floats(input_shapes))
            elif NNSMITH_LIMNF_V == '1':
                tmp_n_floats_cons = self.n_floats_cons + \
                    [nnsmith_le(node.n_floats(input_shapes),
                                self.limit_float // 16)]

        for shape in output_shapes:
            for c in shape.gt_zero():
                constraints.append(c)

        self.cur_node = node
        constraints.extend(self.extra_constraints(node, ishape_indices))
        if self.limnf:
            if NNSMITH_LIMNF_V == '0':
                check_res = self.check_sat(
                    *constraints, nnsmith_le(tmp_n_floats, self.limit_float))
            elif NNSMITH_LIMNF_V == '1':
                check_res = self.check_sat(
                    *constraints, *tmp_n_floats_cons)
        else:
            check_res = self.check_sat(*constraints)
        if check_res == z3.unknown:  # Timeout thing.
            self.on_timeout(node, ishape_indices)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.solver.add(c)
        if self.limnf:
            if NNSMITH_LIMNF_V == '0':
                self.n_floats = tmp_n_floats
            elif NNSMITH_LIMNF_V == '1':
                self.n_floats_cons = tmp_n_floats_cons

        self.insert_node(node, ishape_indices, output_shapes)
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
                    *[None for _ in signature(tar_t).parameters]).inp_dims
                out_dims = src_t(
                    *[None for _ in signature(src_t).parameters]).out_dims

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
                (type(node).__name__, type(self.abstract_graph.nodes[self.alive_shapes[idx][0]]['op']).__name__))


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
        if self.lb == None and self.ub == None:
            return None, None
        if self.ub == None:  # one-sided
            return self.to_linear(self.lb), None
        lb = self.sample()
        ub = self.sample()
        if lb > ub:
            lb, ub = ub, lb
        if lb == ub:
            ub = lb + 1
        return lb, ub


PARAM_CONFIG0 = {
    'NCHWConv2d': {
        'kernel_h_size': [Bin(i, i + 1, scale='log', base=2) for i in range(8)],
        'kernel_w_size': [Bin(i, i + 1, scale='log', base=2) for i in range(8)],
        'stride': [Bin(i, i + 1, scale='log', base=2) for i in range(8)],
        'padding': [Bin(i, i + 1, scale='log', base=2) for i in range(8)] + [Bin(0, 1)],
        # 'in_channels': [Bin(i, i + 1, scale='log', base=2) for i in range(8)] +
        # [Bin(8, None, scale='log', base=2)],
        'in_channels': [],
        'out_channels': [],  # skip
    },
}

PARAM_CONFIG1 = {
    'NCHWConv2d': {
        'kernel_h_size': [Bin(i, i + 1, scale='log', base=2) for i in range(8)],
        'kernel_w_size': [Bin(i, i + 1, scale='log', base=2) for i in range(8)],
        'stride': [Bin(i, i + 1, scale='log', base=2) for i in range(8)],
        'padding': [Bin(i, i + 1, scale='log', base=2) for i in range(8)] + [Bin(0, 1)],
        'out_channels': [Bin(i, i + 1, scale='log', base=2) for i in range(8)] +
        [Bin(8, None, scale='log', base=2)],
        'in_channels': [],  # skip
    },
    # last bin is eseentially no constraint, to ensure -1 can be included
    'Reshape': defaultdict(lambda: [Bin(i, i + 1, scale='log', base=2) for i in range(8)] + [Bin(None, None)]),
}
PARAM_CONFIG1['Reshape1D'] = PARAM_CONFIG1['Reshape']
PARAM_CONFIG1['Reshape2D'] = PARAM_CONFIG1['Reshape']
PARAM_CONFIG1['Reshape3D'] = PARAM_CONFIG1['Reshape']
PARAM_CONFIG1['Reshape4D'] = PARAM_CONFIG1['Reshape']
PARAM_CONFIG1['Reshape5D'] = PARAM_CONFIG1['Reshape']
PARAM_CONFIG1['Reshape6D'] = PARAM_CONFIG1['Reshape']
PARAM_CONFIG2 = {
    'NCHWConv2d': {
        'kernel_h_size': [Bin(1, 256, scale='linear')],
        'kernel_w_size': [Bin(1, 256, scale='linear')],
        'stride': [Bin(1, 256, scale='linear')],
        'padding': [Bin(1, 256, scale='linear')] + [Bin(0, 1)],
        'out_channels': [Bin(1, 256, scale='linear')] +
        [Bin(256, None, scale='linear')],
        'in_channels': [],  # skip
    },
}


def __GROUP_RESHAPE(node, ishape_indices, construct_param_dict, bin=True):
    bins = [Bin(i, i + 1, scale='log', base=2)
            for i in range(8)] + [Bin(None, None)]
    ret = []

    src_group = node.src_group
    dst_group = node.dst_group
    ng = node.ng
    assert len(src_group) == len(dst_group) == ng, (src_group, dst_group)

    construct_params = list(construct_param_dict.keys())
    if bin:
        for gid in range(ng):
            ds = dst_group[gid]
            disable = list(range(len(ds)))
            random.shuffle(disable)
            disable = disable[:1]
            for idx, d in enumerate(ds):
                if idx in disable:
                    continue
                key = construct_params[d]
                param = getattr(node, key)
                if len(bins) == 0:
                    continue
                bin_id = random.randint(0, len(bins) - 1)
                lb, ub = bins[bin_id].sample_range()
                ret.extend(range_constrain(param, lb, ub))

    return ret


PARAM_CONFIG3 = copy.deepcopy(PARAM_CONFIG1)
PARAM_CONFIG3['Reshape'] = __GROUP_RESHAPE
PARAM_CONFIG3['Reshape1D'] = __GROUP_RESHAPE
PARAM_CONFIG3['Reshape2D'] = __GROUP_RESHAPE
PARAM_CONFIG3['Reshape3D'] = __GROUP_RESHAPE
PARAM_CONFIG3['Reshape4D'] = __GROUP_RESHAPE
PARAM_CONFIG3['Reshape5D'] = __GROUP_RESHAPE
PARAM_CONFIG3['Reshape6D'] = __GROUP_RESHAPE

PARAM_CONFIG4 = copy.deepcopy(PARAM_CONFIG3)

PARAM_CONFIG5 = copy.deepcopy(PARAM_CONFIG3)


def range_constrain(param, lb, ub):
    ret = []
    if lb is not None:
        ret.append(nnsmith_ge(param, lb))
    if ub is not None:
        ret.append(nnsmith_lt(param, ub))
    return ret


class GuidedGen(PureSymbolGen):
    def __init__(self, summaries=None, scale='log', base=2, default_bins=8, constrain_prob=1, **kwargs):
        super(GuidedGen, self).__init__(**kwargs)

        self.constrain_prob = constrain_prob
        self.base = 2
        self.param_config = [PARAM_CONFIG0, PARAM_CONFIG1,
                             PARAM_CONFIG2, PARAM_CONFIG3, PARAM_CONFIG4, PARAM_CONFIG5][int(os.getenv('NNSMITH_G_CONFIG', 1))]
        if scale == 'log':
            self.default_config = defaultdict(
                lambda: [Bin(i, i + 1, scale=scale, base=base) for i in range(default_bins)] +
                [Bin(default_bins, None, scale=scale, base=base)])
        else:
            assert scale == 'linear', scale
            self.default_config = defaultdict(
                lambda: [Bin(0, 256, scale='linear')] + [Bin(256, None, scale='linear')])
        self.scale = scale
        # self.inp

    def insert_input_node(self, min_dims, dtype=DType.float32) -> ShapeVar:
        ish = super().insert_input_node(min_dims, dtype, constrain_min=False)
        constraints = []
        for i in ish.shape:
            bins = self.default_config[0]
            lb, ub = bins[random.randint(0, len(bins) - 1)].sample_range()
            constraints.extend(range_constrain(i, lb, ub))
        # throw exception for now since this is unlikely to happen
        assert self.check_sat(
            *constraints, nnsmith_le(self.n_floats, self.limit_float)) == z3.sat, 'Input constraints too tight'
        self.solver.add(*constraints)
        return ish

    def extra_constraints(self, node: AbsOpBase, ishape_indices: List[int]):
        if random.uniform(0, 1) > self.constrain_prob:
            return []
        ret = []
        construct_param_dict = signature(node.__init__).parameters
        config = self.param_config.get(
            node.__class__.__name__, None)
        if config is None:
            return ret
        if callable(config):
            return config(node, ishape_indices, construct_param_dict)

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
            ret.extend(range_constrain(param, lb, ub))
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
    parser.add_argument('--merge_op_v', default=None)
    parser.add_argument(
        '--skip', help='Node types to skip. Split by `,`. By default a blacklist for each backend is also appended.', type=str)
    parser.set_defaults(limnf=True)
    parser.add_argument('--no_limnf', dest='limnf', action='store_false',
                        help='Disable the limit on the number of floats')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--no_export', action='store_true')
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
        merge_op_v=None,
        limnf=True,):

    GenCls = {
        'random': PureSymbolGen,
        'guided': GuidedGen,
    }[mode]
    gen = GenCls(min_dims=min_dims,
                 viz_sbs=viz_sbs, seed=seed, verbose=verbose, use_bitvec=use_bitvec, merge_op_v=merge_op_v, limnf=limnf)
    gen.abstract_gen(max_node_size=max_nodes,
                     max_gen_millisec=timeout)
    solution = gen.get_symbol_solutions()

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
        verbose=False,
        merge_op_v=None,
        limnf=True,):
    if verbose:
        strt_time = time.time()

    gen = CoverageTableGen(table=table, state=state, min_dims=min_dims,
                           viz_sbs=viz_sbs, seed=seed, verbose=verbose, use_bitvec=use_bitvec, merge_op_v=merge_op_v, limnf=limnf)

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
    if args.skip is not None:
        config_skip_op(args.skip)

    strt_time = time.time()
    gen, solution = random_model_gen(min_dims=args.min_dims, seed=seed, viz_sbs=args.viz_sbs, max_nodes=args.max_nodes,
                                     use_bitvec=args.use_bitvec, timeout=args.timeout, verbose=args.verbose, mode=args.mode,
                                     limnf=args.limnf, merge_op_v=args.merge_op_v)
    print(
        f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
    print(
        f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')
    srt_time = time.time()
    if args.verbose or args.viz_graph:
        gen.viz(args.output_path + '.png')

    if args.no_export:
        exit(0)
    net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose,
                    alive_shapes=gen.alive_shapes)
    print('Initializing SymbolNet time: {}s'.format(time.time() - srt_time))
    # turn this on so that nan in the intermediate tensors can be detected too
    input_st = time.time()

    sat_inputs = None
    if args.input_gen == 'v3':
        with torch.no_grad():
            net.eval()
            sat_inputs = net.rand_input_gen(use_cuda=args.use_cuda)
            infer_succ = sat_inputs is not None
    elif args.input_gen == 'grad':
        infer_succ = None  # TODO: are we able to know this?
        try:
            sat_inputs = net.grad_input_gen(use_cuda=args.use_cuda)
        except RuntimeError as e:
            if 'does not have a grad_fn' in str(e):
                # means some op are not differentiable.
                pass
            else:
                raise e
    elif args.input_gen == 'none':
        infer_succ = None
    else:
        raise ValueError(f'Unknown input gen {args.input_gen}')

    ed_time = time.time()
    print('Time to generate inputs: {:.3f}s'.format(ed_time - input_st))

    torch2onnx(net, args.output_path, verbose=args.verbose,
               use_cuda=args.use_cuda)

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
