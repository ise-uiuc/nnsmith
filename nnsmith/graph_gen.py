# See https://github.com/Z3Prover/z3/issues/5656
import z3  # Always import z3 first to avoid incompatibility issue.

from multiprocessing import Process
import psutil
from collections import defaultdict
import math
import textwrap

import networkx as nx
from nnsmith.dtype_test import rewrite_op_dtype
import torch
from torch import nn
import numpy as np
import uuid

import pickle
import cloudpickle
from typing import Dict, NamedTuple, Tuple, List, Optional, Set
from inspect import signature
import traceback
import random
import time
import os
import copy

from nnsmith.error import SanityCheck, ConstraintCheck, ConstraintError
from nnsmith.export import torch2onnx
from nnsmith.abstract.op import *
from nnsmith.abstract.op import __MAX_RANK__ as __MAX_RANK__


NNSMITH_LIMNF_V = os.getenv('NNSMITH_LIMNF_V', '0')
assert NNSMITH_LIMNF_V in ['0', '1']
NNSMITH_BV_SIZE = os.getenv('NNSMITH_BV_SIZE', '8')


class RequiredDimNotFound(Exception):
    pass


ALIVE_SHAPE_TYPE = List[Tuple[int, ShapeVar, int]]


InputInfoBase = NamedTuple(
    'InputInfo', [('op', Input), ('oid', int), ('node_id', int), ('input_name', str)])


class InputInfo(InputInfoBase):
    def __repr__(self) -> str:
        return f"InputInfo(op={self.op}<{self.op.shape_var.dtype.value}>, oid={self.oid}, node_id={self.node_id}, input_name={self.input_name})"


__MB_LIM__ = 6 * 1024
# weight of gradients for valid op losses


def random_tensor(shape, dtype, margin=10, base=1, use_cuda=False):
    # center: -margin ~ 0 ~ +margin
    dev = torch.device('cuda' if use_cuda else 'cpu')
    if base == 'center':
        base = -margin / 2
    else:
        assert isinstance(base, int) or isinstance(base, float)

    fp_tensor = base + torch.rand(shape, device=dev) * margin
    if DType.is_float(dtype):
        return fp_tensor.to(dtype)
    else:
        return torch.round(fp_tensor).to(dtype)


class SymbolNet(nn.Module):
    def __init__(self, graph: nx.MultiDiGraph, model: z3.ModelRef, verbose=False, alive_shapes: ALIVE_SHAPE_TYPE = None,
                 record_intermediate=False, use_gradient=False, megabyte_lim=__MB_LIM__, print_grad=0):
        super(SymbolNet, self).__init__()
        self.megabyte_lim = megabyte_lim
        self.verbose = verbose
        self.print_grad = print_grad
        self.tensors = []  # 1) edges; 2) leaf nodes; 3) input -> 0;
        self.ref_cnt = []  # ref cnt -> tensors; erased on 0;
        self.instructions = []  # <Func, <input idx>, <output idx>>
        self.n_output = 0
        self.inp_id_cnt = 0
        self.n_vulnerable_op = 0

        self.using_cuda = None  # not sure

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

                if hasattr(op, 'torch_loss'):
                    self.n_vulnerable_op += 1
            else:  # Should be input node
                SanityCheck.true(type(op) is Input, 'type(op) should be Input')
                SanityCheck.eq(len(output_idx), 1)
                op.idx = self.inp_id_cnt
                self.inp_id_cnt += 1
                self.input_info.append(
                    InputInfo(op=op, oid=output_idx[0], node_id=node_id, input_name=f'i{op.idx}'))

            # concretize shapevars
            if self.alive_shapes is not None:
                ishape_indices = self.graph.nodes[node_id]['ishape_indices']
                shape_indices = self.graph.nodes[node_id]['shape_indices']
                for shape_idx in shape_indices:
                    shape = self.alive_shapes[shape_idx][1].shape
                    dtype = self.alive_shapes[shape_idx][1].dtype
                    shape = [model.eval(i).as_long() if isinstance(
                        i, z3.ExprRef) else i for i in shape]
                    assert shape_idx not in shape_vars, f"{shape_idx} already exists"
                    shape_vars[shape_idx] = ShapeVar(shape, dtype)
                self.concrete_graph.nodes[node_id]['in_svs'] = [
                    shape_vars[i] for i in ishape_indices]
                self.concrete_graph.nodes[node_id]['out_svs'] = [
                    shape_vars[i] for i in shape_indices]
                # ensure n_floats and flops within limit
                tmp_inp = [shape_vars[i] for i in ishape_indices]
                op.shape_fn(tmp_inp)
                op_nfl = op.n_floats(tmp_inp)
                if self.verbose:
                    print(f"op: {op} nfloats: {op_nfl}")
                n_floats += op_nfl
                assert n_floats * 8 <= megabyte_lim * 1024 * \
                    1024, f'Current number of elements ({n_floats/1024/1024}m) exceeded memory limit ({megabyte_lim} MB) Current op: {op}'
                if FLOPS_LIM is not None:
                    assert op.flops(
                        tmp_inp) < FLOPS_LIM, f'Current number of flops ({op.flops(tmp_inp)}m) exceeded limit ({FLOPS_LIM} m). Current op: {op}'

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

    def get_params(self):
        return sum([i['params'] for i in self.optimizer.param_groups], [])

    def _zero_grad(self):
        for p in self.get_params():
            p.grad = None

    def backward(self):
        if self.loss is not None:
            self._zero_grad()
            nonzero = False
            params = self.get_params()
            cur_losses = len([n for n, _ in self.loss if n.endswith('_+')])
            assert cur_losses == 1, f'cur_losses ({cur_losses}) != 1 loss functions found'
            for loss_name, l in self.loss:
                if loss_name.endswith('_-'):
                    continue
                l.backward()
                if self.print_grad >= 2:
                    for name, i in self.interm_grad:
                        msg = f'{i.grad.min()} ~ {i.grad.max()}' if i.grad is not None else 'None'
                        print(
                            f'Iter {self.iter_num} [{loss_name}] {name} grad: {msg}')
                with torch.no_grad():
                    for i, p in enumerate(params):
                        if p.grad is not None and torch.any(p.grad != 0):
                            nonzero = True
            ConstraintCheck.true(nonzero,
                                 'Gradients are all zero. Cannot make progress.')
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
        self.optimizer = torch.optim.Adam(to_train, lr=5e-1)
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

    def get_random_inps(self, **kwargs) -> List[torch.Tensor]:
        # center: -margin ~ 0 ~ +margin
        inputs = []
        for ii in self.input_info:
            inputs.append(random_tensor(ii.op.shape_var.shape,
                          ii.op.shape_var.dtype.value, **kwargs))

        return inputs

    def rand_input_gen(self, max_iter=10, use_cuda=False, **kwargs) -> Optional[List[torch.Tensor]]:
        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        sat_inputs = None

        for _ in range(max_iter):
            inputs = self.get_random_inps(**kwargs, use_cuda=use_cuda)

            if use_cuda:
                self.use_cuda()

            self.forward(*inputs)

            if not self.invalid_found_last:
                sat_inputs = inputs
                break

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def grad_input_gen(self, max_iter=int(os.getenv('NNSMITH_GRAD_ITER', 100)),
                       init_tensors=None, use_cuda=False,
                       max_time=int(os.getenv('NNSMITH_GRAD_TIME', 10)), **kwargs) -> Optional[List[torch.Tensor]]:
        # TODO: trim the param. max_iter is not used; remove getenv
        if init_tensors is None:
            init_tensors = self.get_random_inps(
                **kwargs, use_cuda=use_cuda)

        inputs = []
        for tensor in init_tensors:
            if tensor.data.dtype.is_floating_point:
                inputs.append(torch.nn.parameter.Parameter(tensor.data))
            else:
                inputs.append(tensor.data)

        self.enable_training(extra_trainable=inputs)

        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        if use_cuda:
            inputs = [inp.cuda() for inp in inputs]
            self.use_cuda()

        sat_inputs = None
        st = time.time()
        self.iter_num = 0
        while time.time() - st < max_time:
            self.training_reset()
            self.iter_num += 1

            try:
                _ = self(*inputs)
                if self.invalid_found_last:  # need_to_train
                    self.backward()
                else:
                    sat_inputs = [v.data for v in inputs]
                    break
            except ConstraintError as e:
                if "[NaN] in model inputs!" in str(e) or "[Inf] in model inputs!" in str(e):
                    # flush NaN/Inf in inputs
                    with torch.no_grad():
                        for inp in inputs:
                            inp.copy_(torch.where(
                                inp.isnan(), torch.rand(*inp.shape).to(inp.dtype).to(inp.device), inp))
                    continue
                print(e)
                break

        self.stop_training()
        if sat_inputs is None:
            print('[grad] no valid range found!!!')

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def use_cuda(self):
        self.cuda()
        self.using_cuda = True

    def forward(self, *args, **kwargs):
        # required: input_info, tensors, ref_cnt, instructions, hacked, first_run verbose # alive_shapes, graph
        self.interm_grad = []
        xs = [None] * len(self.input_info)
        for i in range(len(args)):
            xs[i] = args[i]
        for ii in self.input_info:
            if ii.input_name in kwargs:
                xs[ii.op.idx] = kwargs[ii.input_name]
        assert all(x is not None for x in xs), xs
        ConstraintCheck.true(not any(
            [torch.isinf(x).any() for x in xs]), f'[Inf] in model inputs!')
        ConstraintCheck.true(not any(
            [torch.isnan(x).any() for x in xs]), f'[NaN] in model inputs!')
        local_ref_cnt = self.ref_cnt.copy()
        self.tensors = [None for _ in self.tensors]
        self.invalid_found_last = False
        for i in range(len(xs)):
            if xs[i].requires_grad:
                self.interm_grad.append((f'i_{i}', xs[i]))

        for ii in self.input_info:
            self.tensors[ii.oid] = xs[ii.op.idx]

        for inst, inps, outs, op, node_id in self.instructions:
            input_tensors = [self.tensors[idx] for idx in inps]
            if self.using_cuda:
                input_tensors = [inp.cuda() for inp in input_tensors]
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
                    f'--> executing op={op}, node_id={node_id}, inps={inps}, outs={outs}')
                print('\tinputs=')
                for i in input_tensors:
                    print(f'  (shape={i.shape} dtype={i.dtype})')
            outputs = inst(*input_tensors)
            if not isinstance(outputs, list):
                outputs = [outputs]
            if self.print_grad >= 2:
                if outputs[0].requires_grad:
                    for i in range(len(outputs)):
                        outputs[i].retain_grad()
                        self.interm_grad.append(
                            (f'#{node_id}_{op}_{i}', outputs[i]))

            self._check_out_dtype(outputs, node_id, op)

            if self.check_intermediate_numeric or (self.use_gradient and not self.stop_updating_loss):
                if hasattr(op, 'torch_loss'):
                    vul_op_loss = (op.torch_loss(*input_tensors),)
                else:
                    vul_op_loss = ()
                self.invalid_found_last |= not op.numeric_valid(outputs)
                if self.invalid_found_last and (self.use_gradient and not self.stop_updating_loss):
                    if self.print_grad >= 1:
                        for inp_i, inp in enumerate(input_tensors):
                            print(
                                f'[inp]@{inp_i} :: {inp.min().data:.5f} ~ {inp.max().data:.5f}')

                    ConstraintCheck.true(hasattr(
                        op, 'torch_loss'), f'op={op} has no `torch_loss` but produces NaN or INF!')
                    # TODO: some less vulnerable ops (like Mul) may also trigger Inf and will crash the process.
                    # Given its low chance of happening, ignore it for now.
                    msg = ', '.join(
                        [f'loss_{idx}: {l.min().data:.3f} ~ {l.max().data:.3f}' for idx, l in enumerate(vul_op_loss)])
                    if self.print_grad >= 1:
                        print(
                            f'Iter #{self.iter_num} [NaN/Inf] in outputs ~ {op} ~ id {node_id} :: {msg}')

                    new_losses = []
                    for idx, l in enumerate(vul_op_loss):
                        if (l > 0).sum() > 0:
                            new_losses.append(
                                (f'{op}_{idx}_+', torch.sum((l > 0) * l)))
                        if (l <= 0).sum() > 0:
                            new_losses.append(
                                (f'{op}_{idx}_-', torch.sum((l <= 0) * l)))
                    if self.loss is None:
                        self.loss = new_losses
                    else:
                        self.loss.extend(new_losses)
                    self.stop_updating_loss = True
                    return outputs

            if self.verbose:
                print('\toutputs=', ','.join(
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
                 viz_verbose=False, merge_op_v=None, limnf=True, candidates_overwrite=None, forward_prob=None, init_fp=False):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        self.verbose = verbose
        self.viz_verbose = viz_verbose
        auto_infer_in_dtypes(self.verbose)

        if candidates_overwrite is None:
            self.op_candidates = [
                op for op in ALL_OP_TYPES if op not in skip and not op._skip]
        else:
            self.op_candidates = [
                op for op in candidates_overwrite if op not in skip and not op._skip]

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
        self.min_dims = min_dims
        self.n_floats = 0
        self.monotonic_placeholder_id = 0
        self.monotonic_nx_node_idx = 0
        # self.reusable_placeholder_nx_indices = []
        self.last_soln = None
        self.wts = None
        self.merge_op_v = merge_op_v or 'v0'  # v0 as default version
        self.limnf = limnf
        self.n_floats_cons = []

        # <op idx>
        self.placeholders: List[int] = []
        # for all (including newly created tmp) placeholders
        self.insert_init_ph_node(self.create_placeholder(
            len(min_dims), dtype=torch.float32 if init_fp else None))
        self.init_ph_alive = True
        self.forward_prob = 0.5 if forward_prob is None else forward_prob

    def random_rank(self):
        return random.choices(range(__MAX_RANK__ + 1),
                              weights=[1, 1, 1, 1, 2, 1])[0]

    def random_dtype(self):
        wts = [1] * len(DTYPE_ALL)
        for i in DTYPE_FLOATS:
            wts[DTYPE_ALL.index(i)] = 8
        for i in DTYPE_INTS:
            wts[DTYPE_ALL.index(i)] = 2
        return random.choices(DTYPE_ALL, weights=wts)[0]

    def create_placeholder(self, dim, dtype=None):
        syms = self.new_syms(['v%s_%s' % (
            self.monotonic_placeholder_id, k) for k in range(dim)])
        shapevar = ShapeVar(
            shape=syms,
            dtype=dtype if dtype is not None else self.random_dtype())
        self.monotonic_placeholder_id += 1
        ph = Placeholder(shapevar)
        return ph

    # default to no input constraints
    def gen_ph_cons(self, ph: Placeholder) -> List[z3.ExprRef]:
        return []

    def post_process(self):
        '''Called after the graph is finalized. May be used to add parameter guidance.'''
        pass

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
            bv_sizes = list(map(len, random_group(
                int(os.getenv("NNSMITH_BITS", 30)), len(names))))
            assert len(bv_sizes) == len(names)
            return [self.new_sym(name, bvsize) for name, bvsize in zip(names, bv_sizes)]
        else:
            return [self.new_sym(name) for name in names]

    @ abstractmethod
    def insert_init_ph_node(self, min_dims, shape=None, dtype=DType.float32) -> ShapeVar:
        raise NotImplementedError

    @ abstractmethod
    def try_forward_insert_at(self, node: AbsOpBase, ishape_indices: List[int]) -> bool:
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

    def num_op(self) -> int:
        # exclude placeholders.
        return len(self.abstract_graph.nodes) - len(self.placeholders)

    def recompute_n_floats(self):
        self.n_floats = 0
        for i in self.alive_shapes:
            self.n_floats = nnsmith_add(self.n_floats, i[1].nelement())

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        self.max_gen_millisec = max_gen_millisec
        self.cur_phase = 'abstract_gen'
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
        init_time = time.time()
        while time.time() - init_time < max_gen_millisec / 1000 and self.num_op() < max_node_size:
            if self.extra_exit_check():
                break
            node_t = self.pick_next_op_type()
            self.try_insert_node_type(node_t)
        if abs(self.num_op() - max_node_size) >= 3:
            print(
                f'[WARNING]: graph size: {len(self.abstract_graph.nodes)} < expected size: {max_node_size}')
        self.cur_phase = 'post_process'

        self.recompute_n_floats()
        if self.limnf:  # add into solver since graph is finalized to avoid repeated solving
            if NNSMITH_LIMNF_V == '0':
                self.solver.add(nnsmith_le(self.n_floats, self.limit_float))
            elif NNSMITH_LIMNF_V == '1':
                self.solver.add(*self.n_floats_cons)
            assert self.check_sat() == z3.sat

        self.post_process()  # can be used to add more constraints
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

    def check_arith_ref(self, var):
        SanityCheck.true(isinstance(
            var, (z3.BitVecRef, z3.BoolRef, bool)), f"{type(var)}not supported.")
        if not isinstance(var, bool):
            for child in var.children():
                self.check_arith_ref(child)

    def check_sat(self, *assumptions, timeout=None):
        start = time.time()
        if self.use_bitvec:
            for assump in assumptions:
                self.check_arith_ref(assump)

        if self.verbose:
            print('---> total constraints: \n',
                  '\n'.join(sorted(map(str,
                                       list(self.solver.assertions()) + list(assumptions)))))
        if timeout is None:
            cres = self.solver.check(*assumptions)
        else:
            def _run():
                cres = self.solver.check(*assumptions)
                exit({'sat': 0, 'unsat': 1, 'unknown': 2}[str(cres)])

            p = Process(target=_run)
            p.start()
            p.join(timeout=timeout / 1000)
            if p.is_alive():
                cres = z3.unknown
                for child in psutil.Process(p.pid).children(recursive=False):
                    child: psutil.Process  # type: ignore
                    try:
                        child.terminate()
                        child.wait()
                    except psutil.NoSuchProcess:
                        pass
                p.terminate()
                p.join()

            else:
                cres = {0: z3.sat, 1: z3.unsat, 2: z3.unknown}[p.exitcode]

        checking_time = int((time.time() - start) * 1000)
        if checking_time > 3000 and self.cur_node:  # 3s
            warnings.warn(
                f'[WARNING] check {self.cur_node} {checking_time} ms at phase {self.cur_phase}')
        if self.verbose:
            print(cres, '<-- checking time:', checking_time, 'ms')

            if cres == z3.unsat:
                print(f'Unsat core: {self.solver.unsat_core()}')
        if cres == z3.sat:
            if timeout is not None:
                self.solver.check(*assumptions)
            self.last_soln = self.solver.model()
        return cres

    def compute_wts(self):
        self.wts = [1] * len(self.op_candidates)
        normalize_op_t = {'latest': EXPANDED_OP,
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

    def forward_insert_node(self, node: AbsOpBase, ishape_indices: List[int], oshapes: List[ShapeVar] = None, force_shape_indices=None) -> int:
        if oshapes is None:
            input_shapes = [self.alive_shapes[idx][1]
                            for idx in ishape_indices]
            oshapes = node.shape_fn(input_shapes)

        succ_nid = self.get_new_node_id()
        if isinstance(node, Placeholder):
            self.placeholders.append(succ_nid)

        shape_indices = []
        if force_shape_indices is None:
            for i, shape_var in enumerate(oshapes):
                SanityCheck.true(len(shape_var.shape) in node.out_ranks[i], "{}'s dimension size is not {} in {}".format(
                    shape_var.shape, node.out_ranks[i], node.__class__.__name__))
                node.out_ranks[i] = (len(shape_var.shape),)
                shape_idx = len(self.alive_shapes)
                shape_indices.append(shape_idx)
                self.alive_shapes.append((succ_nid, shape_var, i))
                self.dim2shape_idx.setdefault(
                    len(shape_var.shape), []).append(shape_idx)
        else:
            # When taking the position of placeholders, we do not need to add new alive shapes.
            shape_indices = force_shape_indices

        # NOTE: because of backward insertion, we may not be able to limit the symbol size as there will be some
        # trivially equivalent symbols which harms the readability. (e.g., relations like `a = b` is not known).
        # NOTE: `shape_indices` and `ishape_indices` are indices of alive_shapes
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
            self.abstract_graph.add_edge(
                pred_nid, succ_nid, key=str(uuid.uuid1()),
                shape_idx=idx,
                operand_idx=(out_operand_idx, in_operand_idx),
                label=f'{idx}: ({out_operand_idx},{in_operand_idx}) <{svar.dtype}>{svar.shape}' if not self.viz_verbose else '')

        if self.is_viz_sbs:
            self.viz()

        return succ_nid

    def get_new_node_id(self):
        # if self.reusable_placeholder_nx_indices:
        #     return self.reusable_placeholder_nx_indices.pop()
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
        to_occ_alive_shape_idx = [self.id2nxnode(
            nx_nid)['shape_indices'][0] for nx_nid in occ_holder_idx_nx]
        op_nx_idx = self.forward_insert_node(
            node,
            ishape_indices=ishape_indices,
            oshapes=[self.alive_shapes[as_idx][1]
                     for as_idx in to_occ_alive_shape_idx],
            force_shape_indices=to_occ_alive_shape_idx)

        # Insert edges and remove placeholders
        for i, nx_idx in enumerate(occ_holder_idx_nx):
            for (src, dst, key) in list(self.abstract_graph.edges(nx_idx, keys=True)):
                # multi-graph
                edge_info = copy.deepcopy(
                    self.abstract_graph.get_edge_data(src, dst, key=key))
                old_edge_idx = edge_info['shape_idx']
                # recall alive shape:
                # 1. op nx idx
                # 2. shape var
                # 3. out operand idx

                _, svar, _ = self.alive_shapes[old_edge_idx]
                out_operand_idx = i
                in_operand_idx = edge_info['operand_idx'][1]

                # add cur node -> dst
                self.abstract_graph.add_edge(
                    op_nx_idx,
                    dst,
                    key=str(uuid.uuid1()),
                    shape_idx=edge_info['shape_idx'],  # reuse old alive shape
                    operand_idx=(out_operand_idx, in_operand_idx),
                    label=f'{old_edge_idx}: ({out_operand_idx},{in_operand_idx}) <{svar.dtype}>{svar.shape}' if not self.viz_verbose else ''
                )
                self.alive_shapes[old_edge_idx] = (
                    op_nx_idx, svar, out_operand_idx)

                self.abstract_graph.remove_edge(src, dst, key=key)

            # if the PH to occupy has no consumers, we simply reassign its alive shape.
            # NOTE: we assume the first node is a placeholder.
            if self.init_ph_alive:  # update alive_shape[0]
                self.alive_shapes[0] = (op_nx_idx, self.alive_shapes[0][1], 0)
                self.init_ph_alive = False

            # remove placeholders
            self.abstract_graph.remove_node(nx_idx)
            # self.reusable_placeholder_nx_indices.append(nx_idx)
            self.placeholders.remove(nx_idx)

        if self.is_viz_sbs:
            self.viz()

    def try_forward_insert(self, op: AbsOpBase):
        n_inp = len(op.inp_ranks)
        dim_spec_list = []

        if op.same_inp_dims:  # find `n_inp` under the same input shapes.
            rank_set = set(op.inp_ranks[0])

            for ranks in op.inp_ranks[1:]:
                rank_set.intersection_update(set(ranks))

            SanityCheck.ge(len(rank_set), 1)

            final_dim = random.choice(list(rank_set))
            dim_spec_list = [(final_dim,)] * n_inp
        else:  # inputs have different dimension sizes.
            dim_spec_list = op.inp_ranks

        ishape_indices = self.pick_shape_var_idx(
            type(op), dim_spec_list, op.in_dtypes, candidate_shapes=[s[1] for s in self.alive_shapes])

        if self.try_forward_insert_at(op, ishape_indices):
            return True

        return False

    def try_backward_insert(self, op: AbsOpBase):
        # we know that: Y = op(X)
        # S1 - select Y: Y must be a placeholder; (this also means the graph must start w/ a placeholder)
        ph_candidates = []
        for idx in self.placeholders:
            oshape = self.id2nxnode(idx)['op'].out_shape
            if isinstance(op, Expand) and oshape.ndims < op.expand_last_dim:
                continue
            ph_candidates.append(oshape)

        placeholder_indices = self.pick_shape_var_idx(
            type(op), op.out_ranks, op.out_dtypes, candidate_shapes=ph_candidates)

        if self.try_occupy_placeholder(op, placeholder_indices):
            return True

        return False

    def try_insert_node_type(self, node_t, max_shape_var_pick_time=3) -> bool:
        if self.verbose:
            print(f'Inserting node #{len(self.abstract_graph.nodes)}: '
                  f'trying to insert node type {node_t.__name__}')

        try:
            for _ in range(max_shape_var_pick_time):
                # should recreate a new instance since some attributes (like axis) should be initialized for each pick
                op_param_n = node_t.get_num_var_param()
                op_id = len(self.abstract_graph.nodes)
                op_params = [self.new_sym('op%s_%s' % (op_id, k))
                             for k in range(op_param_n)]

                op: AbsOpBase = node_t(*op_params)
                op._param_list = op_params

                if random.uniform(0, 1) < self.forward_prob:
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

    def filter_shapes(self, ndims, dtype, candidate_shapes: List[ShapeVar]):
        cans = range(len(candidate_shapes))

        cans = list(filter(  # filter with ndim
            lambda sid: candidate_shapes[sid].ndims in ndims, cans))
        if len(cans) == 0:
            raise RequiredDimNotFound(
                f'Cannot find a shape variable with #dimensions {ndims}.')

        if dtype is not None:
            cans = list(filter(  # filter with dtype
                lambda sid: candidate_shapes[sid].dtype == dtype, cans))
            if len(cans) == 0:
                raise RequiredDimNotFound(
                    f'Cannot find a shape variable with #dimensions {ndims} and dtype {dtype}.')

        return cans

    def pick_shape(self, node_t, candidates):
        return random.choice(candidates)

    def pick_shape_var_idx(self, node_t, ndim_list: List[Set[int]], dtype_combs_spec: List[DTypeComb], candidate_shapes: List[ShapeVar]) -> List[int]:
        """Randomly pick indices to shape variables from the output pool.

        Args:
            ndim_list (List[Set]): duable dims for each input.

        Returns:
            List[int]: indices to applicable shape variables.
        """

        shape_var_candidates = []
        if self.verbose:
            print('dtype_combs_spec:', dtype_combs_spec)

        all_can_dtypes = []
        for i, ndims in enumerate(ndim_list):
            all_can_dtypes.extend([candidate_shapes[i].dtype for i in self.filter_shapes(
                ndims=ndims, dtype=None, candidate_shapes=candidate_shapes)])
        # only use dtypes currently available after ndim filtering
        dtype_combs = [comb for comb in dtype_combs_spec if all(
            i in all_can_dtypes for i in comb)]
        if len(dtype_combs) == 0:
            raise RequiredDimNotFound('Op %s: Cannot find a shape variable with dim_spec %s and dtype combinations %s.' % (
                node_t, ndim_list, dtype_combs_spec))
        dtype_comb = random.choice(dtype_combs)
        for i, ndims in enumerate(ndim_list):
            candidates = self.filter_shapes(
                ndims=ndims, dtype=dtype_comb[i], candidate_shapes=candidate_shapes)
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
    def insert_init_ph_node(self, ph: Placeholder) -> Placeholder:
        self.forward_insert_node(ph, [], oshapes=[
                                 ph.out_shape])

        for c in ph.out_shape.gt_zero():
            self.solver.add(c)

        if self.limnf:
            if NNSMITH_LIMNF_V == '0':
                self.n_floats = nnsmith_add(
                    self.n_floats, ph.out_shape.nelement())
            elif NNSMITH_LIMNF_V == '1':
                self.n_floats_cons.append(nnsmith_le(
                    ph.out_shape.nelement(), self.limit_float // 16))
        return ph

    # subclasses may override this
    def extra_constraints(self, node: AbsOpBase, input_shapes: List[ShapeVar]):
        return []

    def try_forward_insert_at(self, node: AbsOpBase, ishape_indices: List[int]) -> bool:
        input_shapes = [self.alive_shapes[idx][1] for idx in ishape_indices]
        constraints = node.requires(input_shapes)

        if self.verbose:
            print('---> Trying to solve: ', node, constraints)

        # make a copy
        output_shapes = node.shape_fn(input_shapes)
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
        # constraints.extend(self.extra_constraints(node, input_shapes))
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

        if self.verbose:
            print('>> Forward insertion node: ', node)
            print('\tinputs:', input_shapes)
            print('\toutputs:', output_shapes)

        self.forward_insert_node(node, ishape_indices, output_shapes)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, occ_holder_indices: List[int]) -> bool:
        if self.verbose:
            print(
                f'---> Trying to occupy placeholder: {occ_holder_indices} for node {node}')
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
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(occupied_holder_shapes):
            # oversample rank 4 tensors as they may be more important
            ph = self.create_placeholder(
                rank if rank != -1 else
                self.random_rank(),
                dtype=dtype)
            new_inp_placeholders.append(ph)
            constraints.extend(ph.out_shape.gt_zero())

        input_shapes = [p.out_shape for p in new_inp_placeholders]
        constraints.extend(node.requires(input_shapes))
        output_shapes = node.shape_fn(input_shapes)

        for i, shape in enumerate(output_shapes):
            constraints.extend(shape.eq(occupied_holder_shapes[i]))
            constraints.extend(shape.gt_zero())

        self.cur_node = node
        # constraints.extend(self.extra_constraints(node, input_shapes))

        # TODO: consider nfloats.
        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if self.verbose:
            print('>> Backward insertion node: ', node)
            print('\tinputs:', new_inp_placeholders)
            print('\toutputs:', to_occupy)

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


class Bin:
    def __init__(self, lb, ub, scale='linear', base=None, mul=1):
        self.lb = lb
        self.ub = ub
        assert scale in ['linear', 'log']
        self.scale = scale
        self.base = base
        self.mul = mul

    def to_linear(self, x):
        if self.scale == 'log':
            x = math.pow(self.base, x)
        return int(x) * self.mul

    def sample(self):
        x = random.uniform(self.lb, self.ub)
        return self.to_linear(x)

    def sample_range(self):
        if self.lb == None and self.ub == None:
            return None, None
        if self.ub == None:  # one-sided
            return self.to_linear(self.lb), None
        if self.lb == None:  # one-sided
            return None, self.to_linear(self.ub)
        lb = self.sample()
        ub = self.sample()
        if lb > ub:
            lb, ub = ub, lb
        if lb == ub:
            ub = lb + 1
        return lb, ub


PARAM_CONFIG0 = {  # no guidance on param, only on inputs.
}


def range_constrain(param, lb, ub):
    ret = []
    if lb is not None:
        ret.append(nnsmith_ge(param, lb))
    if ub is not None and os.getenv('NNSMITH_LB', 'off') == 'off':  # HACK
        ret.append(nnsmith_lt(param, ub))
    return [z3.And(*ret)] if len(ret) > 0 else []


def __SLICE_CONSTRAINTS(node, inp_shps: List[ShapeVar], construct_param_dict):
    # NOTE(JK): backward mode is slow at generating a chain of many slice ops.
    # Might be one potential general performance issue. If hit performance bottleneck someday,
    # might want to revisit this (substitute old placeholder symbols might help?)
    inp = inp_shps[0]
    start = getattr(node, 'start')
    end = getattr(node, 'end')
    dim_s = inp.shape[node.extra_attrs['axis']]
    MAX_TICKS = 1024
    ret = []
    lb = 0
    if not isinstance(start, int):
        if random.randint(0, 1) or True:
            # start / (dim_s - 1) \in [l / MAX_TICKS, r / MAX_TICKS]
            # start * MAX_TICKS \in [l * (dim_s-1) , r * (dim_s-1)]
            var = nnsmith_mul(start, MAX_TICKS)
            l, r = Bin(lb, MAX_TICKS).sample_range()
            lb = l
            ret.extend(range_constrain(var, l * (dim_s - 1), r * (dim_s - 1)))

    if not isinstance(end, int):
        if random.randint(0, 1) or True:
            var = nnsmith_mul(end, MAX_TICKS)
            l, r = Bin(lb, MAX_TICKS).sample_range()
            ret.extend(range_constrain(var, l * dim_s, r * dim_s))
    return ret


_DEFAULT_BINS = 6
_DEFAULT_BIN_CONS = [Bin(i, i + 1, scale='log', base=2) for i in range(_DEFAULT_BINS)] + \
    [Bin(_DEFAULT_BINS, None, scale='log', base=2)]
_PAD_BIN_CONS = (
    [Bin(i, i + 1, scale='log', base=2) for i in range(_DEFAULT_BINS)] +  # positive
    [Bin(_DEFAULT_BINS, None, scale='log', base=2)] +  # positive
    # negative
    [Bin(i, i + 1, scale='log', base=2, mul=-1) for i in range(_DEFAULT_BINS)] +
    [Bin(None, _DEFAULT_BINS, scale='log', base=2, mul=-1)] +  # negative
    [Bin(0, 1)]  # 0
)

PARAM_CONFIG1 = {
    'NCHWConv2d': {
        'kernel_h_size': _DEFAULT_BIN_CONS,
        'kernel_w_size': _DEFAULT_BIN_CONS,
        'stride': _DEFAULT_BIN_CONS,
        'padding': _DEFAULT_BIN_CONS + [Bin(0, 1)],
        'out_channels': _DEFAULT_BIN_CONS,
        'in_channels': [],  # skip
    },
    'ConstPad': defaultdict(lambda: _PAD_BIN_CONS),
    'ReplicatePad': defaultdict(lambda: _PAD_BIN_CONS),
    'ReflectPad': defaultdict(lambda: _PAD_BIN_CONS),
    'NearestInterp': defaultdict(lambda: _DEFAULT_BIN_CONS),
    'LinearInterp': defaultdict(lambda: _DEFAULT_BIN_CONS),
    'BilinearInterp': defaultdict(lambda: _DEFAULT_BIN_CONS),
    'BicubicInterp': defaultdict(lambda: _DEFAULT_BIN_CONS),
    'TrilinearInterp': defaultdict(lambda: _DEFAULT_BIN_CONS),
    'Slice': __SLICE_CONSTRAINTS,
}
PARAM_CONFIG1['Linear'] = {
    'ifeat': [],
    'ofeat': PARAM_CONFIG1['NCHWConv2d']['out_channels']
}
PARAM_CONFIG1['AvgPool2d'] = {
    'kernel_h_size': PARAM_CONFIG1['NCHWConv2d']['kernel_h_size'],
    'kernel_w_size': PARAM_CONFIG1['NCHWConv2d']['kernel_w_size'],
    'stride': PARAM_CONFIG1['NCHWConv2d']['stride'],
    'padding': PARAM_CONFIG1['NCHWConv2d']['padding'],
}
PARAM_CONFIG1['MaxPool2d'] = PARAM_CONFIG1['AvgPool2d']


def get_name_param(node: AbsOpBase, construct_param_dict):
    if node.num_var_param is None:
        key_param = [(key, getattr(node, key))
                     for key in construct_param_dict]
    else:
        key_param = [(f'param_var{i}', param)
                     for i, param in enumerate(node._param_list)]
    return key_param


def __GROUP_RESHAPE(node, inp_shps, construct_param_dict, bin=True):
    bins = [Bin(i, i + 1, scale='log', base=2)
            for i in range(_DEFAULT_BINS)] + [Bin(None, None)]
    ret = []

    src_group = node.src_group
    dst_group = node.dst_group
    ng = node.ng
    assert len(src_group) == len(dst_group) == ng, (src_group, dst_group)

    construct_params = get_name_param(node, construct_param_dict)
    if bin:
        for gid in range(ng):
            ds = dst_group[gid]
            disable = list(range(len(ds)))
            random.shuffle(disable)
            disable = disable[:1]
            for idx, d in enumerate(ds):
                if idx in disable:
                    continue
                name, param = construct_params[d]
                if len(bins) == 0:
                    continue
                bin_id = random.randint(0, len(bins) - 1)
                lb, ub = bins[bin_id].sample_range()
                ret.extend(range_constrain(param, lb, ub))

    # print('reshape ng:', ng, 'cons:', ret)
    return ret


PARAM_CONFIG1['Reshape'] = __GROUP_RESHAPE
assert all(i in ALL_OP_STR2TYPE for i in PARAM_CONFIG1.keys())

PARAM_CONFIG2 = copy.deepcopy(PARAM_CONFIG1)
PARAM_CONFIG2['ConstPad'] = defaultdict(lambda: _DEFAULT_BIN_CONS)
PARAM_CONFIG2['ReplicatePad'] = defaultdict(lambda: _DEFAULT_BIN_CONS)
PARAM_CONFIG2['ReflectPad'] = defaultdict(lambda: _DEFAULT_BIN_CONS)


class GuidedGen(PureSymbolGen):
    def __init__(self, summaries=None, scale='log', base=2, default_bins=_DEFAULT_BINS, constrain_prob=None, **kwargs):
        self.constrain_prob = constrain_prob if constrain_prob is not None else float(
            os.getenv('NNSMITH_G_PROB', 1))
        self.base = 2
        self.param_config = {
            '0': PARAM_CONFIG0, '1': PARAM_CONFIG1, '2': PARAM_CONFIG2
        }[os.getenv('NNSMITH_G_CONFIG', '1')]
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
        super(GuidedGen, self).__init__(**kwargs)

    def gen_ph_cons(self, ph: Placeholder):
        constraints = []
        for i in ph.out_shape.shape:
            bins = self.default_config[0]
            lb, ub = bins[random.randint(0, len(bins) - 1)].sample_range()
            constraints.extend(range_constrain(i, lb, ub))
        # throw exception for now since this is unlikely to happen
        # assert self.check_sat(
        #     *constraints, nnsmith_le(self.n_floats, self.limit_float)) == z3.sat, 'Input constraints too tight'
        return constraints

    def extra_constraints(self, node: AbsOpBase, input_shapes: List[ShapeVar]):
        ret = []
        construct_param_dict = signature(node.__init__).parameters
        config = self.param_config.get(
            node.__class__.__name__, None)
        if config is None:
            return ret
        if callable(config):
            return config(node, input_shapes, construct_param_dict)
        for key, param in get_name_param(node, construct_param_dict):
            bins = config[key]
            if len(bins) == 0:
                continue
            bin_id = random.randint(0, len(bins) - 1)
            lb, ub = bins[bin_id].sample_range()
            ret.extend(range_constrain(param, lb, ub))
        return ret

    def post_process(self):
        # collect guidance
        graph = self.abstract_graph
        shuffled_nids = list(graph.nodes)
        random.shuffle(shuffled_nids)
        all_cons = []
        for node_id in shuffled_nids:
            op = graph.nodes[node_id]['op']
            ishape_indices = graph.nodes[node_id]['ishape_indices']
            ishape_vars = [self.alive_shapes[i][1] for i in ishape_indices]
            if isinstance(op, AbsOpBase):
                cons = self.extra_constraints(op, ishape_vars)
            else:
                assert isinstance(op, Placeholder), op
                cons = self.gen_ph_cons(op)
            if len(cons) == 0 or random.uniform(0, 1) > self.constrain_prob:
                continue
            all_cons.extend(cons)

        # apply all guidance at once and back off if failed
        shuffled = list(range(len(all_cons)))
        random.shuffle(shuffled)
        cur_cons = [all_cons[i] for i in shuffled]
        timeout = self.max_gen_millisec / (1 + math.log2(len(cur_cons))) / 3
        while self.check_sat(*cur_cons) != z3.sat:
            cur_cons = cur_cons[:len(cur_cons) // 2]
            if len(cur_cons) == 0:
                break
        self.solver.add(cur_cons)
        if self.verbose:
            print(
                '# guidance applied: {} / {}'.format(len(cur_cons), len(all_cons)))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nodes', type=int, default=10)
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
    parser.add_argument('--limnf', dest='limnf', action='store_true',
                        help='Enable the limit on the number of floats')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--no_export', action='store_true')
    parser.add_argument('--forward_prob', type=float)
    parser.add_argument('--diff_can_overwrite', action='store_true')
    parser.add_argument('--print_grad', type=int, default=0)
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
        skip=None,
        **kwargs):

    GenCls = {
        'random': PureSymbolGen,
        'guided': GuidedGen,
    }[mode]
    if skip is not None:
        config_skip_op(skip)
    gen = GenCls(min_dims=min_dims,
                 viz_sbs=viz_sbs, seed=seed, verbose=verbose, use_bitvec=use_bitvec,
                 **kwargs)
    gen.abstract_gen(max_node_size=max_nodes,
                     max_gen_millisec=timeout)
    solution = gen.get_symbol_solutions()

    return gen, solution


if __name__ == '__main__':
    args = parse_args()

    gen_args = {}
    if args.diff_can_overwrite:
        __DIFF_CACHE__ = 'config/diff.pkl'
        differentiable_ops = rewrite_op_dtype(
            ALL_OP_TYPES, backend=None, diff=True, verbose=False, cache=__DIFF_CACHE__)
        gen_args['candidates_overwrite'] = differentiable_ops
        gen_args['init_fp'] = True

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
                                     limnf=args.limnf, merge_op_v=args.merge_op_v, forward_prob=args.forward_prob, **gen_args)
    print(
        f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
    print(
        f'{time.time() - strt_time}s to generate a graph w/ {gen.num_op()} operators')
    if args.verbose:
        print('solution:', solution)
    srt_time = time.time()
    if args.verbose or args.viz_graph:
        gen.viz(args.output_path + '.png')

    if args.no_export:
        exit(0)
    net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose,
                    alive_shapes=gen.alive_shapes, print_grad=args.print_grad)
    print('Initializing SymbolNet time: {}s'.format(time.time() - srt_time))
    import onnx
    onnx.checker.check_model(
        onnx.load(args.output_path), full_check=True)
    input_st = time.time()

    sat_inputs = None
    if args.input_gen == 'v3' or args.input_gen == 'random':
        with torch.no_grad():
            net.eval()
            sat_inputs = net.rand_input_gen(use_cuda=args.use_cuda)
            infer_succ = sat_inputs is not None
    elif args.input_gen == 'grad':
        net.eval()
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
    print('self.invalid_found_last=', net.invalid_found_last)
    assert AbsOpBase.numeric_valid(net(*sat_inputs))
    print('Time to generate inputs: {:.3f}s'.format(ed_time - input_st))

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

    if sat_inputs is not None:
        ret_inputs = {}
        for i, name in enumerate(net.input_spec):
            ret_inputs[name] = sat_inputs[i].cpu().numpy()
        sat_inputs = ret_inputs
    pickle.dump(sat_inputs, open(
        args.output_path + '-sat_inputs.pkl', 'wb'))
    torch2onnx(net, args.output_path, verbose=args.verbose,
               use_cuda=args.use_cuda)

    net.to_picklable()
    cloudpickle.dump(net, open(args.output_path +
                     '-net.pkl', 'wb'), protocol=4)
