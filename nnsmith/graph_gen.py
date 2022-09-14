import copy
import logging
import math
import os
import random
import textwrap
import time
import traceback
import uuid
from abc import abstractmethod
from collections import defaultdict, namedtuple
from inspect import signature
from typing import Dict, List, Set, Tuple, Type

import networkx as nx
import z3

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import *
from nnsmith.abstract.op import (
    __MAX_RANK__,
    AbsOpBase,
    AbsTensor,
    Expand,
    Placeholder,
    concretize_op,
    random_group,
)
from nnsmith.error import ConstraintError, SanityCheck
from nnsmith.logging import MGEN_LOG, SMT_LOG
from nnsmith.util import HAS_PYGRAPHVIZ, set_seed, viz_dot

NNSMITH_LIMNF_V = os.getenv("NNSMITH_LIMNF_V", "0")
assert NNSMITH_LIMNF_V in ["0", "1"]
NNSMITH_BV_SIZE = os.getenv("NNSMITH_BV_SIZE", "8")


class RequiredDimNotFound(Exception):
    pass


__MB_LIM__ = 6 * 1024
__TEXTWRAP_WIDTH__ = 30


TensorCtx = namedtuple("TensorCtx", ["op_id", "type", "output_idx"])
# Every tensor is represented by a (unique) integer key.


def concretize_graph(
    graph: nx.MultiDiGraph, dataflow: List[TensorCtx], model: z3.ModelRef
) -> Tuple[nx.MultiDiGraph, Dict[int, AbsTensor]]:
    concrete_shapes: Dict[int, AbsTensor] = {}

    # freeze node with static attributes in label;
    for node_id in nx.topological_sort(graph):
        node = graph.nodes[node_id]
        op = concretize_op(node["op"], model)

        # concretize shapes;
        itensors = [concrete_shapes[df_idx] for df_idx in node["itensor_idx"]]
        otensors = op.checked_type_transfer(itensors)
        op.input_like = itensors
        op.output_like = otensors

        node["op"] = op
        node["label"] = textwrap.fill(f"#{node_id} ~ {op}", width=__TEXTWRAP_WIDTH__)

        for i, df_idx in enumerate(node["otensor_idx"]):
            concrete_shapes[df_idx] = otensors[i]

    # freeze edge with static attributes in label;
    for src, dst, idx in graph.edges(keys=True):
        edge = graph.edges[src, dst, idx]
        out_operand_idx, in_operand_idx = edge["operand_idx"]
        svar = concrete_shapes[edge["shape_idx"]]
        edge["label"] = textwrap.fill(
            f"{out_operand_idx}→{in_operand_idx} {svar.dtype.short()}!{svar.shape}",
            width=__TEXTWRAP_WIDTH__,
        )

    return graph, concrete_shapes


class SimpleGenerator:
    def __init__(
        self,
        opset,
        init_rank=4,
        megabyte_lim=__MB_LIM__,
        seed=None,
        limnf=True,
        forward_prob=None,
        init_fp=False,
    ):
        assert len(opset) > 0, "opset must not be empty"
        if seed is not None:
            set_seed(seed)
            z3.set_param(
                "smt.phase_selection",
                5,
                # TODO(@ganler): re-enable this when having a usable op memory estimator.
                # "smt.arith.random_initial_value",
                # True,
                "smt.random_seed",
                seed,
                "sat.random_seed",
                seed,
                "sat.phase",
                "random",
                "memory_max_size",
                50 * 1024,  # MB
            )

        self.op_candidates = opset
        self.solver = z3.Solver()

        # 4 bytes per float (assume we use float64)
        self.limit_float = 1024**2 * megabyte_lim // 8

        # Node -> op: AbsOpBase
        # Edge -> shape_idx:-> self.alive_shapes
        self.abstract_graph = nx.MultiDiGraph()

        # Flat view of consumable tensors which makes random selection easier.
        # List of namedtuple("TensorCtx", ["op_id", "type", "output_idx"])
        #                                   int    AbsTensor  int
        self.tensor_dataflow: List[TensorCtx] = []

        # dim size -> list[shape idx -> output_tensor_pool]
        self.dim2shape_idx: Dict[int, List[int]] = {}

        self.use_bitvec = False  # Only consider integer domain for now.
        self.init_rank = init_rank
        self.n_floats = 0
        self.monotonic_placeholder_id = 0
        self.monotonic_nx_node_idx = 0
        # self.reusable_placeholder_nx_indices = []
        self.last_solution = None
        self.limnf = limnf
        self.n_floats_cons = []

        # <op idx>
        self.placeholders: List[int] = []
        # for all (including newly created tmp) placeholders

        self.insert_init_ph_node(
            self.create_placeholder(init_rank, dtype=DType.float32 if init_fp else None)
        )
        self.init_ph_alive = True
        self.forward_prob = 0.5 if forward_prob is None else forward_prob

    def random_rank(self):
        return random.choices(range(__MAX_RANK__ + 1), weights=[1, 1, 1, 1, 2, 1])[0]

    def random_dtype(self):
        # more floats than ints.
        wts = [1] * len(DTYPE_ALL)
        for i in DTYPE_FLOATS:
            wts[DTYPE_ALL.index(i)] = 8
        for i in DTYPE_INTS:
            wts[DTYPE_ALL.index(i)] = 2
        return random.choices(DTYPE_ALL, weights=wts)[0]

    def create_placeholder(self, rank, dtype=None):
        syms = self.new_syms(
            ["v%s_%s" % (self.monotonic_placeholder_id, k) for k in range(rank)]
        )
        shapevar = AbsTensor(
            shape=syms, dtype=dtype if dtype is not None else self.random_dtype()
        )
        self.monotonic_placeholder_id += 1
        ph = Placeholder(shapevar)
        return ph

    # default to no input constraints
    def gen_ph_cons(self, ph: Placeholder) -> List[z3.ExprRef]:
        return []

    def post_process(self):
        """Called after the graph is finalized. May be used to add parameter guidance."""
        pass

    def new_sym(self, name, bv_size=None):
        return z3.Int(name)

    def new_syms(self, names):
        if self.use_bitvec:
            bv_sizes = list(
                map(len, random_group(int(os.getenv("NNSMITH_BITS", 30)), len(names)))
            )
            assert len(bv_sizes) == len(names)
            return [self.new_sym(name, bvsize) for name, bvsize in zip(names, bv_sizes)]
        else:
            return [self.new_sym(name) for name in names]

    @abstractmethod
    def insert_init_ph_node(
        self, init_rank, shape=None, dtype=DType.float32
    ) -> AbsTensor:
        raise NotImplementedError

    @abstractmethod
    def try_forward_insert_at(self, node: AbsOpBase, itensor_idx: List[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def try_occupy_placeholder(
        self, node: AbsOpBase, placeholder_indices: List[int]
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_solutions(self) -> List:
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
        for _, val, _ in self.tensor_dataflow:
            self.n_floats = nnsmith_add(self.n_floats, val.nelement())

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        z3.set_param("timeout", max_gen_millisec // 3)

        init_time = time.time()

        # starts generation.
        while (
            time.time() - init_time < max_gen_millisec / 1000
            and self.num_op() < max_node_size
        ):
            if self.extra_exit_check():
                break
            node_t = self.pick_next_op_type()
            self.try_insert_node_type(node_t)
        if abs(self.num_op() - max_node_size) >= 3:
            MGEN_LOG.warning(
                f"[WARNING]: graph size: {len(self.abstract_graph.nodes)} < expected size: {max_node_size}"
            )

        self.recompute_n_floats()
        if (
            self.limnf
        ):  # add into solver since graph is finalized to avoid repeated solving
            if NNSMITH_LIMNF_V == "0":
                self.solver.add(nnsmith_le(self.n_floats, self.limit_float))
            elif NNSMITH_LIMNF_V == "1":
                self.solver.add(*self.n_floats_cons)
            assert self.check_sat() == z3.sat

        self.post_process()  # can be used to add more constraints

        # init graph placeholders
        shuffled_placeholder = self.placeholders
        self.abstract_graph.nodes[shuffled_placeholder[0]][
            "op"
        ] = self.abstract_graph.nodes[shuffled_placeholder[0]]["op"].to_input()
        for holder_idx in shuffled_placeholder[1:]:
            if random.randint(0, 1):
                self.abstract_graph.nodes[holder_idx]["op"] = self.abstract_graph.nodes[
                    holder_idx
                ]["op"].to_const()
            else:
                self.abstract_graph.nodes[holder_idx]["op"] = self.abstract_graph.nodes[
                    holder_idx
                ]["op"].to_input()

    def check_arith_ref(self, var):
        SanityCheck.true(
            isinstance(var, (z3.BitVecRef, z3.BoolRef, bool)),
            f"{type(var)} not supported.",
        )
        if not isinstance(var, bool):
            for child in var.children():
                self.check_arith_ref(child)

    def check_sat(self, *assumptions):
        start = time.time()
        if self.use_bitvec:
            for assump in assumptions:
                self.check_arith_ref(assump)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            if self.solver.assertions():
                SMT_LOG.debug(
                    f"existing constraints: {', '.join(map(str, self.solver.assertions()))}"
                )
            if assumptions:
                SMT_LOG.debug(f"new constraints: {', '.join(map(str, assumptions))}")

        cres = self.solver.check(*assumptions)

        checking_time = int((time.time() - start) * 1000)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"{cres} <-- checking time: {checking_time} ms")

            if cres == z3.unsat:
                SMT_LOG.debug(f"Unsat core: {self.solver.unsat_core()}")

        if cres == z3.sat:
            self.last_solution = self.solver.model()

        return cres

    def pick_next_op_type(self):
        return random.choices(self.op_candidates, k=1)[0]

    def forward_insert_node(
        self,
        node: AbsOpBase,
        itensor_idx: List[int],
        oshapes: List[AbsTensor] = None,
        force_otensor_idx=None,
    ) -> int:
        if oshapes is None:
            input_shapes = [self.tensor_dataflow[idx].type for idx in itensor_idx]
            oshapes = node.checked_type_transfer(input_shapes)

        succ_nid = self.get_new_node_id()
        if isinstance(node, Placeholder):
            self.placeholders.append(succ_nid)

        otensor_idx = []
        if force_otensor_idx is None:
            for i, abs_tensor in enumerate(oshapes):
                SanityCheck.true(
                    len(abs_tensor.shape) in node.out_ranks[i],
                    "{}'s dimension size is not {} in {}".format(
                        abs_tensor.shape, node.out_ranks[i], node.__class__.__name__
                    ),
                )
                node.out_ranks[i] = (len(abs_tensor.shape),)
                shape_idx = len(self.tensor_dataflow)
                otensor_idx.append(shape_idx)
                self.tensor_dataflow.append(
                    TensorCtx(op_id=succ_nid, type=abs_tensor, output_idx=i)
                )
                self.dim2shape_idx.setdefault(len(abs_tensor.shape), []).append(
                    shape_idx
                )
        else:
            # When taking the position of placeholders, we do not need to add new alive shapes.
            otensor_idx = force_otensor_idx

        # NOTE: because of backward insertion, we may not be able to limit the symbol size as there will be some
        # trivially equivalent symbols which harms the readability. (e.g., relations like `a = b` is not known).
        # NOTE: `otensor_idx` and `itensor_idx` are indices of alive_shapes
        self.abstract_graph.add_node(
            succ_nid,
            op=node,
            nin=len(itensor_idx),
            nout=len(oshapes),
            otensor_idx=otensor_idx,
            itensor_idx=itensor_idx,
            label=textwrap.fill(
                f"#{succ_nid} ~ {node}",
                width=__TEXTWRAP_WIDTH__,
            ),
        )

        for in_operand_idx, idx in enumerate(itensor_idx):
            pred_nid, svar, out_operand_idx = self.tensor_dataflow[idx]
            self.abstract_graph.add_edge(
                pred_nid,
                succ_nid,
                key=str(uuid.uuid1()),
                shape_idx=idx,
                operand_idx=(out_operand_idx, in_operand_idx),
                label=f"{out_operand_idx}→{in_operand_idx} {svar.dtype.short()}!{svar.shape}",
            )

        return succ_nid

    def get_new_node_id(self):
        # if self.reusable_placeholder_nx_indices:
        #     return self.reusable_placeholder_nx_indices.pop()
        ret = self.monotonic_nx_node_idx
        self.monotonic_nx_node_idx += 1
        return ret

    def id2nxnode(self, id):
        return self.abstract_graph.nodes[id]

    def backward_insert_node(
        self, node, input_nodes: List[Union[int, Placeholder]], occupied_idx
    ):
        # self.placeholder idx -> nx graph node idx
        occ_holder_idx_nx = [self.placeholders[i] for i in occupied_idx]

        itensor_idx = []
        for input_node in input_nodes:
            # Insert Placeholder in `input_nodes`
            if isinstance(input_node, Placeholder):
                nid = self.get_new_node_id()
                shape_idx = len(self.tensor_dataflow)
                self.tensor_dataflow.append(
                    TensorCtx(op_id=nid, type=input_node.out_shape, output_idx=0)
                )
                self.dim2shape_idx.setdefault(input_node.out_shape.ndims, []).append(
                    shape_idx
                )
                self.abstract_graph.add_node(
                    nid,
                    op=input_node,
                    nin=0,
                    nout=1,
                    itensor_idx=[],
                    otensor_idx=[shape_idx],
                    label=textwrap.fill(
                        f"#{nid} ~ {input_node}",
                        width=__TEXTWRAP_WIDTH__,
                    ),
                )
                itensor_idx.append(shape_idx)
                self.placeholders.append(nid)
            else:
                itensor_idx.append(input_node)

        # Insert node
        to_occ_alive_shape_idx = [
            self.id2nxnode(nx_nid)["otensor_idx"][0] for nx_nid in occ_holder_idx_nx
        ]
        op_nx_idx = self.forward_insert_node(
            node,
            itensor_idx=itensor_idx,
            oshapes=[
                self.tensor_dataflow[as_idx][1] for as_idx in to_occ_alive_shape_idx
            ],
            force_otensor_idx=to_occ_alive_shape_idx,
        )

        # Insert edges and remove placeholders
        for i, nx_idx in enumerate(occ_holder_idx_nx):
            for (src, dst, key) in list(self.abstract_graph.edges(nx_idx, keys=True)):
                # multi-graph
                edge_info = copy.deepcopy(
                    self.abstract_graph.get_edge_data(src, dst, key=key)
                )
                old_edge_idx = edge_info["shape_idx"]
                # recall alive shape:
                # 1. op nx idx
                # 2. shape var
                # 3. out operand idx

                _, svar, _ = self.tensor_dataflow[old_edge_idx]
                out_operand_idx = i
                in_operand_idx = edge_info["operand_idx"][1]

                # add cur node -> dst
                self.abstract_graph.add_edge(
                    op_nx_idx,
                    dst,
                    key=str(uuid.uuid1()),
                    shape_idx=edge_info["shape_idx"],  # reuse old alive shape
                    operand_idx=(out_operand_idx, in_operand_idx),
                    label=f"{out_operand_idx}→{in_operand_idx} {svar.dtype.short()}!{svar.shape}",
                )
                self.tensor_dataflow[old_edge_idx] = TensorCtx(
                    op_nx_idx, svar, out_operand_idx
                )

                self.abstract_graph.remove_edge(src, dst, key=key)

            # if the PH to occupy has no consumers, we simply reassign its alive shape.
            # NOTE: we assume the first node is a placeholder.
            if self.init_ph_alive:  # update alive_shape[0]
                self.tensor_dataflow[0] = TensorCtx(
                    op_nx_idx, self.tensor_dataflow[0][1], 0
                )
                self.init_ph_alive = False

            # remove placeholders
            self.abstract_graph.remove_node(nx_idx)
            # self.reusable_placeholder_nx_indices.append(nx_idx)
            self.placeholders.remove(nx_idx)

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

        itensor_idx = self.pick_tensor_idx(
            type(op),
            dim_spec_list,
            op.in_dtypes,
            candidate_shapes=[s[1] for s in self.tensor_dataflow],
        )

        if self.try_forward_insert_at(op, itensor_idx):
            return True

        return False

    def try_backward_insert(self, op: AbsOpBase):
        # we know that: Y = op(X)
        # S1 - select Y: Y must be a placeholder; (this also means the graph must start w/ a placeholder)
        ph_candidates = []
        for idx in self.placeholders:
            oshape = self.id2nxnode(idx)["op"].out_shape
            if isinstance(op, Expand) and oshape.ndims < op.expand_last_dim:
                continue
            ph_candidates.append(oshape)

        placeholder_indices = self.pick_tensor_idx(
            type(op), op.out_ranks, op.out_dtypes, candidate_shapes=ph_candidates
        )

        if self.try_occupy_placeholder(op, placeholder_indices):
            return True

        return False

    def try_insert_node_type(self, node_t, max_tensor_pick_time=3) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"@[Node #{len(self.abstract_graph.nodes)}] <-- "
                f"trying to insert node type {node_t.__name__}"
            )

        try:
            for _ in range(max_tensor_pick_time):
                # should recreate a new instance since some attributes (like axis) should be initialized for each pick
                op_param_n = node_t.get_num_var_param()
                op_id = len(self.abstract_graph.nodes)
                op_params = [
                    self.new_sym("op%s_%s" % (op_id, k)) for k in range(op_param_n)
                ]

                op: AbsOpBase = node_t(*op_params)
                op._param_list = op_params

                if random.uniform(0, 1) < self.forward_prob:
                    if self.try_forward_insert(op):
                        return True
                else:
                    if self.try_backward_insert(op):
                        return True
        except RequiredDimNotFound:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False
        except ConstraintError:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False

        return False

    def filter_shapes(self, ndims, dtype, candidate_shapes: List[AbsTensor]):
        cans = range(len(candidate_shapes))

        cans = list(
            filter(  # filter with ndim
                lambda sid: candidate_shapes[sid].ndims in ndims, cans
            )
        )
        if len(cans) == 0:
            raise RequiredDimNotFound(
                f"Cannot find a shape variable with #dimensions {ndims}."
            )

        if dtype is not None:
            cans = list(
                filter(  # filter with dtype
                    lambda sid: candidate_shapes[sid].dtype == dtype, cans
                )
            )
            if len(cans) == 0:
                raise RequiredDimNotFound(
                    f"Cannot find a shape variable with #dimensions {ndims} and dtype {dtype}."
                )

        return cans

    def pick_shape(self, node_t, candidates):
        return random.choice(candidates)

    def pick_tensor_idx(
        self,
        node_t,
        ndim_list: List[Set[int]],
        dtype_combs_spec: List[Tuple[DType, ...]],
        candidate_shapes: List[AbsTensor],
    ) -> List[int]:
        """Randomly pick indices to shape variables from the output pool.

        Args:
            ndim_list (List[Set]): duable dims for each input.

        Returns:
            List[int]: indices to applicable shape variables.
        """

        abs_tensor_candidates = []
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f"Input data types candidates: {dtype_combs_spec}")

        all_can_dtypes = []
        for i, ndims in enumerate(ndim_list):
            all_can_dtypes.extend(
                [
                    candidate_shapes[i].dtype
                    for i in self.filter_shapes(
                        ndims=ndims, dtype=None, candidate_shapes=candidate_shapes
                    )
                ]
            )
        # only use dtypes currently available after ndim filtering
        dtype_combs = [
            comb for comb in dtype_combs_spec if all(i in all_can_dtypes for i in comb)
        ]
        if len(dtype_combs) == 0:
            raise RequiredDimNotFound(
                "Op %s: Cannot find a shape variable with dim_spec %s and dtype combinations %s."
                % (node_t, ndim_list, dtype_combs_spec)
            )
        dtype_comb = random.choice(dtype_combs)
        for i, ndims in enumerate(ndim_list):
            candidates = self.filter_shapes(
                ndims=ndims, dtype=dtype_comb[i], candidate_shapes=candidate_shapes
            )
            abs_tensor_candidates.append(self.pick_shape(node_t, candidates))

        return abs_tensor_candidates


class PureSymbolGen(SimpleGenerator):
    def insert_init_ph_node(self, ph: Placeholder) -> Placeholder:
        self.forward_insert_node(ph, [], oshapes=[ph.out_shape])

        for c in ph.out_shape.gt_zero():
            self.solver.add(c)

        if self.limnf:
            if NNSMITH_LIMNF_V == "0":
                self.n_floats = nnsmith_add(self.n_floats, ph.out_shape.nelement())
            elif NNSMITH_LIMNF_V == "1":
                self.n_floats_cons.append(
                    nnsmith_le(ph.out_shape.nelement(), self.limit_float // 16)
                )
        return ph

    # subclasses may override this
    def extra_constraints(self, node: AbsOpBase, input_shapes: List[AbsTensor]):
        return []

    def try_forward_insert_at(self, node: AbsOpBase, itensor_idx: List[int]) -> bool:
        input_shapes = [self.tensor_dataflow[idx][1] for idx in itensor_idx]
        constraints = node.checked_requires(input_shapes)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        output_shapes = node.checked_type_transfer(input_shapes)
        if self.limnf:
            if NNSMITH_LIMNF_V == "0":
                tmp_n_floats = nnsmith_add(self.n_floats, node.n_floats(input_shapes))
            elif NNSMITH_LIMNF_V == "1":
                tmp_n_floats_cons = self.n_floats_cons + [
                    nnsmith_le(node.n_floats(input_shapes), self.limit_float // 16)
                ]

        for shape in output_shapes:
            for c in shape.gt_zero():
                constraints.append(c)

        # constraints.extend(self.extra_constraints(node, input_shapes))
        if self.limnf:
            if NNSMITH_LIMNF_V == "0":
                check_res = self.check_sat(
                    *constraints, nnsmith_le(tmp_n_floats, self.limit_float)
                )
            elif NNSMITH_LIMNF_V == "1":
                check_res = self.check_sat(*constraints, *tmp_n_floats_cons)
        else:
            check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.solver.add(c)
        if self.limnf:
            if NNSMITH_LIMNF_V == "0":
                self.n_floats = tmp_n_floats
            elif NNSMITH_LIMNF_V == "1":
                self.n_floats_cons = tmp_n_floats_cons

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {input_shapes}")
            MGEN_LOG.debug(f"\toutputs: {output_shapes}")

        self.forward_insert_node(node, itensor_idx, output_shapes)
        return True

    def try_occupy_placeholder(
        self, node: AbsOpBase, occ_holder_indices: List[int]
    ) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {occ_holder_indices} for node {node}"
            )
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        to_occupy = [
            self.id2nxnode(self.placeholders[i])["op"] for i in occ_holder_indices
        ]

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
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            new_inp_placeholders.append(ph)
            constraints.extend(ph.out_shape.gt_zero())

        input_shapes = [p.out_shape for p in new_inp_placeholders]
        constraints.extend(node.checked_requires(input_shapes))
        output_shapes = node.checked_type_transfer(input_shapes)

        for i, shape in enumerate(output_shapes):
            constraints.extend(shape.eq(occupied_holder_shapes[i]))
            constraints.extend(shape.gt_zero())

        # constraints.extend(self.extra_constraints(node, input_shapes))

        # TODO: consider nfloats.
        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {new_inp_placeholders}")
            MGEN_LOG.debug(f"\toutputs: {to_occupy}")

        for c in constraints:
            self.solver.add(c)

        self.backward_insert_node(node, new_inp_placeholders, occ_holder_indices)

        return True

    def get_solutions(self) -> List:
        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        return self.last_solution


class Bin:
    def __init__(self, lb, ub, scale="linear", base=None, mul=1):
        self.lb = lb
        self.ub = ub
        assert scale in ["linear", "log"]
        self.scale = scale
        self.base = base
        self.mul = mul

    def to_linear(self, x):
        if self.scale == "log":
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


PARAM_CONFIG0 = {}  # no guidance on param, only on inputs.


def range_constrain(param, lb, ub):
    ret = []
    if lb is not None:
        ret.append(nnsmith_ge(param, lb))
    if ub is not None and os.getenv("NNSMITH_LB", "off") == "off":  # HACK
        ret.append(nnsmith_lt(param, ub))
    return [z3.And(*ret)] if len(ret) > 0 else []


def __SLICE_CONSTRAINTS(node, inp_shps: List[AbsTensor], construct_param_dict):
    # NOTE(JK): backward mode is slow at generating a chain of many slice ops.
    # Might be one potential general performance issue. If hit performance bottleneck someday,
    # might want to revisit this (substitute old placeholder symbols might help?)
    inp = inp_shps[0]
    start = getattr(node, "start")
    end = getattr(node, "end")
    dim_s = inp.shape[node.extra_attrs["axis"]]
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
_DEFAULT_BIN_CONS = [
    Bin(i, i + 1, scale="log", base=2) for i in range(_DEFAULT_BINS)
] + [Bin(_DEFAULT_BINS, None, scale="log", base=2)]
_PAD_BIN_CONS = (
    [Bin(i, i + 1, scale="log", base=2) for i in range(_DEFAULT_BINS)]
    + [Bin(_DEFAULT_BINS, None, scale="log", base=2)]  # positive
    +  # positive
    # negative
    [Bin(i, i + 1, scale="log", base=2, mul=-1) for i in range(_DEFAULT_BINS)]
    + [Bin(None, _DEFAULT_BINS, scale="log", base=2, mul=-1)]
    + [Bin(0, 1)]  # negative  # 0
)

PARAM_CONFIG1 = {
    "NCHWConv2d": {
        "kernel_h_size": _DEFAULT_BIN_CONS,
        "kernel_w_size": _DEFAULT_BIN_CONS,
        "dilation_h": _DEFAULT_BIN_CONS,
        "dilation_w": _DEFAULT_BIN_CONS,
        "stride": _DEFAULT_BIN_CONS,
        "padding": _DEFAULT_BIN_CONS + [Bin(0, 1)],
        "out_channels": _DEFAULT_BIN_CONS,
        "in_channels": [],  # skip
    },
    "ConstPad": defaultdict(lambda: _PAD_BIN_CONS),
    "ReplicatePad": defaultdict(lambda: _PAD_BIN_CONS),
    "ReflectPad": defaultdict(lambda: _PAD_BIN_CONS),
    "NearestInterp": defaultdict(lambda: _DEFAULT_BIN_CONS),
    "LinearInterp": defaultdict(lambda: _DEFAULT_BIN_CONS),
    "BilinearInterp": defaultdict(lambda: _DEFAULT_BIN_CONS),
    "BicubicInterp": defaultdict(lambda: _DEFAULT_BIN_CONS),
    "TrilinearInterp": defaultdict(lambda: _DEFAULT_BIN_CONS),
    "Slice": __SLICE_CONSTRAINTS,
}
PARAM_CONFIG1["Linear"] = {
    "ifeat": [],
    "ofeat": PARAM_CONFIG1["NCHWConv2d"]["out_channels"],
}
PARAM_CONFIG1["AvgPool2d"] = {
    "kernel_h_size": PARAM_CONFIG1["NCHWConv2d"]["kernel_h_size"],
    "kernel_w_size": PARAM_CONFIG1["NCHWConv2d"]["kernel_w_size"],
    "stride": PARAM_CONFIG1["NCHWConv2d"]["stride"],
    "padding": PARAM_CONFIG1["NCHWConv2d"]["padding"],
}
PARAM_CONFIG1["MaxPool2d"] = PARAM_CONFIG1["AvgPool2d"]


def get_name_param(node: AbsOpBase, construct_param_dict):
    if node.num_var_param is None:
        key_param = [(key, getattr(node, key)) for key in construct_param_dict]
    else:
        key_param = [
            (f"param_var{i}", param) for i, param in enumerate(node._param_list)
        ]
    return key_param


def __GROUP_RESHAPE(node, inp_shps, construct_param_dict, bin=True):
    bins = [Bin(i, i + 1, scale="log", base=2) for i in range(_DEFAULT_BINS)] + [
        Bin(None, None)
    ]
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

    return ret


PARAM_CONFIG1["Reshape"] = __GROUP_RESHAPE

PARAM_CONFIG2 = copy.deepcopy(PARAM_CONFIG1)
PARAM_CONFIG2["ConstPad"] = defaultdict(lambda: _DEFAULT_BIN_CONS)
PARAM_CONFIG2["ReplicatePad"] = defaultdict(lambda: _DEFAULT_BIN_CONS)
PARAM_CONFIG2["ReflectPad"] = defaultdict(lambda: _DEFAULT_BIN_CONS)


class GuidedGen(PureSymbolGen):
    def __init__(
        self,
        summaries=None,
        scale="log",
        base=2,
        default_bins=_DEFAULT_BINS,
        constrain_prob=None,
        **kwargs,
    ):
        self.constrain_prob = (
            constrain_prob
            if constrain_prob is not None
            else float(os.getenv("NNSMITH_G_PROB", 1))
        )
        self.base = 2
        self.param_config = {
            "0": PARAM_CONFIG0,
            "1": PARAM_CONFIG1,
            "2": PARAM_CONFIG2,
        }[os.getenv("NNSMITH_G_CONFIG", "1")]
        if scale == "log":
            self.default_config = defaultdict(
                lambda: [
                    Bin(i, i + 1, scale=scale, base=base) for i in range(default_bins)
                ]
                + [Bin(default_bins, None, scale=scale, base=base)]
            )
        else:
            assert scale == "linear", scale
            self.default_config = defaultdict(
                lambda: [Bin(0, 256, scale="linear")] + [Bin(256, None, scale="linear")]
            )
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

    def extra_constraints(self, node: AbsOpBase, input_shapes: List[AbsTensor]):
        ret = []
        construct_param_dict = signature(node.__init__).parameters
        config = self.param_config.get(node.__class__.__name__, None)
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
            op = graph.nodes[node_id]["op"]
            itensor_idx = graph.nodes[node_id]["itensor_idx"]
            itensors = [self.tensor_dataflow[i].type for i in itensor_idx]
            if isinstance(op, AbsOpBase):
                cons = self.extra_constraints(op, itensors)
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
        while self.check_sat(*cur_cons) != z3.sat:
            cur_cons = cur_cons[: len(cur_cons) // 2]
            if len(cur_cons) == 0:
                break
        self.solver.add(cur_cons)
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f"# guidance applied: {len(cur_cons)} / {len(all_cons)}")


def random_model_gen(
    opset: Set[Type[AbsOpBase]],
    init_rank=4,
    max_nodes=5,
    seed=None,
    timeout_ms=20000,
    **kwargs,
):
    gen = PureSymbolGen(
        opset=opset,
        init_rank=init_rank,
        seed=seed,
        **kwargs,
    )
    gen.abstract_gen(max_node_size=max_nodes, max_gen_millisec=timeout_ms)

    return gen


def viz(G, filename: str = None):
    if HAS_PYGRAPHVIZ:
        viz_dot(nx.nx_agraph.to_agraph(G).to_string(), filename)
