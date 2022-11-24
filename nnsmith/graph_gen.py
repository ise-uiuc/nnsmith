import logging
import random
import time
import traceback
from abc import abstractmethod
from typing import List, Optional, Set, Tuple, Type

import z3

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import *
from nnsmith.abstract.op import AbsOpBase, AbsTensor, Expand, Placeholder, rank_all
from nnsmith.error import ConstraintError, SanityCheck
from nnsmith.gir import GraphIR, InstExpr, InstIR
from nnsmith.logging import MGEN_LOG, SMT_LOG
from nnsmith.util import HAS_PYGRAPHVIZ, set_seed, viz_dot


class RequiredDimNotFound(Exception):
    pass


__MB_LIM__ = 6 * 1024
__TEXTWRAP_WIDTH__ = 30


def concretize_graph(ir: GraphIR, model: z3.ModelRef) -> GraphIR:
    return ir.concretize(model)


class SimpleGenerator:
    def __init__(
        self,
        opset,
        seed=None,
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

        self.ir = GraphIR()

        self.monotonic_placeholder_id = 0
        self.last_solution: Optional[z3.ModelRef] = None

        # Names of current placeholders
        self.placeholders: List[str] = []
        # for all (including newly created tmp) placeholders

        self.insert_init_ph_node(
            self.create_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        )
        self.init_ph_alive = True
        self.forward_prob = 0.5 if forward_prob is None else forward_prob

    def random_rank(self):
        return random.choice(rank_all())

    def random_dtype(self):
        # more floats than ints.
        wts = [1] * len(DTYPE_ALL)
        for i in DTYPE_FLOATS:
            wts[DTYPE_ALL.index(i)] = 4
        for i in DTYPE_INTS:
            wts[DTYPE_ALL.index(i)] = 1
        return random.choices(DTYPE_ALL, weights=wts)[0]

    def create_placeholder(self, rank, dtype=None) -> Placeholder:
        syms = self.new_syms(
            ["v%s_%s" % (self.monotonic_placeholder_id, k) for k in range(rank)]
        )
        ph = Placeholder(
            AbsTensor(
                shape=syms, dtype=dtype if dtype is not None else self.random_dtype()
            )
        )
        self.monotonic_placeholder_id += 1
        return ph

    def new_sym(self, name):
        return z3.Int(name)

    def new_syms(self, names):
        return [self.new_sym(name) for name in names]

    def insert_init_ph_node(self, ph: Placeholder) -> InstIR:
        inst = self.forward_insert_node(ph, [])

        for c in ph.ttype.gt_zero():
            self.solver.add(c)

        return inst

    @abstractmethod
    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_sat_model(self) -> List:
        raise NotImplementedError

    def extra_exit_check(self) -> bool:
        """
        Returns:
            bool: add more checks to determine whether to exit the generation.
        """
        return False

    def num_op(self) -> int:
        # exclude placeholders.
        return self.ir.n_compute_inst()

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

        # init graph placeholders
        SanityCheck.gt(len(self.placeholders), 0)

        def determine_ph_type(ph: str, to_input: bool):
            SanityCheck.true(ph in self.placeholders)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            if to_input:
                ph_inst.iexpr.op = ph_inst.iexpr.op.to_input()
            else:
                ph_inst.iexpr.op = ph_inst.iexpr.op.to_const()

        determine_ph_type(self.placeholders[0], True)  # At lease make one input.
        for ph in self.placeholders[1:]:
            determine_ph_type(ph, random.randint(0, 1))

    def check_sat(self, *assumptions):
        start = time.time()

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
        return random.choice(self.op_candidates)

    def forward_insert_node(self, node: AbsOpBase, input_vars: List[str]) -> InstIR:
        new_inst = self.ir.add_inst(InstExpr(op=node, args=input_vars))

        if isinstance(node, Placeholder):
            # Add placeholder's return varname.
            self.placeholders.append(new_inst.retval())

        return new_inst

    def backward_insert_node(
        self, node, input_vars: List[str], ph_to_replace: List[str]
    ) -> InstIR:
        new_inst = self.forward_insert_node(node, input_vars=input_vars)

        # replace all uses of ph_to_replace
        # and delete the unused placeholders.
        for ph, rv in zip(ph_to_replace, new_inst.retvals()):
            self.ir.replace_alluse(ph, rv)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            self.ir.remove_unused(ph_inst)
            self.placeholders.remove(ph)

        return new_inst

    def try_forward_insert(self, op: AbsOpBase) -> bool:
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

        varnames = self.pick_var_group(
            dim_spec_list,
            op.in_dtypes,
        )

        if self.try_forward_insert_at(op, varnames):
            return True

        return False

    def try_backward_insert(self, op: AbsOpBase):
        # we know that: Y = op(X)
        # S1 - select Y: Y must be a placeholder; (this also means the graph must start w/ a placeholder)
        phvars = self.pick_var_group(
            op.out_ranks,
            op.out_dtypes,
            var_candidates=[
                name
                for name in self.placeholders
                if not isinstance(op, Expand)
                or self.ir.vars[name].ndims < op.expand_last_dim
            ],
        )

        if self.try_occupy_placeholder(op, phvars):
            return True

        return False

    def try_insert_node_type(
        self, node_t: Type[AbsOpBase], max_tensor_pick_time=3
    ) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"@[Node #{self.ir.n_inst()}] <-- "
                f"trying to insert node type {node_t.__name__}"
            )

        try:
            for _ in range(max_tensor_pick_time):
                # should recreate a new instance since some attributes (like axis) should be initialized for each pick
                op_param_n = node_t.get_num_var_param()
                op_id = self.ir.n_inst()
                op_params = [
                    self.new_sym("op%s_%s" % (op_id, k)) for k in range(op_param_n)
                ]

                op: AbsOpBase = node_t(*op_params)

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

    def var_filter(self, ndims, dtype, candidates: List[str]):
        cans = candidates

        cans = list(
            filter(  # filter with ndim
                lambda vname: self.ir.vars[vname].ndims in ndims, cans
            )
        )
        if len(cans) == 0:
            raise RequiredDimNotFound(
                f"Cannot find a shape variable with #dimensions {ndims}."
            )

        if dtype is not None:
            cans = list(
                filter(  # filter with dtype
                    lambda vname: self.ir.vars[vname].dtype == dtype, cans
                )
            )
            if len(cans) == 0:
                raise RequiredDimNotFound(
                    f"Cannot find a shape variable with #dimensions {ndims} and dtype {dtype}."
                )

        return cans

    def pick_var_group(
        self,
        ndim_list: List[Set[int]],
        dtype_combs_spec: List[Tuple[DType, ...]],
        var_candidates: Optional[List[str]] = None,
    ) -> List[str]:
        """Randomly pick a group of variables that satisfy one of the `dtype_combs_spec` and `ndim_list`.

        Returns:
            List[str]: Satisfiable group of variable names.
        """

        if var_candidates is None:
            var_candidates = list(self.ir.vars.keys())

        abs_tensor_candidates = []
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f"Input data types candidates: {dtype_combs_spec}")

        viable_dtypes = []
        for i, ndims in enumerate(ndim_list):
            viable_dtypes.extend(
                [
                    self.ir.vars[vname].dtype
                    for vname in self.var_filter(
                        ndims=ndims, dtype=None, candidates=var_candidates
                    )
                ]
            )
        # only use dtypes currently available after ndim filtering
        dtype_combs = [
            comb for comb in dtype_combs_spec if all(dt in viable_dtypes for dt in comb)
        ]
        if len(dtype_combs) == 0:
            raise RequiredDimNotFound(
                "Op %s: Cannot find a shape variable with dim_spec %s and dtype combinations %s."
                % (ndim_list, dtype_combs_spec)
            )
        dtype_comb = random.choice(dtype_combs)
        for i, ndims in enumerate(ndim_list):
            candidates = self.var_filter(
                ndims=ndims, dtype=dtype_comb[i], candidates=var_candidates
            )
            abs_tensor_candidates.append(random.choice(candidates))

        return abs_tensor_candidates


class PureSymbolGen(SimpleGenerator):
    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.gt_zero():
                constraints.append(c)

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.solver.add(c)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        otensors = [self.ir.vars[name] for name in phvars]

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
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.create_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.gt_zero())

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.gt_zero())

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        for c in constraints:
            self.solver.add(c)

        # succ.
        input_vars = []

        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        node.bind_input_like(itensors)
        node.bind_output_like(inferred_otensors)

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def get_sat_model(self) -> z3.ModelRef:
        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        return self.last_solution


def random_model_gen(
    opset: Set[Type[AbsOpBase]],
    max_nodes=5,
    seed=None,
    timeout_ms=20000,
    **kwargs,
):
    gen = PureSymbolGen(
        opset=opset,
        seed=seed,
        **kwargs,
    )
    gen.abstract_gen(max_node_size=max_nodes, max_gen_millisec=timeout_ms)

    return gen


def viz(ir: GraphIR, filename: str = None):
    if HAS_PYGRAPHVIZ:
        viz_dot(ir.to_dot(), filename)
