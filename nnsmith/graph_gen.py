import logging
import random
import time
import traceback
import warnings
from abc import abstractmethod
from typing import List, Optional, Set, Tuple, Type

import z3

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import *
from nnsmith.abstract.op import (
    AbsOpBase,
    AbsTensor,
    Expand,
    Placeholder,
    concretize_op,
    rank_all,
)
from nnsmith.error import ConstraintError, SanityCheck
from nnsmith.gir import GraphIR, InstExpr, InstIR
from nnsmith.logging import MGEN_LOG, SMT_LOG
from nnsmith.util import HAS_PYGRAPHVIZ, set_seed, viz_dot


def concretize_graph(ir: GraphIR, model: z3.ModelRef) -> GraphIR:
    return ir.concretize(model)


class BaseGen:
    def __init__(
        self,
        opset: List[AbsOpBase],
        seed: Optional[int] = None,
        forward_prob: Optional[float] = None,
        concr_ph_dim_rng: Tuple[int, int] = (1, 64),
        max_elem_per_tensor: int = 2**16,
        rank_choices=None,
        dtype_choices=None,
    ):
        assert len(opset) > 0, "opset must not be empty"
        if seed is not None:
            set_seed(seed)

        self.seed = seed
        self.op_candidates = opset
        self.ir = GraphIR()
        self.monotonic_placeholder_id = 0

        # Names of current placeholders
        self.placeholders: List[str] = []
        # for all (including newly created tmp) placeholders
        self.forward_prob = 0.5 if forward_prob is None else forward_prob
        self.concr_ph_dim_rng = concr_ph_dim_rng
        self.max_elem_per_tensor = max_elem_per_tensor
        self.rank_choices = rank_choices if rank_choices else rank_all()
        # analyze the dtypes used by the opset
        dtype_top = set()
        for op in opset:
            dtype_top.update({dt for dtc in op.in_dtypes + op.out_dtypes for dt in dtc})

        self.dtype_choices = (
            [
                dt if isinstance(dt, DType) else DType.from_str(dt)
                for dt in dtype_choices
            ]
            if dtype_choices
            else DTYPE_GEN_ALL
        )

        self.dtype_choices = list(dtype_top.intersection(self.dtype_choices))
        assert len(self.dtype_choices) > 0, "dtype_choices must not be empty"
        assert len(self.rank_choices) > 0, "rank_choices must not be empty"

    def random_rank(self):
        return random.choice(self.rank_choices)

    def tensor_type_constraints(
        self, atensor: AbsTensor
    ) -> List[Union[z3.BoolRef, bool]]:
        return [atensor.nelement() <= self.max_elem_per_tensor]

    @abstractmethod
    def assume(self, c: Union[z3.BoolRef, bool]):
        pass

    def make_symbolic_placeholder(self, rank, dtype=None) -> Placeholder:
        syms = self.new_syms(
            [f"ph{self.monotonic_placeholder_id}_{k}" for k in range(rank)]
        )
        ph = Placeholder(
            AbsTensor(
                shape=syms,
                dtype=dtype if dtype is not None else self.random_dtype_gen(),
            )
        )
        self.monotonic_placeholder_id += 1
        return ph

    def make_random_concrete_placeholder(self, rank, dtype=None):
        l, r = self.concr_ph_dim_rng
        shape = []
        product = 1
        for _ in range(rank):
            v = random.randint(l, r)
            if product * v > self.max_elem_per_tensor:
                v = 1
            shape.append(v)
            product *= v

        random.shuffle(shape)  # shuffle

        ph = Placeholder(
            AbsTensor(
                shape=shape,
                dtype=dtype if dtype is not None else self.random_dtype_gen(),
            )
        )
        return ph

    def random_dtype_gen(self):
        # more floats than ints.
        # ~ in DTYPE_GEN_ALL and in self.dtype_choices
        dtypes = [dt for dt in DTYPE_GEN_ALL if dt in self.dtype_choices]
        assert (
            len(dtypes) > 0
        ), "Empty INTERSECT(DTYPE_GEN_ALL, dtype_choices). Please relax dtype_choices."

        wts = [1] * len(dtypes)
        for dt in DTYPE_GEN_FLOATS:
            if dt in dtypes:
                wts[dtypes.index(dt)] = 4
        return random.choices(dtypes, weights=wts)[0]

    def new_sym(self, name):
        return z3.Int(name)

    def new_syms(self, names):
        return [self.new_sym(name) for name in names]

    def insert_init_ph_node(self, ph: Placeholder) -> InstIR:
        inst = self.forward_insert_node(ph, [])

        for c in ph.ttype.sym_gt_conc_ge_zero():
            self.assume(c)

        return inst

    @abstractmethod
    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def make_concrete(self) -> GraphIR:
        raise NotImplementedError

    def extra_exit_check(self, max_node_size) -> bool:
        """
        Returns:
            bool: add more checks to determine whether to exit the generation.
        """
        return False

    def num_op(self) -> int:
        # exclude placeholders.
        return self.ir.n_compute_inst()

    def try_insert(self):
        node_t = self.pick_next_op_type()
        self.try_insert_node_type(node_t)

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        z3.set_param("timeout", max_gen_millisec // 3)

        assert max_node_size > 0, "max_node_size must be positive"

        init_time = time.time()

        # starts generation.
        while (
            time.time() - init_time < max_gen_millisec / 1000
            and self.num_op() < max_node_size
        ):
            if self.extra_exit_check(max_node_size):
                break
            self.try_insert()

        # init graph placeholders
        SanityCheck.gt(len(self.placeholders), 0)

        def determine_ph_type(ph: str, to_input: bool):
            SanityCheck.true(ph in self.placeholders)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            if to_input:
                ph_inst.iexpr.op = ph_inst.iexpr.op.input()
            else:
                ph_inst.iexpr.op = ph_inst.iexpr.op.const()

        determine_ph_type(self.placeholders[0], True)  # At lease make one input.
        for ph in self.placeholders[1:]:
            determine_ph_type(ph, random.randint(0, 1))

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
            dim_spec_list, op.in_dtypes, ndim_relation=op.irank_relation
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
            ndim_relation=op.orank_relation,
        )

        if self.try_occupy_placeholder(op, phvars):
            return True

        return False

    def try_insert_node_type(
        self, node_t: Type[AbsOpBase], max_tensor_pick_time=3
    ) -> bool:
        MGEN_LOG.debug(
            f"@[Node #{self.ir.n_inst()}] <-- trying to insert node type {node_t.__name__}"
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
        except ConstraintError:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False

        return False

    def filter_rank_dtype(self, ndims, dtype, candidates: List[str]) -> List[str]:
        cans = candidates

        cans = list(
            filter(  # filter with ndim
                lambda vname: self.ir.vars[vname].ndims in ndims, cans
            )
        )
        if len(cans) == 0:
            raise ConstraintError(f"Cannot find candidate to sat rank of {ndims}.")

        if dtype is not None:
            cans = list(
                filter(  # filter with dtype
                    lambda vname: self.ir.vars[vname].dtype == dtype, cans
                )
            )
            if len(cans) == 0:
                raise ConstraintError(
                    f"Cannot find candidate to sat rank of {ndims} and dtype {dtype}."
                )

        return cans

    def pick_var_group(
        self,
        ndim_list: List[Set[int]],
        dtype_combs: List[Tuple[DType, ...]],
        var_candidates: Optional[List[str]] = None,
        ndim_relation=None,
    ) -> List[str]:
        """Randomly pick a group of variables that satisfy one of the `dtype_combs` and `ndim_list`.
        Returns:
            List[str]: Satisfiable group of variable names.
        """

        if var_candidates is None:
            var_candidates = list(self.ir.vars.keys())

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            for cand in var_candidates:
                MGEN_LOG.debug(
                    f"Candidate: {cand}: {self.ir.vars[cand].dtype} ~ {self.ir.vars[cand].ndims}"
                )
            MGEN_LOG.debug(f"Input data ranks candidates: {ndim_list}")
            MGEN_LOG.debug(f"Input data types candidates: {dtype_combs}")

        ir_dtypes = []
        for i, ndims in enumerate(ndim_list):
            ir_dtypes.extend(
                [
                    self.ir.vars[vname].dtype
                    for vname in self.filter_rank_dtype(
                        ndims=ndims, dtype=None, candidates=var_candidates
                    )
                ]
            )

        # possibility check: must be generatable dtypes.
        if set(ir_dtypes).isdisjoint(set(DTYPE_GEN_ALL)):
            raise ConstraintError(f"Unsupported dtypes: {ir_dtypes}")

        # only use dtypes currently available after ndim filtering
        dcombs = [c for c in dtype_combs if all(dt in ir_dtypes for dt in c)]
        if len(dcombs) == 0:
            raise ConstraintError(
                f"No candidate w/ rank@{ndim_list} & dtype@{dtype_combs}"
            )

        # randomized enumeration over dtype_combs
        random.shuffle(dcombs)
        for dcomb in dcombs:
            if ndim_relation is None:
                ret = []
                for i, ndims in enumerate(ndim_list):
                    candidates = self.filter_rank_dtype(
                        ndims=ndims, dtype=dcomb[i], candidates=var_candidates
                    )
                    ret.append(random.choice(candidates))
                return ret
            else:
                # candidates for 0-indexed tensor
                topcands = self.filter_rank_dtype(
                    ndims=ndim_list[0], dtype=dcomb[0], candidates=var_candidates
                )
                random.shuffle(topcands)
                for tcand in topcands:
                    ret = [tcand]
                    for i, ndims in enumerate(ndim_list[1:]):
                        required_ndim = ndim_relation[i + 1](self.ir.vars[tcand].ndims)
                        if required_ndim not in ndim_list[i + 1]:
                            break
                        self.filter_rank_dtype(
                            ndims=[required_ndim],
                            dtype=dcomb[i + 1],
                            candidates=var_candidates,
                        )
                    if len(ret) == len(ndim_list):
                        return ret

        raise ConstraintError("Cannot find desired combinations of tensor variables.")


def check_sat(solver: z3.Solver, *assumptions) -> z3.CheckSatResult:
    start = time.time()

    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        if solver.assertions():
            SMT_LOG.debug(
                f"existing constraints: {', '.join(map(str, solver.assertions()))}"
            )
        if assumptions:
            SMT_LOG.debug(f"new constraints: {', '.join(map(str, assumptions))}")

    cres = solver.check(*assumptions)

    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(
            f"{cres} <-- checking time: {int((time.time() - start) * 1000)} ms"
        )

        if cres == z3.unsat:
            SMT_LOG.debug(f"Unsat core: {solver.unsat_core()}")

    return cres


def set_z3_state(seed=None):
    z3.set_param(
        "smt.phase_selection",
        5,
        "smt.arith.random_initial_value",
        True,
        "smt.random_seed",
        seed,
        "sat.random_seed",
        seed,
        "sat.phase",
        "random",
        "memory_max_size",
        50 * 1024,  # MB
    )


class SymbolicGen(BaseGen):
    def __init__(
        self,
        opset,
        seed=None,
        init_fp=False,
        symbolic_init=True,
        **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        self.solver = z3.Solver()
        self.last_solution: Optional[z3.ModelRef] = None

        # Insert the first node.
        if symbolic_init:
            ph = self.make_symbolic_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        else:
            ph = self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )

        self.insert_init_ph_node(ph)
        for pred in self.tensor_type_constraints(ph.ttype):
            self.assume(pred)

    def assume(self, c: z3.BoolRef):
        self.solver.add(c)

    def check_sat(self, *assumptions):
        cres = check_sat(self.solver, *assumptions)
        if cres == z3.sat:
            self.last_solution = self.solver.model()
        return cres

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

        # limit output tensor size
        for aten in otensors:
            constraints.extend(self.tensor_type_constraints(aten))

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.assume(c)

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
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.gt_zero())
            constraints.extend(self.tensor_type_constraints(ph.ttype))

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
            self.assume(c)

        # succ.
        input_vars = []

        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        node.bind_input_like(itensors)
        node.bind_output_like(inferred_otensors)

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def make_concrete(self) -> GraphIR:
        SanityCheck.gt(len(self.ir.insts), 0, "Empty graph!")
        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        self.ir.concretize(self.last_solution)
        return self.ir


class SymboliSingleIOGen(SymbolicGen):
    """Generate a model which has only one input and one output tensor"""

    def __init__(self, *args, **kwargs):
        forward_prob = kwargs.pop("forward_prob", None)
        if forward_prob is not None:
            # only warn once
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(
                    "`forward_prob` is not supported in SymboliSingleIOGen which is always 1."
                    "Why: the implementation first just generates forward graph and then cuts backward."
                )
        kwargs["forward_prob"] = 1.0
        super().__init__(*args, **kwargs)

    def eliminate_extra_outputs(self):
        """Find the minimal cut to make the graph has only one output tensor."""
        prev_size = None
        while prev_size != self.ir.n_inst():
            prev_size = self.ir.n_inst()
            cuts = self.ir.leaf_cut_chains()
            cuts = sorted(cuts, key=lambda x: len(x))
            for cut in cuts[:-1]:
                for inst in cut:
                    self.ir.remove_unused(inst)
            self.ir.assert_wellform()
        SanityCheck.eq(len(self.ir.leaf_var()), 1, "Failed to eliminate extra outputs!")

    def abstract_gen(self, **kwargs):
        SymbolicGen.abstract_gen(self, **kwargs)
        self.eliminate_extra_outputs()


class ConcolicGen(BaseGen):
    """Different from SymbolicGen, the graph after an insertion is `concrete` in ConcolicGen.
    However, each step when inserting a node, we symbolically find a satisfiable solution for it.
    """

    def __init__(
        self,
        opset,
        seed=None,
        init_fp=False,
        **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        # Insert the first node.
        self.insert_init_ph_node(
            self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        )

    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        solver = z3.Solver()

        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.sym_gt_conc_ge_zero():
                constraints.append(c)

        check_res = check_sat(solver, *constraints)

        if check_res != z3.sat:
            return False

        # materialize otensors and attributes.
        node = concretize_op(node, solver.model())
        otensors = node.checked_type_transfer(itensors)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        # Shape checker.
        # NOTE: No need to check input shape as they are already checked for being in the graph.
        for i, ten in enumerate(otensors):
            if not all(self.tensor_type_constraints(ten)):
                MGEN_LOG.debug(f"{i}-th output type constraint failed: {ten}")
                return False

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )

        # TODO: In backward insertion, reusing existing tensors is not implemented.

        # Concrete tensors.
        solver = z3.Solver()

        otensors = [self.ir.vars[name] for name in phvars]
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.sym_gt_conc_ge_zero())

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.sym_gt_conc_ge_zero())

        check_res = check_sat(solver, *constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        model = solver.model()
        # succ.
        itensors = []
        for i, ph in enumerate(phs_as_op_inputs):
            # materialize ph
            phs_as_op_inputs[i] = concretize_op(ph, model)
            itensors.append(phs_as_op_inputs[i].ttype)

        # Input Shape checker.
        # NOTE: No need to check output because they are already in the graph thus valid.
        for i, ten in enumerate(itensors):
            if not all(self.tensor_type_constraints(ten)):
                MGEN_LOG.debug(f"{i}-th input type constraint failed: {ten}")
                return False

        node = concretize_op(node, model)
        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        # Apply insertion.
        input_vars = []
        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def assume(self, c: bool):
        # semantically equivalent to `assert c`.
        ConstraintCheck.true(c, "Assumption failed")

    def make_concrete(self) -> GraphIR:
        return self.ir


def model_gen(
    opset: Set[Type[AbsOpBase]],
    method: str = "symbolic",
    max_nodes=5,
    seed=None,
    timeout_ms=10000,
    **kwargs,
):
    assert max_nodes > 0, "max_nodes must >= 1"

    symbolic_init = not method.endswith("-cinit")
    if method.startswith("symbolic"):
        gen = SymbolicGen(opset, seed, symbolic_init=symbolic_init, **kwargs)
    elif "concolic" == method:
        gen = ConcolicGen(opset, seed, **kwargs)
    elif method.startswith("single-io"):
        gen = SymboliSingleIOGen(opset, seed, symbolic_init=symbolic_init, **kwargs)
    else:
        raise ValueError(f"Unknown method {method}. Try `symbolic` or `concolic`.")

    gen.abstract_gen(max_node_size=max_nodes, max_gen_millisec=timeout_ms)

    return gen


def viz(ir: GraphIR, filename: str = None):
    if HAS_PYGRAPHVIZ:
        viz_dot(ir.to_dot(), filename)
