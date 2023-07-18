from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from z3 import ModelRef

from nnsmith.abstract.op import (
    AbsOpBase,
    AbsTensor,
    Constant,
    Input,
    Placeholder,
    concretize_op,
)
from nnsmith.logging import CORE_LOG


@dataclass
class InstExpr:
    op: Union[AbsOpBase, Placeholder]
    args: Tuple[str]

    def __str__(self):
        return f"{self.op}({', '.join(self.args)})"

    def n_input(self):
        assert len(self.args) == self.op.n_input()
        return len(self.args)

    def n_output(self):
        return self.op.n_output()


def _make_new_id_from_used(used_ids: Set[int]) -> int:
    for i in range(2**16):
        if i not in used_ids:
            return i

    raise RuntimeError("Cannot find a new id within [0, 2**16)")


def id_maker(fallback, irctx: Union["GraphIR", List["InstIR"]] = None):
    if isinstance(irctx, GraphIR):
        return _make_new_id_from_used({inst.identifier for inst in irctx.insts})
    elif isinstance(irctx, list):
        return _make_new_id_from_used({inst.identifier for inst in irctx})
    else:
        return fallback


def id_checker(id):
    assert id >= 0, "Identifier must be non-negative"


VARNAME_SPLITTER = "_"


class InstIR:
    def __init__(
        self,
        iexpr: InstExpr,
        identifier: Optional[int] = None,
        irctx: Union["GraphIR", List["InstIR"]] = None,
    ) -> None:
        self.iexpr = iexpr
        self.users: List[Set[InstIR]] = [set() for _ in range(self.iexpr.n_output())]
        if identifier is None:
            self.identifier = id_maker(fallback=id(self), irctx=irctx)
        else:
            id_checker(identifier)
            self.identifier = identifier

        if irctx is not None:
            if isinstance(irctx, GraphIR):
                for arg in set(self.iexpr.args):
                    inst_id, ret_idx = InstIR.var_inst_idx(arg)
                    irctx.find_inst_by_id(inst_id).users[ret_idx].add(self)
            elif isinstance(irctx, List):
                for arg in set(self.iexpr.args):
                    inst_id, ret_idx = InstIR.var_inst_idx(arg)
                    for inst in irctx:
                        if inst.identifier == inst_id:
                            inst.users[ret_idx].add(self)

    def __str__(self):
        return (
            f"{', '.join(self.retvals())} = {self.iexpr} \t# inst id: {self.identifier}"
        )

    def no_users(self):
        return all(len(u) == 0 for u in self.users)

    def leaf_var(self) -> List[str]:
        ret = []
        for idx, users in enumerate(self.users):
            if len(users) == 0:
                ret.append(self.retval(idx))
        return ret

    def n_input(self):
        return self.iexpr.n_input()

    def n_output(self):
        return self.iexpr.n_output()

    @staticmethod
    def var_inst_idx(varname: str) -> Tuple[int, int]:
        # parse v[inst-id]_[index]
        tokens = varname[1:].split(VARNAME_SPLITTER)
        return int(tokens[0]), int(tokens[1])

    @staticmethod
    def retval_string(inst_id: int, ret_idx: int) -> str:
        return f"v{inst_id}{VARNAME_SPLITTER}{ret_idx}"

    def retval(self, index=0) -> str:
        assert index < self.n_output(), f"Only has {self.n_output()} outputs in {self}"
        return self.retval_string(self.identifier, index)

    def retvals(self) -> List[str]:
        return [self.retval(i) for i in range(self.n_output())]

    def is_user_of(self, inst: "InstIR", ret_idx: Optional[int] = None) -> bool:
        usee_names = list(inst.retvals())
        if ret_idx is not None:
            if ret_idx >= len(usee_names) or ret_idx < 0:
                raise ValueError(
                    f"Invalid ret_idx: {ret_idx}. Valid domain is [0, {len(usee_names)})"
                )
            return usee_names[ret_idx] in self.iexpr.args
        else:
            return any(u in self.iexpr.args for u in usee_names)


# -----------------------------------------------------------------
#                       Graph IR Structure
# -----------------------------------------------------------------
#       vars:    Dict[ var_name, AbsTensor      ]
#       insts:   List[cps(AbsOp, List[var_name])]
# -----------------------------------------------------------------
#                         Well-formedness
# -----------------------------------------------------------------
#  1. Return name: ${inst.id}_${index(ret)}. e.g., op[0]_0, ...
#  3. Return name is unique.
# -----------------------------------------------------------------


@dataclass
class GraphIR:
    vars: Dict[str, AbsTensor] = field(default_factory=dict)  # VarName  -> AbsTensor
    insts: List[InstIR] = field(
        default_factory=list
    )  # Index -> InstIR := <ID, InstExpr, RetNames..>

    def __str__(self) -> str:
        ret = ""
        for inst in self.insts:
            ret += f"{inst}\n"

        return ret

    def pretty(self) -> str:
        inst_remap = {inst.identifier: f"{idx}" for idx, inst in enumerate(self.insts)}

        ret = ""
        for inst in self.insts:
            pretty_args = []
            for arg in inst.iexpr.args:
                inst_id, ret_idx = InstIR.var_inst_idx(arg)
                pretty_args.append(InstIR.retval_string(inst_remap[inst_id], ret_idx))
            pretty_retvals = [
                InstIR.retval_string(inst_remap[inst.identifier], ret_idx)
                for ret_idx in range(inst.n_output())
            ]

            ret += f"{', '.join(pretty_retvals)} = {inst.iexpr.op}({', '.join(pretty_args)})"
            ret += f" \t# inst id: {inst_remap[inst.identifier]}\n"  # Comment

        return ret

    def n_inst(self) -> int:
        return len(self.insts)

    def n_compute_inst(self) -> int:
        return sum(
            1
            for inst in self.insts
            if not isinstance(inst.iexpr.op, (Input, Constant, Placeholder))
        )

    def n_var(self) -> int:
        return len(self.vars)

    def leaf_inst(self) -> List[InstIR]:
        return [inst for inst in self.insts if inst.no_users()]

    def leaf_var(self) -> List[str]:
        # Non-leaf instructions can produce leaf variables.
        lvs = []
        for inst in self.insts:
            for lv in inst.leaf_var():
                if lv not in lvs:
                    lvs.append(lv)
        return lvs

    def input_var(self) -> List[str]:
        return [
            inst.retval() for inst in self.insts if isinstance(inst.iexpr.op, Input)
        ]

    def add_inst(self, iexpr: InstExpr) -> InstIR:
        new_inst = InstIR(iexpr, irctx=self)

        # Infer the output type if iexpr.op is not binded.
        otensors = iexpr.op.output_like
        if any([t is None for t in otensors]):
            otensors = iexpr.op.checked_type_transfer(
                [self.vars[arg] for arg in iexpr.args]
            )

        # make new values
        for ridx, abstensor in enumerate(otensors):
            vname = new_inst.retval(ridx)
            assert vname not in self.vars, "Variable name is not unique: " + vname
            self.vars[vname] = abstensor

        min_user_idx = 0
        # update users
        for arg in set(iexpr.args):
            assert arg in self.vars, "Variable not defined: " + arg
            inst_id, ret_idx = InstIR.var_inst_idx(arg)
            for idx, may_prod in enumerate(self.insts):
                if inst_id == may_prod.identifier:
                    may_prod.users[ret_idx].add(new_inst)
                    min_user_idx = max(min_user_idx, idx + 1)
                    break

        # insert it to somewhere that follows good topological order
        # i.e., right after the last arg.
        self.insts.insert(min_user_idx, new_inst)
        return new_inst

    def find_inst_by_id(self, obj_id: int) -> Optional[InstIR]:
        for inst in self.insts:
            if inst.identifier == obj_id:
                return inst
        return None

    def replace_alluse(self, oldvar: str, newvar: str, type_check=True) -> None:
        # Change one variable to another new variable.
        assert oldvar in self.vars, "Old var undefined: " + oldvar
        assert newvar in self.vars, "New var undefined: " + newvar
        # check type
        if (
            type_check
            and self.vars[oldvar] is not None
            and self.vars[newvar] is not None
        ):
            assert self.vars[oldvar].weak_compare(
                self.vars[newvar]
            ), f"Type mismatch: {self.vars[oldvar]} != {self.vars[newvar]}"
        # 1. replace all user site of oldvar to newvar.
        old_inst_id, old_ret_idx = InstIR.var_inst_idx(oldvar)
        old_inst = self.find_inst_by_id(old_inst_id)
        assert old_inst is not None, "Old var undefined: " + oldvar
        for ouser in old_inst.users[old_ret_idx]:
            # change all use of oldvar to newvar
            ouser.iexpr.args = [newvar if a == oldvar else a for a in ouser.iexpr.args]
        # 2. change users of oldvar -> newvar.
        new_inst_id, new_ret_idx = InstIR.var_inst_idx(newvar)
        new_inst = self.find_inst_by_id(new_inst_id)
        new_inst.users[new_ret_idx] = old_inst.users[old_ret_idx]
        old_inst.users[old_ret_idx] = set()  # reset

    def replace_arg(self, inst: InstIR, arg_idx: int, newvar: str, type_check=True):
        # Change one variable to another new variable.
        assert newvar in self.vars, "New var undefined: " + newvar
        assert (
            0 <= arg_idx < len(inst.iexpr.args)
        ), f"Invalid argument index {arg_idx} for {inst}"
        oldvar = inst.iexpr.args[arg_idx]

        # check type
        if (
            type_check
            and self.vars[oldvar] is not None
            and self.vars[newvar] is not None
        ):
            assert (
                self.vars[oldvar] == self.vars[newvar]
            ), f"Type mismatch: {self.vars[oldvar]} != {self.vars[newvar]}"
        # 1. replace current user site of oldvar to newvar.
        inst.iexpr.args[arg_idx] = newvar
        # 2. dec ref for oldvar
        if oldvar not in inst.iexpr.args:
            old_inst_id, old_ret_idx = InstIR.var_inst_idx(oldvar)
            old_inst = self.find_inst_by_id(old_inst_id)
            old_inst.users[old_ret_idx].remove(inst)
        # 3. inc ref for newvar
        new_inst_id, new_ret_idx = InstIR.var_inst_idx(newvar)
        new_inst = self.find_inst_by_id(new_inst_id)
        new_inst.users[new_ret_idx].add(inst)

    def remove_unused(self, inst: InstIR) -> None:
        # Remove an instruction which is deemed unused.
        assert inst in self.insts, f"Instruction not in graph: {inst}"
        assert inst.no_users(), f"{inst} has users {inst.users}."
        # remove users
        for other in self.insts:
            if other != inst:
                for users in other.users:
                    if inst in users:
                        users.remove(inst)
        # remove values
        for val in inst.retvals():
            del self.vars[val]
        # remove inst
        self.insts.remove(inst)

    def assert_wellform(self):
        # TODO: Check connectivity.
        defined = set()
        for inst in self.insts:
            for arg in inst.iexpr.args:
                assert arg in self.vars, f"Variable not defined: {arg}"
                assert arg in defined, f"Variable not defined yet: {arg}"
                # check usee.
                usee_id, ret_idx = InstIR.var_inst_idx(arg)
                usee = self.find_inst_by_id(usee_id)
                assert (
                    inst in usee.users[ret_idx]
                ), f"Use-Def chain broken: {usee} should be used by {inst}"

            for rv in inst.retvals():
                assert rv in self.vars, f"Return var not in self.vars: {rv}"
                assert rv not in defined, f"Variable re-defined: {rv}"

            # check user.
            for ret_idx, users in enumerate(inst.users):
                val = inst.retval(ret_idx)
                for user in users:
                    assert (
                        val in user.iexpr.args
                    ), f"Use-Def chain broken: {inst} should be used by {user}"

            defined.update(inst.retvals())

    def _topological_sort(self):
        defined = set()

        def swap(i, j):
            self.insts[i], self.insts[j] = self.insts[j], self.insts[i]

        ptr = 0
        while ptr < len(self.insts):
            frontier = []
            for idx in range(ptr, len(self.insts)):
                inst = self.insts[idx]
                if all(arg in defined for arg in inst.iexpr.args):
                    frontier.append(idx)
            if len(frontier) == 0:
                CORE_LOG.error(f"Bad IR:\n{self.pretty()}")
                raise RuntimeError("Cyclic dependency detected.")
            for idx in frontier:
                swap(ptr, idx)
                defined.update(self.insts[ptr].retvals())
                ptr += 1

    def _udchain_repair(self):
        for inst in self.insts:
            # Add used;
            for arg in inst.iexpr.args:
                usee_id, ret_idx = InstIR.var_inst_idx(arg)
                usee = self.find_inst_by_id(usee_id)
                usee.users[ret_idx].add(inst)
            # Remove unused;
            for ret_idx, users in enumerate(inst.users):
                val = inst.retval(ret_idx)
                for user in list(users):
                    if val not in user.iexpr.args:
                        users.remove(user)

    def wellform_repair(self):
        # 1. Repair use-def chain;
        self._udchain_repair()
        # 2. Repair topological order;
        self._topological_sort()

    def concretize(self, model: ModelRef) -> None:
        """Concretize self with a z3 model."""
        for inst in self.insts:
            # Concretize operators
            op = concretize_op(inst.iexpr.op, model)

            # Concretize output tensors;
            itensors = [self.vars[vname] for vname in inst.iexpr.args]
            otensors = op.checked_type_transfer(itensors)
            op.bind_input_like(itensors)
            op.bind_output_like(otensors)

            # Write back op to insts.
            inst.iexpr.op = op

            # Write back tensors to vars.
            for vname, tensor in zip(inst.retvals(), otensors):
                self.vars[vname] = tensor

    def to_dot(self) -> str:
        """Convert to graphviz dot format."""
        # digraph D {
        #     node [shape=Mrecord];
        #     Input [label="{Input|<f0> out}"];
        #     Add   [label="{{<f0> lhs| <f1> rhs}|Add|{<f2> out}}"];
        #     Input:f0 -> Add:f0;
        #     Input:f0 -> Add:f1;
        # }
        text = "digraph D {\n"
        text += "  node [shape=Mrecord];\n"

        def render_node(inst: InstIR):
            label = "{"
            extra = ""
            if not isinstance(inst.iexpr.op, (Input, Constant, Placeholder)):
                # Inputs
                label += "{"
                label += "|".join(
                    [f"<i{idx}> {arg}" for idx, arg in enumerate(inst.iexpr.args)]
                )
                label += "}|"
            elif isinstance(inst.iexpr.op, Input):
                extra += "fillcolor=cadetblue1,style=filled,"
            elif isinstance(inst.iexpr.op, Constant):
                extra += "fillcolor=lightpink,style=filled,"
            elif isinstance(inst.iexpr.op, Placeholder):
                extra += "fillcolor=lightgray,style=filled,"

            # Op
            label += f"{inst.iexpr.op}|".replace("{", r"\{").replace("}", r"\}")
            # Outputs
            label += "{"
            label += "|".join(
                [f"<o{idx}> {rv}" for idx, rv in enumerate(inst.retvals())]
            )

            label += r"}}"

            return f'  {inst.identifier} [label="{label}",{extra}];\n'

        # render nodes
        for inst in self.insts:
            text += render_node(inst)

        # render edges
        for inst in self.insts:
            for idx, arg in enumerate(inst.iexpr.args):
                usee_id, ret_idx = InstIR.var_inst_idx(arg)
                text += f'  {usee_id}:o{ret_idx} -> {inst.identifier}:i{idx} [label="{self.vars[arg].pretty()}"];\n'

        text += "}\n"
        return text

    def leaf_cut_chains(self) -> List[List[InstIR]]:
        cuts = []
        for leaf in self.leaf_inst():
            assert (
                len(leaf.retvals()) == 1
            ), "find_leaf_cut_chains only support single-output operators"
            pivot = 0
            removed = [leaf]
            while pivot < len(removed):
                inst = removed[pivot]
                pivot += 1
                for arg in inst.iexpr.args:
                    inst_id, _ = InstIR.var_inst_idx(arg)
                    arg_inst = self.find_inst_by_id(inst_id)
                    if arg_inst.users[0].issubset(removed) and arg_inst not in removed:
                        removed.append(arg_inst)
            cuts.append(removed)
        return cuts
