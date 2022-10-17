from dataclasses import dataclass
from logging import PlaceHolder
from typing import Dict, List, Tuple

from nnsmith.abstract.op import AbsOpBase, AbsTensor, Constant, Input
from nnsmith.materialize import Instruction, Schedule


@dataclass
class InstIR:
    op: AbsOpBase
    args: Tuple[str]


# Information for a Mutant:
# A graph of operators:
#  - each node is an operator;
#  - all values are concrete;

# TODO(@ganler): migrate NNSmith graph generation to GraphIR.
# - [x] Schedule -> GraphIR
# - [ ] NetworkX -> GraphIR
# ------------------------------------------------------
#                  Graph IR Structure
# ------------------------------------------------------
#       values: Dict[name, AbsTensor]
#       defs:   Dict[name, cps(AbsOp, List[name])]       # FIXME: consider multiple outputs;
#       users:  Dict[name, List[name]]
# ------------------------------------------------------
@dataclass
class GraphIR:
    values: Dict[str, AbsTensor]
    defs: Dict[str, InstIR]
    users: Dict[str, List[str]]

    def __str__(self) -> str:
        ret = ""
        for name, inst in self.defs.items():
            ret += f"{name}\t = {inst.op} :: {inst.args}\n"

        return ret

    def __len__(self):
        return len(self.values)

    def n_compute_ops(self) -> int:
        return sum(
            1
            for inst in self.defs.values()
            if not isinstance(inst.op, (Input, Constant, PlaceHolder))
        )

    def leafs(self) -> List[str]:
        return [name for name in self.values if 0 == len(self.users[name])]

    def expand_users(self, name: str) -> List[str]:
        ret = []

        def dfs(name: str):
            if name in ret:
                return
            ret.append(name)
            for arg in self.users[name]:
                dfs(arg)

        dfs(name)

        return ret[1:]  # The first does not count.

    def to_schedule(self) -> Schedule:
        instructions: List[Instruction] = []

        self.check()

        name2key = {name: i for i, name in enumerate(self.values)}

        for name, inst in self.defs.items():
            this_key = name2key[name]
            inst.op.input_like = [self.values[arg] for arg in inst.args]
            inst.op.output_like = [self.values[name]]
            instructions.append(
                Instruction(
                    op=inst.op,
                    inputs=[name2key[arg] for arg in inst.args],
                    outputs=[this_key],  # FIXME(@ganler): multiple outputs
                )
            )

        return Schedule(
            instructions,
            input_keys=[
                name2key[n]
                for n, inst in self.defs.items()
                if isinstance(inst.op, Input)
            ],
            leaf_keys=[name2key[n] for n in self.leafs()],
            key2type={key: self.values[n] for n, key in name2key.items()},
        )

    @staticmethod
    def from_schedule(schedule: Schedule) -> "GraphIR":
        defs = {}
        values = {}
        users = {}

        for inst in schedule.instructions:
            assert len(inst.outputs) == 1  # FIXME: multiple outputs
            name = str(inst.outputs[0])
            defs[name] = InstIR(op=inst.op, args=tuple(str(arg) for arg in inst.inputs))
            values[name] = schedule.key2type[inst.outputs[0]]
            if name not in users:
                users[name] = []
            for arg in inst.inputs:
                users.setdefault(str(arg), []).append(name)

        graph = GraphIR(values=values, defs=defs, users=users)
        graph.normalize()
        return graph

    def check(self):
        assert (
            set(self.values.keys()) == set(self.defs.keys()) == set(self.users.keys())
        ), "Key inconsistency in `values`, `defs`, and `users`."
        assert self.is_legal(), "Graph is not legal."

    def is_legal(self) -> bool:
        """Check if the graph is legal.

        Returns:
            bool: True if the graph is legal.
        """
        name2idx = {name: i for i, name in enumerate(self.defs)}
        for name, users in self.users.items():
            for user in users:
                if name2idx[name] >= name2idx[user]:
                    return False
        return True

    def normalize(self) -> Dict[str, str]:  # Return name remap.
        """Normalize the GraphIR by sorting and renaming them with topological order where inputs are first.

        Returns:
            Dict[str, str]: The mapping from old names to new names.
        """
        visited = set()
        topo_names = []

        # inputs go first.
        for n, inst in self.defs.items():
            if isinstance(inst.op, Input):
                topo_names.append(n)
                visited.add(n)

        def dfs(name: str):
            if name in visited:
                return
            visited.add(name)
            for arg in self.defs[name].args:
                dfs(arg)
            topo_names.append(name)

        for name in self.defs:
            dfs(name)

        varremap = {}
        for i, name in enumerate(topo_names):
            varremap[name] = f"v{i}"

        self.values = {varremap[name]: self.values[name] for name in topo_names}
        self.defs = {
            varremap[name]: InstIR(
                op=self.defs[name].op,
                args=tuple(varremap[arg] for arg in self.defs[name].args),
            )
            for name in topo_names
        }
        self.users = {
            varremap[name]: [varremap[arg] for arg in self.users[name]]
            for name in topo_names
        }

        return varremap
