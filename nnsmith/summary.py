from collections import Counter, defaultdict
from inspect import signature
import pandas as pd
import numpy as np
import networkx as nx
from nnsmith.abstract import op as Op
import pickle


class SummaryBase:
    def update(self, graph: nx.MultiDiGraph):
        raise NotImplementedError

    def dump(self, output_path):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    def report(self):  # will be recorded after every iteration. can be used to report customized coverage metrics
        return {}


def _record(d, key, itr):
    if key not in d:
        d[key] = []
    d[key].append(itr)


class ParamShapeSummary(SummaryBase):
    def __init__(self) -> None:
        super().__init__()
        # op_name -> {param_name: {value1: itr1, value2: itr2, ...], in_shapes_0: {shape1: itr1, shape2: itr2, ...}, ...}, where itrx is the iteration number of this record
        self.data = {}
        for op_t in Op.ALL_OP_TYPES + [Op.Input, Op.Constant]:
            op_name = op_t.__name__
            self.data[op_name] = {}
            for i in range(len(op_t.in_dtypes[0])):  # arity
                self.data[op_name][f'in_shapes_{i}'] = {}
            nouts = len(op_t.out_dtypes[0])
            for i in range(nouts):  # num_outputs
                self.data[op_name][f'out_shapes_{i}'] = {}
            if issubclass(op_t, Op.Input):
                continue

            if op_t.num_var_param is not None:
                # input is a variable list.
                for i in range(op_t.get_num_var_param()):
                    self.data[op_name][f'param_var{i}'] = {}
            else:
                construct_param_dict = signature(op_t).parameters
                for key in construct_param_dict:
                    self.data[op_name]['param_' + key] = {}

    def update(self, graph: nx.MultiDiGraph, itr):
        for node_id in graph.nodes:
            op = graph.nodes[node_id]['op']  # type: Op.AbsOpBase
            op_name = op.__class__.__name__
            in_svs = graph.nodes[node_id]['in_svs']
            out_svs = graph.nodes[node_id]['out_svs']
            for i, sv in enumerate(in_svs):
                _record(
                    self.data[op_name][f'in_shapes_{i}'], tuple(sv.shape), itr)
            for i, sv in enumerate(out_svs):
                _record(
                    self.data[op_name][f'out_shapes_{i}'], tuple(sv.shape), itr)

            if isinstance(op, Op.Input):
                continue

            construct_param_dict = signature(op.__init__).parameters
            if op.num_var_param is not None:
                # input is a variable list.
                key = list(construct_param_dict.keys())[0]
                vlist = getattr(op, key)
                for i, value in enumerate(vlist):
                    _record(self.data[op_name][f'param_var{i}'], value, itr)
            else:
                for key in construct_param_dict:
                    _record(self.data[op_name]['param_' + key],
                        getattr(op, key), itr)
                

    def dump(self, output_path):
        pickle.dump(self.data, open(output_path, 'wb'))


class GraphSummary(SummaryBase):
    def __init__(self, level=0) -> None:
        """level=0: all leaf nodes; level=1: repeated nodes are merged
        """
        super().__init__()
        self.level = level
        self.node_name = self.merge_nodes(level)

        self.edge_cnt = {}  # edge -> [itr1, itr2, ...]
        self.node_cnt = {}  # node -> [itr1, itr2, ...]
        self.input_comb_cnt = {}  # node -> input_comb -> [itr1, itr2, ...]
        _ALL_OP_TYPES = Op.ALL_OP_TYPES + [Op.Input, Op.Constant]
        for op_t in _ALL_OP_TYPES:
            op_name = self.node_name[op_t.__name__]
            self.node_cnt[op_name] = []
            for op1_t in _ALL_OP_TYPES:
                self.edge_cnt[(self.node_name[op1_t.__name__], op_name)] = []
            self.input_comb_cnt[op_name] = {}

        self.tot_nodes = len(set(self.node_name.values()))
        self.tot_edges = len(self.edge_cnt)
        assert self.tot_edges == self.tot_nodes ** 2

        self.tot_ic_op = {}  # rough computation of total input combinations
        for op_t in _ALL_OP_TYPES:
            op_name = self.node_name[op_t.__name__]
            if op_name not in self.tot_ic_op:
                self.tot_ic_op[op_name] = {}
            arity = len(op_t.in_dtypes[0])
            self.tot_ic_op[op_name][arity] = self.tot_nodes ** arity
        for op_name in self.tot_ic_op:
            self.tot_ic_op[op_name] = sum(self.tot_ic_op[op_name].values())
        self.tot_ic = sum(self.tot_ic_op.values())

    def report(self):
        return {
            'input_comb_cnt': sum(len(v) for k, v in self.input_comb_cnt.items()),
            'edge_cnt': len([_ for _, iters in self.edge_cnt.items() if len(iters) > 0]),
            'node_cnt': len([_ for _, iters in self.node_cnt.items() if len(iters) > 0]),
            # 'tot_input_comb_cnt': self.tot_ic,
            # 'tot_edge_cnt': len(self.edge_cnt),
            # 'tot_node_cnt': len(self.node_cnt)
        }

    def merge_nodes(self, level):
        _ALL_OP_TYPES = Op.ALL_OP_TYPES + [Op.Input, Op.Constant]
        node_name = {}
        for op_t in _ALL_OP_TYPES:
            op_name = op_t.__name__
            node_name[op_name] = op_name

        if level == 0:
            pass
        elif level == 1:
            # merge similar nodes
            sink = Op.EXPANDED_OP
            for s in sink:
                for op_t in _ALL_OP_TYPES:
                    if issubclass(op_t, s):
                        node_name[op_t.__name__] = s.__name__
        return node_name

    def __repr__(self) -> str:
        return super().__repr__() + f'_lv{self.level}'

    def update(self, graph: nx.MultiDiGraph, itr):
        for node_id in graph.nodes:
            op = graph.nodes[node_id]['op']  # type: Op.AbsOpBase
            op_name = self.node_name[op.__class__.__name__]
            _record(self.node_cnt, op_name, itr)
            src_op_names = [self.node_name[graph.nodes[u]['op'].__class__.__name__]
                            for u, _ in graph.in_edges(node_id, data=False)]
            _record(self.input_comb_cnt[op_name], tuple(src_op_names), itr)
            for name in src_op_names:
                _record(self.edge_cnt, (name, op_name), itr)

    def dump(self, output_path):
        pickle.dump({
            'input_comb_cnt': self.input_comb_cnt,
            'edge_cnt': self.edge_cnt,
            'node_cnt': self.node_cnt,
        }, open(output_path, 'wb'))
