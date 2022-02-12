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


class ParamShapeSummary(SummaryBase):
    def __init__(self) -> None:
        super().__init__()
        self.data = {}
        for op_t in Op.ALL_OP_TYPES:
            op_name = op_t.__name__
            self.data[op_name] = {}
            for i in range(len(op_t.in_dtypes[0])):  # arity
                self.data[op_name][f'in_shapes_{i}'] = Counter()
            construct_param_dict = signature(op_t).parameters
            for key in construct_param_dict:
                self.data[op_name]['param_' + key] = Counter()

    def update(self, graph: nx.MultiDiGraph):
        for node_id in range(len(graph.nodes)):
            op = graph.nodes[node_id]['op']  # type: Op.AbsOpBase
            if isinstance(op, Op.Input):
                continue
            op_name = op.__class__.__name__
            in_svs = graph.nodes[node_id]['in_svs']
            # out_svs = graph.nodes[node_id]['out_svs']
            for i, sv in enumerate(in_svs):
                self.data[op_name][f'in_shapes_{i}'].update(
                    {tuple(sv.shape): 1})

            construct_param_dict = signature(op.__init__).parameters
            for key in construct_param_dict:
                self.data[op_name]['param_' +
                                   key].update({getattr(op, key): 1})

    def dump(self, output_path):
        pickle.dump(self.data, open(output_path, 'wb'))


class GraphSummary(SummaryBase):
    def __init__(self, level=0) -> None:
        """level=0: all leaf nodes; level=1: repeated nodes are merged
        """
        super().__init__()
        self.level = level
        self.node_name = self.merge_nodes(level)

        self.edge_cnt = Counter()
        self.node_cnt = Counter()
        self.input_comb_cnt = {}
        _ALL_OP_TYPES = Op.ALL_OP_TYPES + [Op.Input]
        for op_t in _ALL_OP_TYPES:
            op_name = self.node_name[op_t.__name__]
            self.node_cnt.update({op_name: 0})
            for op1_t in _ALL_OP_TYPES:
                self.edge_cnt.update(
                    {(self.node_name[op1_t.__name__], op_name): 0})
            self.input_comb_cnt[op_name] = Counter()

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
            'edge_cnt': len([_ for _, count in self.edge_cnt.items() if count > 0]),
            'node_cnt': len([_ for _, count in self.node_cnt.items() if count > 0]),
            'tot_input_comb_cnt': self.tot_ic,
            'tot_edge_cnt': len(self.edge_cnt),
            'tot_node_cnt': len(self.node_cnt)
        }

    def merge_nodes(self, level):
        _ALL_OP_TYPES = Op.ALL_OP_TYPES + [Op.Input]
        node_name = {}
        for op_t in _ALL_OP_TYPES:
            op_name = op_t.__name__
            node_name[op_name] = op_name

        if level == 0:
            pass
        elif level == 1:
            # merge concat
            sink = [Op.Concat, Op.Constant, Op.Expand, Op.Reshape, Op.ArgMax,
                    Op.ArgMin, Op.ReduceMax, Op.ReduceMin, Op.ReduceMean, Op.SqueezeBase,
                    Op.ReduceSum]
            for s in sink:
                for op_t in _ALL_OP_TYPES:
                    if issubclass(op_t, s):
                        node_name[op_t.__name__] = s.__name__
        return node_name

    def __repr__(self) -> str:
        return super().__repr__() + f'_lv{self.level}'

    def update(self, graph: nx.MultiDiGraph):
        for node_id in range(len(graph.nodes)):
            op = graph.nodes[node_id]['op']  # type: Op.AbsOpBase
            op_name = self.node_name[op.__class__.__name__]
            self.node_cnt.update({op_name: 1})
            src_op_names = [self.node_name[graph.nodes[u]['op'].__class__.__name__]
                            for u, _ in graph.in_edges(node_id, data=False)]
            self.input_comb_cnt[op_name].update({tuple(src_op_names): 1})
            for name in src_op_names:
                self.edge_cnt.update({(name, op_name): 1})

    def dump(self, output_path):
        pickle.dump({
            'input_comb_cnt': self.input_comb_cnt,
            'edge_cnt': self.edge_cnt,
            'node_cnt': self.node_cnt,
        }, open(output_path, 'wb'))
