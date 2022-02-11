from collections import Counter
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
