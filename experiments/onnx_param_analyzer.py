"""Just to figure out operators types and connections.
"""

from collections import Counter
import os
from multiprocessing import Pool, cpu_count

import onnx
import pandas as pd
import pickle

OP_PARAM_NAMES = {  # use as a sanity check. All ops should have the same set of param names.
    'Conv': ['kernel_shape', 'strides', 'pads', 'dilations'],
}

UNDEFINED = 0
FLOAT = 1
INT = 2
STRING = 3
TENSOR = 4
GRAPH = 5
SPARSE_TENSOR = 11
TYPE_PROTO = 13

FLOATS = 6
INTS = 7
STRINGS = 8
TENSORS = 9
GRAPHS = 10
SPARSE_TENSORS = 12
TYPE_PROTOS = 14


OP_TO_SHOW = ['Conv', 'Dense', 'Pad', 'Reshape']


def parse_attr(attr):
    # message AttributeProto {
    #   // Note: this enum is structurally identical to the OpSchema::AttrType
    #   // enum defined in schema.h.  If you rev one, you likely need to rev the other.
    #   enum AttributeType {
    #     UNDEFINED = 0;
    #     FLOAT = 1;
    #     INT = 2;
    #     STRING = 3;
    #     TENSOR = 4;
    #     GRAPH = 5;
    #     SPARSE_TENSOR = 11;
    #     TYPE_PROTO = 13;

    #     FLOATS = 6;
    #     INTS = 7;
    #     STRINGS = 8;
    #     TENSORS = 9;
    #     GRAPHS = 10;
    #     SPARSE_TENSORS = 12;
    #     TYPE_PROTOS = 14;
    #   }
    #   optional string name = 1;           // namespace Attribute
    #   optional string ref_attr_name = 21;
    #   optional string doc_string = 13;
    #   optional AttributeType type = 20;   // discriminator that indicates which field below is in use
    #   optional float f = 2;               // float
    #   optional int64 i = 3;               // int
    #   optional bytes s = 4;               // UTF-8 string
    #   optional TensorProto t = 5;         // tensor value
    #   optional GraphProto g = 6;          // graph
    #   optional SparseTensorProto sparse_tensor = 22;  // sparse tensor value
    #   optional TypeProto tp = 14;          // type proto

    #   repeated float floats = 7;          // list of floats
    #   repeated int64 ints = 8;            // list of ints
    #   repeated bytes strings = 9;         // list of UTF-8 strings
    #   repeated TensorProto tensors = 10;  // list of tensors
    #   repeated GraphProto graphs = 11;    // list of graph
    #   repeated SparseTensorProto sparse_tensors = 23; // list of sparse tensors
    #   repeated TypeProto type_protos = 15;// list of type protos
    # }
    if not hasattr(attr, 'type'):
        print('warning: no type found for', attr)
        return None
    if attr.type == INT:
        return attr.i
    elif attr.type == INTS:
        return tuple(int(i) for i in attr.ints)
    elif attr.type == TENSOR:
        return None  # skip
    elif attr.type == STRING:
        return None  # skip
    elif attr.type == FLOAT:
        return None  # skip
    else:  # TODO: handle subgraph...
        print(f'Warning: skip type `{attr.type}` for\n{attr}')
        return None


def to_tuple(op_type, attrs):
    global OP_PARAM_NAMES
    if op_type not in OP_PARAM_NAMES:
        OP_PARAM_NAMES[op_type] = sorted(attrs.keys())
    else:
        if sorted(attrs.keys()) != OP_PARAM_NAMES[op_type]:
            print(
                f'Warning: {op_type} has different param names, {sorted(attrs.keys())} vs {OP_PARAM_NAMES[op_type]}')
    return tuple(sorted(attrs.items(), key=lambda x: x[0]))


def analyze_one_count(model_path):
    if 'FAILURE' in model_path:
        return {}

    model = onnx.load(model_path)

    info = {}
    for node in model.graph.node:
        # print(node)
        if node.op_type not in info:
            info[node.op_type] = Counter()

        attrs = {}
        for attr in node.attribute:
            v = parse_attr(attr)
            if v is not None:
                attrs[attr.name] = v
        t = to_tuple(node.op_type, attrs)
        # print('attrs=\n', attrs)
        # print('tuple=', t)
        assert t == tuple(t), f'{t} not comparable'
        info[node.op_type].update({t: 1})

    ish = Counter()
    for node in model.graph.input:
        if not hasattr(node.type, 'tensor_type'):
            print('Warning: skip unknown input:\n', node)
            continue
        shape = tuple(int(dim.dim_value)
                      for dim in node.type.tensor_type.shape.dim)
        # print(shape)
        ish.update({shape: 1})
    res = {'input shape': ish}
    for op in OP_TO_SHOW:
        res[op + '_param'] = info.get(op, Counter())
    return res


def analyze_one(*args, **kwargs):
    a = analyze_one_count(*args, **kwargs)
    # drop the frequency information
    return {k: set(v.keys()) for k, v in a.items()}


def analyze_folders(folders, cache_dir=None, force=False, n_limit=None):
    res = []

    __CACHE_FILE__ = 'onnx_analysis_cache.pkl'
    for folder in folders:
        if os.path.exists(os.path.join(cache_dir, __CACHE_FILE__)) and not force and cache_dir is not None:
            print('===> {} already exists.'.format(
                os.path.join(folder, __CACHE_FILE__)))
            with open(os.path.join(cache_dir, __CACHE_FILE__), 'rb') as fp:
                return pickle.load(fp)

    times = []
    file_hubs = []
    least_time = None
    assert n_limit is None or len(n_limit) == len(folders)
    for i, folder in enumerate(folders):
        df = pd.read_csv(os.path.join(folder, 'gentime.csv'),
                         usecols=[0, 1], header=None)
        ts = df[0].to_numpy()

        times.append(ts)
        files = df[1].tolist()
        if n_limit is not None:
            files = files[:n_limit[i]]
        files = [os.path.join(folder, f) for f in files]
        file_hubs.append(files)

        time_cost = ts.sum()
        if least_time is None or time_cost < least_time:
            least_time = time_cost

    # only consider rows that < least_time
    for i, ts in enumerate(times):
        file_hubs[i] = file_hubs[i][:(ts < least_time).sum()]

    for files in file_hubs:
        sets = {}
        for new in map(analyze_one, files):
            for k, v in new.items():
                if k not in sets:
                    sets[k] = set()
                sets[k] |= v
        res.append(sets)

    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, __CACHE_FILE__), 'wb') as fp:
            pickle.dump(res, fp)
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=str, nargs='+', required=True)
    parser.add_argument('--tags', type=str, nargs='+', default=None)
    # should compare models within same generation duration.
    parser.add_argument('--nlim', type=int, nargs='+', default=None)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split('-')[0] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    results = analyze_folders(
        args.folders, cache_dir=args.output, force=args.force, n_limit=args.nlim)
    for tag, sets in zip(args.tags, results):
        print(f'{tag}:\t')
        for k, v in sets.items():
            print(f'\t{k}: {len(v)}')

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

    node_list = []  # TODO: should call it params
    edge_list = []  # TODO: should call it input shapes
    for i in range(len(results)):
        nodes = results[i]['Conv_param']  # TODO: more ops
        edges = results[i]['input shape']
        node_list.append(nodes)
        edge_list.append(edges)

    if len(node_list) == 2:
        venn2(subsets=node_list, set_labels=[
              f'$\\bf{{{t}}}$' for t in args.tags])
        venn2_circles(subsets=node_list, linestyle='dashed')
    elif len(node_list) == 3:
        v = venn3(subsets=node_list, set_labels=[
                  f'$\\bf{{{t}}}$' for t in args.tags])
        hatches = ['\\', '.', '*']
        circles = ['MediumVioletRed', 'SeaGreen', 'Lavender']
        for idx, id in enumerate(['100', '010', '001', '111']):
            if v.get_label_by_id(id) is None:
                continue

            cnt = int(v.get_label_by_id(id).get_text())

            if id != '111':
                v.get_patch_by_id(id).set_alpha(0.5)
                v.get_patch_by_id(id).set_hatch(hatches[idx])
                v.get_patch_by_id(id).set_edgecolor(circles[idx])
                v.get_patch_by_id(id).set_linewidth(2)
                v.get_patch_by_id(id).set_linestyle('--')

    plt.title("Venn Diagram of Covered ONNX Operators")
    plt.savefig(
        f'{os.path.join(args.output, "onnx_node_venn")}.png', bbox_inches='tight')
    plt.savefig(
        f'{os.path.join(args.output, "onnx_node_venn")}.pdf', bbox_inches='tight')
    plt.close()

    if len(node_list) == 2:
        venn2(subsets=edge_list, set_labels=[
              f'$\\bf{{{t}}}$' for t in args.tags])
        venn2_circles(subsets=edge_list, linestyle='dashed')
    elif len(node_list) == 3:
        v = venn3(subsets=edge_list, set_labels=[
                  f'$\\bf{{{t}}}$' for t in args.tags])
        hatches = ['\\', '.', '*']
        circles = ['MediumVioletRed', 'SeaGreen', 'Lavender']
        for idx, id in enumerate(['100', '010', '001', '111']):
            if v.get_label_by_id(id) is None:
                continue

            cnt = int(v.get_label_by_id(id).get_text())

            if id != '111':
                v.get_patch_by_id(id).set_alpha(0.5)
                v.get_patch_by_id(id).set_hatch(hatches[idx])
                v.get_patch_by_id(id).set_edgecolor(circles[idx])
                v.get_patch_by_id(id).set_linewidth(2)
                v.get_patch_by_id(id).set_linestyle('--')
    plt.title("Venn Diagram of Covered ONNX Operators Edges")
    plt.savefig(
        f'{os.path.join(args.output, "onnx_edge_venn")}.png', bbox_inches='tight')
    plt.savefig(
        f'{os.path.join(args.output, "onnx_edge_venn")}.pdf', bbox_inches='tight')
    plt.close()
