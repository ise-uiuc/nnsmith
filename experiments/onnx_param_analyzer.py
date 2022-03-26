"""Just to figure out operators types and connections.
"""

from collections import Counter
import os
from multiprocessing import Pool, cpu_count
import traceback
from typing import Dict

import onnx
import pandas as pd
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from onnx import helper, shape_inference

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
        print('warning: no type found for', attr.name)
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
    elif attr.type == GRAPH:  # TODO: handle subgraph...
        return None
    else:
        print(f'Warning: skip type `{attr.type}` for\n{attr}')
        return None


def attr_to_tuple(op_type, attrs):
    return tuple(sorted(attrs.items(), key=lambda x: x[0]))


def ish_to_tuple(ish):
    return tuple(sorted(ish.items(), key=lambda x: x[0]))


def analyze_shape(model):
    ret: Dict[str, tuple] = {}  # name->shape
    model = shape_inference.infer_shapes(model, True, True, True)
    for vi in model.graph.value_info:
        if not hasattr(vi.type, 'tensor_type'):
            print('Warning: skip non-tensor ValueInfo: ', vi)
            continue
        dims = vi.type.tensor_type.shape.dim
        ret[vi.name] = tuple(int(dim.dim_value) for dim in dims)

    for node in list(model.graph.input) + list(model.graph.output):
        if not hasattr(node.type, 'tensor_type'):
            print('Warning: skip unknown input/output:\n', node)
            continue
        shape = tuple(int(dim.dim_value)
                      for dim in node.type.tensor_type.shape.dim)
        assert node.name not in ret, f'{node.name} already in ret, {node}'
        ret[node.name] = shape

    for node in model.graph.initializer:
        shape = tuple(int(dim)
                      for dim in node.dims)
        if node.name in ret:
            assert ret[node.name] == shape, f'{node.name} already in ret but shape mismatch, {node} vs {ret[node.name]}'
        ret[node.name] = shape
    return ret


def analyze_one(model_path):
    if 'FAILURE' in model_path:
        return {}
    try:
        model = onnx.load(model_path)
        shape_dict = analyze_shape(model)
    except Exception as e:
        print('-------------> Skip model', model_path, 'due to exception:')
        traceback.print_exc()
        return

    info = {}
    for node in model.graph.node:
        # print(node)

        attrs = {}
        for attr in node.attribute:
            v = parse_attr(attr)
            if v is not None:
                attrs[attr.name] = v
        attr_t = attr_to_tuple(node.op_type, attrs)

        ish = {idx: shape_dict[i]
               for idx, i in enumerate(node.input) if i != ""}
        ish_t = ish_to_tuple(ish)

        t = (*ish_t, *attr_t)

        # if len(attr_t) == 0:
        #     continue
        if node.op_type not in info:
            info[node.op_type + '_attr_ish'] = Counter()
            info[node.op_type + '_attr'] = Counter()
            info[node.op_type + '_ish'] = Counter()
        # print('attrs=\n', attrs)
        # print('tuple=', t)
        assert t == tuple(t), f'{t} not comparable'
        info[node.op_type + '_attr_ish'].update({t: 1})
        info[node.op_type + '_attr'].update({attr_t: 1})
        info[node.op_type + '_ish'].update({ish_t: 1})

    ish = Counter()
    for node in model.graph.input:
        if not hasattr(node.type, 'tensor_type'):
            print('Warning: skip unknown input:\n', node)
            continue
        shape = tuple(int(dim.dim_value)
                      for dim in node.type.tensor_type.shape.dim)
        # print(shape)
        ish.update({shape: 1})
    res = {'PlaceHolder_ish': ish}
    res.update(info)
    return res


def analyze_folders(folders, cache_dir=None, force=False, n_limit=None):
    res = []

    __CACHE_FILE__ = 'onnx_param_cache.pkl'
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
        cnts = {}
        with Pool(min(cpu_count(), len(files))) as p:
            for new in p.imap_unordered(analyze_one, files):
                if new is None:
                    continue
                for op_name, cnt in new.items():
                    if op_name not in cnts:
                        cnts[op_name] = Counter()
                    cnts[op_name].update(cnt)

        res.append(cnts)

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
    parser.add_argument('--ops', type=str, nargs='+',
                        default=None, help='Pass operator names to show.')
    args = parser.parse_args()

    ops = args.ops
    if ops is None:
        ops = ['MaxPool', 'AveragePool', 'Conv', 'Dense', 'Slice']
        # Resahpe and Pad's parameters are specified using a tensor so unfortunately hard to parse. Skip for now.
    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split('-')[0] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    results = analyze_folders(
        args.folders, cache_dir=args.output, force=args.force, n_limit=args.nlim)

    df = pd.DataFrame()

    def to_df(cnts, suffix):
        cnts = {k[:-len(suffix)]: v for k,
                v in cnts.items() if k.endswith(suffix)}
        return pd.DataFrame({
            'name': cnts.keys(),
            'count': [len(cnt) for cnt in cnts.values()],
            'ratio': [len(cnt) / sum(cnt.values()) for cnt in cnts.values()],
            'cat': [suffix] * len(cnts)
        })

    for tag, cnts in zip(args.tags, results):
        print(f'{tag}:\t')
        for op_name, cnt in cnts.items():
            print(f'\t{op_name}: {len(cnt)}')

        df_op_param = to_df(cnts, '_attr')
        df_op_ish = to_df(cnts, '_ish')
        df_op_param_ish = to_df(cnts, '_attr_ish')
        df1 = pd.concat([df_op_param, df_op_ish, df_op_param_ish])
        df1['fuzzers'] = tag
        df = df.append(df1, ignore_index=True)

    def print_most_common(d):
        s = sum(d.values())
        for k, v in d.most_common():
            print(f'`{k}`: count={v} ratio={v / s}')

    # print_most_common(results[0]['MaxPool_ish'])
    # print_most_common(results[0]['MaxPool_attr_ish'])
    # print_most_common(results[0]['MaxPool_attr'])

    def plot_one_cat(df, cat, name):
        df = df[df['cat'] == cat]
        for c in ['Count', 'Ratio']:
            _c = c.lower()
            df_ops = df[df.name.map(lambda x: x in ops)]
            plt.title(
                f"{c} of unqiue {name} combination for different ONNX operators")
            sns.barplot(x='name', y=f'{_c}', hue='tag', data=df_ops)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, f'{_c}_{cat}.png'))
            plt.close()

            df_ops = df[df.name.map(lambda x: x != 'input shape')]
            fig, ax = plt.subplots(figsize=(20, 6))
            plt.title(
                f"{c} of unqiue {name} combination for different ONNX operators")
            sns.barplot(x='name', y=f'{_c}', hue='tag', data=df_ops, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, f'{_c}_params_{cat}.png'))
            plt.close()

    plot_one_cat(df, '_attr', 'attribute')
    plot_one_cat(df, '_ish', 'input shapes')
    plot_one_cat(df, '_attr_ish', 'attribute and input shapes')
