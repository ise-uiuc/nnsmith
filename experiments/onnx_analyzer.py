"""Just to figure out operators types and connections.
"""

import os
from multiprocessing import Pool, cpu_count

import onnx
import pandas as pd
import pickle


def analyze_one(model_path):
    if 'FAILURE' in model_path:
        return set(), set()

    model = onnx.load(model_path)
    output_to_op_t = {}

    nodes = set()
    edges = set()

    for node in model.graph.node:
        nodes.add(node.op_type)
        for o in node.output:
            output_to_op_t[o] = node.op_type

    for node in model.graph.node:
        for i in node.input:
            if i in output_to_op_t:
                edges.add((output_to_op_t[i], node.op_type))

    return nodes, edges


def analyze_folders(folders, cache_dir=None, force=False):
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
    for folder in folders:
        df = pd.read_csv(os.path.join(folder, 'gentime.csv'),
                         usecols=[0, 1], header=None)
        ts = df[0].to_numpy()

        times.append(ts)
        files = df[1].tolist()
        files = [os.path.join(folder, f) for f in files]
        file_hubs.append(files)

        time_cost = ts.sum()
        if least_time is None or time_cost < least_time:
            least_time = time_cost

    # only consider rows that < least_time
    for i, ts in enumerate(times):
        file_hubs[i] = file_hubs[i][:(ts < least_time).sum()]

    for files in file_hubs:
        nodes = set()
        edges = set()

        with Pool(min(cpu_count(), len(files))) as p:
            for n, e in p.imap_unordered(analyze_one, files):
                nodes |= n
                edges |= e

        res.append((nodes, edges))

    if cache_dir is not None:
        with open(os.path.join(cache_dir, __CACHE_FILE__), 'wb') as fp:
            pickle.dump(res, fp)
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=str, nargs='+', required=True)
    parser.add_argument('--tags', type=str, nargs='+', default=None)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.tags is None:
        args.tags = [os.path.split(f)[-1] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    results = analyze_folders(
        args.folders, cache_dir=args.output, force=args.force)
    for tag, (nodes, edges) in zip(args.tags, results):
        print(f'{tag}:\t nodes: {len(nodes)};\t edges: {len(edges)}')

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn2_circles

    node_list = []
    edge_list = []
    for i in range(len(results)):
        nodes, edges = results[i]
        node_list.append(nodes)
        edge_list.append(edges)

    venn2(subsets=node_list, set_labels=args.tags)
    venn2_circles(subsets=node_list, linestyle='dashed')
    plt.title("Venn Diagram of Covered ONNX Operators")
    plt.savefig(f'{os.path.join(args.output, "onnx_node_venn")}.png', bbox_inches='tight')
    plt.savefig(f'{os.path.join(args.output, "onnx_node_venn")}.pdf', bbox_inches='tight')
    plt.close()

    venn2(subsets=edge_list, set_labels=args.tags)
    venn2_circles(subsets=edge_list, linestyle='dashed')
    plt.title("Venn Diagram of Covered ONNX Operators Edges")
    plt.savefig(f'{os.path.join(args.output, "onnx_edge_venn")}.png', bbox_inches='tight')
    plt.savefig(f'{os.path.join(args.output, "onnx_edge_venn")}.pdf', bbox_inches='tight')
    plt.close()
