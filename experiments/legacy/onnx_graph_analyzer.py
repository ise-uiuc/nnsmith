"""Just to figure out operators types and connections.
"""

from collections import Counter
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, Set
import re
import pickle

import onnx
import pandas as pd

from tvm import relay

from nnsmith.backends import BackendFactory


def relay_op_cluster(mod, ignore_arg=False, verbose=False, use_counter=False):
    mod = relay.transform.InferType()(mod)
    op2type = {}

    def visit(node):
        def comment_remover(text):
            def replacer(match):
                s = match.group(0)
                if s.startswith("/"):
                    return " "  # note: a space and not an empty string
                else:
                    return s

            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE,
            )
            return re.sub(pattern, replacer, text)

        # the trick is: make a signature string. lmao.
        if isinstance(node, relay.Call):
            statement = comment_remover(str(node).splitlines()[-1]).replace(" ", "")
            num_args = len(node.type_args)
            attr_str = ",".join(statement[:-1].split(",")[num_args:])
            op_str = str(node.op)

            if attr_str == "":
                attr_str = None

            if ignore_arg:
                arg_type_str = None
            else:
                arg_type_str = str(node.type_args).replace(" ", "")

            hash_str = f"{arg_type_str}@{attr_str}"
            if verbose:
                print(f"[DEBUG] statement={statement}")
                print(f"[DEBUG] op_str={op_str}")
                print(f"[DEBUG] arg_type_str={arg_type_str}")
                print(f"[DEBUG] attr_str={attr_str}")
                print(f"[DEBUG] hash_str={hash_str}")

            if use_counter:
                op2type.setdefault(op_str, Counter()).update({hash_str: 1})
            else:
                op2type.setdefault(op_str, set()).update({hash_str: 1})

    for func in mod.functions.values():
        relay.analysis.post_order_visit(func, lambda node: visit(node))
    return op2type


def analyze_one_relay(model_path, use_counter=False) -> Dict[str, Set[str]]:
    """Return <op name> -> tag (a string)"""
    if "FAILURE" in model_path:
        return {}

    onnx_model = BackendFactory.get_onnx_proto(model_path)
    inp_spec, _ = BackendFactory.analyze_onnx_io(onnx_model)
    shape_dict = {name: inp_spec[name].shape for name in inp_spec}
    mod, _ = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    return relay_op_cluster(mod)


def analyze_one(model_path):
    if "FAILURE" in model_path:
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


def analyze_folders(folders, cache_dir=None, force=False, n_limit=None):
    res = []

    __CACHE_FILE__ = "onnx_analysis_cache.pkl"
    for folder in folders:
        if (
            os.path.exists(os.path.join(cache_dir, __CACHE_FILE__))
            and not force
            and cache_dir is not None
        ):
            print(
                "===> {} already exists.".format(os.path.join(folder, __CACHE_FILE__))
            )
            with open(os.path.join(cache_dir, __CACHE_FILE__), "rb") as fp:
                return pickle.load(fp)

    times = []
    file_hubs = []
    least_time = None
    assert n_limit is None or len(n_limit) == len(folders)
    for i, folder in enumerate(folders):
        df = pd.read_csv(
            os.path.join(folder, "gentime.csv"), usecols=[0, 1], header=None
        )
        ts = df[0].to_numpy()

        times.append(ts)
        files = df[1].tolist()
        if n_limit is not None:
            files = files[: n_limit[i]]
        files = [os.path.join(folder, f) for f in files]
        file_hubs.append(files)

        time_cost = ts.sum()
        if least_time is None or time_cost < least_time:
            least_time = time_cost

    # only consider rows that < least_time
    for i, ts in enumerate(times):
        file_hubs[i] = file_hubs[i][: (ts < least_time).sum()]

    for files in file_hubs:
        nodes = set()
        edges = set()

        with Pool(min(cpu_count(), len(files))) as p:
            for n, e in p.imap_unordered(analyze_one, files):
                nodes |= n
                edges |= e

        res.append((nodes, edges))

    if cache_dir is not None:
        with open(os.path.join(cache_dir, __CACHE_FILE__), "wb") as fp:
            pickle.dump(res, fp)
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", type=str, nargs="+", required=True)
    parser.add_argument("--tags", type=str, nargs="+", default=None)
    # should compare models within same generation duration.
    parser.add_argument("--nlim", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split("-")[0] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    results = analyze_folders(
        args.folders, cache_dir=args.output, force=args.force, n_limit=args.nlim
    )
    for tag, (nodes, edges) in zip(args.tags, results):
        print(f"{tag}:\t nodes: {len(nodes)};\t edges: {len(edges)}")

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

    node_list = []
    edge_list = []
    for i in range(len(results)):
        nodes, edges = results[i]
        node_list.append(nodes)
        edge_list.append(edges)

    if len(node_list) == 2:
        venn2(subsets=node_list, set_labels=[f"$\\bf{{{t}}}$" for t in args.tags])
        venn2_circles(subsets=node_list, linestyle="dashed")
    elif len(node_list) == 3:
        v = venn3(subsets=node_list, set_labels=[f"$\\bf{{{t}}}$" for t in args.tags])
        hatches = ["\\", ".", "*"]
        circles = ["MediumVioletRed", "SeaGreen", "Lavender"]
        for idx, id in enumerate(["100", "010", "001", "111"]):
            if v.get_label_by_id(id) is None:
                continue

            cnt = int(v.get_label_by_id(id).get_text())

            if id != "111":
                v.get_patch_by_id(id).set_alpha(0.5)
                v.get_patch_by_id(id).set_hatch(hatches[idx])
                v.get_patch_by_id(id).set_edgecolor(circles[idx])
                v.get_patch_by_id(id).set_linewidth(2)
                v.get_patch_by_id(id).set_linestyle("--")

    plt.title("Venn Diagram of Covered ONNX Operators")
    plt.savefig(
        f'{os.path.join(args.output, "onnx_node_venn")}.png', bbox_inches="tight"
    )
    plt.savefig(
        f'{os.path.join(args.output, "onnx_node_venn")}.pdf', bbox_inches="tight"
    )
    plt.close()

    if len(node_list) == 2:
        venn2(subsets=edge_list, set_labels=[f"$\\bf{{{t}}}$" for t in args.tags])
        venn2_circles(subsets=edge_list, linestyle="dashed")
    elif len(node_list) == 3:
        v = venn3(subsets=edge_list, set_labels=[f"$\\bf{{{t}}}$" for t in args.tags])
        hatches = ["\\", ".", "*"]
        circles = ["MediumVioletRed", "SeaGreen", "Lavender"]
        for idx, id in enumerate(["100", "010", "001", "111"]):
            if v.get_label_by_id(id) is None:
                continue

            cnt = int(v.get_label_by_id(id).get_text())

            if id != "111":
                v.get_patch_by_id(id).set_alpha(0.5)
                v.get_patch_by_id(id).set_hatch(hatches[idx])
                v.get_patch_by_id(id).set_edgecolor(circles[idx])
                v.get_patch_by_id(id).set_linewidth(2)
                v.get_patch_by_id(id).set_linestyle("--")
    plt.title("Venn Diagram of Covered ONNX Operators Edges")
    plt.savefig(
        f'{os.path.join(args.output, "onnx_edge_venn")}.png', bbox_inches="tight"
    )
    plt.savefig(
        f'{os.path.join(args.output, "onnx_edge_venn")}.pdf', bbox_inches="tight"
    )
    plt.close()
