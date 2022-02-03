from time import time
from nnsmith.backends import DiffTestBackend
import tvm
from tvm import relay
from tvm.relay.transform.transform import DefuseOps
import sys

from tvm.ir.expr import RelayExpr
from tvm.relay.analysis import post_order_visit
from tvm.relay import ExprMutator, TensorType, TupleType
from tvm.relay.expr import Call
from tvm.topi.utils import get_const_tuple
from tvm.relay.frontend.common import infer_shape, infer_type
from graphviz import Digraph
import onnx


def call_node_infer_type(node):
    """infer the output types of call node"""
    infer_out = infer_type(node)
    out_type = infer_out._checked_type_
    if isinstance(out_type, TensorType):
        types = [out_type]
    elif isinstance(out_type, TupleType):
        types = list(out_type.fields)
    else:
        raise RuntimeError(
            "Unsupported output type %s in operator %s" % (
                type(out_type), node.op.nae)
        )
    return types


def flatten_attrs_til_expr(node, prefix):
    # if prefix == 'args': print(prefix, node, type(node))
    if not isinstance(node, (RelayExpr, tvm.ir.container.Array)):
        return []
    if isinstance(node, RelayExpr):
        return [(node, prefix)]
    res = []
    for i, child in enumerate(node):
        res = res + flatten_attrs_til_expr(child, prefix + f'_{i}')
    return res


def get_children(node):
    res = []
    for name in dir(node):
        if name == "checked_type" or not hasattr(node, name):
            continue
        child = getattr(node, name)
        res += flatten_attrs_til_expr(child, name)
    return res

# will use node_id to init if it's passed in


def relay_assign_id(expr, init_node_id=None):
    if init_node_id is None:
        node_id = {}
    else:
        node_id = dict(init_node_id)

    def visit(node):
        if node in node_id:
            return
        node_id[node] = len(node_id)
    post_order_visit(expr, visit)
    return node_id


def is_ew(op):
    return (op.name in ['add', 'nn.relu', 'multiply', 'divide', 'sqrt'] or
            op.get_attr('TOpPattern') == relay.op.OpPattern.ELEMWISE
            )


def annotate_layout(expr, node_id=None):
    layout = {}

    def make_same(lhs, rhs):
        if lhs in layout:
            lhs, rhs = rhs, lhs
        if lhs in layout:
            assert layout[lhs] == layout[rhs]
        elif rhs in layout:
            layout[lhs] = layout[rhs]

    def backward_prop(node):
        if isinstance(node, Call) and is_ew(node.op):
            for arg in node.args:
                # if we find one argument that maintains its shape intact, probably
                # so is the layout. Note that it's fine if the layout info is incorrect,
                # as it's only a hint
                if infer_shape(arg) == infer_shape(node):
                    memo = layout.get(arg, None)
                    make_same(node, arg)
                    if layout.get(node, None) != memo:
                        assert memo is None
                        backward_prop(arg)

    def visit(node):
        if not isinstance(node, Call):
            return
        if node.op.name == "layout_transform":
            layout[node] = node.attrs.get_str("dst_layout")
            layout[node.args[0]] = node.attrs.get_str("src_layout")
            assert len(node.args) == 1
        elif node.op.name == "nn.global_avg_pool2d":
            layout[node.args[0]] = layout[node] = node.attrs.get_str("layout")
            assert len(node.args) == 1
        elif node.op.name in ["nn.contrib_conv2d_NCHWc", "nn.conv2d"]:
            out_layout = node.attrs.get_str("out_layout")
            data_layout = node.attrs.get_str("data_layout")
            kernel_layout = node.attrs.get_str("kernel_layout")
            if out_layout == "":
                out_layout = data_layout
            layout[node] = out_layout
            layout[node.args[0]] = data_layout
            layout[node.args[1]] = kernel_layout

    post_order_visit(expr, visit)
    for node in node_id.keys():
        backward_prop(node)

    if node_id is not None:  # switch to id based mapping
        layout = {node_id[k]: v for k, v in layout.items()}
    return layout


class Namer(object):

    def __init__(self, node_id, nodeid2layout=None, auto_infer_type=False) -> None:
        self.nodeid2layout = nodeid2layout
        self.node_id = node_id
        self.auto_infer_type = auto_infer_type

    @staticmethod
    def get_base_name(node):
        if isinstance(node, relay.expr.Call):
            # name = 'CallNode(' + node.op.name + ')'
            name = node.op.name
        elif isinstance(node, relay.expr.Var):
            name = 'Var(' + node.name_hint + ')'
        elif isinstance(node, relay.expr.TupleGetItem):
            name = f'TupleGetItemNode(index={node.index})'
        else:
            # name = repr(node) # TOO SLOW?
            name = type(node).__name__
            if len(name) > 47:
                name = name[:47] + '...'
        return name

    def get_type_info(self, node):
        type_info = ''
        st = time()
        try:
            type_info = repr(node.checked_type) + "; "
        except:
            if self.auto_infer_type:
                try:
                    type_info = repr(infer_type(node).checked_type) + "; "
                except Exception as e:
                    print(e, file=sys.stderr)
        # print('checked-type time=', time() - st)
        if self.nodeid2layout is not None:
            layout = self.nodeid2layout.get(self.node_id[node], None)
        else:
            layout = None
        if layout is not None:
            type_info += layout + "; "
        return type_info

    def __call__(self, node):
        name, type_info = self.get_base_name(node), self.get_type_info(node)
        if type_info != '':
            name += "\n: " + type_info
        return '#' + str(self.node_id[node]) + ': ' + name


def visualize(expr_or_mod, show_full_bn=False, node_id=None,
              layout=None, show_full_tuple=True, auto_infer_type=False, name_me=None):
    if isinstance(expr_or_mod, tvm.IRModule):
        expr = expr_or_mod["main"].body
    else:
        expr = expr_or_mod
    assert isinstance(expr, relay.Expr), expr
    dot = Digraph(format='svg')
    dot.attr(rankdir='BT')
    dot.attr('node', shape='box')

    if node_id is None:
        assert layout is None
        node_id = relay_assign_id(expr)
        layout = annotate_layout(expr, node_id)

    if name_me is None:
        name_me = Namer(node_id, layout, auto_infer_type)

    def visit(node):
        if isinstance(node, tvm.ir.Op):
            return
        assert isinstance(node, relay.expr.RelayExpr), node
        dot.node(str(node_id[node]), name_me(node))
        children = get_children(node)
        for (c, label) in children:
            if label == 'op':
                continue
            if not show_full_bn and isinstance(node, relay.expr.Call) and \
                    node.op.name == 'nn.batch_norm' and label != 'args_0':
                continue
            if not show_full_tuple and isinstance(node, relay.expr.Tuple):
                continue
            dot.edge(str(node_id[c]), str(node_id[node]), label=label)
    post_order_visit(expr, visit)
    return dot


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='onnx model path')
    parser.add_argument('--output', type=str, required=True,
                        help='output figure path')
    parser.add_argument('--select', type=int, nargs='+',
                        help='select output tensors to visualize')
    args = parser.parse_args()
    onnx_model = DiffTestBackend.get_onnx_proto(args.model)
    inp_spec, onames = DiffTestBackend.analyze_onnx_io(onnx_model)
    shape_dict = {name: inp_spec[name].shape for name in inp_spec}
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape_dict, freeze_params=True)
    if args.select is not None:
        func = mod["main"]
        func_1 = relay.Function(
            func.params, relay.Tuple([func.body.fields[i] for i in args.select]))
        mod = tvm.IRModule.from_expr(func_1)
    mod = relay.transform.InferType()(mod)

    visualize(mod, show_full_bn=True).render(cleanup=True, outfile=args.output)
