from typing import List
import warnings
import pickle
import os
import random

import z3
import torch
import numpy as np
import onnx

from nnsmith.abstract.op import ALL_OP_TYPES, ALL_OP_STR2TYPE, AbsOpBase, DType, ShapeVar, concretize, Input, Constant
from nnsmith.backends import DiffTestBackend
from nnsmith.input_gen import gen_one_input

def _differentiable_test(model, available_idtypes, concrete_input_shapes, oranks, verbose=False):
    model.eval()

    success_idtypes = list()
    success_odtypes = set()

    for itypes in available_idtypes:
        assert len(itypes) == len(
            concrete_input_shapes), f'{len(itypes)} != {len(concrete_input_shapes)}'
        
        torch_inputs = []

        for s, dt in zip(concrete_input_shapes, itypes):
            data = torch.from_numpy(np.ones(s, dtype=str(dt)))
            if DType.is_float(dt):
                torch_inputs.append(torch.nn.parameter.Parameter(data))
            else:
                torch_inputs.append(data)
        
        if os.getenv('USE_CUDA') == '1':
            torch_inputs = [i.cuda() for i in torch_inputs]
            model = model.cuda()

        o = model(*torch_inputs)

        differntiable = False
        if len(oranks) == 1:
            differntiable = o.grad_fn is not None
        else:
            for i in range(oranks):
                differntiable = differntiable or o[i].grad_fn is not None

        if not differntiable:
            if verbose:
                print(f'=====> [Undifferntiable] at {itypes}')
            continue

        otypes = []
        if len(oranks) == 1:
            otypes = [o.dtype]
        else:
            for i in range(oranks):
                otypes.append(o[i].dtype)

        for i in range(len(otypes)):
            otypes[i] = DType.from_str(str(otypes[i]).split('.')[-1])

        success_idtypes.append(itypes)
        success_odtypes.add(tuple(otypes))
        if verbose:
            print(f'=====> [Success] {itypes}')
    success_odtypes = list(success_odtypes)
    
    return success_idtypes, success_odtypes


def _inference_test(model, available_idtypes, concrete_input_shapes, oranks, verbose=False):
    success_idtypes = list()
    success_odtypes = set()

    model.eval()
    for itypes in available_idtypes:
        assert len(itypes) == len(
            concrete_input_shapes), f'{len(itypes)} != {len(concrete_input_shapes)}'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    onnx_model_path = 'ort.onnx'
                    torch_inputs = [torch.from_numpy(np.ones(s, dtype=str(dt))) for s, dt in zip(
                        concrete_input_shapes, itypes)]
                    o = model(*torch_inputs)
                    torch.onnx.export(
                        model, tuple(torch_inputs),
                        onnx_model_path,
                        opset_version=14)
                    onnx_model = onnx.load(onnx_model_path)
                    input_spec, _ = DiffTestBackend.analyze_onnx_io(
                        onnx_model)
                    eval_inputs = gen_one_input(input_spec, 1, 1)
                    backend.predict(onnx_model, eval_inputs)
        except Exception as e:
            if verbose:
                print(f'=====> [Failure] at {itypes}')
            if 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented' in str(type(e)) or \
                    "Unexpected data type for" in str(e):
                continue
            if 'DiagnosticError: one or more error diagnostics were emitted, please check diagnostic render for output' in str(type(e)):
                continue
            raise e
        success_idtypes.append(itypes)
        otypes = []
        if len(oranks) == 1:
            otypes = [o.dtype]
        else:
            for i in range(oranks):
                otypes.append(o[i].dtype)
        for i in range(len(otypes)):
            otypes[i] = DType.from_str(str(otypes[i]).split('.')[-1])
        success_odtypes.add(tuple(otypes))
        if verbose:
            print(f'=====> [Success] {itypes}')
    success_odtypes = list(success_odtypes)
    
    return success_idtypes, success_odtypes

def rewrite_op_dtype(ops: List[AbsOpBase], diff=False, backend=None, verbose=False, cache=None):
    ret_ops = []

    class TestNet(torch.nn.Module):
        def __init__(self, op: AbsOpBase) -> None:
            super().__init__()
            self.op = None
            self.mlist = torch.nn.ModuleList()
            if isinstance(op, torch.nn.Module):
                self.mlist.append(op.torch())
            else:
                self.op = op.torch()

        def forward(self, *args):
            if self.op is None:
                return self.mlist[0](*args)
            else:
                return self.op(*args)

    make_cache = cache is not None and not os.path.exists(cache)
    reuse_cache = cache is not None and os.path.exists(cache)

    if backend is None and not diff:
        assert reuse_cache, 'Must provide backend if cache is not provided.'

    cache_dict = {}
    if reuse_cache:
        with open(cache, 'rb') as f:
            if verbose:
                print(f'Reusing cache {cache}')
            cache_dict = pickle.load(f)

    for idx, node_t in enumerate(ops):
        if node_t is Input or node_t is Constant:
            continue  # meaningless to test

        if reuse_cache:
            success_idtypes, success_odtypes = cache_dict[str(node_t)]
            node_t.in_dtypes = success_idtypes
            node_t.out_dtypes = success_odtypes
            continue

        if verbose:
            print(f'===> Trying {node_t} # {idx}')
        available_idtypes = node_t.in_dtypes

        op_param_n = node_t.get_num_var_param()
        op_params = [z3.Int('v%s-%s' % (idx, k))
                     for k in range(op_param_n)]
        op = node_t(*op_params)

        solver = z3.Solver()

        inputs = []
        for i, ranks in enumerate(op.inp_ranks):
            if op.same_inp_dims and inputs:
                rank = inputs[0].ndims
            else:
                rank = random.choice(ranks)
            shape = ShapeVar(shape=[z3.Int('s%s' % (k))
                             for k in range(rank)], dtype=available_idtypes[0][i])
            inputs.append(shape)
            solver.add(*shape.gt_zero())

        solver.add(*op.requires(inputs))

        # solve
        assert solver.check(
        ) == z3.sat, f"Cannot solve the problem in {node_t}"
        m = solver.model()

        # rewrite
        concrete_op = concretize(op, m)
        concrete_input_shapes = []
        for inp in inputs:
            shape = []
            for s in inp.shape:
                shape.append(m.eval(s).as_long())
            concrete_input_shapes.append(shape)

        model = TestNet(concrete_op)
        
        if verbose:
            print(f'=====> [Testing] {node_t}')

        if diff:
            success_idtypes, success_odtypes = _differentiable_test(model, available_idtypes, concrete_input_shapes, op.out_ranks, verbose)
        else:
            success_idtypes, success_odtypes = _inference_test(model, available_idtypes, concrete_input_shapes, op.out_ranks, verbose)

        node_t.in_dtypes = success_idtypes
        node_t.out_dtypes = success_odtypes

        if len(success_idtypes) != 0 and len(success_odtypes) != 0:
            ret_ops.append(node_t)

        if make_cache:
            cache_dict[str(node_t)] = (success_idtypes, success_odtypes)

    if make_cache:
        with open(cache, 'wb') as f:
            if verbose:
                print(f'Writing cache to {cache}')
            pickle.dump(cache_dict, f)

    if reuse_cache:
        ret_ops = [ALL_OP_STR2TYPE[op_t.split("'")[1].split('.')[-1]] for op_t in cache_dict.keys()]

    return ret_ops

if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='config/ort_cpu_dtype.pkl')
    parser.add_argument('--seed', default=233, type=int)
    parser.add_argument('--diff', action='store_true')
    args = parser.parse_args()

    if args.cache == 'None':
        args.cache = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from nnsmith.backends.ort_graph import ORTExecutor
    backend = ORTExecutor(opt_level=3)

    rewrite_op_dtype(ALL_OP_TYPES, backend=backend, diff=args.diff, verbose=True, cache=args.cache)
