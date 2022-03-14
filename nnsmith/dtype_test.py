from typing import List
from inspect import signature
import warnings
import pickle
import os

import z3
import torch
import numpy as np
import onnx

from nnsmith.abstract.op import ALL_OP_TYPES, AbsOpBase, DType, ShapeVar, concretize, Input, Constant
from nnsmith.backends import DiffTestBackend
from nnsmith.input_gen import gen_one_input


def rewrite_op_dtype(ops: List[AbsOpBase], backend=None, verbose=False, cache=None):
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

    if backend is None:
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

        op_param_n = signature(node_t).parameters
        op_params = [z3.Int('v%s-%s' % (idx, k))
                     for k in range(len(op_param_n))]
        op = node_t(*op_params)

        solver = z3.Solver()

        inputs = []
        for i, rank in enumerate(op.inp_ranks):
            if rank == -1:
                rank = 2
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

        success_idtypes = list()
        success_odtypes = set()

        model = TestNet(concrete_op)
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
                    print(f'=====> [Failure] {itypes}')
                if 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented' in str(type(e)) or \
                        "Unexpected data type for" in str(e):
                    continue
                raise e
            success_idtypes.append(itypes)
            otypes = []
            if len(op.out_ranks) == 1:
                otypes = [o.dtype]
            else:
                for i in range(node_t.out_ranks):
                    otypes.append(o[i].dtype)
            for i in range(len(otypes)):
                otypes[i] = DType.from_str(str(otypes[i]).split('.')[-1])
            success_odtypes.add(tuple(otypes))
            if verbose:
                print(f'=====> [Success] {itypes}')
        success_odtypes = list(success_odtypes)
        if verbose:
            print(
                f'ORT Inferred: {node_t}\t{success_idtypes} -> {success_odtypes}')

        node_t.in_dtypes = success_idtypes
        node_t.out_dtypes = success_odtypes
        if make_cache:
            cache_dict[str(node_t)] = (success_idtypes, success_odtypes)

    if make_cache:
        with open(cache, 'wb') as f:
            if verbose:
                print(f'Writing cache to {cache}')
            pickle.dump(cache_dict, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='config/ort_cpu_dtype.pkl')
    args = parser.parse_args()

    from nnsmith.backends.ort_graph import ORTExecutor
    backend = ORTExecutor(opt_level=3)

    rewrite_op_dtype(ALL_OP_TYPES, backend, verbose=True, cache=args.cache)
