from itertools import product
from tqdm import tqdm
from z3.z3 import ExprRef
from nnsmith.abstract.op import *
from nnsmith import graph_gen
from nnsmith.error import ConstraintError
from nnsmith.graph_gen import PureSymbolGen, SymbolNet, parse_args
from nnsmith.export import torch2onnx
import time
import tempfile
import os


def test_slice():
    def test_slice_helper_torch(inp, start, end, axis, step, dtype):
        dtype = dtype.value
        inp_t = torch.empty(inp, dtype=dtype)
        if axis == 0:
            res = inp_t[start:end:step, ...]
        elif axis == 1:
            res = inp_t[:, start:end:step, ...]
        elif axis == 2:
            res = inp_t[:, :, start:end:step, ...]
        elif axis == 3:
            res = inp_t[:, :, :, start:end:step, ...]
        elif axis == 4:
            res = inp_t[:, :, :, :, start:end:step, ...]
        elif axis == 5:
            res = inp_t[:, :, :, :, :, start:end:step, ...]
        elif axis == 6:
            res = inp_t[:, :, :, :, :, :, start:end:step, ...]
        else:
            assert False
        if any(i == 0 for i in list(res.shape)):
            raise ValueError('Empty dimension')
        return list(res.shape)

    def test_slice_helper_ours(inp, start, end, axis, step, dtype):
        inp_sv = ShapeVar(inp, dtype)
        dim_s = inp[axis]
        if start < 0:
            start = start + dim_s
        if end < 0:
            end = end + dim_s
        if isinstance(start, int):
            start = z3.IntVal(start)
        if isinstance(end, int) and end != Slice.INT_MAX:
            end = z3.IntVal(end)
        sl = Slice(start, end, step)
        sl.extra_attrs['axis'] = axis
        sl.extra_attrs['ndims'] = len(inp)
        sl.extra_attrs['region'] = random.choice(['left', 'mid', 'right'])
        out_sv = sl.shape_fn([inp_sv])[0]
        cons = z3.And(*sl.requires([inp_sv]))
        cons = z3.And(cons, *out_sv.gt_zero())
        if z3.is_false(z3.simplify(cons)):
            raise ConstraintError(f'Constraint {cons} is false')
        return [z3.simplify(i).as_long() if isinstance(i, z3.ExprRef) else i for i in out_sv.shape]

    def test(inp, start, end, axis, step, dtype):
        err_torch, err_ours = False, False
        e0, e1 = None, None
        try:
            out_shape = test_slice_helper_torch(
                inp, start, end, axis, step, dtype)
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            e0 = e
            err_torch = True
        try:
            out_shape_ours = test_slice_helper_ours(
                inp, start, end, axis, step, dtype)
        except ConstraintError as e:
            e1 = e
            err_ours = True
        assert err_torch == err_ours, f'{err_torch}(err_torch) != {err_ours}(err_ours)' + \
            f' when testing in_shape={inp} start={start} end={end} axis={axis} step={step} {dtype};\nTorch exception={e0}\nOur exception={e1}'
        if err_torch:
            return
        assert out_shape == out_shape_ours, f'{out_shape} != {out_shape_ours}' + \
            f' when testing in_shape={inp} start={start} end={end} axis={axis} step={step} {dtype}'

    for inp in tqdm(list(product([1, 7], repeat=1)) + list(product([1, 7], repeat=2)) + list(product([1, 7], repeat=3))):
        for axis in range(len(inp)):
            for start in list(range(-inp[axis], inp[axis])):  # + [-2**63]:
                for end in list(range(-inp[axis], inp[axis] + 1)) + [2**63 - 1]:
                    # for dtype in DTYPE_ALL:
                    dtype = DType.float32
                    step = random.randint(1, inp[axis])
                    test(inp, start, end, axis, step, dtype)

    i0, i1, i2 = z3.Ints('i0 i1 i2')
    for inp in tqdm([[i0], [i0, i1], [i0, i1, i2]]):
        for trials in tqdm(range(100)):
            start, end, step = z3.Ints('start end step')
            dtype = DType.float32
            min_dims = [random.randint(1, 37) for _ in range(len(inp))]
            inp_sv = ShapeVar(inp, dtype)
            sl = Slice(start, end, step)
            cons = []
            for i in range(len(inp_sv.shape)):
                cons.append(nnsmith_ge(inp_sv.shape[i], min_dims[i]))
            cons.extend(sl.requires([inp_sv]))
            cons.extend(sl.shape_fn([inp_sv])[0].gt_zero())
            s = z3.Solver()
            s.add(*cons)
            assert s.check() == z3.sat
            model = s.model()
            start = sl.start if isinstance(
                sl.start, int) else model.evaluate(start).as_long()
            end = sl.end if isinstance(
                sl.end, int) else model.evaluate(end).as_long()
            axis = sl.extra_attrs['axis']
            step = model.evaluate(step).as_long()
            inp_concrete = [model.evaluate(
                i).as_long() for i in inp_sv.shape]
            print(
                f'testing in_shape={inp_concrete} start={start} end={end} axis={axis} step={step} {dtype}')
            test(inp_concrete, start, end, axis, step, dtype)


test_slice()
print('passed')
