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

auto_infer_in_dtypes()


def test_cast():
    for src_t in DTYPE_ALL:
        for dst_t in DTYPE_ALL:
            shapes = [(2, 3)]
            shapes_var = [ShapeVar(s, src_t) for s in shapes]
            cast = Cast()
            cast.extra_attrs['to'] = dst_t
            a = torch.zeros(shapes[0], dtype=shapes_var[0].dtype.value)
            my_sh = tuple(cast.shape_fn(shapes_var)[0].shape)
            assert shapes[0] == my_sh, f'{shapes[0]} != {my_sh}'
            assert cast.torch()(a).dtype == cast.extra_attrs['to'].value


def test_gemm():
    def test_gemm_helper_torch(inp, mat1, mat2, beta, alpha, dtype):
        dtype = dtype.value
        inp_t = torch.empty(inp, dtype=dtype)
        mat1_t = torch.empty(mat1, dtype=dtype)
        mat2_t = torch.empty(mat2, dtype=dtype)
        out_t = torch.addmm(inp_t, mat1_t, mat2_t, beta=beta, alpha=alpha)
        return list(out_t.shape)

    def test_gemm_helper_ours(inp, mat1, mat2, beta, alpha, dtype):
        inp_sv = ShapeVar(inp, dtype)
        mat1_sv = ShapeVar(mat1, dtype)
        mat2_sv = ShapeVar(mat2, dtype)
        gemm = Gemm()
        gemm.extra_attrs['beta'] = beta
        gemm.extra_attrs['alpha'] = alpha
        cons = z3.simplify(z3.And(*gemm.requires([inp_sv, mat1_sv, mat2_sv])))
        if z3.is_false(cons):
            raise ConstraintError(f'Constraint {cons} is false')
        out_sv = gemm.shape_fn([inp_sv, mat1_sv, mat2_sv])[0]
        return [z3.simplify(i).as_long() if isinstance(i, z3.ExprRef) else i for i in out_sv.shape]

    for dtype in DTYPE_NON_BOOLS:
        for inp in list(product([1, 2], repeat=1)) + list(product([1, 2], repeat=2)) + list(product([1, 2], repeat=3)):
            for mat1 in list(product([1, 2], repeat=2)):
                for mat2 in list(product([1, 2], repeat=2)):
                    beta, alpha = random.uniform(-10,
                                                 10), random.uniform(-10, 10)
                    if dtype in DTYPE_INTS:
                        beta, alpha = int(beta), int(alpha)
                        err_torch, err_ours = False, False
                        e0, e1 = None, None
                        try:
                            out_shape = test_gemm_helper_torch(
                                inp, mat1, mat2, beta, alpha, dtype)
                        except Exception as e:
                            e0 = e
                            err_torch = True
                        try:
                            out_shape_ours = test_gemm_helper_ours(
                                inp, mat1, mat2, beta, alpha, dtype)
                        except ConstraintError as e:
                            e1 = e
                            err_ours = True
                        assert err_torch == err_ours, f'{err_torch}(err_torch) != {err_ours}(err_ours)' + \
                            f' when testing {inp} {mat1} {mat2} {beta} {alpha} {dtype};\nTorch exception={e0}\nOur exception={e1}'
                        if err_torch:
                            continue
                        assert out_shape == out_shape_ours, f'{out_shape} != {out_shape_ours}' + \
                            f' when testing {inp} {mat1} {mat2} {beta} {alpha} {dtype}'


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
        sl = Slice(start, end, step)
        sl.extra_attrs['axis'] = axis
        sl.extra_attrs['ndims'] = len(inp)
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

    for inp in list(product([1, 7], repeat=1)) + list(product([1, 7], repeat=2)) + list(product([1, 7], repeat=3)):
        for axis in range(len(inp)):
            for start in list(range(-inp[axis], inp[axis])) + [-2**63]:
                for end in list(range(-inp[axis], inp[axis] + 1)) + [2**63 - 1]:
                    # for dtype in DTYPE_ALL:
                    dtype = DType.float32
                    step = random.randint(1, inp[axis])
                    test(inp, start, end, axis, step, dtype)

    i0, i1, i2 = z3.Ints('i0 i1 i2')
    for inp in [[i0], [i0, i1, i2]]:
        for trials in range(10):
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


def test_gemm2():
    gen = PureSymbolGen()
    gen.insert_input_node([3, 4, 5, 6])
    gen.insert_input_node([5, 6, 7, 8])
    gen.insert_input_node([7, 8, 9, 10])
    assert gen.try_insert_node_type_at(Reshape2D, [0])
    assert gen.try_insert_node_type_at(Reshape2D, [1])
    assert gen.try_insert_node_type_at(Reshape2D, [2])
    assert gen.try_insert_node(Gemm(), [3, 3, 3])
    assert gen.try_insert_node(Gemm(), [3, 4, 5])
    solution = gen.get_symbol_solutions()
    net = SymbolNet(gen.abstract_graph, solution, verbose=False)
    net.eval()
    gen.viz('output.onnx.png')
    # net.set_input_spec(input_shape)
    torch2onnx(model=net, filename='output.onnx', verbose=False)


def test_with_graph_gen():
    class NewOpOrientedGen(graph_gen.PureSymbolGen):
        def pick_next_op_type(self):
            wts = []
            for op in self.op_candidates:
                if issubclass(op, (Gemm, Slice)):
                    wts.append(50)
                elif issubclass(op, (Softmax, Cast, GELU)):
                    wts.append(5)
                else:
                    wts.append(1)
            return random.choices(self.op_candidates, wts)[0]

    def gen_once(idx, gen_cls):
        args = parse_args()
        seed = args.seed
        if seed is None:
            seed = random.getrandbits(32)
        print(f"Using seed {seed}")
        random.seed(seed)
        strt_time = time.time()
        gen = gen_cls(min_dims=args.min_dims,
                      viz_sbs=args.viz_sbs, seed=seed, verbose=args.verbose, use_bitvec=args.use_bitvec)
        gen.abstract_gen(max_node_size=args.max_nodes,
                         max_gen_millisec=args.timeout)
        print(
            f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')

        solution = gen.get_symbol_solutions()
        # print(
        #     f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
        # print(solution)

        gen.viz(d + f'/output-{idx}.onnx.png')

        net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose)
        net.eval()
        torch2onnx(model=net, filename=d +
                   f'/output-{idx}.onnx', verbose=args.verbose)
        del net, gen

    d = tempfile.mkdtemp(dir='.')
    print('creating tmp dir:', d)
    for i in tqdm(range(100)):
        gen_once(i, NewOpOrientedGen)
    os.system(f'rm -rf {d}')
    d = tempfile.mkdtemp(dir='.')
    print('creating tmp dir:', d)
    for i in tqdm(range(100)):
        gen_once(i, PureSymbolGen)
    os.system(f'rm -rf {d}')


test_cast()
test_gemm()
test_gemm2()
test_slice()
test_with_graph_gen()
print('All tests passed.')
