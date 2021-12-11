from nnsmith.abstract.op import *
from nnsmith import graph_gen
from nnsmith.graph_gen import PureSymbolGen, SymbolNet, parse_args
from nnsmith.export import torch2onnx
import time
import tempfile
import os


def test_bcast():
    # bcast tests
    p0, p1, p2, p3, p4, p5 = z3.Ints('p0 p1 p2 p3 p4 p5')
    shapes = (2,), (3, 1), (1, 1, 1)
    shapes_var = [ShapeVar(s, DType.float32) for s in shapes]
    assert list(torch.broadcast_shapes(*shapes)
                ) == broadcast_shapes(*(i.shape for i in shapes_var))
    shapes = (p0,), (p1, 3), (1, 1, 1)
    shapes_var = [ShapeVar(s, DType.float32) for s in shapes]
    # print(broadcast_cons(*shapes_var))
    shapes = (p0,), (p1, 3), (1, 1, 2)
    shapes_var = [ShapeVar(s, DType.float32) for s in shapes]
    # print(broadcast_cons(*shapes_var))
    assert z3.is_false(z3.simplify(
        broadcast_cons(*(i.shape for i in shapes_var))[0]))

    for x1 in [p0, 1, 3]:
        for x2 in [p1, 1, 3]:
            shapes = (x1,), (x2,)
            shapes_var = [ShapeVar(s, DType.float32) for s in shapes]
            cons1 = broadcast_cons(*(i.shape for i in shapes_var))
            cons2 = broadcast_cons_binary(*(i.shape for i in shapes_var))
            s = z3.Solver()
            assert s.check(z3.And(*cons1) != z3.And(*cons2)) == z3.unsat


def test_bcast_add():
    # Add
    a = torch.randn(2, 1, 4, 5)
    b = torch.randn(3, 1, 5)
    c = a + b
    assert c.shape == torch.Size(Add().shape_fn(
        [ShapeVar(list(a.shape), DType.float32), ShapeVar(list(b.shape), DType.float32)])[0].shape)

    i0, i1, i2, i3 = z3.Ints('i0 i1 i2 i3')
    ash = ShapeVar([i0, i1, 5], DType.float32)
    bsh = ShapeVar([3, i2, 1, i3], DType.float32)
    csh = Add().shape_fn([ash, bsh])[0]
    cons = Add()._requires([ash, bsh])
    cons.extend([i >= 1 for i in ash.shape])
    cons.extend([i >= 1 for i in bsh.shape])
    cons.extend([i >= 1 for i in csh.shape])
    s = z3.Solver()
    s.add(*cons)
    assert s.check() == z3.sat
    # print(s.model())

    s.add(i1 > 3)
    assert s.check() == z3.sat
    # print(s.model())

    s.add(i3 > 3)
    assert s.check() == z3.sat
    # print(s.model())

    s.add(i3 > 5)
    assert s.check() == z3.unsat


def test_bcast_with_graph_gen():
    class BcastOrientedGen(graph_gen.PureSymbolGen):
        def pick_next_op_type(self):
            wts = []
            for op in self.op_candidates:
                if issubclass(op, (Input, Constant)):
                    wts.append(5)
                elif op.bcastable:
                    wts.append(50)
                else:
                    wts.append(1)
            return random.choices(self.op_candidates, wts)[0]

    def gen_once(idx, gen_cls):
        args = parse_args()
        seed = args.seed
        if seed is None:
            # If we have not selected a seed, choose random one.
            seed = random.getrandbits(32)
        print(f"Using seed {seed}")
        random.seed(seed)

        z3.set_param(
            "smt.phase_selection",
            5,
            "smt.arith.random_initial_value",
            True,
            "sat.phase",
            "random",
        )

        strt_time = time.time()
        gen = gen_cls(min_dims=args.min_dims,
                      viz_sbs=args.viz_sbs, seed=seed, verbose=args.verbose, use_bitvec=args.use_bitvec)
        gen.abstract_gen(max_node_size=args.max_nodes,
                         max_gen_millisec=args.timeout)
        print(
            f'{time.time() - strt_time}s to generate a graph w/ {len(gen.abstract_graph.nodes())} nodes')

        solution = gen.get_symbol_solutions()
        print(
            f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
        print(solution)

        gen.viz(d + f'/output-{idx}.onnx.png')

        # input_shape = gen.concretize_input_shape(solution)
        # print(f'Input shape: {input_shape}')

        net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose)
        net.eval()
        # net.set_input_spec(input_shape)
        torch2onnx(model=net, filename=d + f'/output-{idx}.onnx', verbose=True)
        del net, gen
    d = tempfile.mkdtemp(dir='.')
    print('creating tmp dir:', d)
    for i in range(100):
        gen_once(i, BcastOrientedGen)
    os.system(f'rm -rf {d}')
    d = tempfile.mkdtemp(dir='.')
    print('creating tmp dir:', d)
    for i in range(100):
        gen_once(i, PureSymbolGen)
    os.system(f'rm -rf {d}')


test_bcast()
test_bcast_add()
test_bcast_with_graph_gen()
