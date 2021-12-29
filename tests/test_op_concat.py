from nnsmith.abstract.op import *
from nnsmith import graph_gen
from nnsmith.graph_gen import PureSymbolGen, SymbolNet, parse_args
from nnsmith.export import torch2onnx
import time
import tempfile
import os

auto_infer_in_dtypes()


def test_concat():
    # bcast tests
    p0, p1, p2, p3, p4, p5 = z3.Ints('p0 p1 p2 p3 p4 p5')
    shapes = (2, 1), (3, 1), (1, 1)
    shapes_var = [ShapeVar(s, DType.float32) for s in shapes]
    torch_sh = list(torch.concat([torch.zeros(i) for i in shapes]).shape)
    my_sh = Concat0D_3().shape_fn(shapes_var)[0].shape
    assert torch_sh == my_sh, (torch_sh, my_sh)
    assert all(Concat0D_3().requires(shapes_var))

    shapes = (1, 1, 2), (2, 3, 1), (1, 1, 1)
    shapes_var = [ShapeVar(s, DType.float32) for s in shapes]
    assert not all(Concat0D_3().requires(shapes_var))


def test_concat_v2():

    gen = PureSymbolGen(verbose=True)
    gen.insert_input_node([1, 10, 20, 30])
    gen.insert_input_node([1, 10, 20, 30])
    # gen.insert_input_node([1, 10, 20, 30])
    # gen.insert_input_node([1, 10, 20, 30])
    assert gen.try_insert_node(Concat0D_2(), [0, 1])
    assert gen.try_insert_node(Concat0D_2(), [0, 2])
    assert gen.try_insert_node(Add(), [0, 3])
    solution = gen.get_symbol_solutions()
    gen.viz('output.onnx.png')
    net = SymbolNet(gen.abstract_graph, solution, verbose=False)
    net.eval()
    # net.set_input_spec(input_shape)
    torch2onnx(model=net, filename='output.onnx', verbose=True)


def test_concat_with_graph_gen():
    class ConcatOrientedGen(graph_gen.PureSymbolGen):
        def pick_next_op_type(self):
            wts = []
            for op in self.op_candidates:
                if isinstance(op, Concat):
                    wts.append(50)
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
        print(
            f'{len(solution)} symbols and {len(gen.solver.assertions())} constraints.')
        print(solution)

        gen.viz(d + f'/output-{idx}.onnx.png')

        net = SymbolNet(gen.abstract_graph, solution, verbose=args.verbose)
        net.eval()
        torch2onnx(model=net, filename=d + f'/output-{idx}.onnx', verbose=True)
        del net, gen

    d = tempfile.mkdtemp(dir='.')
    print('creating tmp dir:', d)
    for i in range(100):
        gen_once(i, ConcatOrientedGen)
    os.system(f'rm -rf {d}')
    d = tempfile.mkdtemp(dir='.')
    print('creating tmp dir:', d)
    for i in range(100):
        gen_once(i, PureSymbolGen)
    os.system(f'rm -rf {d}')


test_concat()
test_concat_v2()
test_concat_with_graph_gen()
print('All tests passed.')
