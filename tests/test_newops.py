from tqdm import tqdm
from nnsmith.abstract.op import *
from nnsmith import graph_gen
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


def test_with_graph_gen():
    class NewOpOrientedGen(graph_gen.PureSymbolGen):
        def pick_next_op_type(self):
            wts = []
            for op in self.op_candidates:
                if issubclass(op, Cast):
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
test_with_graph_gen()
print('All tests passed.')
