from pathlib import Path
import pickle
from subprocess import CalledProcessError, check_call
import numpy as np
import pandas as pd
from tqdm import tqdm
from nnsmith.backends import DiffTestBackend
from nnsmith.graph_gen import PureSymbolGen, SymbolNet, torch2onnx
from nnsmith.input_gen import InputGenBase, InputGenV1, InputGenV3
import time
import random


def safe_wrapper(func):
    def wrapper(*args, **kwargs):
        succ = False
        while not succ:
            try:
                res = func(*args, **kwargs)
                succ = True
            except CalledProcessError:
                pass
        return res
    return wrapper


# ATTENTION: This changes random seed
# @util.forked  # call this function in a forked process to work around unknown z3 issues
def gen_model_and_range(
        output_path,
        seed=None,
        # input_gen: InputGenBase = InputGenV3(),
        max_node_size=10,
        max_gen_millisec=2000,
        **kwargs):
    """Generate a model and its input range.

    Parameters
    ----------
        output_path : str
            onnx model output path
        seed : int
            random seed. When None, it will randomly generate a seed.

    Returns
    -------
        model : onnx.ModelProto
            generated model
        rngs : List[Tuple[float, float]]
            input ranges
        stats : dict 
            statistics

    Example usage:

    for i in range(10):
        print(gen_model_and_range(
            './output.onnx', seed=i, max_node_size=20)[1:])
    """
    # gen_model_st = time.time()
    # if seed is None:
    #     seed = random.getrandbits(32)
    # random.seed(seed)
    # gen = PureSymbolGen(**kwargs)
    # gen.abstract_gen(max_node_size=max_node_size,
    #                  max_gen_millisec=max_gen_millisec)
    # gen.viz(output_path + '.png')
    # solution = gen.get_symbol_solutions()
    # net = SymbolNet(gen.abstract_graph, solution,
    #                 alive_shapes=gen.alive_shapes)
    # net.eval()

    # torch2onnx(model=net, filename=output_path)
    # model = DiffTestBackend.get_onnx_proto(output_path)

    # input_st = time.time()
    # rngs = input_gen.infer_domain(model)
    # infer_succ = rngs is not None
    # ed_time = time.time()

    # stats = {
    #     'gen_succ': True,
    #     'infer_succ': infer_succ,
    #     'elpased_time': ed_time - gen_model_st,
    #     'gen_model_time': input_st - gen_model_st,
    #     'infer_domain_time': ed_time - input_st,
    #     'seed': seed,
    # }

    if seed is None:
        seed = random.getrandbits(32)
    kwargs_str = ' '.join([f'--{k} {v}' for k, v in kwargs.items()])
    check_call(f'python -u -m nnsmith.graph_gen --output_path {output_path}'
               f' --seed {seed} --max_nodes {max_node_size} --timeout {max_gen_millisec} '
               f'{kwargs_str} 2>&1', shell=True)
    model = DiffTestBackend.get_onnx_proto(output_path)
    stats = pickle.load(open(output_path + '-stats.pkl', 'rb'))
    rngs = stats['rngs']
    return model, rngs, stats


gen_model_and_range_safe = safe_wrapper(gen_model_and_range)


def _main(root: str, num_models, max_nodes, input_gen: InputGenBase):
    st_time = time.time()
    profile = []
    profile_inputs = []
    root = Path(root)  # type: Path
    if len(list(root.glob('model*'))) > 0:
        raise Exception(f'Folder {root} already exists')
    model_root = root / 'model_input'
    model_root.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(num_models)):
        model_path = model_root / f'{i}' / 'model.onnx'
        model_path.parent.mkdir(exist_ok=True, parents=True)
        succ = False
        atmpts = 0
        while not succ:
            atmpts += 1
            e1 = None
            infer_succ = gen_succ = False
            input_st = np.nan
            seed = int(time.time() * 1000)
            print(f'seeding {seed}, attempt {atmpts}')
            gen_model_st = time.time()
            stats = {}
            try:
                # check_call(
                #     f'python -u -m nnsmith.graph_gen --output_path {model} --seed {seed} {gen_args} 2>&1', shell=True)
                # gen_succ = True
                # input_st = time.time()

                # # infer input range
                # rngs = input_gen.infer_domain(
                #     DiffTestBackend.get_onnx_proto(str(model)))
                # infer_succ = rngs is not None
                # pickle.dump(rngs, open(model.parent / 'domain.pkl', 'wb'))

                # succ = True
                _, rngs, stats = gen_model_and_range(
                    str(model_path), seed=seed, max_node_size=max_nodes)
                succ = True
            except Exception as e:
                e1 = e
                import traceback
                traceback.print_exc()
            finally:
                ed_time = time.time()
                x = {
                    'model': str(model_path),
                    'atmpts': atmpts,
                    'gen_succ': gen_succ,
                    'infer_succ': infer_succ,
                    'elpased_time': ed_time - gen_model_st,
                    'gen_model_time': input_st - gen_model_st,
                    'infer_domain_time': ed_time - input_st,
                    'seed': seed,
                }
                x.update(stats)
                profile.append(x)
            if e1 is not None:
                profile[-1]['error'] = e1
        pickle.dump(rngs, open(model_path.parent / 'domain.pkl', 'wb'))
    print('gen_model elapsed time={}s'.format(time.time() - st_time))
    profile = pd.DataFrame(profile)
    profile.to_pickle(Path(root) / 'gen_model_profile.pkl')
    profile.to_csv(Path(root) / 'gen_model_profile.csv')

    # profile_inputs = pd.DataFrame(profile_inputs)
    # profile_inputs = profile_inputs.sort_values(
    #     by=['succ_rate'], ascending=True)
    # profile_inputs.to_pickle(Path(root) / 'gen_input_profile.pkl')
    # profile_inputs.to_csv(Path(root) / 'gen_input_profile.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='tmp/seed1')
    parser.add_argument('--num_models', type=int, default=2)
    # parser.add_argument('--num_inputs', type=int, default=2)
    parser.add_argument('--input_only', action='store_true')
    parser.add_argument('--model', type=str, nargs='*',
                        help='Generate input for specific model, specified as the path to the .onnx file.')
    parser.add_argument('--input_gen_method', type=str, default='v3')
    parser.add_argument('--max_nodes', type=int)
    args = parser.parse_args()
    gen_inputs_func = {
        'v1': InputGenV1(),
        # 'v2': InputGenV2(),
        'v3': InputGenV3(),
    }[args.input_gen_method]
    if not args.input_only:
        _main(args.root, args.num_models,
              args.max_nodes, gen_inputs_func)
    # gen_inputs_for_all(args.root, args.num_inputs, args.model, gen_inputs_func)
