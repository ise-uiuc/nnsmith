from pathlib import Path
import pickle
from subprocess import CalledProcessError, check_call
import multiprocessing as mp
import psutil
import time
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from nnsmith.abstract.op import ALL_OP_TYPES

from nnsmith.backends import DiffTestBackend
from nnsmith.graph_gen import SymbolNet, torch2onnx, random_model_gen, table_model_gen
from nnsmith.input_gen import InputGenBase, InputGenV1, InputGenV3


class ModelGenSubProcesssError(Exception):
    pass


def safe_wrapper(func):
    def wrapper(*args, **kwargs):
        succ = False
        while not succ:
            try:
                res = func(*args, **kwargs)
                succ = True
            except CalledProcessError:
                pass
            except ModelGenSubProcesssError:
                pass
        return res
    return wrapper


@safe_wrapper
def forked_execution(
        gen_method, output_path, seed=None, max_nodes=10, max_gen_millisec=2000, table=None, **kwargs):
    if seed is None:
        seed = random.getrandbits(32)

    def subprocess_call(ipc_dict):
        random.seed(seed if seed is not None else random.getrandbits(32))

        if gen_method == 'random':
            gen, solution = random_model_gen(
                max_nodes=max_nodes, timeout=max_gen_millisec)
        elif gen_method == 'table':
            gen, solution = table_model_gen(
                table=table,
                state=ipc_dict['state'],
                max_nodes=max_nodes, timeout=max_gen_millisec)
            abs_graph = gen.abstract_graph
            unique_set = set()
            for src, dst in abs_graph.edges():
                pair = (ALL_OP_TYPES.index(type(abs_graph.nodes[src]['op'])), ALL_OP_TYPES.index(
                    type(abs_graph.nodes[dst]['op'])) - 1)
                unique_set.add(pair)
            ipc_dict['edges'] = unique_set
        else:
            assert False, f'Unknown gen_method: {gen_method}'

        net = SymbolNet(gen.abstract_graph, solution, verbose=False,
                        alive_shapes=gen.alive_shapes)
        net.eval()
        torch2onnx(model=net, filename=output_path, verbose=False)
        model = DiffTestBackend.get_onnx_proto(output_path)

        input_gen = InputGenV3()
        rngs = input_gen.infer_domain(model)
        ipc_dict['ranges'] = rngs

    with mp.Manager() as manager:
        ipc_dict = manager.dict()
        ipc_dict['state'] = manager.dict()
        ipc_dict['state']['unsolvable'] = manager.list()
        ipc_dict['edges'] = set()
        p = mp.Process(target=subprocess_call, args=(ipc_dict,))

        p_duration = None
        try:
            process_time_tolerance = max_gen_millisec * 2 / 1000
            p.start()
            p_strt_time = time.time()
            p.join(process_time_tolerance)
            p_duration = time.time() - p_strt_time
        finally:
            if p.is_alive():
                for child in psutil.Process(p.pid).children(recursive=False):
                    child: psutil.Process  # type: ignore
                    try:
                        child.terminate()
                        child.wait()
                    except psutil.NoSuchProcess:
                        pass
                p.terminate()
                p.join()

                if p_duration is not None and p_duration >= process_time_tolerance:
                    print('We got timeout due to process hang.')
                    raise ModelGenSubProcesssError(
                        f'process hang {process_time_tolerance}')

            for src, dst in ipc_dict['state']['unsolvable']:
                table.on_unsolvable(src, dst)

        assert not p.is_alive()

        if p.exitcode != 0:
            print(
                f'Return code not zero in model generation process: {p.exitcode}')
            raise ModelGenSubProcesssError(
                'return code not zero: {}'.format(p.exitcode))

        return ipc_dict['ranges'], ipc_dict['state'], ipc_dict['edges']


# TODO(from Jiawei @Jinkun): stop using this implementation.
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
    if seed is None:
        seed = random.getrandbits(32)
    kwargs_str = ' '.join([f'--{k} {v}' for k, v in kwargs.items()])
    check_call(f'python -u -m nnsmith.graph_gen --output_path {output_path}'
               f' --seed {seed} --max_nodes {max_node_size} --timeout {max_gen_millisec} '
               f'{kwargs_str} 2>&1', shell=True, timeout=max_gen_millisec * 2 / 1000)
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
        cont = input('The output directory already exists. Continue? y/n.')
        if cont == 'n':
            raise Exception(f'Folder {root} already exists')
        elif cont == 'y':
            pass
        else:
            raise Exception('Invalid input')
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
                    # 'elapsed_time': ed_time - gen_model_st,
                    'gen_model_time': input_st - gen_model_st,
                    'infer_domain_time': ed_time - input_st,
                    'seed': seed,
                }
                x.update(stats)
                x['elapsed_time'] = ed_time - gen_model_st
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
