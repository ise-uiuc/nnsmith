import gc
import os
from pathlib import Path
import pickle
import cloudpickle
from subprocess import check_call
import multiprocessing as mp
import traceback
import psutil
import time
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from nnsmith.abstract.op import ALL_OP_STR2TYPE, ALL_OP_TYPES, _op_set_use_cuda
from nnsmith.backends import DiffTestBackend
from nnsmith.error import SanityCheck
from nnsmith.graph_gen import SymbolNet, random_model_gen
from nnsmith.export import torch2onnx
import util

# pool should be better when backends are also running.
# Using fork for consistency with previous setting.
_DEFAULT_FORK_METHOD = 'fork'


class ModelGenSubProcesssError(Exception):
    pass


def safe_wrapper(func):
    def wrapper(*args, **kwargs):
        succ = False
        while not succ:
            try:
                res = func(*args, **kwargs)
                succ = True
            except Exception as e:
                traceback.print_exc()
                print('retrying...')
        return res
    return wrapper


def subprocess_call(gen_method, seed, max_nodes, max_gen_millisec, inp_gen, output_path, use_bitvec, merge_op_v, limnf, use_cuda, ipc_dict):
    _op_set_use_cuda(use_cuda)
    ipc_dict['seed'] = seed
    profile = ipc_dict['profile']

    gen_model_st = time.time()
    if gen_method == 'random':
        gen, solution = random_model_gen(
            max_nodes=max_nodes, timeout=max_gen_millisec, use_bitvec=use_bitvec, merge_op_v=merge_op_v, limnf=limnf, seed=seed)
    elif gen_method == 'guided':
        from nnsmith.graph_gen import GuidedGen
        gen = GuidedGen(
            summaries=ipc_dict['state']['summaries'], use_bitvec=use_bitvec, merge_op_v=merge_op_v, limnf=limnf, seed=seed)
        gen.abstract_gen(max_node_size=max_nodes,
                         max_gen_millisec=max_gen_millisec)
        solution = gen.get_symbol_solutions()
    else:
        SanityCheck.true(False, f'Unknown gen_method: {gen_method}')
    symnet_st = time.time()
    profile['model_gen_t'] = symnet_st - gen_model_st
    net = SymbolNet(gen.abstract_graph, solution, verbose=False,
                    alive_shapes=gen.alive_shapes)
    gen_input_st = time.time()
    profile['symnet_init_t'] = gen_input_st - symnet_st

    sat_inputs = None
    if inp_gen == 'random':
        with torch.no_grad():
            net.eval()
            sat_inputs = net.rand_input_gen(use_cuda=use_cuda)
    elif inp_gen == 'grad':
        net.eval()
        sat_inputs = net.grad_input_gen(use_cuda=use_cuda)
    elif inp_gen == 'none':
        sat_inputs = None
    else:
        raise ValueError(f'Unknown inp_gen: {inp_gen}')

    if sat_inputs is not None:
        ret_inputs = {}
        for i, name in enumerate(net.input_spec):
            ret_inputs[name] = sat_inputs[i].cpu().numpy()
        ipc_dict['sat_inputs'] = ret_inputs
    else:
        ipc_dict['sat_inputs'] = None

    export_t_s = time.time()
    profile['input_gen_t'] = export_t_s - gen_input_st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch2onnx(net, output_path, use_cuda=use_cuda,
                   dummy_inputs=sat_inputs)
    dump_t_s = time.time()
    profile['export_t'] = dump_t_s - export_t_s

    cloudpickle.dump(net.concrete_graph, open(
        output_path + '-graph.pkl', 'wb'), protocol=4)
    profile['dump_t'] = time.time() - dump_t_s


# @safe_wrapper
def _forked_execution(
        gen_method, output_path, seed=None, max_nodes=10, max_gen_millisec=2000, save_torch=False, inp_gen='random',
        use_bitvec=False, summaries=None, merge_op_v=None, limnf=True, use_cuda=False):
    if seed is None:
        seed = random.getrandbits(32)

    with mp.Manager() as manager:
        # NOTE: Please only try to transfer primitive data types. e.g., str.
        # That is why I use `ALL_OP_STR2TYPE` to map strings to original types.
        # You might want to use `dill` to serialize some special stuff, e.g., lambda.
        # Also see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
        mp.set_forkserver_preload(
            ['collections', 'math', 'textwrap', 'z3', 'networkx', 'torch', 'torch.nn', 'numpy', 'pickle', 'cloudpickle', 'inspect', 'nnsmith.abstract.op.*'])
        ipc_dict = manager.dict()
        ipc_dict['state'] = manager.dict()
        ipc_dict['state']['unsolvable'] = manager.list()
        ipc_dict['state']['summaries'] = summaries
        ipc_dict['edges'] = set()
        ipc_dict['profile'] = manager.dict()
        nnsmith_fork = os.environ.get(
            'NNSMITH_FORK', _DEFAULT_FORK_METHOD)  # specify the fork method.
        if nnsmith_fork == 'pool':
            nnsmith_fork = 'fork'
        if nnsmith_fork == 'forkserver':
            # TODO(JK): integrate the initializations (skip op, infer type, etc.) and rich panel into forkserver
            warnings.warn(
                '`--skip` option may not have any effect in forkserver mode. Subprocess call output may be covered by the panel.')
        if nnsmith_fork != 'inprocess':  # let's try to get rid of fork
            p = mp.get_context(nnsmith_fork).Process(
                target=subprocess_call, args=(gen_method, seed, max_nodes, max_gen_millisec, inp_gen, output_path, use_bitvec, merge_op_v, limnf, use_cuda, ipc_dict,))

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

            SanityCheck.false(
                p.is_alive(), 'Process should be terminated but still alive.')

            if p.exitcode != 0:
                print(
                    f'Return code not zero in model generation process: {p.exitcode}')
                raise ModelGenSubProcesssError(
                    'return code not zero: {}'.format(p.exitcode))
        else:
            subprocess_call(gen_method, seed, max_nodes,
                            max_gen_millisec, inp_gen, output_path, use_bitvec, merge_op_v, limnf, use_cuda, ipc_dict)
        # make ipc_dict serializable
        del ipc_dict['state']

        return ipc_dict['sat_inputs'], ipc_dict['edges'], ipc_dict['seed'], dict(ipc_dict['profile'])


forkpool_execution = util.forkpool_execution(_forked_execution)


def forked_execution(*args, **kwargs):
    nnsmith_fork = os.environ.get(
        'NNSMITH_FORK', _DEFAULT_FORK_METHOD)  # specify the fork method.

    return _forked_execution(*args, **kwargs) if nnsmith_fork != 'pool' else forkpool_execution(*args, **kwargs)


# TODO(from Jiawei @Jinkun): stop using this implementation.
def gen_model_and_range(
        output_path,
        seed=None,
        input_gen: str = 'v3',
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
               f' --seed {seed} --max_nodes {max_node_size} --timeout {max_gen_millisec} --viz_graph --input_gen {input_gen}'
               f'{kwargs_str} 2>&1', shell=True, timeout=max_gen_millisec * 2 / 1000)
    model = DiffTestBackend.get_onnx_proto(output_path)
    stats = pickle.load(open(output_path + '-stats.pkl', 'rb'))
    rngs = stats['rngs']
    return model, rngs, stats


gen_model_and_range_safe = safe_wrapper(gen_model_and_range)


def _main(root: str, num_models, max_nodes, input_gen: str, seed=None, timeout=2000):
    if seed is not None:
        random.seed(seed)
    st_time = time.time()
    profile = []
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
            seed = random.getrandbits(32)
            print(f'seeding {seed}, attempt {atmpts}')
            gen_model_st = time.time()
            stats = {}
            try:
                _, rngs, stats = gen_model_and_range(
                    str(model_path), seed=seed, max_node_size=max_nodes, max_gen_millisec=timeout)
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
                    'succ': succ,
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
    parser.add_argument('--seed', type=int)
    parser.add_argument('--timeout', type=int, default=2000)
    args = parser.parse_args()
    if not args.input_only:
        _main(args.root, args.num_models,
              args.max_nodes, args.input_gen_method, seed=args.seed, timeout=args.timeout)
    # gen_inputs_for_all(args.root, args.num_inputs, args.model, gen_inputs_func)
