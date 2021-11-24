from queue import Empty
from typing import List, Union, Dict, Tuple
import onnx
import numpy as np
from numpy import testing

from nnsmith.backends import DiffTestBackend
from nnsmith.error import *
import glob
import pickle
import multiprocessing
from pathlib import Path
import time
from tqdm import tqdm


def assert_allclose(obtained: Dict[str, np.ndarray], desired: Dict[str, np.ndarray], obtained_name: str, oracle_name: str, nan_as_err=True):
    err_msg = ''
    if obtained is None:
        err_msg += f'{obtained_name} crashed'
    if desired is None:
        err_msg += f'{oracle_name} crashed'
    if err_msg != '':
        raise CrashError(err_msg)

    if set(obtained.keys()) != set(desired.keys()):
        print('Key sets differ')
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} have different output tensor names')

    if nan_as_err:
        for index, key in enumerate(obtained):
            err_msg = ''
            if np.isnan(obtained[key]).any():
                err_msg += f'{obtained_name} has NaN, '
            if np.isnan(desired[key]).any():
                err_msg += f'{oracle_name} has NaN'
            if err_msg != '':
                err_msg = f'At tensor #{index}: ' + err_msg
                # print(err_msg) # Mute.
                raise NaNError(err_msg)

    try:
        index = -1
        assert set(obtained.keys()) == set(desired.keys())
        index = 0
        for key in obtained:
            testing.assert_allclose(
                obtained[key], desired[key], rtol=1e-02, atol=1e-05)
            index += 1
    except AssertionError as err:
        # print(err) # Mute.
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor: {str(err)}')


def run_backend_same_proc(model_path: str, input_path: str, backend: DiffTestBackend):
    """This function is for debugging purpose.
    Run the backend on the same process.
    """
    inputs = pickle.load(Path(input_path).open('rb'))
    outputs = backend.predict(model_path, inputs)


def summarize(outputs: Dict[str, np.ndarray]):
    m = {k + '_mean': np.mean(o) for k, o in outputs.items()}
    # TODO(JK): figure out how to deal with nan
    m.update({k + '_nanmean': np.nanmean(o) for k, o in outputs.items()})
    m.update({k + '_num_nan': np.sum(np.isnan(o)) for k, o in outputs.items()})
    return m


def run_backend(root: str, backend: DiffTestBackend, timeout: int):
    def run(q: multiprocessing.Queue, r: multiprocessing.Queue):
        while True:
            task = q.get()
            # q.task_done()
            if task is None:
                r.put(True)
                break
            # add a placeholder for the output. If the backend crashes, the output will be None
            model, inp_path, out_path = task
            pickle.dump({'exit_code': 1, 'outputs': None}, out_path.open('wb'))
            # type: List[Dict[str, np.ndarray]]
            inputs = pickle.load(inp_path.open('rb'))
            outputs = backend.predict(model, inputs)
            outputs = summarize(outputs)
            pickle.dump({'exit_code': 0, 'outputs': outputs},
                        out_path.open('wb'))
            r.put(True)

    def re_start_worker():
        """(Re)start a (dead) worker process"""
        nonlocal p, q, r
        q = multiprocessing.Queue()
        r = multiprocessing.Queue()
        if p is not None:
            p.kill()
            p.join()
        p = multiprocessing.Process(target=run, args=(q, r))
        p.start()

    def run_task(task):
        assert q.empty()
        q.put(task)
        st_time = time.time()
        done = False
        while st_time + timeout > time.time():
            try:
                done = r.get(block=False)
            except Empty:
                pass
            if done:
                break
            if not p.is_alive():
                break
            time.sleep(0.01)
        # timeout; skip this task and restart the worker
        if not done:
            re_start_worker()

    p = None  # type: multiprocessing.Process
    q = None  # type: multiprocessing.Queue
    r = None  # type: multiprocessing.Queue
    re_start_worker()
    model_root = Path(root) / 'model_input'
    output_dir = Path(root) / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(list(model_root.glob('*/')))
    print(f'Found {len(model_paths)} models at {model_root}')
    for model_path in tqdm(model_paths):
        model_name = model_path.name
        for inp_path in sorted(list(model_path.glob(f'input.*.pkl'))):
            idx = inp_path.stem.split('.')[-1]
            out_path = output_dir / \
                f'{model_name}/{backend.__class__.__name__}.output.{idx}.pkl'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            task = (str(model_path / 'model.onnx'), inp_path, out_path)
            print(f'testing {model_path} on input {inp_path}')
            run_task(task)

    run_task(None)
    p.join()


def difftest(root: str):
    """
    This function compares the outputs of each backend and generate bug reports.
    Note: `run_backend` must run before this function.
    Args:
        output_dir: path to the directory storing the outputs.
    """
    # file structure:
    # root: /path/to/root
    # model_root: ${root}/model_and_input/
    # - all models: ${model_root}/${model_name}/model.onnx
    # - i-th input: ${model_root}/${model_name}/input.${i}.pkl
    # output_dir: ${root}/output
    # - i-th output: ${output_dir}/${model_name}/output.${i}.pkl

    # input and output pickle format:
    # inputs.pkl: Dict[str, np.ndarray]
    # outputs.pkl: {'exit_code': 0, 'outputs': outputs}, where outputs is of type Dict[str, np.ndarray]

    root = Path(root)
    output_dir = root / 'output'
    # numerical consistency check
    report = []
    for model_path in sorted(output_dir.glob('*/')):  # reproducibility
        model_name = model_path.name

        def get_meta_info():
            a = list(model_path.glob(f'*.output.*.pkl'))
            bknd_names = set(map(lambda x: x.name.split('.')[0], a))
            num_out = len(a) // len(bknd_names)
            assert num_out == len(list(
                (output_dir.parent / 'model_input' /
                 model_name).glob(f'input.*.pkl')
            )), 'inputs and outputs are not matched. Do you forget to run_backends?'
            assert len(a) % len(bknd_names) == 0, \
                f'{model_name} has {len(a)} outputs, but {len(bknd_names)} backends which cannot divide'
            return num_out, bknd_names

        def get_output(backend_name: str, idx: str) -> Tuple[Dict[str, np.ndarray], str]:
            out_path = output_dir / \
                f'{model_name}/{backend_name}.output.{idx}.pkl'
            return pickle.load(out_path.open('rb'))['outputs'], str(out_path)

        # TODO(JK): use more advanced oracle (e.g., clustering?) if this does not work well
        num_out, bknd_names = get_meta_info()
        bknd_names = sorted(bknd_names)  # sort for reproducibility
        for i in range(num_out):
            oracle_path = None
            for backend in bknd_names:
                output, out_path = get_output(backend, i)
                if oracle_path is None:
                    # read oracle's data (assume first backend as oracle)
                    oracle = output
                    oracle_path = out_path
                    oracle_name = Path(out_path).name.split('.')[0]
                    continue
                try:
                    assert_allclose(output, oracle, out_path, oracle_path)
                except ModeledError as err:
                    report.append({
                        'model_idx': model_path.name,
                        'input_path': str(output_dir.parent / 'model_input' / model_name / f'input.{i}.pkl'),
                        'backend': backend,
                        'oracle': oracle_name,
                        'input_idx': i,
                        'output_backend': out_path,
                        'output_oracle': oracle_path,
                        'error': err})
                    print(err)
    import json
    pickle.dump(report, open(root / 'report.pkl', 'wb'))
    for i in report:
        i['error'] = str(i['error'])
    json.dump(report, open(root / 'report.json', 'w'), indent=2)
    if len(report) > 0:
        print(f'{len(report)} differences found!!!')
    else:
        print('No differences found!')


if __name__ == '__main__':  # generate bug reports.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./tmp')
    args = parser.parse_args()
    difftest(args.root)
