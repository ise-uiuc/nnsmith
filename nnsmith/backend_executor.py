from nnsmith.backends import DiffTestBackend
import multiprocessing
import pickle
from pathlib import Path
from typing import List, Union, Dict, Tuple
from queue import Empty
import time
import numpy as np
from tqdm import tqdm
from nnsmith import util
import sys
import traceback


class CrashExecutor(DiffTestBackend):
    """For testing purposes"""

    def predict(self, model, inputs):
        assert False


class HangExecutor(DiffTestBackend):
    """For testing purposes"""

    def predict(self, model, inputs):
        while True:
            pass


class BackendCreator:
    NAME_MAP = {
        'ort': 'ORTExecutor',
        'tvm-llvm': 'TVMExecutorLLVM',
        'tvm-cuda': 'TVMExecutor',
        'xla': 'XLAExecutor',
        'trt': 'TRTBackend',
    }

    def __init__(self, name):
        self.name = name
        self.dump_name = self.NAME_MAP[name]

    def __call__(self, *args, **kwargs):
        name = self.name
        if name == 'ort':
            from nnsmith.backends.ort_graph import ORTExecutor
            return ORTExecutor()
        elif name == 'tvm-llvm':
            from nnsmith.backends.tvm_graph import TVMExecutor
            return TVMExecutor(target='llvm')
        elif name == 'tvm-cuda':
            from nnsmith.backends.tvm_graph import TVMExecutor
            return TVMExecutor(target='cuda')
        elif name == 'xla':
            from nnsmith.backends.xla_graph import XLAExecutor
            return XLAExecutor(device='CUDA')
        elif name == 'trt':
            from nnsmith.backends.trt_graph import TRTBackend
            return TRTBackend()
        elif name == 'crash':
            return CrashExecutor()
        elif name == 'hang':
            return HangExecutor()
        else:
            raise ValueError(f'unknown backend: {name}')


def run_backend_same_proc(model_path: str, input_path: str, backend: BackendCreator, dump_raw: str):
    """This function is for debugging purpose.
    Run the backend on the same process.
    """
    backend = backend()
    if input_path is not None:
        inputs = pickle.load(Path(input_path).open('rb'))
        outputs = backend.predict(model_path, inputs)
    else:
        outputs = []
        for inp_path in tqdm(sorted(list(Path(model_path).parent.glob(f'input.*.pkl')))):
            inputs = pickle.load(inp_path.open('rb'))
            outputs.append(backend.predict(model_path, inputs))
    if dump_raw is not None:
        pickle.dump(outputs, open(dump_raw, 'wb'))


def summarize(outputs: Dict[str, np.ndarray]):
    m = {k + '_mean': np.mean(o) for k, o in outputs.items()}
    # TODO(JK): figure out how to deal with nan
    m.update({k + '_nanmean': np.nanmean(o) for k, o in outputs.items()})
    m.update({k + '_num_nan': np.sum(np.isnan(o)) for k, o in outputs.items()})
    return m


def run_backend(root: str, backend_creator: BackendCreator, timeout: int, selected_models: List[str] = None):
    def run(q: multiprocessing.Queue, r: multiprocessing.Queue):
        backend = backend_creator()
        while True:
            task = q.get()
            # q.task_done()
            if task is None:
                r.put(True)
                break
            # add a placeholder for the output. If the backend crashes, the output will be None
            model, inp_path, out_path = task
            pickle.dump({'exit_code': 1, 'outputs': None},
                        out_path.open('wb'))
            inputs = pickle.load(inp_path.open('rb'))  # type: List[Dict[str, np.ndarray]] # noqa
            with util.stdout_redirected(f"{out_path}.stdout"), \
                    util.stdout_redirected(f"{out_path}.stderr", sys.stderr):
                try:  # try block to catch the exception message before the rediction exits
                    outputs = backend.predict(model, inputs)
                    outputs = summarize(outputs)
                except:
                    traceback.print_exc()
                    return
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
    model_folders = sorted(list(model_root.glob('*/')))
    if selected_models is not None:
        model_folders = [m for m in model_folders if m.name in selected_models]
    print(f'Found {len(model_folders)} models at {model_root}')
    for model_folder in tqdm(model_folders):
        model_name = model_folder.name
        for inp_path in sorted(list(model_folder.glob(f'input.*.pkl'))):
            idx = inp_path.stem.split('.')[-1]
            out_path = output_dir / \
                f'{model_name}/{bknd.dump_name}.output.{idx}.pkl'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            task = (str(model_folder / 'model.onnx'), inp_path, out_path)
            print(f'testing {model_folder} on input {inp_path}')
            run_task(task)

    run_task(None)
    p.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./tmp')
    parser.add_argument('--backend', type=str, required=True,
                        help='One of ort, trt, tvm, and xla')
    parser.add_argument('--timeout', type=int, default=5 * 60,
                        help='timeout in seconds')
    parser.add_argument('--model', type=str,
                        help='For debugging purpose: path to onnx model;')
    parser.add_argument('--input', type=str,
                        help='For debugging purpose: path to input pkl file. If not specified, it will run on all inputs found within the same folder as the model')
    parser.add_argument(
        '--dump_raw', help='For debugging purposes. Dumps the raw output instead of summary to the specified path')
    parser.add_argument('--select_model', nargs='*',
                        help='Run the selected models only')
    # TODO: Add support for passing backend-specific options
    args = parser.parse_args()
    if args.root is None and args.select_model is not None:
        raise ValueError('--root is required when --select_model is used')

    bknd = BackendCreator(args.backend)
    if args.model is None:
        run_backend(args.root, bknd, args.timeout, args.select_model)
    else:
        run_backend_same_proc(args.model, args.input, bknd, args.dump_raw)
