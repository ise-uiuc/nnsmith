import multiprocessing
from pathlib import Path
import pickle
import sys
import time
import traceback

from tqdm import tqdm
from nnsmith import input_gen, util
from nnsmith.backends import DiffTestBackend
from nnsmith.backend_executor import BackendCreator, summarize
from typing import List, Union, Dict, Tuple
from queue import Empty
import psutil


def run_backend(root: str, output_dir: str, backend_creator: BackendCreator, timeout: int, selected_models: List[str] = None, gen_input: int = None):
    def run(q: multiprocessing.Queue, r: multiprocessing.Queue):
        backend = backend_creator()
        while True:
            task = q.get()
            # q.task_done()
            if task is None:
                r.put(True)
                break
            # add a placeholder for the output. If the backend crashes or the input is skipped, the output will be None
            model, inp_path, out_path, skip = task
            pickle.dump({'exit_code': 1, 'outputs': None, 'infer_succ': None},
                        out_path.open('wb'))
            if skip:  # crashed before on the same model. so skip later same-model tasks
                r.put(True)
                continue
            if isinstance(inp_path, Path):
                inputs = pickle.load(inp_path.open('rb'))  # type: List[Dict[str, np.ndarray]] # noqa
                infer_succ = None
            else:
                rngs, input_spec, seed = inp_path
                # used to filter out false positive due to nan
                infer_succ = (rngs is not None)
                inputs = input_gen.gen_one_input_rngs(input_spec, rngs, seed)
            with util.stdout_redirected(f"{out_path}.stdout"), \
                    util.stdout_redirected(f"{out_path}.stderr", sys.stderr):
                try:  # try block to catch the exception message before the rediction exits
                    outputs = backend.predict(model, inputs)
                    outputs = summarize(outputs)
                except:
                    traceback.print_exc()
                    return
            pickle.dump({'exit_code': 0, 'outputs': outputs, 'infer_succ': infer_succ},
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
        """Returns true if the backend is crashed"""
        assert q.empty()
        q.put(task)
        st_time = time.time()
        done = False
        crashed = False
        while st_time + timeout > time.time():
            try:
                done = r.get(block=False)
            except Empty:
                pass
            if done:
                break
            if not p.is_alive():
                crashed = True
                break
            time.sleep(0.01)
        # timeout; skip this task and restart the worker
        if not done or not p.is_alive():
            re_start_worker()
        return crashed

    p = None  # type: multiprocessing.Process
    q = None  # type: multiprocessing.Queue
    r = None  # type: multiprocessing.Queue
    re_start_worker()
    model_root = Path(root) / 'model_input'
    if output_dir is None:
        output_dir = Path(root) / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_folders = sorted(list(model_root.glob('*/')))
    if selected_models is not None:
        model_folders = [Path(i).parent for i in selected_models]
    print(
        f'Found {len(model_folders)} models at {model_root} with selected_model {selected_models}')
    for model_folder in tqdm(model_folders):
        crashed = False
        model_name = model_folder.name
        inp_spec = DiffTestBackend.analyze_onnx_io(
            DiffTestBackend.get_onnx_proto(str(model_folder / f'model.onnx')))[0]
        if gen_input is None:
            assert len(list(model_folder.glob(f'input.*.pkl'))
                       ) > 0, 'No input data found. Do you want to gen input on the fly but forget to specify gen_input?'
            for inp_path in sorted(list(model_folder.glob(f'input.*.pkl'))):
                idx = inp_path.stem.split('.')[-1]
                out_path = output_dir / \
                    f'{model_name}/{bknd.dump_name}.output.{idx}.pkl'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                task = (str(model_folder / 'model.onnx'),
                        inp_path, out_path, crashed)
                print(f'testing {model_folder} on input {inp_path}')
                if run_task(task):
                    crashed = True
        else:
            rngs = pickle.load(
                (model_folder / 'domain.pkl').open('rb'))
            for idx in range(gen_input):
                out_path = output_dir / \
                    f'{model_name}/{bknd.dump_name}.output.{idx}.pkl'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                inp_info = rngs, inp_spec, idx
                task = (str(model_folder / 'model.onnx'),
                        inp_info, out_path, crashed)
                print(f'testing {model_folder} on input #{idx}')
                if run_task(task):
                    crashed = True

    assert p.is_alive()
    q.put(None)
    p.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./tmp')
    parser.add_argument('--backend', type=str, required=True,
                        help='One of ort, trt, tvm, and xla')
    parser.add_argument('--timeout', type=int, default=5 * 60,
                        help='timeout in seconds')
    parser.add_argument(
        '--dump_raw', help='For debugging purposes. Dumps the raw output instead of summary to the specified path')
    parser.add_argument('--select_model', nargs='*',
                        help='Paths to model.onnx. Run the selected models only. The output will be generated at args.output_dir/<model_name>/<backend>.output.pkl')
    parser.add_argument(
        '--output_dir', help='by default it is args.root/output')
    parser.add_argument('--gen_input', type=int, default=10,
                        help='Should be an int, means number of input data to test (will be randomly generated on the fly)')
    parser.add_argument('--seed', type=int, default=0,
                        help='to generate random input data')

    # TODO: Add support for passing backend-specific options
    args = parser.parse_args()
    if args.root is None and args.select_model is not None:
        raise ValueError('--root is required when --select_model is used')

    bknd = BackendCreator(args.backend)
    st = time.time()
    if args.model is None:
        run_backend(args.root, args.output_dir, bknd, args.timeout,
                    args.select_model, args.gen_input)
    print(f'Total time: {time.time() - st}')
