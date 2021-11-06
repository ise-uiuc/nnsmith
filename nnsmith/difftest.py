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

def assert_allclose(obtained: Dict[str, np.ndarray], desired: Dict[str, np.ndarray], obtained_name: str, oracle_name: str):
    try:
        index = -1
        assert obtained is not None, f'{obtained_name} crashed'
        assert desired is not None, f'{oracle_name} crashed'
        assert set(obtained.keys()) == set(desired.keys())
        index = 0
        for key in obtained:
            testing.assert_allclose(obtained[key], desired[key], rtol=1e-02, atol=1e-05)
            index += 1
    except AssertionError as err:
        print(err)
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor:')


def run_backend(model_root: str, backend: DiffTestBackend):
    def run(model: Path):
        inputs = pickle.load(inp_path.open('rb')) # type: List[Dict[str, np.ndarray]]
        outputs = backend.predict(model, inputs)
        pickle.dump({'exit_code': 0, 'outputs': outputs}, out_path.open('wb'))

    model_root = Path(model_root) # type: Path
    output_dir = model_root.parent/'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_path in model_root.glob('*/'):
        model_name = model_path.name
        for inp_path in model_path.glob(f'input.*.pkl'):
            idx = inp_path.stem.split('.')[-1]
            out_path = output_dir / f'{model_name}/{backend.__class__.__name__}.output.{idx}.pkl'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # TODO(JK): reuse process, redirect stdout/stderr to file?
            p = multiprocessing.Process(target=run, 
                args=(str(model_path/'model.onnx'),))
            p.start()
            p.join()
            if p.exitcode != 0:
                pickle.dump({'exit_code': p.exitcode, 'outputs': None}, out_path.open('wb'))

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
    for model_path in output_dir.glob('*/'):
        model_name = model_path.name

        def get_meta_info():
            a = list(model_path.glob(f'*.output.*.pkl'))
            bknd_names = set(map(lambda x: x.name.split('.')[0], a))
            num_out = len(a) // len(bknd_names)
            assert len(a) % len(bknd_names) == 0, \
                f'{model_name} has {len(a)} outputs, but {len(bknd_names)} backends which cannot divide'
            return num_out, bknd_names

        def get_output(backend_name: str, idx: str) -> Tuple[Dict[str, np.ndarray], str]:
            out_path = output_dir / f'{model_name}/{backend_name}.output.{idx}.pkl'
            return pickle.load(out_path.open('rb'))['outputs'], str(out_path)

        # TODO(JK): use more advanced oracle (e.g., clustering?) if this does not work well
        num_out, bknd_names = get_meta_info()
        for i in range(num_out):
            oracle = None
            for backend in bknd_names:
                output, out_path = get_output(backend, i)
                if oracle is None:
                    # read oracle's data (assume first backend as oracle)
                    oracle = output
                    oracle_path = out_path
                    oracle_name = out_path.split('.')[0]
                    continue
                try:
                    assert_allclose(output, oracle, out_path, oracle_path)
                except IncorrectResult as err:
                    report.append({
                        'model_path': str(model_path), 
                        'backend': backend, 
                        'oracle': oracle_name,
                        'input_idx': i,
                        'error': str(err)})
                    print(err)
    import json
    json.dump(report, open(root /'report.json', 'w'), indent=2)