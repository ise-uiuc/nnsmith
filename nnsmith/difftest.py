from .backends.ort_graph import ORTExecutor
from .backends.tvm_graph import TVMExecutor
from .backends.xla_graph import XLAExecutor
# from .backends.trt_graph import TRTBackend
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
        assert set(obtained.keys()) == set(desired.keys())
        index = 0
        for key in obtained:
            testing.assert_allclose(obtained[key], desired[key], rtol=1e-02, atol=1e-05)
            index += 1
    except AssertionError as err:
        print(err)
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor:')


def run_backend(model_root: Path, backend: DiffTestBackend, output_dir: Path):
    def run(model: Path):
        inputs = pickle.load(inp_path.open('rb')) # type: List[Dict[str, np.ndarray]]
        outputs = backend.predict(model, inputs)
        pickle.dump({'exit_code': 0, 'outputs': outputs}, out_path.open('wb'))

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
                pickle.dump({'exit_code': p.exitcode, 'outputs': []}, out_path.open('wb'))

def difftest(model_root: str, backends: Union[List[DiffTestBackend], None] = None,
    output_dir: str = None):
    # file structure:
    # root: /path/to/root
    # model_root: ${root}/model_and_input/
    # - all models: ${model_root}/${model_name}/model.onnx
    # - i-th input: ${model_root}/${model_name}/input.${i}.pkl
    # output_dir: default to ${root}/output

    # input and output pickle format:
    # inputs.pkl: Dict[str, np.ndarray] 
    # outputs.pkl: {'exit_code': 0, 'outputs': outputs}, where outputs is of type Dict[str, np.ndarray]

    model_root = Path(model_root) # type: Path
    root = model_root.parent
    output_dir = Path(output_dir or root/'outputs') # type: Path
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    backends = backends or [ORTExecutor(), TVMExecutor(), XLAExecutor(), TRTBackend()]
    for backend in backends:
        run_backend(model_root, backend, output_dir)

    # numerical consistency check
    report = []
    for model_path in Path(model_root).glob('*/'):
        model_name = model_path.name

        def get_output(backend_name: str, idx: str) -> Tuple[Dict[str, np.ndarray], str]:
            out_path = output_dir / f'{model_name}/{backend_name}.output.{idx}.pkl'
            return pickle.load(out_path.open('rb'))['outputs'], str(out_path)

        # read oracle's data (assume first backend as oracle)
        prefix = output_dir / model_name
        backend_name = backends[0].__class__.__name__
        num_out = len(list(prefix.glob(f'{backend_name}.output.*.pkl')))
        oracle, oracle_path = [], []
        for i in range(num_out):
            output, out_path = get_output(backend_name, i)
            oracle.append(output)
            oracle_path.append(out_path)

        for backend in backends[1:]:
            for i in range(num_out):
                output, out_path = get_output(backend.__class__.__name__, i)
                try:
                    assert_allclose(output, oracle[i], out_path, oracle_path[i])
                except IncorrectResult as err:
                    report.append({
                        'model_path': str(model_path), 
                        'backend': backend.__class__.__name__, 
                        'input_idx': i,
                        'oracle': oracle_path[i],
                        'error': str(err)})
                    print(err)
    import json
    json.dump(report, open(output_dir /'report.json', 'w'), indent=2)

if __name__ == '__main__':
    import wget
    import os
    import numpy as np
    from onnxsim import simplify
    from tvm import relay
    import tvm
    from tvm.contrib.target.onnx import to_onnx

    # 2-input & 2-output static model.
    def get_model():
        x = relay.var("x", shape=(1, 3, 224, 224))
        y = relay.var("y", shape=(1, 2))
        mod = tvm.IRModule.from_expr(relay.Function([x, y], relay.Tuple([x, y])))
        return to_onnx(mod, {}, 'model')

    model = get_model()
    model = DiffTestBackend.get_onnx_proto(model)
    model, check = simplify(
        model, input_shapes={'x': [1, 3, 224, 224], 'y': [1, 2]})

    difftest(model, {'x': np.zeros((1, 3, 224, 224), dtype='float32'), 
        'y': np.array([[1, 2]], dtype='float32')},
        backends=[TVMExecutor(), XLAExecutor()])
    print('test1 passed')


    filename = 'mobilenetv2.onnx'
    if not os.path.exists('mobilenetv2.onnx'):
        filename = wget.download(
            'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx', out='mobilenetv2.onnx')
    sim_model, check = simplify(DiffTestBackend.get_onnx_proto(
        filename), input_shapes={'input': [1, 3, 224, 224]})
    difftest(sim_model, {'input': np.zeros((1, 3, 224, 224), dtype='float32')},
        backends=[TVMExecutor(), XLAExecutor()])
    print('test2 passed')

