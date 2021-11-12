import numpy as np
import os
from pathlib import Path
from nnsmith.difftest import DiffTestBackend
import pickle
from subprocess import check_call

def gen_inputs(model, num_inputs):
    inp_spec = DiffTestBackend.analyze_onnx_io(model)
    inps = []
    for i in range(num_inputs):
        inp = {}
        for name, shape in inp_spec[0].items():
            # random range is [-0.5, 0.5]
            inp[name] = np.random.rand(*shape.shape).astype(shape.dtype) - 0.5
        inps.append(inp)
    return inps


def gen_inputs_for_all(root, num_inputs=2):
    for model in Path(root).glob('model_input/*/*.onnx'):
        print(model)
        inps = gen_inputs(DiffTestBackend.get_onnx_proto(str(model)), num_inputs)
        for i, inp in enumerate(inps):
            model_path = Path(model).parent
            pickle.dump(inp, (model_path/f'input.{i}.pkl').open("wb"))

def gen_models(root: str, num_models):
    root = Path(root)
    if root.exists():
        raise Exception(f'Folder {root} already exists')
    model_root = root / 'model_input'
    model_root.mkdir(exist_ok=True, parents=True)
    for i in range(num_models):
        model = model_root / f'{i}' / 'model.onnx'
        model.parent.mkdir(exist_ok=True, parents=True)
        check_call(f'python -m nnsmith.gen --output_path {model}', shell=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='tmp/seed1')
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--num_inputs', type=int, default=2)
    args = parser.parse_args()
    gen_models(args.root, args.num_models)
    gen_inputs_for_all(args.root, args.num_inputs)
