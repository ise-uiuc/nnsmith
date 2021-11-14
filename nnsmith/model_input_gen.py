import numpy as np
import os
from pathlib import Path
from nnsmith.difftest import DiffTestBackend
import pickle
from subprocess import check_call
from tqdm import tqdm
import torch
from typing import List, Dict
from nnsmith.backends.ort_graph import ORTExecutor
import pandas as pd
import time

MAX_TRIALS = 20


def gen_one_input(inp_spec):
    inp = {}
    for name, shape in inp_spec.items():
        # random range is [-0.5, 0.5]
        inp[name] = np.random.rand(
            *shape.shape).astype(shape.dtype)  # * 2 - 0.5
    return inp


def has_nan(output: Dict[str, np.ndarray]):
    for k, o in output.items():
        if np.isnan(o).any():
            # print(f'NaN in {k}')
            return True
    return False


def gen_inputs(model, num_inputs, model_path):
    rf_model = ORTExecutor(opt_level=0)  # reference model
    inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]
    trials, succ = 0, 0
    for i in tqdm(range(num_inputs)):
        for ntrials in range(MAX_TRIALS):
            inp = gen_one_input(inp_spec)
            out = rf_model.predict(model, inp)
            if not has_nan(out):
                break
        trials += ntrials + 1
        succ += not has_nan(out)
        pickle.dump(inp, (Path(model_path) / f'input.{i}.pkl').open("wb"))

    print('succ rate:', np.mean(succ / trials))
    return succ, trials


def gen_inputs_for_all(root, num_inputs=2, models=None):
    profile = []
    models = models or sorted(list(Path(root).glob('model_input/*/*.onnx')))
    for model in tqdm(models):
        print(model)
        succ, trials = gen_inputs(
            DiffTestBackend.get_onnx_proto(str(model)), num_inputs, Path(model).parent)
        profile.append((model, succ, trials, succ / trials))
    profile = sorted(profile, key=lambda x: x[-1])
    profile = pd.DataFrame(
        profile, columns=['model', 'succ', 'trials', 'succ_rate'])
    profile.to_pickle(Path(root) / 'gen_profile.pkl')
    profile.to_csv(Path(root) / 'gen_profile.csv')


def gen_models(root: str, num_models):
    root = Path(root)
    if root.exists():
        raise Exception(f'Folder {root} already exists')
    model_root = root / 'model_input'
    model_root.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(num_models)):
        model = model_root / f'{i}' / 'model.onnx'
        model.parent.mkdir(exist_ok=True, parents=True)
        seed = int(time.time() * 1000)
        print(f'seeding {seed}')
        check_call(
            f'python -m nnsmith.gen --output_path {model} --seed {seed}', shell=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='tmp/seed1')
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--num_inputs', type=int, default=2)
    parser.add_argument('--input_only', action='store_true')
    parser.add_argument('--model', type=str, nargs='*',
                        help='Generate input for specific model')
    args = parser.parse_args()
    if not args.input_only:
        gen_models(args.root, args.num_models)
    gen_inputs_for_all(args.root, args.num_inputs, args.model)
