import numpy as np
import os
from pathlib import Path
from nnsmith.backends import DiffTestBackend
import pickle
from subprocess import check_call
from tqdm import tqdm
import torch
from typing import List, Dict
from nnsmith.backends.ort_graph import ORTExecutor
import pandas as pd
import time


class InputGenBase:
    MAX_TRIALS = 2
    DEFAULT_RNG = (0, 1)

    @staticmethod
    def gen_one_input(inp_spec, l, r):
        inp = {}
        for name, shape in inp_spec.items():
            inp[name] = np.random.uniform(
                low=l, high=r, size=shape.shape).astype(shape.dtype)
        return inp

    @staticmethod
    def has_nan(output: Dict[str, np.ndarray]):
        for k, o in output.items():
            if np.isnan(o).any():
                # print(f'NaN in {k}')
                return True
        return False

    # overload me; can return stats for further analysis
    def gen_inputs(self, model, num_inputs, model_path) -> Dict:
        raise NotImplementedError

    @classmethod
    def _gen_inputs(cls, model, num_inputs, model_path, rngs=None, max_trials=MAX_TRIALS):
        rngs = rngs or [cls.DEFAULT_RNG]
        rf_exe = ORTExecutor(opt_level=0)  # reference model
        inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]
        trials, succ = 0, 0
        for i in tqdm(range(num_inputs)):
            for ntrials in range(max_trials):
                rng_idx = np.random.randint(len(rngs))
                inp = cls.gen_one_input(
                    inp_spec, l=rngs[rng_idx][0], r=rngs[rng_idx][1])
                out = rf_exe.predict(model, inp)
                if not cls.has_nan(out):
                    break
            trials += ntrials + 1
            succ += not cls.has_nan(out)
            pickle.dump(inp, (Path(model_path) / f'input.{i}.pkl').open("wb"))

        print('succ rate:', np.mean(succ / trials))
        return {'succ': succ, 'trials': trials, 'succ_rate': succ / trials}


class InputGenV1(InputGenBase):
    def __init__(self) -> None:
        super().__init__()

    def gen_inputs(self, *args, **kwargs):
        return super()._gen_inputs(*args, **kwargs)


class InputGenV2(InputGenBase):
    L = -1e2
    R = 1e2
    EPS = 1e-3
    THRES = 1

    def __init__(self, max_rng_trials=3) -> None:
        super().__init__()
        self.max_rng_trials = max_rng_trials

    def get_range(self, model):
        def check(l, r):
            succ = 0
            for ntrials in range(self.max_rng_trials):
                inp = self.gen_one_input(inp_spec, l, r)
                out = rf_exe.predict(model, inp)
                succ += not self.has_nan(out)
            return succ / (ntrials + 1) >= self.THRES

        rf_exe = ORTExecutor(opt_level=0)  # reference model
        inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]

        a = np.linspace(-1, 1, 10)
        print('inital serach points=', a)

        def binary_search(l, r, checker):
            while r - l > self.EPS:
                mid = (l + r) / 2
                if checker(mid):
                    r = mid
                else:
                    l = mid
            return r

        valid_rngs = []
        for valid_point in a:
            included = False
            for rng in valid_rngs:
                if rng[0] <= valid_point <= rng[1]:
                    included = True
            if included or not check(valid_point, valid_point):
                continue
            l = binary_search(self.L, valid_point,
                              lambda l: check(l, valid_point))
            r = -binary_search(-self.R, -valid_point, lambda r: check(l, -r))
            # fix boundary to avoid redundant binary search
            # [0, 0.99999] -> [0, 1] if 1 is valid, thus saving binary search on 1
            for v2 in a:
                if abs(l - v2) <= self.EPS:
                    l = min(l, v2)
                if abs(r - v2) <= self.EPS:
                    r = max(r, v2)
            valid_rngs.append((l, r))

        if len(valid_rngs) == 0:
            print('no valid range found!!!')
            valid_rngs = None
        return valid_rngs

    def gen_inputs(self, model, num_inputs, model_path):
        st = time.time()
        rngs = self.get_range(model)
        get_range_time = time.time() - st
        st = time.time()
        stats = self._gen_inputs(model, num_inputs, model_path, rngs)
        gen_input_time = time.time() - st
        print('get_range_time=', get_range_time, 'gen_input_time=',
              gen_input_time, 'valid ranges=', rngs)
        stats['get_range_time'] = get_range_time
        stats['gen_input_time'] = gen_input_time
        return stats


class InputGenV3(InputGenBase):
    L = -10
    R = 10
    EPS = 1e-3
    THRES = 1

    def __init__(self, max_rng_trials=3) -> None:
        super().__init__()
        self.max_rng_trials = max_rng_trials
        # reference model
        self.rf_exe = ORTExecutor(opt_level=0, providers=[
                                  'CPUExecutionProvider'])

    def load_model(self, model):
        self.model = model
        self.inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]

    def check(self, l, r):
        succ = 0
        for ntrials in range(self.max_rng_trials):
            inp = self.gen_one_input(self.inp_spec, l, r)
            out = self.rf_exe.predict(self.model, inp)
            succ += not self.has_nan(out)
            remain = self.max_rng_trials - ntrials - 1
            if (succ + remain) / self.max_rng_trials < self.THRES:
                return False  # fast failure
        return True  # succ / (ntrials + 1) >= self.THRES

    def get_range(self, model):

        a = np.linspace(-1, 1, 10)
        print('inital serach points=', a)

        def binary_search(l, r, checker, return_on_first=True):
            # asssume monotonicity in the form of
            # False False False ... True ... True
            # l     l+1   l+2   ... mid  ... r
            if l == r or checker(l):
                return l
            while r - l > self.EPS:
                mid = (l + r) / 2
                if checker(mid):
                    r = mid
                    if return_on_first:
                        return r
                else:
                    l = mid
            return r

        valid_rngs = []
        for valid_point in a:
            included = False
            for rng in valid_rngs:
                if rng[0] <= valid_point <= rng[1]:
                    included = True
            if included or not self.check(valid_point, valid_point):
                continue

            # try expand previous range
            if len(valid_rngs) > 0 and self.check(valid_rngs[-1][0], valid_point):
                valid_rngs[-1] = (valid_rngs[-1][0], valid_point)
                continue

            last_r = self.L if len(valid_rngs) == 0 else valid_rngs[-1][1]
            l = binary_search(last_r, valid_point,
                              lambda l: self.check(l, valid_point))
            r = -binary_search(-self.R, -valid_point,
                               lambda r: self.check(l, -r))
            last_r = r
            valid_rngs.append((l, r))

        if len(valid_rngs) == 0:
            print('no valid range found!!!')
            valid_rngs = None
        return valid_rngs

    def gen_inputs(self, model, num_inputs, model_path):
        self.load_model(model)
        st = time.time()
        if self.check(0, 1):
            rngs = [(0, 1)]
        else:
            rngs = self.get_range(model)
        get_range_time = time.time() - st

        st = time.time()
        stats = self._gen_inputs(model, num_inputs, model_path, rngs)
        gen_input_time = time.time() - st

        print('get_range_time=', get_range_time, 'gen_input_time=',
              gen_input_time, 'valid ranges=', rngs)
        stats['get_range_time'] = get_range_time
        stats['gen_input_time'] = gen_input_time
        return stats


def gen_inputs_for_one(num_inputs, model, input_gen: InputGenBase = InputGenV1()):
    st = time.time()
    stats = input_gen.gen_inputs(
        DiffTestBackend.get_onnx_proto(str(model)), num_inputs, Path(model).parent)
    stats.update({
        'model': model,
        'elpased_time': time.time() - st}
    )
    return stats


def gen_inputs_for_all(root, num_inputs=2, models=None, input_gen: InputGenBase = InputGenV1()):
    profile = []
    models = models or sorted(list(Path(root).glob('model_input/*/*.onnx')))
    for model in tqdm(models):
        print(model)
        profile.append(gen_inputs_for_one(num_inputs, model, input_gen))
    # profile = pd.DataFrame(
        # profile, columns=['model', 'succ', 'trials', 'succ_rate', 'elapsed_time'])
    profile = pd.DataFrame(profile)
    profile = profile.sort_values(by=['succ_rate'], ascending=True)
    profile.to_pickle(Path(root) / 'gen_profile.pkl')
    profile.to_csv(Path(root) / 'gen_profile.csv')


def gen_models_inputs(root: str, num_models, gen_args, num_inputs, input_gen):
    st_time = time.time()
    profile = []
    profile_inputs = []
    root = Path(root)  # type: Path
    if len(list(root.glob('model*'))) > 0:
        raise Exception(f'Folder {root} already exists')
    model_root = root / 'model_input'
    model_root.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(num_models)):
        st = time.time()
        succ = False
        atmpts = 0
        while not succ:
            atmpts += 1
            e1 = None
            try:
                model = model_root / f'{i}' / 'model.onnx'
                model.parent.mkdir(exist_ok=True, parents=True)
                seed = int(time.time() * 1000)
                print(f'seeding {seed}')
                check_call(
                    f'python -u -m nnsmith.graph_gen --output_path {model} --seed {seed} {gen_args} 2>&1', shell=True)
                input_st = time.time()
                profile_inputs.append(gen_inputs_for_one(
                    num_inputs, model, input_gen))
                print('gen_input time={}s'.format(time.time() - input_st))
                succ = True
            except Exception as e:
                e1 = e
                print(e)
            finally:
                profile.append({
                    'model': model,
                    'atmpts': atmpts,
                    'succ': succ,
                    'elpased_time': time.time() - st,
                    'seed': seed,
                })
            if e1 is not None:
                profile[-1]['error'] = e1
    print('gen_model elapsed time={}s'.format(time.time() - st_time))
    profile = pd.DataFrame(profile)
    profile.to_pickle(Path(root) / 'gen_model_profile.pkl')
    profile.to_csv(Path(root) / 'gen_model_profile.csv')

    profile_inputs = pd.DataFrame(profile_inputs)
    profile_inputs = profile_inputs.sort_values(
        by=['succ_rate'], ascending=True)
    profile_inputs.to_pickle(Path(root) / 'gen_input_profile.pkl')
    profile_inputs.to_csv(Path(root) / 'gen_input_profile.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='tmp/seed1')
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--num_inputs', type=int, default=2)
    parser.add_argument('--input_only', action='store_true')
    parser.add_argument('--model', type=str, nargs='*',
                        help='Generate input for specific model, specified as the path to the .onnx file.')
    parser.add_argument('--input_gen_method', type=str, default='v3')
    parser.add_argument('--model_gen_args')
    args = parser.parse_args()
    gen_inputs_func = {
        'v1': InputGenV1(),
        'v2': InputGenV2(),
        'v3': InputGenV3(),
    }[args.input_gen_method]
    if not args.input_only:
        gen_models_inputs(args.root, args.num_models,
                          args.model_gen_args, args.num_inputs, gen_inputs_func)
    gen_inputs_for_all(args.root, args.num_inputs, args.model, gen_inputs_func)
