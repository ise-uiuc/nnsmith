import numpy as np
import os
from pathlib import Path

import onnx
from nnsmith.backends import DiffTestBackend
import pickle
from subprocess import check_call
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import time


def gen_one_input(inp_spec, l, r, seed=None):
    if seed is not None:
        np.random.seed(seed)  # TODO: use standalone random generator
    inp = {}
    for name, shape in inp_spec.items():
        inp[name] = np.random.uniform(
            low=l, high=r, size=shape.shape).astype(shape.dtype)
    return inp


Range = Tuple[float, float]


def gen_one_input_rngs(inp_spec, rngs: Union[List[Range], None], seed=None):
    """rngs is a list of tuples (low, high). When rngs is None (which means no valid range found), this falls back to use low=0, high=1 as a workaround"""
    if rngs is None:
        rngs = [(0, 1)]
    return gen_one_input(inp_spec, *rngs[np.random.randint(len(rngs))], seed)


def has_nan(output: Dict[str, np.ndarray]):
    for k, o in output.items():
        if np.isnan(o).any():
            # print(f'NaN in {k}')
            return True
    return False


class InputGenBase:
    MAX_TRIALS = 2
    DEFAULT_RNG = (0, 1)

    # overload me; can return stats for further analysis
    def gen_inputs(self, model, num_inputs, model_path) -> Dict:
        raise NotImplementedError

    # overload me; return valid ranges for further analysis

    def infer_domain(self, model: onnx.ModelProto) -> List[Range]:
        raise NotImplementedError

    @classmethod
    def _gen_inputs(cls, model, num_inputs, model_path, rngs=None, max_trials=MAX_TRIALS):
        from nnsmith.backends.ort_graph import ORTExecutor
        rngs = rngs or [cls.DEFAULT_RNG]
        rf_exe = ORTExecutor(opt_level=0)  # reference model
        inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]
        trials, succ = 0, 0
        for i in tqdm(range(num_inputs)):
            for ntrials in range(max_trials):
                rng_idx = np.random.randint(len(rngs))
                inp = gen_one_input(
                    inp_spec, l=rngs[rng_idx][0], r=rngs[rng_idx][1])
                out = rf_exe.predict(model, inp)
                if not has_nan(out):
                    break
            trials += ntrials + 1
            succ += not has_nan(out)
            pickle.dump(inp, (Path(model_path) / f'input.{i}.pkl').open("wb"))

        print('succ rate:', np.mean(succ / trials))
        return {'succ': succ, 'trials': trials, 'succ_rate': succ / trials}


class InputGenV1(InputGenBase):
    def __init__(self) -> None:
        super().__init__()

    def gen_inputs(self, *args, **kwargs):
        return super()._gen_inputs(*args, **kwargs)

    def infer_domain(self, model):
        return [(0, 1)]


# class InputGenV2(InputGenBase):
#     L = -1e2
#     R = 1e2
#     EPS = 1e-3
#     THRES = 1

#     def __init__(self, max_rng_trials=3) -> None:
#         super().__init__()
#         self.max_rng_trials = max_rng_trials

#     def _get_range(self, model):
#         def check(l, r):
#             succ = 0
#             for ntrials in range(self.max_rng_trials):
#                 inp = self.gen_one_input(inp_spec, l, r)
#                 out = rf_exe.predict(model, inp)
#                 succ += not self.has_nan(out)
#             return succ / (ntrials + 1) >= self.THRES

#         rf_exe = ORTExecutor(opt_level=0)  # reference model
#         inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]

#         a = np.linspace(-1, 1, 10)
#         # print('inital serach points=', a)

#         def binary_search(l, r, checker):
#             while r - l > self.EPS:
#                 mid = (l + r) / 2
#                 if checker(mid):
#                     r = mid
#                 else:
#                     l = mid
#             return r

#         valid_rngs = []
#         for valid_point in a:
#             included = False
#             for rng in valid_rngs:
#                 if rng[0] <= valid_point <= rng[1]:
#                     included = True
#             if included or not check(valid_point, valid_point):
#                 continue
#             l = binary_search(self.L, valid_point,
#                               lambda l: check(l, valid_point))
#             r = -binary_search(-self.R, -valid_point, lambda r: check(l, -r))
#             # fix boundary to avoid redundant binary search
#             # [0, 0.99999] -> [0, 1] if 1 is valid, thus saving binary search on 1
#             for v2 in a:
#                 if abs(l - v2) <= self.EPS:
#                     l = min(l, v2)
#                 if abs(r - v2) <= self.EPS:
#                     r = max(r, v2)
#             valid_rngs.append((l, r))

#         if len(valid_rngs) == 0:
#             print('no valid range found!!!')
#             valid_rngs = None
#         return valid_rngs

#     def gen_inputs(self, model, num_inputs, model_path):
#         st = time.time()
#         rngs = self._get_range(model)
#         get_range_time = time.time() - st
#         st = time.time()
#         stats = self._gen_inputs(model, num_inputs, model_path, rngs)
#         gen_input_time = time.time() - st
#         print('get_range_time=', get_range_time, 'gen_input_time=',
#               gen_input_time, 'valid ranges=', rngs)
#         stats['get_range_time'] = get_range_time
#         stats['gen_input_time'] = gen_input_time
#         return stats

class NaNChecker():
    THRES = 1

    def __init__(self, max_rng_trials=3) -> None:
        from nnsmith.backends.ort_graph import ORTExecutor
        super().__init__()
        self.max_rng_trials = max_rng_trials
        # reference model
        self.rf_exe = ORTExecutor(opt_level=0, providers=[
                                  'CPUExecutionProvider'])

    def load_model(self, model):
        self.model = model
        self.inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]

    def __call__(self, l, r):
        succ = 0
        for ntrials in range(self.max_rng_trials):
            inp = gen_one_input(self.inp_spec, l, r)
            out = self.rf_exe.predict(self.model, inp)
            succ += not has_nan(out)
            remain = self.max_rng_trials - ntrials - 1
            if (succ + remain) / self.max_rng_trials < self.THRES:
                return False  # fast failure
        return True  # succ / (ntrials + 1) >= self.THRES


class InputGenV3(InputGenBase):
    L = -10
    R = 10
    EPS = 1e-3

    def __init__(self, nan_checker=None) -> None:
        super().__init__()
        self.nan_checker = nan_checker or NaNChecker()

    def _get_range(self):
        a = np.linspace(-1, 1, 10)

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
            if included or not self.nan_checker(valid_point, valid_point):
                continue

            # try expand previous range
            if len(valid_rngs) > 0 and self.nan_checker(valid_rngs[-1][0], valid_point):
                valid_rngs[-1] = (valid_rngs[-1][0], valid_point)
                continue

            last_r = self.L if len(valid_rngs) == 0 else valid_rngs[-1][1]
            l = binary_search(last_r, valid_point,
                              lambda l: self.nan_checker(l, valid_point))
            r = -binary_search(-self.R, -valid_point,
                               lambda r: self.nan_checker(l, -r))
            last_r = r
            valid_rngs.append((l, r))

        if len(valid_rngs) == 0:
            print('no valid range found!!!')
            valid_rngs = None
        return valid_rngs

    def gen_inputs(self, model, num_inputs, model_path):
        self.nan_checker.load_model(model)
        st = time.time()
        rngs = self.infer_domain(model)
        get_range_time = time.time() - st

        st = time.time()
        stats = self._gen_inputs(model, num_inputs, model_path, rngs)
        gen_input_time = time.time() - st

        # print('get_range_time=', get_range_time, 'gen_input_time=',
        #       gen_input_time, 'valid ranges=', rngs)
        stats['get_range_time'] = get_range_time
        stats['gen_input_time'] = gen_input_time
        return stats

    def infer_domain(self, model):
        self.nan_checker.load_model(model)
        if self.nan_checker(0, 1):
            rngs = [(0, 1)]
        else:
            rngs = self._get_range()

        return rngs


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
    import pandas as pd
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
