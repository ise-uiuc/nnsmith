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


def gen_one_input_rngs(inp_spec: Union[str, Dict], rngs: Union[str, List[Range], None], seed=None) -> Dict:
    """
    Parameters
    ----------
    `inp_spec` can be either a string or a dictionary. When it's a string, it's the a path to the ONNX model.

    `rngs` can be 
    - a list of tuples (low, high).
    - None, which means no valid range found, this falls back to use low=0, high=1 as a workaroun
    - a string, which is interpreted as a path to a pickled file.
    """
    if rngs is None:
        rngs = [(0, 1)]
    elif isinstance(rngs, str):
        rngs = pickle.load(open(rngs, 'rb'))
    if isinstance(inp_spec, str):  # in this case the inp_spec is a path to a the model proto
        inp_spec = DiffTestBackend.analyze_onnx_io(
            DiffTestBackend.get_onnx_proto(inp_spec))[0]
    return gen_one_input(inp_spec, *rngs[np.random.randint(len(rngs))], seed)


def is_invalid(output: Dict[str, np.ndarray]):
    for k, o in output.items():
        if np.isnan(o).any() or np.isinf(o).any():
            return True
    return False


class InputGenBase:
    MAX_TRIALS = 2
    DEFAULT_RNG = (0, 1)

    # overload me; return valid ranges for further analysis
    def infer_domain(self, model: onnx.ModelProto) -> List[Range]:
        raise NotImplementedError


class InputGenV1(InputGenBase):
    def __init__(self) -> None:
        super().__init__()

    def infer_domain(self, model):
        return [(0, 1)]


class NumericChecker:
    def load_model(self, model):
        self.model = model
        self.inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]

    def all_valid(self, l, r):
        succ = 0
        for ntrials in range(self.max_rng_trials):
            inp = gen_one_input(self.inp_spec, l, r)
            out = self.rf_exe.predict(self.model, inp)
            succ += not is_invalid(out)
            remain = self.max_rng_trials - ntrials - 1
            if (succ + remain) / self.max_rng_trials < self.THRES:
                return False  # fast failure
        return True  # succ / (ntrials + 1) >= self.THRES


class ORTNumericChecker(NumericChecker):
    '''WARNING: this class does not check intermidiate tensors.'''
    THRES = 1

    def __init__(self, max_rng_trials=3) -> None:
        from nnsmith.backends.ort_graph import ORTExecutor
        super().__init__()
        self.max_rng_trials = max_rng_trials
        # reference model
        self.rf_exe = ORTExecutor(opt_level=0, providers=[
                                  'CPUExecutionProvider'])


class TorchNumericChecker(NumericChecker):
    '''WARNING: this class does not check intermidiate tensors.'''
    THRES = 1

    class TorchExecutor:
        def __init__(self, torch_model) -> None:
            self.torch_model = torch_model

        def predict(self, not_used, inp):
            import torch
            torch_inp = []
            # keys are 'i0', 'i1', 'i2', ... sort by int comparison to be consistent with the ordering
            for k in sorted(inp.keys(), key=lambda s: int(s[1:])):
                torch_inp.append(torch.from_numpy(inp[k]))
            return {str(i): t.cpu().numpy() for i, t in enumerate(self.torch_model(*torch_inp))}

    def __init__(self, torch_model, max_rng_trials=3) -> None:
        super().__init__()
        self.max_rng_trials = max_rng_trials
        self.torch_model = torch_model
        self.rf_exe = TorchNumericChecker.TorchExecutor(torch_model)


class InputGenV3(InputGenBase):
    L = -10
    R = 10
    EPS = 1e-3

    def __init__(self, numeric_checker: NumericChecker = None) -> None:
        super().__init__()
        self.numeric_checker = numeric_checker or ORTNumericChecker()

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
            if included or not self.numeric_checker.all_valid(valid_point, valid_point):
                continue

            # try expand previous range
            if len(valid_rngs) > 0 and self.numeric_checker.all_valid(valid_rngs[-1][0], valid_point):
                valid_rngs[-1] = (valid_rngs[-1][0], valid_point)
                continue

            last_r = self.L if len(valid_rngs) == 0 else valid_rngs[-1][1]
            l = binary_search(last_r, valid_point,
                              lambda l: self.numeric_checker.all_valid(l, valid_point))
            r = -binary_search(-self.R, -valid_point,
                               lambda r: self.numeric_checker.all_valid(l, -r))
            last_r = r
            valid_rngs.append((l, r))

        if len(valid_rngs) == 0:
            print('no valid range found!!!')
            valid_rngs = None
        return valid_rngs

    def infer_domain(self, model):
        self.numeric_checker.load_model(model)
        if self.numeric_checker.all_valid(0, 1):
            rngs = [(0, 1)]
        else:
            rngs = self._get_range()

        return rngs


def gen_one_input_for_model(model: Union[str, onnx.GraphProto], input_gen=None, seed=None):
    """Convenient wrapper for gen_one_input_rngs. This function requires only one input that specified the model. 
    Under the hood, it parses the model to extract the input spec and invokes input_gen to infer the domain."""
    if input_gen is None:
        input_gen = InputGenV3()
    inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]
    return gen_one_input_rngs(inp_spec, input_gen.infer_domain(model), seed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Domain inferencer')
    parser.add_argument('--model')
    parser.add_argument('--output_path', default='./domain.pkl')
    args = parser.parse_args()

    model = DiffTestBackend.get_onnx_proto(args.model)

    input_gen = InputGenV3()
    input_st = time.time()
    rngs = input_gen.infer_domain(model)
    pickle.dump(rngs, open(args.output_path, 'wb'))
