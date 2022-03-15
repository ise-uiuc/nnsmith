from nnsmith.backends import DiffTestBackend

import numpy as np
import pickle

from typing import List, Dict, Tuple, Union


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

