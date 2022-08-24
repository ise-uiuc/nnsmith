import os
import shutil
from typing import Dict

import numpy as np


def mkdir(dir, yes=False):
    if os.path.exists(dir):
        decision = ""
        if yes:
            decision = "y"
        while decision.lower() not in ["y", "n"]:
            decision = input(
                "Report folder already exists. Press [Y/N] to continue or exit..."
            )
        if decision.lower() == "n":
            raise RuntimeError(
                f"{dir} already exist... Remove it or use a different name."
            )
        else:
            shutil.rmtree(dir)

    os.makedirs(dir)


def gen_one_input(inp_spec, l, r, seed=None):
    if seed is not None:
        np.random.seed(seed)  # TODO: use standalone random generator
    inp = {}
    for name, shape in inp_spec.items():
        inp[name] = np.random.uniform(low=l, high=r, size=shape.shape).astype(
            shape.dtype
        )
    return inp


def is_invalid(output: Dict[str, np.ndarray]):
    for _, o in output.items():
        if np.isnan(o).any() or np.isinf(o).any():
            return True
    return False
