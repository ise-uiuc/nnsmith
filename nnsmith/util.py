import os
import shutil
import random
from typing import Callable, Dict, List

import numpy as np


SEED_SETTERS = {
    "random": random.seed,
    "numpy": np.random.seed,
}


def register_seed_setter(
    name: str,
    fn: Callable[[int], None],
    overwrite=False,
):
    if not overwrite:
        assert name not in SEED_SETTERS, f"{name} is already registered"
    SEED_SETTERS[name] = fn


def set_seed(seed: int, names: List = None):
    if names is None:
        names = SEED_SETTERS.keys()
    for name in names:
        SEED_SETTERS[name](seed)


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


def is_invalid(output: Dict[str, np.ndarray]):
    for _, o in output.items():
        if np.isnan(o).any() or np.isinf(o).any():
            return True
    return False
