import os
import random
import shutil
from typing import Callable, Dict, List

import numpy as np
from termcolor import colored


def succ_print(*args):
    return print(*[colored(x, "green") for x in args])


def fail_print(*args):
    return print(*[colored(x, "red") for x in args])


def note_print(*args):
    return print(*[colored(x, "yellow") for x in args])


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
            note_print(
                "Report folder already exists. Press [Y/N] to continue or exit..."
            )
            decision = input()
        if decision.lower() == "n":
            fail_print(f"{dir} already exist... Remove it or use a different name.")
            raise RuntimeError("Folder already exists")
        else:
            shutil.rmtree(dir)

    os.makedirs(dir)


def is_invalid(output: Dict[str, np.ndarray]):
    for _, o in output.items():
        if np.isnan(o).any() or np.isinf(o).any():
            return True
    return False
