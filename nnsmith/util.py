import os
import random
import shutil
from typing import Callable, Dict, List

import numpy as np
import pydot
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


def mkdir(dir: os.PathLike, yes=False):
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


_DOT_EXIST = shutil.which("dot") is not None
_CONDA_EXIST = shutil.which("conda") is not None
_APT_EXIST = shutil.which("apt") is not None
_BREW_EXIST = shutil.which("brew") is not None


def _check_dot_install():
    if not _DOT_EXIST:
        note_print("`dot` not found.")
        if _CONDA_EXIST or _APT_EXIST or _BREW_EXIST:
            note_print("To install via:")
            if _CONDA_EXIST:
                note_print(" conda:\t conda install -c anaconda graphviz -y")

            if _APT_EXIST:
                note_print(" apt:\t sudo apt install graphviz -y")

            if _BREW_EXIST:
                note_print(" brew:\t brew install graphviz")

        note_print("Also see: https://graphviz.org/download/")
        return False

    return True


def viz_dot(dotobj: pydot.Dot, filename: str = None):
    if _check_dot_install():
        if filename is None:
            filename = f"graph.png"
        if filename.endswith("png"):
            dotobj.write_png(filename)
        elif filename.endswith("svg"):
            dotobj.write_svg(filename)
        else:
            raise ValueError(f"Unsupported image format: {filename}")
    else:
        note_print(
            f"Skipping visualizing `{filename}` due to missing `dot` (graphviz)."
        )
