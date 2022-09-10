import os
import random
import shutil
from typing import Callable, Dict, List

import numpy as np
from termcolor import colored

try:
    import pygraphviz as pgv

    HAS_PYGRAPHVIZ = True
except ImportError:
    import warnings

    warnings.warn(
        "Install pygraphviz for visualization: https://pygraphviz.github.io/documentation/stable/install.html\n"
        "Currently graph visualization is not enabled."
    )
    pgv = None
    HAS_PYGRAPHVIZ = False


from nnsmith.logging import VIZ_LOG


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

_CALL_ONCE = False


def _check_dot_install():
    global _CALL_ONCE
    if not _DOT_EXIST and not _CALL_ONCE:
        _CALL_ONCE = True
        VIZ_LOG.warn("`dot` not found.")
        if _CONDA_EXIST or _APT_EXIST or _BREW_EXIST:
            VIZ_LOG.warn("To install via:")
            if _CONDA_EXIST:
                VIZ_LOG.warn(" conda:\t conda install -c anaconda graphviz -y")

            if _APT_EXIST:
                VIZ_LOG.warn(" apt:\t sudo apt install graphviz -y")

            if _BREW_EXIST:
                VIZ_LOG.warn(" brew:\t brew install graphviz")

        VIZ_LOG.warn("Also see: https://graphviz.org/download/")
        return False

    return True


def viz_dot(dotobj, filename: str = None):
    if _check_dot_install():
        if filename is None:
            filename = f"graph.png"

        if isinstance(dotobj, str):
            dotobj = pgv.AGraph(dotobj)

        dotobj.layout("dot")
        dotobj.draw(filename)
