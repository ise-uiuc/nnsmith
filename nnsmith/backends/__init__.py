from typing import List, Union, Dict, Tuple
import os

import pickle
import numpy as np

from nnsmith.util import gen_one_input

from nnsmith.backends.factory import BackendFactory


def mk_factory(name, device="cpu", optmax=True, **kwargs):
    if name == "ort" or name == "onnxruntime":
        from nnsmith.backends.onnxruntime import ORTFactory

        return ORTFactory(device=device, optmax=optmax)
    elif name == "tvm":
        from nnsmith.backends.tvm import TVMFactory

        return TVMFactory(device=device, optmax=optmax, executor="graph")
    elif name == "trt":
        from nnsmith.backends.tensorrt import TRTFactory

        return TRTFactory()
    else:
        raise ValueError(f"unknown backend: {name}")
