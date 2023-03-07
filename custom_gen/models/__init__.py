import json
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from typing import Any, Dict, List, Type, TypeVar

import numpy as np
from multipledispatch import dispatch

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsOpBase, Constant
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.materialize import Model
from nnsmith.error import SanityCheck
from nnsmith.gir import GraphIR
from nnsmith.util import HAS_PYGRAPHVIZ, viz_dot

class ModelCust(Model):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def init(name, backend_target=None) -> Type["Model"]:
        if name is None:
            raise ValueError(
                "Model type cannot be None. Use `model.type=[torch|onnx|tensorflow]`."
            )
        if name == "torch-onnx":
            from models.torch import TorchModelExportable

            return TorchModelExportable
        elif name == "tensorflow-onnx":
            from models.tensorflow import TFModelExportable

            return TFModelExportable
        elif name == "torch":
            from nnsmith.materialize.torch import TorchModel

            # PyTorch CPU - GPU implementation are quite the same.
            return TorchModel
        elif name == "onnx":
            # device agnoistic
            from nnsmith.materialize.onnx import ONNXModel

            return ONNXModel
        elif name == "tensorflow":
            from nnsmith.materialize.tensorflow import TFModelCPU, TFModelGPU

            if backend_target == "gpu" or backend_target == "cuda":
                # XLA must align device location of eager mode execution.
                return TFModelGPU
            else:
                return TFModelCPU

        raise ValueError(
            f"Unsupported: ModelType={name} for BackendTarget={backend_target}"
        )