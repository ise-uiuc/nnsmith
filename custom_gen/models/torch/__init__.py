import pickle
from os import PathLike
from typing import Dict, List, Type

import torch

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import SymbolNet
from nnsmith.materialize.torch import TorchModel
from nnsmith.util import register_seed_setter

class TorchModelExportable(TorchModel):
    def __init__(self) -> None:
        super().__init__()

    def export_onnx(self):
        print(self.torch_model)
