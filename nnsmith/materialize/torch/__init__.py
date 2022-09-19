import pickle
from os import PathLike
from typing import Dict, List, Type

import torch

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.materialize import Model, Oracle, Schedule
from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import SymbolNet
from nnsmith.util import register_seed_setter


class TorchModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.torch_model: SymbolNet = None
        self.sat_inputs = None

    @classmethod
    def from_schedule(cls, schedule: Schedule, **kwargs) -> "TorchModel":
        ret = cls()
        ret.torch_model = SymbolNet(schedule, **kwargs)
        return ret

    @staticmethod
    def schedule_name() -> str:
        return "schedule.pkl"

    def refine_weights(self) -> None:
        self.torch_model.enable_proxy_grad()
        searcher = PracticalHybridSearch(self.torch_model)
        # TODO(@ganler): Can we directly get both inputs and outputs?
        _, inputs = searcher.search(
            max_time_ms=20,
            max_sample=2,
        )
        if inputs:
            self.sat_inputs = inputs
        self.torch_model.disable_proxy_grad()

    def make_oracle(self) -> Oracle:
        with torch.no_grad():
            self.torch_model.eval()
            # fall back to random inputs if no solution is found.
            if self.sat_inputs is None:
                inputs = self.torch_model.get_random_inps()
            else:
                inputs = self.sat_inputs
            outputs = self.torch_model.forward(**inputs)

        # numpyify
        input_dict = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
        output_dict = {}
        for oname, val in zip(self.output_like.keys(), outputs):
            output_dict[oname] = val.cpu().detach().numpy()

        return Oracle(input_dict, output_dict, provider="torch[cpu] eager")

    def dump(self, path: PathLike):
        torch.save(self.torch_model.state_dict(), path)
        schedule_path = path.replace(
            TorchModel.name_prefix() + TorchModel.name_suffix(),
            TorchModel.schedule_name(),
        )
        with open(schedule_path, "wb") as f:
            pickle.dump(self.torch_model.schedule, f)

    @classmethod
    def load(cls, path: PathLike) -> "TorchModel":
        ret = cls()
        schedule_path = path.replace(
            cls.name_prefix() + cls.name_suffix(),
            cls.schedule_name(),
        )
        with open(schedule_path, "rb") as f:
            schedule = pickle.load(f)
        torch_model = SymbolNet(schedule)
        torch_model.load_state_dict(torch.load(path))
        ret.torch_model = torch_model
        return ret

    @staticmethod
    def name_suffix() -> str:
        return ".pth"

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return self.torch_model.input_like

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return self.torch_model.output_like

    @property
    def native_model(self) -> SymbolNet:
        return self.torch_model

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return ALL_TORCH_OPS

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("torch", torch.manual_seed, overwrite=True)
