from typing import Dict
import pickle

import torch

from nnsmith.graph_gen import Schedule
from nnsmith.abstract.op import AbsTensor
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import SymbolNet


class TorchModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.torch_model: SymbolNet = None
        self.sat_inputs = None

    @staticmethod
    def from_schedule(schedule: Schedule, **kwargs) -> "TorchModel":
        ret = TorchModel()
        ret.torch_model = SymbolNet(schedule, **kwargs)
        return ret

    @staticmethod
    def schedule_name() -> str:
        return "schedule.pkl"

    def refine_weights(self) -> None:
        self.torch_model.enable_proxy_grad()
        searcher = PracticalHybridSearch(self.torch_model)
        # TODO(@ganler): Can we directly get both inputs and outputs?
        n_try, inputs = searcher.search(
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

        return Oracle(input_dict, output_dict)

    def dump(self, path: str):
        torch.save(self.torch_model.state_dict(), path)
        schedule_path = path.replace(
            TorchModel.name_prefix() + TorchModel.name_suffix(),
            TorchModel.schedule_name(),
        )
        with open(schedule_path, "wb") as f:
            pickle.dump(self.torch_model.schedule, f)

    @staticmethod
    def load(path: str) -> "TorchModel":
        ret = TorchModel()
        schedule_path = path.replace(
            TorchModel.name_prefix() + TorchModel.name_suffix(),
            TorchModel.schedule_name(),
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
