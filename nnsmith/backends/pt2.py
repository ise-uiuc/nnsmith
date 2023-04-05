from typing import Dict, Tuple

import numpy as np
import torch
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.torch import TorchModel
from nnsmith.materialize.torch.symbolnet import FxTracing


class PT2(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False, **kwargs):
        super().__init__(target, optmax)
        if self.target == "cpu":
            self.device = torch.device("cpu")
        elif self.target == "cuda":
            self.device = torch.device("cuda")
        else:
            raise ValueError(
                f"Unknown target: {self.target}. Only `cpu` and `cuda` are supported."
            )

        # get backend from kwargs or inductor by default
        self.backend = kwargs.get("backend", "inductor")
        self.mode = kwargs.get("mode")

    @property
    def system_name(self) -> str:
        return "pt2"

    @dispatch(TorchModel)
    def make_backend(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device).eval()
        with torch.no_grad():
            with FxTracing():
                traced = torch.fx.symbolic_trace(torch_net)
                compiled = torch.compile(
                    traced, fullgraph=True, backend=self.backend, mode=self.mode
                )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            input_ts = [torch.from_numpy(v).to(self.device) for _, v in inputs.items()]
            with torch.no_grad():
                output: Tuple[torch.Tensor] = compiled(*input_ts)
            return {
                k: v.cpu().detach().resolve_conj().numpy()
                if v.is_conj()
                else v.cpu().detach().numpy()
                for k, v in zip(torch_net.output_like.keys(), output)
            }

        return closure
