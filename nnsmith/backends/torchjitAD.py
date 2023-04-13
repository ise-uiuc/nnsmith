import os
import warnings
from typing import Dict, Tuple, List


import numpy as np
import torch
from multipledispatch import dispatch
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.autograd.forward_ad as fwAD

from nnsmith.backends.factory import BackendCallable, BackendFactory, BackendInput, BackendOutput
from nnsmith.materialize.torch import TorchModel
from nnsmith.abstract.op import AbsTensor
from nnsmith.materialize.torch.symbolnet import random_tensor


# Check https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# for more PyTorch-internal options.
NNSMITH_PTJIT_OPT_MOBILE = os.getenv("NNSMITH_PTJIT_OPT_MOBILE", "0") == "1"


class TorchJITAD(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False, ad="", **kwargs):
        super().__init__(target, optmax)
        if self.target == "cpu":
            self.device = torch.device("cpu")
        elif self.target == "cuda":
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(
                f"Unknown target: {self.target}. Only `cpu` and `cuda` are supported."
            )
        
        self.ad = ad

    @property
    def system_name(self) -> str:
        return "torchjitAD"

    def make_backend_forward(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device).train()
        with fwAD.dual_level():
            with torch.no_grad():
                for _, param in torch_net.named_parameters():
                        dual = fwAD.make_dual(param.to(self.device), torch.ones_like(param).to(self.device))
                        param.copy_(dual)
            trace_inp = [ts.to(self.device) for ts in torch_net.get_random_inps().values()]
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    category=torch.jit.TracerWarning,
                )
                exported = torch.jit.trace(
                    torch_net,
                    trace_inp,
                )
                if self.target == "cpu" and NNSMITH_PTJIT_OPT_MOBILE:
                    exported = optimize_for_mobile(exported)
        
        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            with fwAD.dual_level():
                with torch.no_grad():
                    for _, param in torch_net.named_parameters():
                            dual = fwAD.make_dual(param.to(self.device), torch.ones_like(param).to(self.device))
                            param.copy_(dual)
                input_ts = [torch.from_numpy(v).to(self.device) for _, v in inputs.items()]
                dual_output = exported(*input_ts)
                output_dict = {}
                for k, v in zip(torch_net.output_like.keys(), dual_output):
                    primal, tangent = fwAD.unpack_dual(v)
                    primal = primal.cpu().detach().resolve_conj().numpy() if primal.is_conj() else primal.cpu().detach().numpy()
                    # get the Jacobian-vector product
                    if tangent is not None:
                        tangent = tangent.cpu().detach().resolve_conj().numpy() if tangent.is_conj() else tangent.cpu().detach().numpy()
                    output_dict[k] = primal
                    output_dict[k + "_jvp"] = tangent
                
                return output_dict

        return closure

    def make_backend_backward(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device).train()
        trace_inp = [ts.to(self.device) for ts in torch_net.get_random_inps().values()]
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=torch.jit.TracerWarning,
            )
            exported = torch.jit.trace(
                torch_net,
                trace_inp,
            )
            if self.target == "cpu" and NNSMITH_PTJIT_OPT_MOBILE:
                exported = optimize_for_mobile(exported)

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, ...]]:
            input_ts = [torch.from_numpy(v).to(self.device) for _, v in inputs.items()]
            outputs = exported(*input_ts)
            params = {k : v for k, v in exported.named_parameters()}
            output_dict = {}
            for name, output in zip(torch_net.output_like.keys(), outputs):
                if output.requires_grad is False:
                    output = output.cpu().detach().resolve_conj().numpy() if output.is_conj() else output.cpu().detach().numpy()
                    output_dict[name] = output
                    continue
                # if the output is differentiate
                # get Vector-Jacobian product
                out_grad = torch.autograd.grad(outputs=output, inputs=params.values(), grad_outputs=torch.ones_like(output),retain_graph=True, allow_unused=True)
                for k, v in zip(params.keys(), out_grad):
                    if v is None:
                        output_dict[name + "_vjp_" + k] = None
                    else:
                        output_dict[name + "_vjp_" + k] = v.cpu().detach().resolve_conj().numpy() if v.is_conj() else v.cpu().detach().numpy()
                output = output.cpu().detach().resolve_conj().numpy() if output.is_conj() else output.cpu().detach().numpy()
                output_dict[name] = output

            return output_dict


        return closure

    @dispatch(TorchModel)
    def make_backend(self, model: TorchModel) -> BackendCallable:
        if self.ad == "forward":
            return self.make_backend_forward(model)
        elif self.ad == "backward":
            return self.make_backend_backward(model)
        else:
            raise ValueError(f"Unknown AD mod {self.ADMod}. Try `forward` or `backward`.")
        
    @property
    def import_libs(self) -> List[str]:
        return ["import torch"]

    def emit_compile(self, opt_name: str, mod_name: str, inp_name: str) -> str:
        return f"{opt_name} = torch.jit.trace({mod_name}, [torch.from_numpy(v).to('{self.device.type}') for v in {inp_name}])"

    def emit_run(self, out_name: str, opt_name: str, inp_name: str) -> str:
        return f"""{out_name} = {opt_name}(*[torch.from_numpy(v).to('{self.device.type}') for v in {inp_name}])
{out_name} = [v.cpu().detach() for v in {out_name}] # torch2numpy
{out_name} = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in {out_name}] # torch2numpy"""
