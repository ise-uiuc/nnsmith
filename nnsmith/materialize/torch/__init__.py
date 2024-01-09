import inspect
import pickle
from abc import ABC, abstractmethod
from os import PathLike
from typing import Dict, List, Optional, Type

import torch
from torch import fx, nn

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import FxTracing, SymbolNet
from nnsmith.util import register_seed_setter


# FIXME(@ganler): handle (Sequential, ModuleList, ModuleDict) precisely
# FIXME(@ganler): experimental; May not work.
def _safe_repr(mod: nn.Module):
    if isinstance(mod, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
        return False
    # Heuristic 2: multi-line repr
    if mod.__repr__().count("\n") > 0:
        return False
    return True


def _render_constructor(name: str, mod: nn.Module):
    if _safe_repr(mod):
        return f"torch.nn.{mod.__repr__()}"

    # TODO(@ganler): Report them to PyTorch
    # Dealing exceptional cases
    if isinstance(mod, nn.MultiheadAttention):
        kvs = {
            "embed_dim": mod.embed_dim,
            "num_heads": mod.num_heads,
            "dropout": mod.dropout,
            "bias": mod.in_proj_bias is not None,
            "add_bias_kv": mod.bias_k is not None,
            "add_zero_attn": mod.add_zero_attn,
            "kdim": mod.kdim,
            "vdim": mod.vdim,
            "batch_first": mod.batch_first,
        }
        sig = inspect.signature(nn.MultiheadAttention)
        kvs = {k: v for k, v in kvs.items() if sig.parameters[k].default != v}
        return f"torch.nn.MultiheadAttention({ ', '.join(f'{k}={v}' for k, v in kvs.items()) })"

    return f"torch.load('{name}.pth') # {type(mod).__name__}"


def numpify(v: Optional[torch.Tensor]):
    if v is None:
        return None
    return (
        v.cpu().detach().resolve_conj().numpy()
        if v.is_conj()
        else v.cpu().detach().numpy()
    )


class TorchModel(Model, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.torch_model: SymbolNet = None
        self.sat_inputs = None

    @classmethod
    @abstractmethod
    def device(cls) -> torch.device:
        pass

    @property
    def version(self) -> str:
        return torch.__version__

    @classmethod
    def from_gir(cls: Type["TorchModel"], ir: GraphIR, **kwargs) -> "TorchModel":
        ret = cls()
        ret.torch_model = SymbolNet(ir, **kwargs).to(cls.device())
        return ret

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    def refine_weights(self) -> None:
        self.torch_model.enable_proxy_grad()
        use_cuda = self.device().type == "cuda"
        searcher = PracticalHybridSearch(self.torch_model, use_cuda=use_cuda)
        # TODO(@ganler): Can we directly get both inputs and outputs?
        _, inputs = searcher.search(
            max_time_ms=20,
            max_sample=2,
        )
        if inputs:
            self.sat_inputs = inputs
        self.torch_model.disable_proxy_grad()

    def make_oracle(self) -> Oracle:
        inputs = self.sat_inputs or self.torch_model.get_random_inps()

        # Return NumPy array as outputs
        input_dict = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
        output_dict = {}

        # TODO(@ganler, @FatPigeorz): can we make sure of determinism?
        if self.needs_grad_check():
            self.torch_model.train()
            params = {k: v for k, v in self.torch_model.named_parameters()}
            outputs = self.torch_model.forward(
                *[v.to(self.device()) for _, v in inputs.items()]
            )
            for name, output in zip(self.output_like.keys(), outputs):
                output_dict[name] = numpify(output)
                if output.requires_grad:
                    # Get Vector-Jacobian Product (VJP)
                    out_grad = torch.autograd.grad(
                        outputs=output,
                        inputs=self.torch_model.parameters(),
                        grad_outputs=torch.ones_like(output),
                        retain_graph=True,
                        allow_unused=True,
                    )

                    for k, v in zip(params.keys(), out_grad):
                        output_dict[name + "_vjp_" + k] = numpify(v)

        else:  # No gradient required.
            with torch.no_grad():
                self.torch_model.eval()
                outputs = self.torch_model.forward(
                    *[v.to(self.device()) for _, v in inputs.items()]
                )

                for oname, val in zip(self.output_like.keys(), outputs):
                    output_dict[oname] = numpify(val)

        return Oracle(input_dict, output_dict, provider="torch[cpu] eager")

    def dump(self, path: PathLike):
        torch.save(self.torch_model.state_dict(), path)
        gir_path = path.replace(
            TorchModel.name_prefix() + TorchModel.name_suffix(),
            TorchModel.gir_name(),
        )
        with open(gir_path, "wb") as f:
            pickle.dump(self.torch_model.ir, f)

    @classmethod
    def load(cls, path: PathLike) -> "TorchModel":
        ret = cls()
        gir_path = path.replace(
            cls.name_prefix() + cls.name_suffix(),
            cls.gir_name(),
        )
        with open(gir_path, "rb") as f:
            ir = pickle.load(f)
        torch_model = SymbolNet(ir).to(cls.device())
        torch_model.load_state_dict(torch.load(path), strict=False)
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

    @property
    def import_libs(self) -> List[str]:
        return ["import torch", "import numpy as np", "import pickle"]

    def emit_input(self, inp_name: str, path: Optional[PathLike] = None):
        if path is not None:  # Assume NumPy tensors as inputs
            return f"{inp_name} = [v for _, v in pickle.load(open('{path}', 'rb'))['input'].items()]"

        # Path is None. Generate inputs from scratch.
        tensor_text = []
        for at in self.input_like.values():
            tensor_text.append(f"np.zeros({at.shape}, dtype='{at.dtype}')")

        return f"{inp_name} = [{', '.join(tensor_text)}]"

    def emit_weight(self, mod_name: str, path: Optional[PathLike] = None):
        if path is None:
            return "# No specific weights to load. Just let it self-initialize."

        return f"{mod_name}.load_state_dict(torch.load('{path}'), strict=False)"

    def emit_def(self, mod_name: str, mod_cls: str) -> str:
        with FxTracing():
            if isinstance(self.native_model, SymbolNet):
                fn_args = f"self, {', '.join(self.native_model.input_map.keys())}"
                fn_kwargs = ", ".join([f"{k}={k}" for k in self.native_model.input_map])

                tmp = self.native_model.forward
                self.native_model.forward = (
                    f"lambda {fn_args}: SymbolNet.forward(self, {fn_kwargs})"
                )
                traced = fx.symbolic_trace(self.native_model)
                self.native_model.forward = tmp
            else:
                traced = fx.symbolic_trace(self.native_model)

        tab = " " * 4
        mod_text = ""
        # _parameters only shows the upper-level parameters
        # while
        # named_parameters() shows all parameters recursively
        for name, param in self.native_model._parameters.items():
            mod_text += (
                f"{2*tab}self.{name} = torch.nn.Parameter"
                + f"(torch.empty({list(param.shape)}, dtype={param.dtype}), requires_grad={param.requires_grad})\n"
            )

        for name, mod in self.native_model.named_children():
            if name == "":
                continue
            mod_text += f"{2*tab}self.{name} = {_render_constructor(name, mod)}\n"

        # fwd code is `traced.code` with 2 tabs in front of each line
        fwd_code = tab + ("\n" + tab).join(traced.code.strip().splitlines())

        return f"""class {mod_cls}(torch.nn.Module):
    def __init__(self):
        super().__init__()
{mod_text}
{fwd_code}

{mod_name} = {mod_cls}()
"""

    def emit_run(self, out_name: str, inp_name: str, mod_name: str) -> str:
        return f"""{out_name} = {mod_name}(*[torch.from_numpy(v).to('{self.device().type}') for v in {inp_name}])
{out_name} = [v.cpu().detach() for v in {out_name}] # torch2numpy
{out_name} = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in {out_name}] # torch2numpy"""

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("torch", torch.manual_seed, overwrite=True)


class TorchModelCPU(TorchModel):
    @classmethod
    def device(cls) -> torch.device:
        return torch.device("cpu")


class TorchModelCUDA(TorchModel):
    @classmethod
    def device(cls) -> torch.device:
        return torch.device("cuda")
