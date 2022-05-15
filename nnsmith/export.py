import warnings

import torch
import torch.onnx

from nnsmith.abstract.op import DType, ShapeVar
from nnsmith.graph_gen import SymbolNet


# Torch is actually not an ideal choice for graph generation,
# as it is based on dynamic graph construction.
def torch2onnx(model: SymbolNet, filename, verbose=False, use_cuda=False, dummy_inputs=None, do_constant_folding=False):
    """Convert PyTorch model to ONNX format.
    """
    proxy_enabled = model.proxy_enabled
    if proxy_enabled:
        model.disable_proxy_grad()

    dev = torch.device('cuda' if use_cuda else 'cpu')
    # Get dynamic axis sizes & input names
    dynamic_axes = {}
    input_names = list(model.input_spec.keys())
    for name in input_names:
        dshape = [i for i, v in enumerate(model.input_spec[name]) if v == -1]
        if len(dshape) > 0:
            dynamic_axes[name] = dshape
    output_names = [f'o{i}' for i in range(model.n_output)]

    # TODO: explicitly model outputs.
    # output_names = list(model.output_spec.keys())
    # for name in output_names:
    #     dshape = [i for i, v in enumerate(model.output_spec[name]) if v != -1]
    #     if len(dshape) > 0:
    #         dynamic_axes[name] = dshape

    # Dummy inputs
    if dummy_inputs is None:
        dummy_inputs = []
        for _, svar in model.plausible_input_shape.items():
            dummy_inputs.append(torch.rand(
                size=svar.shape, device=dev).to(dtype=svar.dtype.value))
    if verbose:
        print(f"Generated model:\n{model}")

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter(
                "default" if verbose else "ignore", category=torch.jit.TracerWarning)
            model.to(dev)
            model.eval()
            torch.onnx.export(
                model, tuple(dummy_inputs),
                filename,
                input_names=input_names,
                output_names=output_names,
                verbose=verbose,
                dynamic_axes=dynamic_axes,
                do_constant_folding=do_constant_folding,
                opset_version=14)

    if proxy_enabled:  # Re-enable proxy grad
        model.enable_proxy_grad()

    return input_names, output_names


if __name__ == "__main__":
    def test_torch2onnx(net):
        try:
            torch2onnx(net, "test.onnx")
        finally:
            import os
            if 'NO_DEL' not in os.environ:
                if os.path.isfile("test.onnx"):
                    os.remove("test.onnx")

    class DyNet(torch.nn.Module):
        def __init__(self):
            super(DyNet, self).__init__()
            # Following attributes are required to export ONNX model.
            self.plausible_input_shape = {"i0": ShapeVar(
                [1, 1, 3]), "i1": ShapeVar([2, 3, 3], dtype=DType.float64)}
            self.input_spec = {"i0": [-1, -1, 3], "i1": [2, 3, 3]}
            self.n_output = 2

        @torch.no_grad()
        def forward(self, x, y):
            return x, y

    test_torch2onnx(DyNet())

    class StaticNet(torch.nn.Module):
        def __init__(self):
            super(StaticNet, self).__init__()
            # Following attributes are required to export ONNX model.
            self.plausible_input_shape = {"i0": ShapeVar(
                [1, 1, 3]), "i1": ShapeVar([1, 1, 3])}
            self.input_spec = {"i0": [1, 1, 3], "i1": [1, 1, 3]}
            self.n_output = 1

        @torch.no_grad()
        def forward(self, x, y):
            return x + y

    test_torch2onnx(StaticNet())
