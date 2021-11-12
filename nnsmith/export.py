import torch
import torch.onnx


# Torch is actually not an ideal choice for graph generation,
# as it is based on dynamic graph construction.
# TODO: Use CUDA to accelerate the export process.
def torch2onnx(model, filename):
    """Convert PyTorch model to ONNX format.
    """
    # Get dynamic axis sizes & input names
    dynamic_axes = {}

    input_names = list(model.input_spec.keys())
    for name in input_names:
        dshape = [i for i, v in enumerate(model.input_spec[name]) if v == -1]
        if len(dshape) > 0:
            dynamic_axes[name] = dshape

    # TODO: explicitly model outputs.
    # output_names = list(model.output_spec.keys())
    # for name in output_names:
    #     dshape = [i for i, v in enumerate(model.output_spec[name]) if v != -1]
    #     if len(dshape) > 0:
    #         dynamic_axes[name] = dshape

    # Dummy inputs
    dummy_inputs = []
    for _, v in model.plausible_input_shape.items():
        dummy_inputs.append(torch.zeros(v))

    torch.onnx.export(
        model, tuple(dummy_inputs),
        filename,
        input_names=input_names,
        output_names=[f'o{i}' for i in range(model.n_output)],
        verbose=True,
        dynamic_axes=dynamic_axes)


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
            self.plausible_input_shape = {"i0": [1, 1, 3], "i1": [2, 3, 3]}
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
            self.plausible_input_shape = {"i0": [1, 1, 3], "i1": [1, 1, 3]}
            self.input_spec = {"i0": [1, 1, 3], "i1": [1, 1, 3]}
            self.n_output = 1

        @torch.no_grad()
        def forward(self, x, y):
            return x + y

    test_torch2onnx(StaticNet())
