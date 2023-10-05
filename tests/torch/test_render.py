import subprocess

import pytest
import torch
from torch import nn

from nnsmith.abstract import AbsTensor, AvgPool2d, DType
from nnsmith.backends.pt2 import PT2
from nnsmith.backends.torchjit import TorchJIT
from nnsmith.gir import GraphIR, InstExpr, Placeholder
from nnsmith.materialize import Render
from nnsmith.materialize.torch import TorchModelCPU


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.bn = nn.BatchNorm2d(3)
        self.linear = nn.Linear(62, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.linear(x)
        return x


def test_model_def_clean0():
    model = TorchModelCPU()
    model.torch_model = CNN()

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.linear = torch.nn.Linear(in_features=62, out_features=3, bias=True)

    def forward(self, x):
        conv = self.conv(x);  x = None
        bn = self.bn(conv);  conv = None
        linear = self.linear(bn);  bn = None
        return linear

m = M()
"""
    )


# RNN model
def test_model_def_clean1():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, 3)
            self.linear = nn.Linear(3, 3)

        def forward(self, x):
            x = self.lstm(x)
            x = self.linear(x)
            return x

    model = TorchModelCPU()
    model.torch_model = M()

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(3, 3)
        self.linear = torch.nn.Linear(in_features=3, out_features=3, bias=True)

    def forward(self, x):
        lstm = self.lstm(x);  x = None
        linear = self.linear(lstm);  lstm = None
        return linear

m = M()
"""
    )


# Transformer model
def test_model_def_clean2():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention0 = nn.MultiheadAttention(3, 3)
            self.attention1 = nn.MultiheadAttention(3, 3)
            self.linear = nn.Linear(3, 3)

        def forward(self, x):
            x = self.attention0(x, x, x)
            x = self.attention1(x, x, x)
            x = self.linear(x)
            return x

    model = TorchModelCPU()
    model.torch_model = M()

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention0 = torch.nn.MultiheadAttention(embed_dim=3, num_heads=3, kdim=3, vdim=3)
        self.attention1 = torch.nn.MultiheadAttention(embed_dim=3, num_heads=3, kdim=3, vdim=3)
        self.linear = torch.nn.Linear(in_features=3, out_features=3, bias=True)

    def forward(self, x):
        attention0 = self.attention0(x, x, x);  x = None
        attention1 = self.attention1(attention0, attention0, attention0);  attention0 = None
        linear = self.linear(attention1);  attention1 = None
        return linear

m = M()
"""
    )


def test_model_def_seq():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(3, 3, 3),
                nn.Linear(3, 3),
                nn.BatchNorm2d(3),
            )

        def forward(self, x):
            return self.seq(x)

    model = TorchModelCPU()
    model.torch_model = M()

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.load('seq.pth') # Sequential

    def forward(self, x):
        seq_0 = getattr(self.seq, "0")(x);  x = None
        seq_1 = getattr(self.seq, "1")(seq_0);  seq_0 = None
        seq_2 = getattr(self.seq, "2")(seq_1);  seq_1 = None
        return seq_2

m = M()
"""
    )


def test_model_def_mod_dict():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod_dict = torch.nn.ModuleDict(
                {
                    "conv": torch.nn.Conv2d(3, 3, 3),
                    "linear": torch.nn.Linear(3, 3),
                    "bn": torch.nn.BatchNorm2d(3),
                }
            )

        def forward(self, x):
            x = self.mod_dict.conv(x)
            x = self.mod_dict.linear(x)
            x = self.mod_dict.bn(x)
            return x

    model = TorchModelCPU()
    model.torch_model = M()

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_dict = torch.load('mod_dict.pth') # ModuleDict

    def forward(self, x):
        mod_dict_conv = self.mod_dict.conv(x);  x = None
        mod_dict_linear = self.mod_dict.linear(mod_dict_conv);  mod_dict_conv = None
        mod_dict_bn = self.mod_dict.bn(mod_dict_linear);  mod_dict_linear = None
        return mod_dict_bn

m = M()
"""
    )


def test_model_def_nnsmith():
    ir = GraphIR()
    ir.add_inst(
        InstExpr(
            AvgPool2d(kh=3, kw=3, stride=3, padding=2),
            [
                ir.add_inst(
                    InstExpr(
                        Placeholder(
                            ttype=AbsTensor([1, 3, 64, 64], DType.float32)
                        ).input(),
                        [],
                    )
                ).retval()
            ],
        )
    )

    model = TorchModelCPU.from_gir(ir)

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=3, padding=2)

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        m1 = self.m1(getitem);  getitem = None
        return (m1,)

m = M()
"""
    )


def test_render_model_only():
    model = TorchModelCPU()
    model.torch_model = CNN()

    model.torch_model.input_like = {"x": AbsTensor([1, 3, 64, 64], DType.float32)}

    render = Render()
    render.emit_model(model)
    render.emit_input(model)

    # pickle is not used (no `ModuleList` in the code)
    # so no need to import pickle
    assert (
        render.render()
        == R"""
import numpy as np
import torch
import pickle

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.linear = torch.nn.Linear(in_features=62, out_features=3, bias=True)

    def forward(self, x):
        conv = self.conv(x);  x = None
        bn = self.bn(conv);  conv = None
        linear = self.linear(bn);  bn = None
        return linear

m = M()


# Initialize weight
# None

# Initialize input
inp = [np.zeros([1, 3, 64, 64], dtype='float32')]

# Compile the model
# None

# Eager run
m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
m_out = [v.cpu().detach() for v in m_out] # torch2numpy
m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out] # torch2numpy

# Compiled run
# None

# Differential testing
# None
"""
    )


def test_render_e2e_param_pt2():
    model = TorchModelCPU()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.const = torch.nn.Parameter(
                torch.empty([1], dtype=torch.int16), requires_grad=False
            )

        def forward(self, x):
            squeeze = self.const.squeeze(0)
            mul = torch.mul(squeeze, x)
            expand = mul.expand(1)
            expand_1 = mul.expand(1, 1, 1, 1)
            max_1 = torch.max(expand_1, x)
            return (expand, max_1)

    model.torch_model = M()

    model.torch_model.input_like = {"x": AbsTensor([], DType.int16)}

    render = Render()
    render.emit_model(model)
    render.emit_input(model)
    render.emit_backend(PT2(target="cpu", optmax=True))

    rendered = render.render()

    # pickle is not used (no `ModuleList` in the code)
    # so no need to import pickle
    assert (
        rendered
        == R"""
import numpy as np
import torch
import pickle

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.const = torch.nn.Parameter(torch.empty([1], dtype=torch.int16), requires_grad=False)

    def forward(self, x):
        const = self.const
        squeeze = const.squeeze(0);  const = None
        mul = torch.mul(squeeze, x);  squeeze = None
        expand = mul.expand(1)
        expand_1 = mul.expand(1, 1, 1, 1);  mul = None
        max_1 = torch.max(expand_1, x);  expand_1 = x = None
        return (expand, max_1)

m = M()


# Initialize weight
# None

# Initialize input
inp = [np.zeros([], dtype='int16')]

# Compile the model
opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)

# Eager run
m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
m_out = [v.cpu().detach() for v in m_out] # torch2numpy
m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out] # torch2numpy

# Compiled run
opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp])
opt_out = [v.cpu().detach() for v in opt_out] # torch2numpy
opt_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in opt_out] # torch2numpy

# Differential testing
for i, (l, r) in enumerate(zip(m_out, opt_out)):
    np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {i}")
"""
    )

    # Run rendered code in a subprocess as a smoke test
    subprocess.run(
        ["python", "-c", rendered],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_render_e2e_cnn_pt2():
    model = TorchModelCPU()
    model.torch_model = CNN()

    model.torch_model.input_like = {"x": AbsTensor([1, 3, 64, 64], DType.float32)}

    render = Render()
    render.emit_model(model)
    render.emit_input(model)
    render.emit_backend(PT2(target="cpu", optmax=True))

    rendered = render.render()

    # pickle is not used (no `ModuleList` in the code)
    # so no need to import pickle
    assert (
        rendered
        == R"""
import numpy as np
import torch
import pickle

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.linear = torch.nn.Linear(in_features=62, out_features=3, bias=True)

    def forward(self, x):
        conv = self.conv(x);  x = None
        bn = self.bn(conv);  conv = None
        linear = self.linear(bn);  bn = None
        return linear

m = M()


# Initialize weight
# None

# Initialize input
inp = [np.zeros([1, 3, 64, 64], dtype='float32')]

# Compile the model
opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)

# Eager run
m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
m_out = [v.cpu().detach() for v in m_out] # torch2numpy
m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out] # torch2numpy

# Compiled run
opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp])
opt_out = [v.cpu().detach() for v in opt_out] # torch2numpy
opt_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in opt_out] # torch2numpy

# Differential testing
for i, (l, r) in enumerate(zip(m_out, opt_out)):
    np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {i}")
"""
    )

    # Run rendered code in a subprocess as a smoke test
    subprocess.run(
        ["python", "-c", rendered],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_render_e2e_cnn_torchjit():
    model = TorchModelCPU()
    model.torch_model = CNN()

    model.torch_model.input_like = {"x": AbsTensor([1, 3, 64, 64], DType.float32)}

    render = Render()
    render.emit_model(model)
    render.emit_input(model)
    render.emit_backend(TorchJIT(target="cpu", optmax=True))

    rendered = render.render()

    # pickle is not used (no `ModuleList` in the code)
    # so no need to import pickle
    assert (
        rendered
        == R"""
import numpy as np
import torch
import pickle

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.linear = torch.nn.Linear(in_features=62, out_features=3, bias=True)

    def forward(self, x):
        conv = self.conv(x);  x = None
        bn = self.bn(conv);  conv = None
        linear = self.linear(bn);  bn = None
        return linear

m = M()


# Initialize weight
# None

# Initialize input
inp = [np.zeros([1, 3, 64, 64], dtype='float32')]

# Compile the model
opt = torch.jit.trace(m, [torch.from_numpy(v).to('cpu') for v in inp])

# Eager run
m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
m_out = [v.cpu().detach() for v in m_out] # torch2numpy
m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out] # torch2numpy

# Compiled run
opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp])
opt_out = [v.cpu().detach() for v in opt_out] # torch2numpy
opt_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in opt_out] # torch2numpy

# Differential testing
for i, (l, r) in enumerate(zip(m_out, opt_out)):
    np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {i}")
"""
    )

    # Run rendered code in a subprocess as a smoke test
    subprocess.run(
        ["python", "-c", rendered],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
