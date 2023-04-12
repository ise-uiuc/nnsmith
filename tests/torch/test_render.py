import pytest
import torch
from torch import nn

from nnsmith.abstract import AbsTensor, AvgPool2d, DType
from nnsmith.gir import GraphIR, InstExpr, Placeholder
from nnsmith.materialize.torch import TorchModelCPU


# CV model
def test_model_def_clean0():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3)
            self.linear = nn.Linear(3, 3)
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            x = self.conv(x)
            x = self.linear(x)
            x = self.bn(x)
            return x

    model = TorchModelCPU()
    model.torch_model = M()

    assert (
        model.emit_def(mod_name="m", mod_cls="M")
        == R"""class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        self.linear = torch.nn.Linear(in_features=3, out_features=3, bias=True)
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        conv = self.conv(x);  x = None
        linear = self.linear(conv);  conv = None
        bn = self.bn(linear);  linear = None
        return bn

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
                            ttype=AbsTensor(shape=[1, 3, 64, 64], dtype=DType.float32)
                        ).as_input(),
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
