# import torch


# class Net(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.param = torch.nn.Parameter(torch.rand((1, 3, 24, 24)))
#         self.intermediate_y = None

#     def forward(self, x: torch.Tensor):
#         x = self.param
#         y = torch.asin(x)
#         self.intermediate_y = y
#         return torch.asin(y)


# net = Net()
# x = torch.ones((1, 3, 24, 24))
# optimizer = torch.optim.Adamax(net.parameters(), lr=0.1)
# loss_fn = torch.nn.MSELoss()


# for _ in range(10):
#     out = net(x)
#     # print('out:', out)
#     # diff = loss_fn(out, torch.tensor(0.))
#     print('out', out.mean())

#     op_loss = loss_fn(net.intermediate_y,
#                       torch.zeros_like(net.intermediate_y))
#     print('vulnerable op loss: ', op_loss)

#     op_loss.backward()
#     optimizer.step()

#     for p in net.parameters():
#         print('expected inp: ', p.mean())

#     optimizer.zero_grad()

import torch
import torch.nn.functional as F


# class Net(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x: torch.Tensor):
#         return torch.relu(x)

# opset 10
# class Net(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x: torch.Tensor):
#         return torch.nn.functional.pad(x, (1, 1))


# opset 11

# import torch

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.clip(x, 0.3, 0.5)


net = Net().cuda()

onnx_model = 'test.onnx'

torch.onnx.export(net, (torch.ones((3, 3), dtype=torch.float16).cuda(),),
                  onnx_model, verbose=True, opset_version=7)


# import onnxruntime
# import torch

# model = torch.nn.PReLU()
# onnx_model = "test.onnx"
# torch.onnx.export(model, (torch.tensor(1.0),), onnx_model, verbose=True)
# i_sess = onnxruntime.InferenceSession(onnx_model)
# i_sess.run([], {"input": torch.tensor(1.0).numpy(),})

# print(f'{onnxruntime.__version__=}; {torch.__version__=}')