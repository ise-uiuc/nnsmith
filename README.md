# NNSmith: DNN Model Generation in the Wild

This project is under heavy development at this point.

## TODOs

- [x] Export pytorch models to ONNX format;
    - Actually PyTorch is not suitable for graph generation. It requires some effort to have an IR from our own and parse it to run in the `forward` function.
- [ ] Static model generation with the following operators (See [Tab. 2](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)):
    - [ ] One-to-one: ReLU ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#relu)) & Add ([[torch]](https://pytorch.org/docs/stable/generated/torch.add.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#add));
    - [ ] One-to-many: Expand ([[torch]](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Expand))
    - [ ] Many-to-many: Conv ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv))
    - [ ] Reorganize: Reshape ([[torch]](https://pytorch.org/docs/stable/generated/torch.reshape.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape))
    - [ ] Shuffle: Transpose ([[torch]](https://pytorch.org/docs/stable/generated/torch.transpose.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#transpose))