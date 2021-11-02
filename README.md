# NNSmith: DNN Model Generation in the Wild

This project is under heavy development at this point.

## TODOs

- [x] Export pytorch models to ONNX format; @jiawei
    - Actually PyTorch is not suitable for graph generation. It requires some effort to have an IR from our own and parse it to run in the `forward` function.
- [x] Data structure of abstract domain of shape analysis (regard it as an abstract interpretation problem). @jiawei
- [x] Shape function and constraints with the following operators (See [Tab. 2](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)): @jiawei
    - [x] One-to-one: ReLU ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#relu)) & Add ([[torch]](https://pytorch.org/docs/stable/generated/torch.add.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#add));
    - [x] One-to-many: Expand ([[torch]](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Expand))
    - [x] Many-to-many: Conv ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv))
    - [x] Reorganize: Reshape ([[torch]](https://pytorch.org/docs/stable/generated/torch.reshape.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape))
    - [x] Shuffle: Transpose ([[torch]](https://pytorch.org/docs/stable/generated/torch.transpose.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#transpose))
- [ ] γ function to map abstract domain to concrete domain (PyTorch's `nn.Module`). @jiawei
- [ ] Random type-wise graph generation. @jiawei
- [ ] Differential testing candidates: Given an ONNX model, get results from DNN libraries/compilers:
    - [ ] Specification: @jinkun
        - Output: Pickle containing Dict[Array]; (naming convension: ${MODEL_NAME}.${FRAMEWORK}.pickle)
        - Input: ONNX model; Ones(shape);
        - Bug report: ptr->model, input;
    - [ ] Oracles:
        - Result consistency (allclose);
        - Performance degradation;
        - Crash;
    - [ ] Differential testing comparison (allclose); @jinkun
    - [ ] TVM (dynamic models: VM/Debug; & graph); @jiawei
    - [ ] ONNXRuntime (new); @jiawei
    - [ ] XLA (ONNX to TF. Compile in XLA mode); @jinkun
    - [ ] TensorRT; @jiawei
    - [ ] Glow; @jinkun
- [ ] Dynamic model testing;
- [ ] Mutating input tensors;

