# NNSmith: DNN Model Generation in the Wild

This project is under heavy development at this point.

## Notes

- To quickly install latest TVM on a linux machine (w/ CUDA 10.2 or higher): 
    - `pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels`
    - See also: https://tlcpack.ai/
- Please visit the following websites to learn about the operator conversion coverage when you decide to add new operators in our generator. That said, always prefer operators that are acceptable for most frameworks.
    - [TensorRT-ONNX Coverage](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)
    - [PyTorch-ONNX Coverage](https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/ONNXOpCoverage.md)
    - [TensorFlow-ONNX Coverage](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)
    - [Glow-ONNX Coverage](https://github.com/pytorch/glow/tree/d7bd6c59e68a105edafe094ee77c987903eb24a5/tests/models/onnxModels)
    - TVM-ONNX Coverage: N/A

## Progress & TODOs

- [x] Export pytorch models to ONNX format; @jiawei
    - Actually PyTorch is not suitable for graph generation. It requires some effort to have an IR from our own and parse it to run in the `forward` function.
- [x] Data structure of abstract domain of shape analysis (regard it as an abstract interpretation problem). @jiawei
- [x] Shape function and constraints with the following operators (See [Tab. 2](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)): @jiawei
    - [x] One-to-one: ReLU ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#relu)) & Add ([[torch]](https://pytorch.org/docs/stable/generated/torch.add.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#add)) & LeakyReLU & PReLU & Sigmoid & Sin & Cos & Asin & Acos & Tan & Atan & Abs & Ceil & Clip & Round & Sqrt & Log & Not
    - [x] One-to-many: Expand ([[torch]](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Expand))
    - [x] Many-to-many: Conv ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv))
    - [x] Reorganize: Reshape ([[torch]](https://pytorch.org/docs/stable/generated/torch.reshape.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape))
    - [x] Shuffle: Transpose ([[torch]](https://pytorch.org/docs/stable/generated/torch.transpose.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#transpose))
- [ ] γ function to map abstract domain to concrete domain (PyTorch's `nn.Module`). @jiawei
- [ ] Random type-wise graph generation. @jiawei
- [ ] Differential testing candidates: Given an ONNX model, get results from DNN libraries/compilers:
    - Specification: @jinkun @jiawei See `nnsmith/backends/__init__.py` for the specification.
        - Output: Output tensors (`Dict[np.ndarray]`);
        - Input: ONNX model; Input tensors (`Dict[np.ndarray]`);
        - Bug report: ptr->model, input tensors;
    - [ ] Oracles:
        - Result consistency (allclose);
        - Performance degradation;
        - Crash;
    - [ ] Differential testing comparison (allclose); @jinkun
    - [x] TVM (dynamic models: VM/Debug; & graph); @jiawei
    - [ ] ONNXRuntime (new); @jiawei
    - [ ] XLA (ONNX to TF. Compile in XLA mode); @jinkun
    - [ ] TensorRT; @jiawei
    - [ ] Glow; @jinkun
- [ ] Model parameters to operator constructors; @jiawei
- [ ] Dynamic model testing;
- [ ] Mutating input tensors;

