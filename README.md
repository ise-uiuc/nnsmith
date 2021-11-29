# NNSmith: DNN Model Generation in the Wild

This project is under heavy development at this point.

**Please** put bug reports/trackings on this [google sheet](https://docs.google.com/spreadsheets/d/15YY88x_JyZWom2YGNW2JO0JdqNVYWzPbaaRyhVxBJ_Y/edit#gid=0).

## Quick Start

```shell
export root='./tmp/seed1' # the path storing (to store) the model and inputs (outputs and bug reports)
# See difftest.py for the spec of the file structure

python ./nnsmith/input_gen.py --root ./tmp/seed1 # generate models and inputs

... # setup your enviroment for ort
python -m nnsmith.backend_executor --root $root --backend ort # test

... # setup your enviroment for xla
python -m nnsmith.backend_executor --root $root --backend xla # test
# ...

# compare the result (all close)
python -m nnsmith.difftest --root $root

# fuzzing
export target=fuzz_report
python nnsmith/fuzz.py --report $target
# Bug report will be put under `$target` (fuzz_report by default).
# Under $target
# - cov_by_time.csv               ~ csv to plot the coverage trend; (use `plot_cov.py`).
# - meta.txt                      ~ metadata.
# - ${ErrType}__${ID}/      
#                    - err.txt    ~ error message
#                    - model.onnx ~ error model

python plot_cov.py -f $target # -cl 80000
# use `-cl` to set the axis bias.
```

## Notes

<details><summary><b>To use ONNXRuntime on GPU & ONNX Simplifier</b> <i>[click to expand]</i></summary>
<div>

```shell
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime 
pip install onnxruntime-gpu # the order matters; and you have to split the install steps;
```

</div>
</details>

<details><summary><b>Misc</b> <i>[click to expand]</i></summary>
<div>

- To quickly install latest TVM on a linux machine (w/ CUDA 10.2 or higher): 
    - `pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels`
    - See also: https://tlcpack.ai/

- Please visit the following websites to learn about the operator conversion coverage when you decide to add new operators in our generator. That said, always prefer operators that are acceptable for most frameworks.
    - [TensorRT-ONNX Coverage](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)
    - [PyTorch-ONNX Coverage](https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/ONNXOpCoverage.md)
    - [TensorFlow-ONNX Coverage](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md)
    - [Glow-ONNX Coverage](https://github.com/pytorch/glow/tree/d7bd6c59e68a105edafe094ee77c987903eb24a5/tests/models/onnxModels)
    - TVM-ONNX Coverage: N/A

</div>
</details>

## Progress & TODOs

- [x] Export pytorch models to ONNX format (`nnsmith/export.py`); @jiawei
    - Actually PyTorch is not suitable for graph generation. It requires some effort to have an IR from our own and parse it to run in the `forward` function.
- [x] Data structure of abstract domain of shape analysis (regard it as an abstract interpretation problem) (`nnsmith/abstract/op.py`). @jiawei
- [x] **Op Batch 1** Shape function and constraints with the following operators (See [Tab. 2](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)): @jiawei
    - [x] One-to-one: ReLU ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#relu)) & Add ([[torch]](https://pytorch.org/docs/stable/generated/torch.add.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#add)) & LeakyReLU & PReLU & Sigmoid & Sin & Cos & Asin & Acos & Tan & Atan & Abs & Ceil & Clip & Round & Sqrt & Log & Not
    - [x] One-to-many: Expand ([[torch]](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Expand))
    - [x] Many-to-many: Conv ([[torch]](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv))
    - [x] Reorganize: Reshape ([[torch]](https://pytorch.org/docs/stable/generated/torch.reshape.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape))
    - [x] Shuffle: Transpose ([[torch]](https://pytorch.org/docs/stable/generated/torch.transpose.html) [[onnx]](https://github.com/onnx/onnx/blob/master/docs/Operators.md#transpose))
- [x] Make operator's parameters also symbolic (`nnsmith/abstract/op.py`). @jiawei
- [x] Random type-wise graph generation (`nnsmith/graph_gen.py`). @jiawei
- [x] γ function to map abstract domain to concrete domain (PyTorch's `nn.Module`). @jiawei @jinkun
    - NOTE: Jiawei tried to convert abstract graph into PyTorch's `nn.Module` and it works. However, due to the implementation issues of PyTorch's JIT tracing, we cannot export the `nn.Module` we created into ONNX model. Therefore, our next plan is to support Keras model generation and then export keras model into ONNX.
    - FIXED: Jinkun added `nn.ModuleList` to trace the layers as a workaround.
- [ ] Differential testing candidates: Given an ONNX model, get results from DNN libraries/compilers:
    - Specification: @jinkun @jiawei See `nnsmith/backends/__init__.py` for the specification.
        - Output: Output tensors (`Dict[np.ndarray]`);
        - Input: ONNX model; Input tensors (`Dict[np.ndarray]`);
        - Bug report: ptr->model, input tensors;
    - [ ] Oracles:
        - [x] Result consistency (allclose);
        - [ ] Performance degradation;
        - [ ] Crash (use sub-process when doing evaluation);
    - [x] Differential testing comparison (allclose); @jinkun
    - [x] TVM (dynamic models: VM/Debug; & graph); @jiawei
    - [x] ONNXRuntime (new); @jiawei
    - [x] XLA (ONNX to TF. Compile in XLA mode); @jinkun refined@jiawei
    - [x] TensorRT; @jiawei
    - [ ] Glow (not prioritized); @jinkun
- [x] Search-based input generation; @jinkun
- [x] Add edge coverage guidance and TVM fuzzing loop; @jiawei (Install TVM's [coverage branch](https://github.com/ganler/tvm/tree/coverage))
- [x] Fuzzing loop for TVM with `rich` monitoring (`nnsmith/fuzz.py`). @jiawei
- **FOUND AN ISSUE**: z3 SMT extremely slow when setting randomized constraints on input shapes.
    - If we don't set random input constraints -> very fast! but those solutions will stick to [1, 1, ..., 1] which is not realistic;
    - If we set those input constraints -> very slow (e.g., up to 25s to generate a 20-node model)... but the generated model is diverse!
- **Proposals**: @jiawei
    - [ ] Half-symbolic generation: only symbolize operators' parameters;
        - Pros: should be faster than prior one;
        - Cons: generated solves might be edge cases but we can add some guiding constraints;
- [ ] **Op Batch 2**: Focuse on multi-input & complex-shape-transfer-func models;
    - [ ] multi-input: And, Sub, Mul, Concat, Div, Greater; @jinkun
    - [x] complex-shape-func: Sum, Min, Max, Mean, ArgMin, ArgMax, Squeeze, Size; @jiawei
- [ ] Coverage-guided fuzzing with relation table. @jiawei
- [ ] Dynamic model testing;
- [ ] Enable multiple inputs;

