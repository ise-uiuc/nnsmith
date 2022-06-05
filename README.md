# NNSmith: DNN Model Generation in the Wild

This project is under heavy development at this point.

Keep active bug tracking and please put bug reports/trackings on this [google sheet](https://docs.google.com/spreadsheets/d/15YY88x_JyZWom2YGNW2JO0JdqNVYWzPbaaRyhVxBJ_Y/edit#gid=0).

## Quick Start

### Coverage Evaluation

#### nnsmith

```shell
# TVM
NNSMITH_DCE=0.1 LIB_PATH='../tvm/build/libtvm.so ../tvm/build/libtvm_runtime.so' \
python nnsmith/fuzz.py --mode guided --time 14400 \
                       --max_nodes 10 --eval_freq 256 \
                       --backend tvm --root nnsmith-tvm

# ORT
NNSMITH_DCE=0.1 LIB_PATH='../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime_providers_shared.so ../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so' \
python nnsmith/fuzz.py --mode guided --time 14400 \
                       --max_nodes 10 --eval_freq 256 \
                       --backend ort --root nnsmith-ort

python experiments/cov_merge.py -f nnsmith-tvm nnsmith-ort  # generate merged_cov.pkl
```

#### LEMON

Please prepare ~ 50GB disk space to store LEMON.

```shell
# step 1: Run LEMON to generate models (https://github.com/ganler/LEMON);
# step 2:
# For TVM
python experiments/lemon_tf2onnx.py --lemon_output_dir /PATH/TO/LEMON/lemon_outputs/ --onnx_dir lemon-onnx
python experiments/cov_eval.py --model_dir lemon-onnx    \
                               --report_folder lemon-tvm \
                               --backend tvm --lib '../tvm/build/libtvm.so ../tvm/build/libtvm_runtime.so' \
                               --llvm-version 14 # if you compile tvm w/ llvm 14 instrumented on ubuntu.
# For ORT:
python experiments/cov_eval.py --model_dir lemon-onnx \
                               --report_folder lemon-ort \
                               --backend ort \
                               --lib '../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime_providers_shared.so ../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so' \
                               --llvm-version 14
python experiments/cov_merge.py -f lemon-tvm lemon-ort # generate merged_cov.pkl
```

#### GraphFuzz

*The original paper does not give it a name so we call it GraphFuzz for convenience.*

```shell
# Make sure ORT dtype support config file is generated.
python nnsmith/dtype_test.py --cache config/ort_cpu_dtype.pkl

# TVM
python experiments/graphfuzz.py --time_budget 14400 --onnx_dir /PATH/TO/LEMON/graphfuzz-tvm-onnx
python experiments/cov_eval.py --model_dir /PATH/TO/LEMON/graphfuzz-tvm-onnx    \
                               --report_folder graphfuzz-tvm \
                               --backend tvm --lib '../tvm/build/libtvm.so ../tvm/build/libtvm_runtime.so' \
                               --llvm-version 14

# ORT
python experiments/graphfuzz.py --time_budget 14400 --onnx_dir /PATH/TO/LEMON/graphfuzz-ort-onnx --ort_cache config/ort_cpu_dtype.pkl
python experiments/cov_eval.py --model_dir /PATH/TO/LEMON/graphfuzz-ort-onnx \
                               --report_folder graphfuzz-ort \
                               --backend ort \
                               --lib '../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime_providers_shared.so ../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so' \
                               --llvm-version 14

python experiments/cov_merge.py -f graphfuzz-tvm graphfuzz-ort # generate merged_cov.pkl
```


#### Visualization

```shell
mkdir results # Store those files in results
# TVM coverage.
python experiments/viz_merged_cov.py --folders nnsmith-tvm graphfuzz-tvm lemon-tvm --tvm --pdf --tags 'NNSmith' 'GraphFuzz [Luo et al.]' 'LEMON' --venn
# ORT coverage.
python experiments/viz_merged_cov.py --folders nnsmith-ort graphfuzz-ort lemon-ort --ort --pdf --tags 'NNSmith' 'GraphFuzz [Luo et al.]' 'LEMON' --venn
python experiments/onnx_analyzer.py --folders lemon-onnx nnsmith-onnx GraphFuzz-onnx # venn graph ~ #op #edge
```

### Examine a single model

```shell
# run <model_path> with tvm (`--device` = cpu by default) backend with auto generated input.
python nnsmith/backend_executor.py --model <model_path> --backend tvm

# supply input domain with <domain_path>
python nnsmith/backend_executor.py --model <model_path> --backend tvm --raw_input <input_path> --input_domain <domain_path>

# supply <input_path> instead of random input generation
python nnsmith/backend_executor.py --model <model_path> --backend tvm --raw_input <input_path>

# do differential testing against `tvm-debug` backend
python nnsmith/backend_executor.py --model <model_path> --backend tvm --cmp_with tvm-debug

```

### Evaluate input searching algorithm

```shell
# Run experiments.
bash experiments/input_search_exp.sh 10
bash experiments/input_search_exp.sh 20
bash experiments/input_search_exp.sh 30

# visualization
python experiments/plot_inp_search_merge.py --root 512-model-10-node-exp \
                                                   512-model-20-node-exp \
                                                   512-model-30-node-exp

# debug
## run with models saved
python experiments/input_search.py --max_nodes 10 --n_model 100 --n_inp_sample 1 --root /path/to/save/model
## to reproduce a specific model
python -u debugging/grad.py /path/to/<model_id>-net.pkl <model_seed>
## to reproduce all models
python experiments/input_search.py --max_nodes 10 --n_model 100 --n_inp_sample 1 --load /path/to/save/model
```

## Notes

<details><summary><b>Minor Coding Spec</b> <i>[click to expand]</i></summary>
<div>

- **Do not put repeated logging/warning in library code.** Fuzzing loop might execute such programs for many times that such logs will create numerous loggings that messes STDOUT.

</div>
</details>

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
- [x] Differential testing candidates: Given an ONNX model, get results from DNN libraries/compilers:
    - Specification: @jinkun @jiawei See `nnsmith/backends/__init__.py` for the specification.
        - Output: Output tensors (`Dict[np.ndarray]`);
        - Input: ONNX model; Input tensors (`Dict[np.ndarray]`); **Not implemented in the Fuzzing Phase**.
        - Bug report: ptr->model, input tensors;
    - [x] Oracles:
        - [x] Result consistency (allclose);
        - [x] Crash (use sub-process when doing evaluation);
    - [x] Differential testing comparison (allclose); @jinkun
    - [ ] **Model device difference: CPU/GPU Differential Testing** @jiawei
    - DL Compiler-Driven Engines:
      - [x] TVM (dynamic models: VM/Debug; & graph); @jiawei **7.5k star; ative community**
      - [x] ONNXRuntime (new); @jiawei **6k star; ative community**
      - [x] XLA (ONNX to TF. Compile in XLA mode); @jinkun refined@jiawei **162k star; ative community**
      - [x] TensorRT; @jiawei **4.7k; NVidia Official; fastest GPU inference tool**
      - [ ] PyTorch JIT; @jiawei **53k star; ative community**
- [x] Search-based input generation; @jinkun
- [x] Add edge coverage guidance and TVM fuzzing loop; @jiawei (Install TVM's [coverage branch](https://github.com/ganler/tvm/tree/coverage))
- [x] Fuzzing loop for TVM with `rich` monitoring (`nnsmith/fuzz.py`). @jiawei
    - See instructions [here](https://github.com/Tzer-AnonBot/tzer/blob/main/tvm_cov_patch/build_tvm.sh).
- **FOUND AN ISSUE**: z3 SMT extremely slow when setting randomized constraints on input shapes.
    - If we don't set random input constraints -> very fast! but those solutions will stick to [1, 1, ..., 1] which is not realistic;
    - If we set those input constraints -> very slow (e.g., up to 25s to generate a 20-node model)... but the generated model is diverse!
- [x] **Op Batch 2**: Focuse on multi-input & complex-shape-transfer-func models;
    - [x] multi-input: And, Sub, Mul, Concat, Div, Greater; @jinkun
    - [x] complex-shape-func: Sum, Min, Max, Mean, ArgMin, ArgMax, Squeeze, Size; @jiawei
- [x] Coverage-guided fuzzing with relation table. @jiawei
- [x] Coverage feedback support for ONNXRuntime (Install ORT's [coverage branch](https://github.com/ganler/onnxruntime/tree/coverage)) @jiawei
    - Make sure you have `/usr/bin/clang++` installed with compiler runtime;
    - `git clone -b coverage git@github.com:ganler/onnxruntime.git --recursive`
    - `./build.sh --config RelWithDebInfo --build_shared_lib --parallel --build_wheel --skip_onnx_tests`
    - `pip install build/Linux/RelWithDebInfo/dist/onnxruntime-1.11.0-cp38-cp38-linux_x86_64.whl --force-reinstall`
- [x] Enable multiple inputs; @jinkun
- [x] **High-Priority** Parameter-wise Fuzzing. i.e., binning @jinkun
- [x] (Experimental) Improve input-searching algorithm @jiawei
    - [x] [Gradient-based Input Searching](https://dl.acm.org/doi/pdf/10.1145/3468264.3468612)
- [x] Implement the re-designed graph construction algorithm (mixed forward/backward construction) @jiawei
- [ ] Enhance fw-bw insertion by reusing outputs in backward insertion mode @jiawei
- [x] LEMON coverage evaluation ([modified](https://github.com/ganler/LEMON) to make it work) @jiawei
- [x] Coverage normalization. @jiawei
- [x] **Op Batch 3**: Softmax, BatchNorm, Linear, Flatten, *Pool2d. @jiawei
- [x] 2-phase evaluation: first generate model quickly; then evaluate them with instrumentation.
    - [x] LEMON
    - [x] nnsmith
    - [x] graph-fuzz
- [x] Implement baseline [graph-fuzz](https://dl.acm.org/doi/abs/10.1109/ICSE43902.2021.00037)
- [x] Migrate to source-level coverage (more information)
- [x] Inference in/out detailed data type by direct backend execution.
- [x] Fix `fuzz.py`
- [x] Generalize vulnerable operator
- [x] Proxy gradient
