## Installation

```shell
python3 -m pip install "nnsmith[torch,onnx,tvm,onnxruntime]" --upgrade
# Or try the HEAD branch:
# pip install "git+https://github.com/ise-uiuc/nnsmith@main#egg=nnsmith[torch,onnx,tvm,onnxruntime]" --upgrade
```

The core functionality of NNSmith is graph generation.
Based on this, we provide powerful functionalities to do fuzzing and bug reporting.
Therefore, the systems and model types you want to fuzz are installed as dependencies as is shown in labels inside "`[]`".

We currently have model formats:
- `torch`
- `onnx`
- `tensorflow` (experimental)

and backends:
- `tvm`: TVM
- `onnxruntime`: ONNXRuntime
- `trt`: TensorRT
- `xla`: XLA
- `tflite`: TFLite
- `torchjit`: PyTorch JIT

Meanwhile, the backend of `xla` and `tflite` is installed as part of TensorFlow.

You can also have your own by extending `nnsmith.materialize.Model` and `nnsmith.backends.BackendFactory`.

## Graph generation

```shell
# Generate 5-node onnx model.
nnsmith.model_gen mgen.max_nodes=5 model.type=onnx debug.viz=true
# See: nnsmith_output/* (default output folder)

# TensorFlow model.
nnsmith.model_gen debug.viz=true model.type=tensorflow

# User-spec. output directory
nnsmith.model_gen debug.viz=true model.type=tensorflow mgen.save=tf_output
```

## Locally debug a model

```python
# Generate a onnx model
nnsmith.model_gen model.type=onnx mgen.max_nodes=5

# Check the model
pip install onnxruntime # use ONNXRuntime to execute the model
nnsmith.model_exec model.type=onnx backend.type=onnxruntime model.path=nnsmith_output/model.onnx
# `model.path` should point to the exact model, instead of a folder.
# It will first compile and run to see if there's any bug.
# By default it will search `oracle.pkl` and do verification.

# Check the model and do diff testing with tvm
nnsmith.model_exec  model.type=onnx                        \
                    backend.type=onnxruntime               \
                    model.path=nnsmith_output/model.onnx   \
                    cmp.with='{type:tvm, optmax:true, target:cpu}'
```

## Data type testing

Many compilers do not support a full set of operators (in ONNX and TensorFlow). Thus, we infer the support set by doing single operator testing.

```shell
# Infer the support set of onnxruntime to ONNX format.
nnsmith.dtype_test model.type="onnx" backend.type="onnxruntime"
# Results are often cached in `~/.cache/nnsmith`.
```

## Fuzzing

```shell
nnsmith.fuzz fuzz.time=30s model.type=onnx backend.type=tvm fuzz.root=fuzz_report debug.viz=true
# Bug reports are stored in `./fuzz_report`.
```

## Limit operator types, ranks and data types

To limit:
- rank only to be 4 (needed by Conv2d);
- data type only to be float32;
- only include Conv2d and ReLU.

```shell
yes | nnsmith.model_gen model.type=torch mgen.method=symbolic-cinit \
                                         mgen.rank_choices="[4]"    \
                                         mgen.dtype_choices="[f32]" \
                                         mgen.include="[core.NCHWConv2d, core.ReLU]" \
                                         debug.viz=true
```

## Add extra constraints

```shell
# Create patch file as `patch.py`
echo 'from nnsmith.abstract.arith import nnsmith_lt
from nnsmith.abstract.extension import patch_requires


@patch_requires("global", "core.NCHWConv2d")
def limit_conv2d(self, _):
    # let the kernels to be > 3
    return [nnsmith_lt(3, self.kernel_h_size), nnsmith_lt(3, self.kernel_w_size)]
' > patch.py
# Apply the patch with `mgen.patch_requires=./tests/mock/requires_patch.py` (can also be a list of paths)
yes | nnsmith.model_gen model.type=torch mgen.method=symbolic-cinit \
                                         mgen.rank_choices="[4]"    \
                                         mgen.dtype_choices="[f32]" \
                                         mgen.include="[core.NCHWConv2d, core.ReLU]" \
                                         mgen.patch_requires=./patch.py \
                                         debug.viz=true
```

## Misc

TensorFlow logging can be very noisy. Use `TF_CPP_MIN_LOG_LEVEL=3` as environmental variable to depress that.
