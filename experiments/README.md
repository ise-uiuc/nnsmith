# Experiments

## Evaluating source-level coverage of NNSmith-generated tests

> **Note** **Developer note**
>
> - TODO(@ganler): we need to simplify the evaluation by memorizing some internal parameters.
> - TODO(@Kristoff-starling): tutorials for TensorFlow.

### Step 1: Compile the System Under Test (SUT) with coverage instrumentation

Taking PyTorch as an example:

**Install PyTorch**

```shell
git clone https://github.com/pytorch/pytorch.git --recursive
cd pytorch
```

**Modify the source code to enable coverage instrumentation**

You may use the patch from [here](https://gist.github.com/ganler/986078a929f08962d966dcc0b8ec0ebe) along with the following commands:

```shell
# Add the patch which enables instrumentation
wget https://gist.githubusercontent.com/ganler/986078a929f08962d966dcc0b8ec0ebe/raw/f00502c86127b4a1867c99c3c2d5879f8c223460/torch_cov.patch
git apply torch_cov.patch
```

> **Warning** **Compatibility of the patch**
>
> Nonetheless, you can manually curate the original patch file from [here](https://gist.github.com/ganler/986078a929f08962d966dcc0b8ec0ebe)
> The patch might not be compatible with the latest commit of PyTorch where you may need some edits to make it work.
> Nonetheless x 2, `git checkout 0692bdd95fc1a448c69a484cf34203b937a9eadc` is verified to work.

> **Warning** **Functionality of the patch**
>
> The patch by default only instrument a subset of PyTorch files:
> + It starts with `${Caffe2_CPU_SRCS}`
> - But removes kernel functions files such as `"aten/src/ATen/native/*"`
> - But removes files that blocks the linkage such as `"torch/csrc/jit/serialization/*"`
> Why don't we instrument kernel files?
> (i)  They are not the focus of compiler fuzzing (such kernels are mostly for eager mode)
> (ii) They slow down the instrumentation by a lot as they are mostly made of deep and nested loops
> Anyhow, you can modify the patch to instrument more files.

**Compile PyTorch**

> **Warning** **Create a new Conda environments**
>
> The commands below will install some new packages (including `torch`) to your current Python/Conda environment.
> You may want to use a new Conda environment to avoid messing up your current environment.

```shell
USE_CPP_CODE_COVERAGE=1 \
CC=clang-14 CXX=clang++-14 \
USE_KINETO=0 BUILD_CAFFE2=0 USE_DISTRIBUTED=0 USE_NCCL=0 BUILD_TEST=0 USE_XNNPACK=0 \
USE_QNNPACK=0 USE_MIOPEN=0 BUILD_CAFFE2_OPS=0 USE_TENSORPIPE=0 \
USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 \
python setup.py develop
```

> **Note** **Modify the commands based on your needs**
>
> - You may change the `CC` and `CXX` to other versions of Clang (but it cannot be GCC!)
> - The script above compiles almost the minimalist version of PyTorch (CPU-only). You may enable more flags to compile more components of PyTorch (thought it could break the compilation!)

**Test your installation**

```shell
python -c "import torch;print(torch.mul(1, 1))"
ls default.profraw # This is the coverage file. You should be able to see it.
```

> **Note** **``libstdc++.so.6: version `GLIBCXX_3.4.30' not found``**
>
> You may encounter such an error when running `import torch` at this step. The is because the `libstdc++` in your local system
> is more advanced/recent than the one in the conda environment.
>
> To address this, you can try `strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30` and see if there is any output.
> If so, you can substitute the `libstdc++` in conda environment with the one in the OS, like below.
>
> ```shell
> export CONDA_LIB_PATH=$(python3 -c "import site, pathlib; print(pathlib.Path(site.getsitepackages()[0]).parent.parent)")
> mv ${CONDA_LIB_PATH}/libstdc++.so.6 ${CONDA_LIB_PATH}/libstdc++.so.6.bk # backup
> ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_LIB_PATH}/libstdc++.so.6
> ```

### Step 2: Run NNSmith over instrumented SUT

Next, we will run NNSmith to dump a bunch of random test-cases for an SUT (say PyTorch 2).

> **Warning** **Use the non-instrumented SUT to run NNSmith first**
>
> We will run NNSmith to generate and validate each test-case -- by validating the test-cases,
> we need to execute the **SUT** (i.e., the compiler) as well. Notably, the **SUT** to run here
> is **NOT** the instrumented version we talked about in Step 1. Instead, we need to use the
> corresponding non-instrumented version in order to mimic the real-world use case -- because
> in real world, if you use NNSmith to test, say PyTorch, you don't need to re-compile PyTorch
> with coverage instrumentation -- you just do `pip install torch` and go with NNSmith.

> **Note** **Use the nightly version of SUT**
>
> We recommend you to use the nightly version of the SUT to run NNSmith in order to find new
> bugs.
>
> ```shell
> # PyTorch
> pip install --index-url https://download.pytorch.org/whl/nightly/cpu --pre torch
> # TensorFlow
> pip install tf-nightly
> ```
> For more please visit [here](https://github.com/ise-uiuc/nnsmith/blob/main/requirements/sys/).

Now we can run the fuzzer:

```shell
# For example:
python nnsmith/cli/fuzz.py  fuzz.time=4h fuzz.root=${PATH_TO_REPORT} \
                            model.type=torch backend.type=pt2        \
                            filter.type="[nan,inf,dup]"              \
                            fuzz.save_test=${PATH_TO_SAVE_TESTS}
```

The step above will save all models and tests generated to `${PATH_TO_SAVE_TESTS}`. It will
take 4 hours to run the fuzzer as we set `fuzz.time=4h`.

### Step 3: Record and compute coverage

By doing step 2, we obtained a set of test-cases in `${PATH_TO_SAVE_TESTS}`. Now we need to
collect the coverage by replaying each test-case on the instrumented SUT.

So, first switch to the (conda) environment that has the instrumented SUT installed. Then, run
the following command by filling the corresponding `MODEL`, `BACKEND`, and `DEVICE` used in
Step 2.

```shell
python experiments/evaluate_models.py --root ${PATH_TO_SAVE_TESTS}        \
                                               --model_type ${MODEL}      \
                                               --backend_type ${BACKEND}  \
                                               --backend_target ${DEVICE}
```

> **Note** **Parallelization**
>
> The replay will be much faster as it can be parallelized (configurable via `--parallel`).
> It also uses an optional argument `--batch-size` which refers the number of test-cases to
> be executed in each thread/process as a batch. The default value is 100 (This is important for future steps).

### Step 4: Coverage analysis

The replay will generate LLVM profile files (`.profraw`) in `${PATH_TO_SAVE_TESTS}/coverage`. We need to further parse them to get detailed coverage information:

```shell
python experiments/process_profraws.py --root ${PATH_TO_SAVE_TESTS}       \
                                       --llvm-config-path  ${LLVM_CONFIG} \
                                       --batch-size        ${BS}          \
                                       --instrumented-libs ${....}
```

- `--llvm-config-path`: Recall the version of clang you used in Step 1. You need to set the `LLVM_CONFIG` to the corresponding version of `llvm-config` (e.g., `clang-14` uses `llvm-config-14` and `clang` just uses `llvm-config`)
- `--batch-size`: this must be set to the same value as in Step 3 (if you use the default one, it is just `100`)
- `--instrumented-libs`: this must be set to the `.so` library being instrumented in Step 2. For PyTorch, the part should be `${TORCH_ROOT}/build/lib/libtorch.so ${TORCH_ROOT}/build/lib/libtorch_cpu.so`. For TVM, it is `${TVM_ROOT}/build/libtvm.so ${TVM_ROOT}/build/libtvm_runtime.so`.

> **Note** **Parallelization**
>
> `--parallel` is also available.

After this, we can visualize the coverage via:

```shell
python experiments/viz_merged_cov.py -o results --folders \
${PATH_TO_SAVE_TESTS}/coverage --tags "NNSmith" --pdf
```

Figures will be stored in where the `-o` option specifies.

If you have multiple experiments to show together:

```shell
python experiments/viz_merged_cov.py -o results --folders \
                                         ${PATH_TO_SAVE_TESTS_1}/coverage \
                                         ${PATH_TO_SAVE_TESTS_2}/coverage \
                                     --tags "Exp 1" "Exp 2" --pdf
```
