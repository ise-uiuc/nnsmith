# NNSmith: DNN Model Generation in the Wild

[![](https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml/badge.svg)](https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml) [![](https://img.shields.io/pypi/v/nnsmith?color=g)](https://pypi.org/project/nnsmith/) [![](https://img.shields.io/pypi/l/nnsmith)](https://github.com/ise-uiuc/nnsmith/blob/main/LICENSE)

## Backend-Model Support

| Backend\Model | ONNX/PyTorch | TensorFlow |
| ------------- | ------------ | ---------- |
| TVM           | ✅            |            |
| ONNXRuntime   | ✅            |            |
| TensorRT      | ✅            |            |
| TFLite        |              | ⚠️          |
| XLA           |              | ⚠️          |

✅: Supported; ⚠️: Beta support; Others are not supported yet -- Contributions are welcome!

## Quick Start

<details><summary><b>Setting up graphviz for debugging</b> <i>[click to expand]</i></summary>
<div>

Graphviz provides `dot` for visualizing graphs in nice pictures. But it needs to be installed via the following methods:

```shell
sudo apt-get install graphviz graphviz-dev      # Linux
brew install graphviz                           # MacOS
conda install --channel conda-forge pygraphviz  # Conda
choco install graphviz                          # Windows
```

Also see [pygraphviz install guidance](https://pygraphviz.github.io/documentation/stable/install.html).

</div>
</details>

```shell
python3 -m pip install "nnsmith[torch,onnx]"                     \
                       --upgrade                                 \
                       --index-url https://test.pypi.org/simple/ \
                       --extra-index-url https://pypi.org/simple/
# Generate a 5-node graph:
nnsmith.model_gen model.max_nodes=5 debug.viz=true
# see "nnsmith_outputs/*"
```

See other commands under `nnsmith/cli`. We use [hydra](https://hydra.cc/) to manage configurations. See `nnsmith/config/main.yaml`.

## Developer Notes

- `pip install -r requirements/core.txt` to run generation and fuzzing;
- `pip install --upgrade -r requirements/sys/[system].txt` to allow generating and running specific frameworks;
  -  **Why "--upgrade"?** In fact, all the sources under `requirements/sys/` are nightly release (except tvm) as we want to "save the world" by catching new bugs;

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

You can use `pre-commit` to simpify development:

- `pip install -r requirements/dev.txt`;
- `pre-commit install`;
- `pre-commit` will run upon a commit; To explicitly run `pre-commit` for all files: `pre-commit run --all-files`.

<details><summary><b>More notes</b> <i>[click to expand]</i></summary>
<div>

*Simplicity is prerequisite for reliability.* --Edsger W. Dijkstra

We want **code simplicity**: keeping minimal dependencies and focusing on a small set of simple APIs to make NNSmith maintainable to developers and reliable to users.

</div>
</details>

Run tests before commit:

```shell
# env of torch & tf will conflict so split their unit tests.
pytest tests/core -s
pytest tests/torch -s
pytest tests/tensorflow -s
```


<!--
### Coverage Evaluation

**WIP: Scripts under `experiments/` are not ready yet due to recent refactors.**

To run coverage evaluation, first compile the DL framework with LLVM's [source-based code coverage](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html). The commands below should be at least compatible with LLVM-14.

<details><summary><b>NNSmith</b> <i>[click to expand]</i></summary>
<div>

```shell
bash experiments/cov_exp.sh
python experiments/cov_merge.py -f nnsmith-tvm-* nnsmith-ort-*  # generate merged_cov.pkl
```

</div>
</details>


<details><summary><b>LEMON</b> <i>[click to expand]</i></summary>
<div>

Please prepare 100GB disk space to store LEMON's outputs.

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

</div>
</details>

<details><summary><b>GraphFuzzer</b> <i>[click to expand]</i></summary>
<div>

*The original [paper](https://conf.researchr.org/details/icse-2021/icse-2021-papers/68/Graph-based-Fuzz-Testing-for-Deep-Learning-Inference-Engines) does not give it a name so we call it GraphFuzzer for convenience.*

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

</div>
</details>

<details><summary><b>Visualization</b> <i>[click to expand]</i></summary>
<div>

```shell
mkdir results # Store those files in results
# TVM coverage.
python experiments/viz_merged_cov.py --folders lemon-tvm graphfuzz-tvm nnsmith-tvm --tvm --pdf --tags 'LEMON' 'GraphFuzzer' 'NNSmith' --venn --output main_result
# ORT coverage.
python experiments/viz_merged_cov.py --folders lemon-ort graphfuzz-ort nnsmith-ort --ort --pdf --tags 'LEMON' 'GraphFuzzer' 'NNSmith' --venn --output main_result
```

</div>
</details>

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
``` -->
