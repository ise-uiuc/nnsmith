<p align="center">
    <img src="https://github.com/ganler/nnsmith-logo/raw/master/nnsmith-logo.svg", width="500">
</p>

<p align="center">
    <a href="https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml"><img src="https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml/badge.svg">
    <a href="https://pypi.org/project/nnsmith/"><img src="https://img.shields.io/pypi/v/nnsmith?color=g">
    <a href="https://github.com/ise-uiuc/nnsmith/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/nnsmith"></a>
</p>

## Backend-Model Support

<center>

| Model\Engine | [TVM](https://github.com/apache/tvm) | [ORT](https://github.com/microsoft/onnxruntime) | [TensorRT](https://github.com/NVIDIA/TensorRT) | [TFLite](https://www.tensorflow.org/lite) | [XLA](https://www.tensorflow.org/xla) | [IREE](https://github.com/iree-org/iree) |
| ------------ | ------------------------------------ | ----------------------------------------------- | ---------------------------------------------- | ----------------------------------------- | ------------------------------------- | ---------------------------------------- |
| PyTorch-ONNX | ✅                                    | ✅                                               | ✅                                              |                                           |                                       |                                          |
| TensorFlow   |                                      |                                                 |                                                | ⚠️                                         | ⚠️                                     | ⚠️                                        |


✅: Supported; ⚠️: Beta support; Others are not supported yet -- Contributions are welcome!

</center>

## Setup

**Install latest stable release:**

```shell
pip install "nnsmith[torch,onnx]" --upgrade
```

<details><summary><b>Install GitHub HEAD: </b> <i>[click to expand]</i></summary>
<div>

```shell
pip install "git+https://github.com/ise-uiuc/nnsmith@main#egg=nnsmith[torch,onnx]" --upgrade
# or pip install "git+ssh://git@github.com/ise-uiuc/nnsmith@main#egg=nnsmith[torch,onnx]" --upgrade
```

</div>
</details>

<details><summary><b>Install latest pre-release: </b> <i>[click to expand]</i></summary>
<div>

```shell
pip install "nnsmith[torch,onnx]"                     \
            --pre --upgrade                           \
            --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/
```

</div>
</details>


## Quick Start

<details><summary><b>Setting up graphviz for debugging</b> <i>[click to expand]</i></summary>
<div>

Graphviz provides `dot` for visualizing graphs in nice pictures. But it needs to be installed via the following methods:

```shell
sudo apt-get install graphviz graphviz-dev      # Linux
brew install graphviz                           # MacOS
conda install --channel conda-forge pygraphviz  # Conda
choco install graphviz                          # Windows

pip install pygraphviz  # Final step.
```

Also see [pygraphviz install guidance](https://pygraphviz.github.io/documentation/stable/install.html).

</div>
</details>

```shell
# Generate a random model in "nnsmith_outputs/*"
nnsmith.model_gen model.type=onnx debug.viz=true
```

See other commands under [`doc/cli`](doc/cli.md). We use [hydra](https://hydra.cc/) to manage configurations. See `nnsmith/config/main.yaml`.

## Developer Notes

- `pip install -r requirements/core.txt` to run generation and fuzzing;
- `pip install --upgrade -r requirements/sys/[system].txt` to allow generating and running specific frameworks;
  -  **Why "--upgrade"?** In fact, all the sources under `requirements/sys/` are nightly release (except tvm) as we want to "save the world" by catching new bugs;

<details><summary><b>Pre-commits</b> <i>[click to expand]</i></summary>
<div>

You can use `pre-commit` to simpify development:

- `pip install -r requirements/dev.txt`;
- `pre-commit install`;
- `pre-commit` will run upon a commit; To explicitly run `pre-commit` for all files: `pre-commit run --all-files`.

</div>
</details>

<details><summary><b>Local development</b> <i>[click to expand]</i></summary>
<div>

- Develop locally by setting `export PYTHONPATH=$PYTHONPATH:$(pwd)` (`pwd` should be this git folder.)
- Set `PYTHONPATH=""` when doing `pip install nnsmith` from online version.

</div>
</details>

<details><summary><b>Simplify the code</b> <i>[click to expand]</i></summary>
<div>

*Simplicity is prerequisite for reliability.* --Edsger W. Dijkstra

We want **code simplicity**: keeping minimal dependencies and focusing on a small set of simple APIs to make NNSmith maintainable to developers and reliable to users.

</div>
</details>

<details><summary><b>Test before commit</b> <i>[click to expand]</i></summary>
<div>

```shell
# env of torch & tf will conflict so split their unit tests.
pytest tests/core -s
pytest tests/torch -s
pytest tests/tensorflow -s
```

</div>
</details>

## Paper

Our paper is accepted by ASPLOS'23 and the pre-print is now available on [![arXiv](https://img.shields.io/badge/arXiv-2207.13066-b31b1b.svg)](https://arxiv.org/abs/2207.13066).
