<p align="center">
    <img src="https://github.com/ganler/nnsmith-logo/raw/master/nnsmith-logo.svg", width="500">
</p>

<p align="center">
    <a href="https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml"><img src="https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml/badge.svg">
    <a href="https://pypi.org/project/nnsmith/"><img src="https://img.shields.io/pypi/v/nnsmith?color=g">
    <a href="https://github.com/ise-uiuc/nnsmith/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/nnsmith"></a>
</p>

## Support table

<center>

| Model\Engine | [TVM](https://github.com/apache/tvm) | [ORT](https://github.com/microsoft/onnxruntime) | [TensorRT](https://github.com/NVIDIA/TensorRT) | [TFLite](https://www.tensorflow.org/lite) | [XLA](https://www.tensorflow.org/xla) | [Torch-JIT](https://pytorch.org/docs/stable/jit.html) |
| ------------ | ------------------------------------ | ----------------------------------------------- | ---------------------------------------------- | ----------------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| ONNX         | ✅                                    | ✅                                               | ✅                                              |                                           |                                       |                                                       |
| TensorFlow   | 🔨                                    |                                                 |                                                | ✅                                         | ✅                                     |                                                       |
| PyTorch      | 🔨                                    | 🔨                                               |                                                |                                           |                                       | 🔨                                                     |




✅: Supported; ⚠️: Experimental support; 🔨: Coming soon;

</center>

## Setup

**Install latest code (GitHub HEAD):**

```shell
pip install "git+https://github.com/ise-uiuc/nnsmith@main#egg=nnsmith[torch,onnx]" --upgrade
# [optional] add more front- and back-ends such as [tf] and [tvm,ort,xla,...] in "[...]"
```

<details><summary><b>Install latest stable release: </b> <i>[expand]</i></summary>
<div>

```shell
pip install "nnsmith[torch,onnx]" --upgrade
```

</div>
</details>

<details><summary><b>Install latest pre-release: </b> <i>[expand]</i></summary>
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

<details><summary><b>Setting up graphviz for debugging</b> <i>[expand]</i></summary>
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
- `pip install --upgrade --pre -r requirements/sys/[system].txt` to allow generating and running specific frameworks;
  -  **Why "--upgrade --pre"?** In fact, all the sources under `requirements/sys/` are nightly release (except tvm) as we want to "save the world" by catching new bugs;

<details><summary><b>Pre-commits</b> <i>[expand]</i></summary>
<div>

You can use `pre-commit` to simpify development:

- `pip install -r requirements/dev.txt`;
- `pre-commit install`;
- `pre-commit` will run upon a commit; To explicitly run `pre-commit` for all files: `pre-commit run --all-files`.

</div>
</details>

<details><summary><b>Local development</b> <i>[expand]</i></summary>
<div>

- Develop locally by setting `export PYTHONPATH=$PYTHONPATH:$(pwd)` (`pwd` should be this git folder.)
- Set `PYTHONPATH=""` when doing `pip install nnsmith` from online version.

</div>
</details>

<details><summary><b>Simplify the code</b> <i>[expand]</i></summary>
<div>

*Simplicity is prerequisite for reliability.* --Edsger W. Dijkstra

We want **code simplicity**: keeping minimal dependencies and focusing on a small set of simple APIs to make NNSmith maintainable to developers and reliable to users.

</div>
</details>

<details><summary><b>Test before commit</b> <i>[expand]</i></summary>
<div>

```shell
# env of torch & tf will conflict so split their unit tests.
pytest tests/core -s
pytest tests/torch -s
pytest tests/tensorflow -s
```

</div>
</details>

## Notes

+ NNSmith is modularized and can be extended as a 3rd-party library, which allows you to patch your own backend and do fuzzing without modifying NNSmith's source code.
+ Meanwhile, feel free to [request](https://github.com/ise-uiuc/nnsmith/issues) a backend support: the project maintainer is happy to support DL systems that care about software reliability and quality to benefit the whole DL software stack.
+ It would be great if you can [let us know](https://github.com/ise-uiuc/nnsmith/issues) if you find new bugs with NNSmith or build a new system inspired by NNSmith.

## Paper

<details><summary><b>NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers.</b> <i>[expand citation]</i></summary>
<div>

```bibtex
@inproceedings{liu2023nnsmith,
  title={Nnsmith: Generating diverse and valid test cases for deep learning compilers},
  author={Liu, Jiawei and Lin, Jinkun and Ruffy, Fabian and Tan, Cheng and Li, Jinyang and Panda, Aurojit and Zhang, Lingming},
  booktitle={Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
  pages={530--543},
  year={2023}
}
```

</div>
</details>

<p align="center">
    <a href="https://dl.acm.org/doi/10.1145/3575693.3575707"><img src="https://img.shields.io/badge/Paper-ASPLOS'23-a55fed.svg"></a>
    <a href="https://arxiv.org/abs/2207.13066"><img src="https://img.shields.io/badge/arXiv-2207.13066-b31b1b.svg"></a>
    <a href="http://nnsmith-asplos.rtfd.io/"><img src="https://img.shields.io/badge/artifact-doc-black.svg"></a>
    <a href="https://github.com/ganler/nnsmith-asplos-artifact"><img src="https://img.shields.io/badge/artifact-git-black.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.7222132"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7222132.svg"></a>
</p>
