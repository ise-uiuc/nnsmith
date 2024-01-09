<div align="center">
    <img src="https://github.com/ganler/nnsmith-logo/raw/master/nnsmith-logo.png" align="right" alt="logo" width="200px"/>
</div>

# NNSmith

[![](https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml/badge.svg)](https://github.com/ise-uiuc/nnsmith/actions/workflows/ci.yaml)
[![](https://img.shields.io/pypi/v/nnsmith?color=g)](https://pypi.org/project/nnsmith/)
[![](https://static.pepy.tech/badge/nnsmith)](https://pepy.tech/project/nnsmith)
[![](https://img.shields.io/pypi/l/nnsmith)](https://github.com/ise-uiuc/nnsmith/blob/main/LICENSE)

🌟NNSmith🌟 is a random DNN generator and a fuzzing infrastructure, primarily designed for automatically validating deep-learning frameworks and compilers.

## Support Table

<div align="center">

| Models | [`tvm`](https://github.com/apache/tvm) | [`pt2`](https://pytorch.org/get-started/pytorch-2.0/) | [`torchjit`](https://pytorch.org/docs/stable/jit.html) | [`tensorrt`](https://github.com/NVIDIA/TensorRT) | [`onnxruntime`](https://github.com/microsoft/onnxruntime) | [`xla`](https://www.tensorflow.org/xla) | [`tflite`](https://www.tensorflow.org/lite) |
| ------------ | ------------------------------------ | ----------------------------------------------- | ---------------------------------------------- | ----------------------------------------- | ------------------------------------- | ----------------------------------------------------- | ------------ |
| ONNX         | ✅                                    |                                                |                                               | ✅ | ✅ |                                                       |  |
| PyTorch |                                      | ✅📈 | ✅📈 |                                          |                                      |                                         |                                             |
| TensorFlow |                                      |                                                       |                                                        |                                           |                                       | ✅                                                    | ✅ |

✅: Supported; 📈: Supports gradient check;

</div>

## Quick Start

**Install latest code (GitHub HEAD):**

```shell
pip install pip --upgrade
pip install "nnsmith[torch,onnx] @ git+https://github.com/ise-uiuc/nnsmith@main" --upgrade
# [optional] add more front- and back-ends such as [tensorflow] and [tvm,onnxruntime,...] in "[...]"
```

<details><summary><b>Install latest stable release </b> <i>[click]</i></summary>
<div>

```shell
pip install "nnsmith[torch,onnx]" --upgrade
```

</div>
</details>

<details><summary><b>Install latest pre-release </b> <i>[click]</i></summary>
<div>

```shell
pip install "nnsmith[torch,onnx]" --upgrade --pre
```

</div>
</details>

<details><summary><b>Setting up graphviz for debugging</b> <i>[click]</i></summary>
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

## Learning More

- 🐛 [**Uncovered bugs**](doc/bugs.md).
- 📚 [**Documentation**](doc/): [CLI](doc/cli.md), [concept](doc/concept.md), [logging](doc/log-and-err.md), and [known issues](doc/known-issues.md).
- 🤗 [**Contributing to NNSmith**](doc/CONTRIBUTING.md)
- 📝 We use [hydra](https://hydra.cc/) to manage configurations. See `nnsmith/config/main.yaml`.

## Papers

<details><summary><b> 📜 NeuRI: Diversifying DNN Generation via Inductive Rule Inference </b> <i>[click :: citation]</i></summary>
<div>

```bibtex
@article{liu2023neuri,
  title = {NeuRI: Diversifying DNN Generation via Inductive Rule Inference},
  author = {Liu, Jiawei and Peng, Jinjun and Wang, Yuyao and Zhang, Lingming},
  journal = {arXiv preprint arXiv:2302.02261},
  year = {2023},
}
```

</div>
</details>

<p align="center">
    <a href="https://arxiv.org/abs/2302.02261"><img src="https://img.shields.io/badge/Paper-FSE'23-a55fed.svg"></a>
    <a href="https://arxiv.org/abs/2302.02261"><img src="https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg"></a>
    <a href="https://github.com/ise-uiuc/neuri-artifact"><img src="https://img.shields.io/badge/artifact-git-black.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.8319975"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8319975.svg"></a>
</p>

<details><summary><b> 📜 NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers </b> <i>[click :: citation]</i></summary>
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
