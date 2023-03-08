# Contributor Guide 🤗

We welcome various sorts of contributions to NNSmith,
including reporting an [issue](https://github.com/ise-uiuc/nnsmith/issues) or submitting a [PR](https://github.com/ise-uiuc/nnsmith/pulls).

## Reporting an issue

We appreciate developers to report current limitations of NNSmith on the [GitHub issue tracking system](https://github.com/ise-uiuc/nnsmith/issues),
including but not limited to:

1. **Bug reports**: unexpected behaviors of NNSmith (e.g., a flaw of the operator rule/specification)
2. **Feature requests**: tell us what could be the promising feature to make NNSmith stronger!

## Submitting a PR

The general flow for submitting a PR (Pull Request):

> (Optional) Submit an issue to talk about the PR (necessary for introducing new features);

1. [Fork](https://github.com/ise-uiuc/nnsmith/fork) NNSmith;
2. Pull your fork: `git clone git@github.com:$[YOUR_FORK]/nnsmith.git`;
3. `cd nnsmith && export PYTHONPATH=$PYTHONPATH:$(pwd)`
4. Coding NNSmith! Then commit and push your code!
5. Submit a PR [here](https://github.com/ise-uiuc/nnsmith/pulls);
6. Code review;
7. Merge!

### Do I need to submit an issue before PR?

- **No**: minor or nice-to-have cases such as typo fixes and bug fixes.
- **Yes**: new features (e.g., extending new backend) and fundamental changes.

### Will my contributions be rejected?

Oftentimes not, rare cases yes (that's why it is suggested to submit an issue for discussion first).

**S-sized contributions** are oftentimes easy-to-accept, including bug/typo fixes, CI improvements, test-case improvements, etc.
as long as it is beneficial and satisfies the properties in the "General coding guidance" section.

**M-sized contributions** such as extending new front-ends/backends/fuzzing strategies/etc. are welcome as well
-- as long as it shows an edge in improvements.
However, for maintainability, it could be moved to the temporary "contrib" folder if it is non-trivial/unclear for being well-maintained.
For example, let's say we supported backend "X" in the "contrib" folder and started to submitting bug reports to the "X" community.
Later on, if "X" community is found to be not interested fixing bugs
-- we don't have to support "X" as backend and consequently we can just drop it.

**L-sized contributions** are those that conflicting the fundamental designs and goals of NNSmith.
For example, NNSmith is fundamentally model generator, and it too much for it to support, for example, "distributed training".
As a result, such changes might not be accepted unless there is a compelling justification
-- but NNSmith is under Apache-2.0 -- you can always make it in the way you like via a fork :).
Of course, some L-sized contributions can still possibly accepted,
such as improving the operator specification or developing a more promising intermediate representation than GraphIR,
as long as we agree on that the benefits (over the efforts) are unquestionable.

## General coding guidance

### `pre-commit`

[`pre-commit`](https://pre-commit.com/) is a convenient tool to check and format your code while committing codes.

To set-up pre-commit:

```shell
pip install -r requirements/dev.txt
pre-commit install
```

Now it will run checking and auto-formatting while you commit:

```shell
git commit ...
# if [NOTHING HAPPENS], you are good to go;
# if [IT FAILS], the auto-formatting is automatically applied;
#                you just need to check, `git add` these changes and re-commit.
```

### Testing

If applicable (e.g., adding a new backend), add a few tests to validate your implementation. Examples can be found:

1. [Python unit-tests](https://github.com/ise-uiuc/nnsmith/tree/main/tests);
2. [End-to-end testing](https://github.com/ise-uiuc/nnsmith/blob/main/.github/workflows/ci.yaml);

To run the Python tests:

```shell
# env of torch & tf (and others) will conflict so split their unit tests.
pytest tests/core -s
pytest tests/torch -s
pytest tests/tensorflow -s
pytest tests/onnxruntime -s
pytest tests/tvm -s
pytest tests/tensorrt -s
```

### Simple code

> “Simplicity is the prerequisite for reliability.” - Edsger W. Dijkstra

Maintaining code is hard, esp. when
(i) initial code owners are not available; and
(ii) the code is too complicated to be understood/modified.
As a result, contributors are recommended to write simple code:
(i) easy-to-understand;
(ii) well-organized and easy-to-extend;
(iii) well-documented if the concept is tricky;
and (iv) avoiding changes that brings low improvement over high complexity.

For example, the complexity of test-case structure is non-trivial in NNSmith;
consequently, initial maintainers spent some amount of effort to make it systematically structured,
so that it will be easier-to-use and extend.
(@ganler: I know it could be boring, but it is indeed important for a long-live project.)

![](https://gist.github.com/ganler/bdf7e867e57c96e8c09ff31cb0b90a1f/raw/4667ad9b7dcb0b77cb722e7025402105560ebf41/datastructure.png)

There are a few more concrete terms to consider:

1. Try not to introduce new dependencies:
    - If we only need "one" function from the prospective dependency, implement it on our own if possible;
    - If we have to use, try to consider "reliable" ones first. For example, those have been tested by millions of developers (such as NumPy).
2. Avoid bring data files in the repository -- it will bloat the codebase, making it harder to distribute.
    - If it is a picture, upstream that to gist or other "storage" repos and use an URL for it.
    - If it is some configuration file or data file, using script to re-generate it (if easy) or we distribute that on ["Releases"](https://github.com/ise-uiuc/nnsmith/releases) (if large).
