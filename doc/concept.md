# Development Guide of NNSmith

## Bug Report Format

*To be standardarized.*

- **Bug-introducing model**: such as `model.onnx` for ONNX models and a saved-model folder in TensorFlow.
- **Oracle**: `oracle.pkl` contains a dictionary
    - `"input"`: A dictionary of input.
    - `"output"`: A dictionary of output if the results are expected to be compared or `None` if the output contains `NaN` or `Inf` as undefined behaviours.
- **Meta information**: `meta.json` meta information.
    - `"system"`: like `"tvm-gpu"` and `"ort-cpu"`
    - `"version"`: a version string hooked from `${SystemPackage}.__version__`
    - `"symptom"`: `"crash"` or `"inconsistency"` or `"timeout"`
    - `"version_id"` (optional): an identifier of the system's version (e.g., git hash or version strings)

## Abstract Operators (AO)

An abstract operator contains information to construct a materialized operator. We will start with a simplified pooling 2D example.

### `__init__`

The initializer of Abstract Operators (AO) takes a list of symbolic integers. In this way, during model generation, we can construct operators by feeding a certain number of symbolic integers.

```python
class Pool2d(UnaryOpBase):
    def __init__(self, kw, kh, stride, pad):
        # Step 1: Invoke daddy's constructor
        super().__init__()

        # Step 2: Take arguments as attributes
        self.kw, self.kh = kw, kh
        self.stride = stride
        self.pad = pad

        # Step 3: Define desired operator input and output ranks
        # Typing: List[Tuple] where each tuple has some ranks
        # [ input.0(ok_rank.0, ok_rank.1, ...), input.1(...), ... ]
        # Why [(4,)]?
        #  1. Pooling2D only takes one input/output => one tuple;
        #  2. Pooling2D only accepts NCHW tensors   => the only viable dim is 4;
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]
```

### `type_transfer(itensors{.shape, .dtype}) => otensors{.shape, .dtype}`

`type_transfer` is a type inference function to infer the output type (shape and data type) given inputs’ type information.

```python
def type_infer(self, itensors: List[AbsTensor]):
    n, c, h, w = itensors[0].shape
    return [ # List
        AbsTensor(shape=[n, c,
                         ((h - self.kh) + 2 * self.pad) // self.stride,
                         ((w - self.kw) + 2 * self.pad) // self.stride,
                        ], dtype=itensors[0].dtype)]
```

### `requires(itensors{.shape, .dtype}) => [constraints, ...]`

`requires` returns constraints (predicates) that must be satisfied when inserting this operator into a computational graph.

```python
def requires(self, itensors: List[AbsTensor]):
    n, c, h, w = itensors[0].shape
    return [ # List
        self.kh >= 1, self.kw >= 1, self.stride >= 1, self.pad >= 0,
        self.kh <= h + 2 * self.pad,
        self.kw <= w + 2 * self.pad,
    ]
```

### Class members

- Viable data types:
    - `inp_dtypes`: Similar to `self.inp_ranks`, it contains a list of independent and viable input data types.
    - `out_dtypes`

### Varadic operator parameter

Sometimes, we want to define an AO that can take variadic numbers of arguments, e.g., `Padding` can take a padding list which can be of 4 pad sizes (for (H and W) x (left-most and right-most)) for 4D tensors (e.g., NCHW) and 6 pad sizes for 5D tensors.

Therefore, we need to let the model generator know how many arguments / symbolic integers a (padding) operator accepts. To specify this information, we overload the class data member `num_var_param: List[int]` which takes a list of integers, each of which is an acceptable number of arguments. For example, to create a padding operator that accepts 4 ~ 6D tensors.

### Constraining viable ranks of different input tensors

Operators like `Concat` require input tensors to have the same ranks. Therefore, we need to somehow constraint the input ranks for `self.inp_ranks` as by default ranks in `self.inp_ranks` are *independent*.

To do so, we set `self.same_inp_dims = True` in initializer:

```python
def __init__(...):
    super().__init__()
    ...
    self.same_inp_dims = True  # But this is not True for Pool2D and many binary operators.
```
