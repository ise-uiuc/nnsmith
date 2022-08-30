import pickle

import pytest

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import MaxPool2d
from nnsmith.abstract.tensor import AbsTensor


def test_dtype_picklable():
    dtype = DType.from_str("float32")
    dtype2 = pickle.loads(pickle.dumps(dtype))
    assert dtype == dtype2


def test_abstensor_picklable():
    shape = [1, 2, 3]
    dtype = DType.from_str("float32")
    tensor = AbsTensor(shape, dtype)
    tensor2 = pickle.loads(pickle.dumps(tensor))
    assert tensor == tensor2
    assert tensor.shape == tensor2.shape
    assert tensor.dtype == tensor2.dtype


def test_absop_picklable():
    maxpool = MaxPool2d(kernel_h_size=2, kernel_w_size=2, stride=1, padding=0)
    maxpool2 = pickle.loads(pickle.dumps(maxpool))
    # TODO(@ganler): make AbsOp comparable.
    # assert maxpool == maxpool2
