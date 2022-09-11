from enum import Enum, unique

import numpy as np


# TODO(@ganler): add float16 support.
@unique
class DType(Enum):
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    bool = "bool"
    complex64 = "complex64"
    complex128 = "complex128"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        s = super().__str__()
        assert s.startswith("DType."), s
        return s[len("DType.") :]

    def short(self) -> str:
        return {
            DType.float16: "f16",
            DType.float32: "f32",
            DType.float64: "f64",
            DType.int8: "i8",
            DType.int16: "i16",
            DType.int32: "i32",
            DType.int64: "i64",
            DType.complex64: "c64",
            DType.complex128: "c128",
            DType.bool: "bool",
        }[self]

    @staticmethod
    def is_float(dtype):  # Don't use string. Make it well-formed.
        return dtype in [DType.float32, DType.float64]

    @staticmethod
    def from_str(s):
        return {
            "f16": DType.float16,
            "f32": DType.float32,
            "f64": DType.float64,
            "i8": DType.int8,
            "i32": DType.int32,
            "i64": DType.int64,
            "c64": DType.complex64,
            "c128": DType.complex128,
            "float32": DType.float32,
            "float64": DType.float64,
            "int8": DType.int8,
            "int32": DType.int32,
            "int64": DType.int64,
            "complex64": DType.complex64,
            "complex128": DType.complex128,
            "bool": DType.bool,
        }[s]

    def numpy(self):
        return {
            DType.float16: np.float16,
            DType.float32: np.float32,
            DType.float64: np.float64,
            DType.int8: np.int8,
            DType.int16: np.int16,
            DType.int32: np.int32,
            DType.int64: np.int64,
            DType.complex64: np.complex64,
            DType.complex128: np.complex128,
            DType.bool: np.bool_,
        }[self]

    # TODO(@ganler): put "torchization" in a separate file.
    def torch(self) -> "torch.dtype":
        import torch

        return {
            DType.float16: torch.float16,
            DType.float32: torch.float32,
            DType.float64: torch.float64,
            DType.int8: torch.int8,
            DType.int16: torch.int16,
            DType.int32: torch.int32,
            DType.int64: torch.int64,
            DType.complex64: torch.complex64,
            DType.complex128: torch.complex128,
            DType.bool: torch.bool,
        }[self]

    @staticmethod
    def from_torch(dtype) -> "DType":
        import torch

        return {
            torch.float16: DType.float16,
            torch.float32: DType.float32,
            torch.float64: DType.float64,
            torch.int8: DType.int8,
            torch.int16: DType.int16,
            torch.int32: DType.int32,
            torch.int64: DType.int64,
            torch.complex64: DType.complex64,
            torch.complex128: DType.complex128,
            torch.bool: DType.bool,
        }[dtype]

    def tensorflow(self) -> "tf.Dtype":
        import tensorflow as tf

        return {
            DType.float16: tf.float16,
            DType.float32: tf.float32,
            DType.float64: tf.float64,
            DType.int8: tf.int8,
            DType.int16: tf.int16,
            DType.int32: tf.int32,
            DType.int64: tf.int64,
            DType.complex64: tf.complex64,
            DType.complex128: tf.complex128,
            DType.bool: tf.bool,
        }[self]

    @staticmethod
    def from_tensorflow(dtype) -> "DType":
        import tensorflow as tf

        return {
            tf.float16: DType.float16,
            tf.float32: DType.float32,
            tf.float64: DType.float64,
            tf.int8: DType.int8,
            tf.int16: DType.int16,
            tf.int32: DType.int32,
            tf.int64: DType.int64,
            tf.complex64: DType.complex64,
            tf.complex128: DType.complex128,
            tf.bool: DType.bool,
        }[dtype]


DTYPE_ALL = [
    DType.float32,
    DType.float64,
    DType.int32,
    DType.int64,
    DType.bool,
]
DTYPE_NON_BOOLS = [dtype for dtype in DTYPE_ALL if dtype != DType.bool]
DTYPE_FLOATS = [DType.float32, DType.float64]
DTYPE_INTS = [DType.int32, DType.int64]
