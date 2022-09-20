import os
import random
import warnings
from io import BytesIO
from os import PathLike
from typing import Dict, Tuple, Union

import onnx
import onnx.checker
import onnx.helper
import torch
import torch.onnx
from onnx.external_data_helper import load_external_data_for_model
from onnx.tools import update_model_dims

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsTensor
from nnsmith.macro import onnx2external_data_dir
from nnsmith.materialize import Schedule
from nnsmith.materialize.torch import SymbolNet, TorchModel


def create_deadcode_onnx(onnx_model, name_mask) -> onnx.ModelProto:
    graph_def = onnx.helper.make_graph(
        nodes=onnx_model.graph.node,  # nodes
        name=onnx_model.graph.name,  # name
        inputs=onnx_model.graph.input,  # inputs
        outputs=[o for o in onnx_model.graph.output if o.name in name_mask],  # outputs
        initializer=onnx_model.graph.initializer,
    )

    model_def = onnx.helper.make_model(
        graph=graph_def, producer_name="nnsmith.deadcode"
    )

    onnx.checker.check_model(model_def, full_check=True)
    return model_def


def torch2onnx(
    model: SymbolNet,
    exportable: Union[str, BytesIO],
    verbose=False,
    dummy_inputs=None,
    do_constant_folding=False,
) -> None:
    """Convert PyTorch model to ONNX format."""
    proxy_enabled = model.proxy_enabled
    if proxy_enabled:
        model.disable_proxy_grad()

    # Dummy inputs
    if dummy_inputs is None:
        dummy_inputs = [
            torch.ones(size=svar.shape).uniform_(1, 2).to(dtype=svar.dtype.torch())
            for _, svar in model.input_like.items()
        ]

    input_names = list(model.input_like.keys())

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter(
                "default" if verbose else "ignore", category=torch.jit.TracerWarning
            )
            warnings.simplefilter(
                "default" if verbose else "ignore", category=UserWarning, append=True
            )
            model.eval()
            torch.onnx.export(
                model,
                tuple(dummy_inputs),
                exportable,
                input_names=input_names,
                output_names=list(model.output_like.keys()),
                verbose=verbose,
                do_constant_folding=do_constant_folding,
                opset_version=14,
            )

    if proxy_enabled:  # Re-enable proxy grad
        model.enable_proxy_grad()


def dtype_from_onnx(onnx_dtype: onnx.TensorProto.DataType) -> DType:
    """Return dtype from ONNX data type."""
    return {
        onnx.TensorProto.DataType.FLOAT: DType.float32,
        onnx.TensorProto.DataType.DOUBLE: DType.float64,
        onnx.TensorProto.DataType.INT8: DType.int8,
        onnx.TensorProto.DataType.INT16: DType.int16,
        onnx.TensorProto.DataType.INT32: DType.int32,
        onnx.TensorProto.DataType.INT64: DType.int64,
        onnx.TensorProto.DataType.BOOL: DType.bool,
        onnx.TensorProto.DataType.COMPLEX64: DType.complex64,
        onnx.TensorProto.DataType.COMPLEX128: DType.complex128,
        onnx.TensorProto.DataType.FLOAT16: DType.float16,
        # onnx.TensorProto.DataType.UINT8: DType.uint8, // unsigned type not supported
        # onnx.TensorProto.DataType.UINT16: DType.uint16,
        # onnx.TensorProto.DataType.UINT32: DType.uint32,
        # onnx.TensorProto.DataType.UINT64: DType.uint64,
        # onnx.TensorProto.DataType.STRING: DType.string, // strings not supported
        # onnx.TensorProto.DataType.BFLOAT16: DType.bfloat16, // bfloat16 not supported
    }[onnx_dtype]


def analyze_onnx_io(
    model: onnx.ModelProto,
) -> Tuple[Dict[str, AbsTensor], Dict[str, AbsTensor]]:
    """Return input and output types in dictionaries."""
    input_types = {}
    output_types = {}
    for input in model.graph.input:
        name = input.name
        dtype = dtype_from_onnx(input.type.tensor_type.elem_type)
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        # assert all shapes are positive integers
        assert all(
            isinstance(dim, int) and dim > 0 for dim in shape
        ), f"Fixed shape needed, but got {shape} for input {name}"
        input_types[name] = AbsTensor(shape=shape, dtype=dtype)
    for output in model.graph.output:
        name = output.name
        dtype = dtype_from_onnx(output.type.tensor_type.elem_type)
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        assert all(
            isinstance(dim, int) and dim > 0 for dim in shape
        ), f"Fixed shape needed, but got {shape} for output {name}"
        output_types[name] = AbsTensor(shape=shape, dtype=dtype)
    return input_types, output_types


def get_onnx_proto(model: Union[onnx.ModelProto, str]) -> onnx.ModelProto:
    if isinstance(model, onnx.ModelProto):
        return model
    else:
        external_data_dir = onnx2external_data_dir(model)
        if os.path.exists(external_data_dir):
            onnx_model = onnx.load(model, load_external_data=False)
            load_external_data_for_model(onnx_model, external_data_dir)
        else:
            onnx_model = onnx.load(model)
        return onnx_model


class ONNXModel(TorchModel):
    def __init__(self, with_torch=True):
        """Initialize a ONNXModel.

        Args:
            with_torch (bool, optional): Whether to load/dump torch related files. Defaults to True.
        """
        super().__init__()
        self.with_torch = with_torch
        self.masked_output_like = None
        self.full_output_like = None
        self.full_input_like = None
        self.onnx_model = None

    def _mask_outputs(self) -> Dict[str, AbsTensor]:
        assert (
            self.torch_model is not None
        ), "Create a concrete model before masking outputs."

        # random mask that has mask_n True, otherwise False.
        after_mask = {
            k: v for k, v in self.full_output_like.items() if random.random() < 0.5
        }
        if len(after_mask) == 0:
            # at least one output must be returned.
            only_key = random.sample(self.full_output_like.keys(), 1)[0]
            after_mask = {only_key: self.full_output_like[only_key]}

        return after_mask

    @staticmethod
    def _dce_prob() -> float:  # \in [0, 1]
        dce_prob = 0.0
        dce_env = os.getenv("NNSMITH_ONNX_DCE")
        if dce_env is not None:
            if not dce_env.replace(".", "", 1).isdigit() or not 0 < float(dce_env) < 1:
                raise ValueError(f"NNSMITH_ONNX_DCE must be [0, 1], but got {dce_env}")
            dce_prob = float(dce_env)
        return dce_prob

    @classmethod
    def from_schedule(cls, schedule: Schedule, **kwargs) -> "ONNXModel":
        ret = cls()  # ONNXModel
        ret.torch_model = SymbolNet(schedule, **kwargs)

        ret.full_input_like = ret.torch_model.input_like
        ret.full_output_like = ret.torch_model.output_like
        ret.masked_output_like = ret.full_output_like

        if random.random() < cls._dce_prob():
            ret.masked_output_like = ret._mask_outputs()

        return ret

    def refine_weights(self) -> None:
        TorchModel.refine_weights(self)
        # weights are set. let's save the model.
        self.onnx_model = self.get_onnx_from_torch()
        if set(self.masked_output_like.keys()) != set(self.full_output_like):
            self.onnx_model = create_deadcode_onnx(
                self.onnx_model, self.masked_output_like.keys()
            )

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return self.full_input_like

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return self.masked_output_like

    def dump(self, path: PathLike) -> None:
        if self.with_torch:
            TorchModel.dump(
                self, path.replace(self.name_suffix(), TorchModel.name_suffix())
            )
            if self.onnx_model is None:
                self.onnx_model = self.get_onnx_from_torch()

        onnx.checker.check_model(self.onnx_model, full_check=True)
        onnx.save(self.onnx_model, path)

    @classmethod
    def load(cls, path: PathLike) -> "ONNXModel":
        ret = cls()
        ret.onnx_model = onnx.load(path)

        torch_path = path.replace(cls.name_suffix(), TorchModel.name_suffix())

        ret.with_torch = False
        full_input_like, full_output_like = analyze_onnx_io(ret.onnx_model)
        ret.full_input_like = full_input_like
        ret.full_output_like = full_output_like
        ret.masked_output_like = ret.full_output_like

        if os.path.exists(torch_path):
            ret.with_torch = True
            ret.torch_model = TorchModel.load(torch_path)
            ret.full_input_like = ret.torch_model.input_like
            ret.full_output_like = ret.torch_model.output_like

        return ret

    @property
    def native_model(self):
        if self.with_torch and self.onnx_model is None:
            self.onnx_model = self.get_onnx_from_torch()
        return self.onnx_model

    def get_onnx_from_torch(self) -> onnx.ModelProto:
        f = BytesIO()
        torch2onnx(self.torch_model, f)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        # freeze input and output shapes.
        onnx_model = update_model_dims.update_inputs_outputs_dims(
            onnx_model,
            {k: v.shape for k, v in self.torch_model.input_like.items()},
            {k: v.shape for k, v in self.torch_model.output_like.items()},
        )
        onnx.checker.check_model(onnx_model, full_check=True)
        return onnx_model

    @staticmethod
    def name_suffix() -> str:
        return ".onnx"
