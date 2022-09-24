from dataclasses import dataclass
from typing import List

import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
from multipledispatch import dispatch
from pycuda.driver import DeviceAllocation

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.extension import patch_requires
from nnsmith.abstract.op import AbsOpBase
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.backends import BackendFactory
from nnsmith.materialize.onnx import ONNXModel


@dataclass
class HostDeviceMem:
    host: np.ndarray
    device: DeviceAllocation


class TRTFactory(BackendFactory):
    def __init__(self, target="cuda", optmax=True, **kwargs):
        super().__init__(target, optmax, **kwargs)

        if target != "cuda":
            raise ValueError("TensorRT backend only supports GPU!")

        if optmax is False:
            # TODO(@ganler): support non-optimized TensorRT by using performing
            # inference over a model that marks all nodes as outputs.
            raise ValueError("There is not O0 mode for TensorRT so far.")

    @property
    def system_name(self) -> str:
        return "tensorrt"

    @dispatch(ONNXModel)
    def make_backend(self, model: ONNXModel):
        import pycuda.autoinit

        onnx_model = model.native_model
        engine = self.build_engine_onnx(onnx_model)

        def closure(inputs):
            (
                trt_inputs,
                trt_outputs,
                trt_bindings,
                stream,
                onames,
                name2idx,
            ) = self.allocate_buffers(engine)
            context = engine.create_execution_context()

            for iname in inputs:
                np.copyto(
                    trt_inputs[name2idx[iname]].host,
                    inputs[iname]
                    .astype(trt.nptype(engine.get_binding_dtype(iname)))
                    .ravel(),
                )

            trt_concrete_outputs = self.do_inference_v2(
                context,
                bindings=trt_bindings,
                inputs=trt_inputs,
                outputs=trt_outputs,
                stream=stream,
            )

            return {
                n: v.reshape(engine.get_binding_shape(n))
                for n, v in zip(onames, trt_concrete_outputs)
            }

        return closure

    @staticmethod
    def build_engine_onnx(onnx_model):
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        if not parser.parse(onnx._serialize(onnx_model)):
            error_msg = ""
            for error in range(parser.num_errors):
                error_msg += str(parser.get_error(error))
            raise RuntimeError(error_msg)
        engine_bytes = builder.build_serialized_network(network, config)
        return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            engine_bytes
        )

    @staticmethod
    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        onames = []
        name2idx = {}
        for idx, binding in enumerate(engine):
            name2idx[binding] = idx
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                onames.append(binding)
        return inputs, outputs, bindings, stream, onames, name2idx

    @staticmethod
    def do_inference_v2(context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    @classmethod
    def skip_dtypes(cls) -> List[DType]:
        # TRT will truncate f64 -> f32 and i64 -> i32
        return [DType.float64, DType.int64]


@patch_requires(TRTFactory.system_name, "core.Pool2d")
def RulePool2d(self: AbsOpBase, _: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
    return [nnsmith_lt(nnsmith_mul(self.kernel_h_size, self.kernel_w_size), 10000)]
