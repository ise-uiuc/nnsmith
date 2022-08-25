import warnings

from multipledispatch import dispatch
import tensorrt as trt
import onnx
import pycuda.driver as cuda
import numpy as np

from nnsmith.backends import BackendFactory
from nnsmith.materialize.onnx import ONNXModel


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTFactory(BackendFactory):
    def __init__(self, device="gpu", optmax=True, **kwargs):
        super().__init__(device, optmax, **kwargs)

        if device != "gpu":
            raise ValueError("TensorRT backend only supports GPU!")

        if optmax is False:
            # TODO(@ganler): support non-optimized TensorRT by using performing
            # inference over a model that marks all nodes as outputs.
            warnings.warn("There is not O0 mode for TensorRT so far.")

    @property
    def system_name(self) -> str:
        return "tensorrt"

    @dispatch(ONNXModel)
    def mk_backend(self, model: ONNXModel):
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
        return builder.build_engine(network, config)

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
