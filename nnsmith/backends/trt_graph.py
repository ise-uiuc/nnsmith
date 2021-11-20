from nnsmith.backends import DiffTestBackend

# See https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html
import tensorrt as trt
import onnx
import pycuda.driver as cuda
import numpy as np


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTBackend(DiffTestBackend):
    def predict(self, model, inputs):
        import pycuda.autoinit
        onnx_model = self.get_onnx_proto(model)
        engine = self.build_engine_onnx(onnx_model)
        trt_inputs, trt_outputs, trt_bindings, stream, onames, name2idx = self.allocate_buffers(
            engine)
        context = engine.create_execution_context()

        for iname in inputs:
            np.copyto(trt_inputs[name2idx[iname]].host, inputs[iname].astype(
                trt.nptype(engine.get_binding_dtype(iname))).ravel())

        trt_concrete_outputs = self.do_inference_v2(
            context, bindings=trt_bindings, inputs=trt_inputs, outputs=trt_outputs, stream=stream)

        return {n: v.reshape(engine.get_binding_shape(n)) for n, v in zip(onames, trt_concrete_outputs)}

    @staticmethod
    def build_engine_onnx(model_file):
        onnx_model = DiffTestBackend.get_onnx_proto(model_file)
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

        config.max_workspace_size = 1 * 1 << 30
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        if not parser.parse(onnx._serialize(onnx_model)):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
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
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
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
        [cuda.memcpy_htod_async(inp.device, inp.host, stream)
         for inp in inputs]
        # Run inference.
        context.execute_async_v2(
            bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream)
         for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]


if __name__ == '__main__':
    import wget
    import os
    import numpy as np
    from onnxsim import simplify

    filename = 'mobilenetv2.onnx'
    if not os.path.exists('mobilenetv2.onnx'):
        filename = wget.download(
            'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx', out='mobilenetv2.onnx')
    backend = TRTBackend()
    sim_model, check = simplify(DiffTestBackend.get_onnx_proto(
        filename), input_shapes={'input': [1, 3, 224, 224]})
    output = backend.predict(
        sim_model, {'input': np.zeros((1, 3, 224, 224))})['output']
    assert output.shape == (1, 1000), "{} != {}".format(
        output.shape, (1, 1000))
    assert output[0, 233] - (-1.34753) < 1e-3
