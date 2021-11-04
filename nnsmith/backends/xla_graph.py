import os
# TODO: Ensure XLA is enabled
os.environ['TF_XLA_FLAGS']="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" # Enable XLA JIT

# See https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md
from onnx_tf.backend import prepare, supports_device
from nnsmith.backends import DiffTestBackend


class XLAExecutor(DiffTestBackend):
    def __init__(self, device: str = 'CPU'):
        """
        Args:
            device (str, optional): 'CPU' or 'CUDA'. Defaults to 'CPU'.
        """
        self.device = device
        assert supports_device(self.device), "Device {} not supported by ONNX-TF".format(self.device)

    def predict(self, model, inputs):
        onnx_model = self.get_onnx_proto(model)
        # prepare tf representation
        tf_rep = prepare(onnx_model, device=self.device)

        inp_spec, out_names = self.analyze_onnx_io(onnx_model)
        shape_dict = {name: inp_spec[name].shape for name in inp_spec}
        for name in shape_dict:
            if shape_dict[name][0] == -1:  # Freeze batch size
                shape_dict[name][0] = 1
                # (JK) I think this should not occur after the concretization?
                print("Freezing batch size to 1 for {}".format(name))

        # FIXME: Enable multiple outputs
        assert len(out_names) == 1, "Only support single output at this moment"

        outputs = tf_rep.run(
            {iname: inputs[iname].astype(inp_spec[iname].dtype) for iname in inputs})
        assert len(outputs) == 1
        output = outputs[0]

        return {out_names[0]: output}


if __name__ == '__main__':
    import wget
    import os
    import numpy as np
    from onnxsim import simplify

    filename = 'mobilenetv2.onnx'
    if not os.path.exists('mobilenetv2.onnx'):
        filename = wget.download(
            'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx', out='mobilenetv2.onnx')
    backend = XLAExecutor()
    sim_model, check = simplify(DiffTestBackend.get_onnx_proto(
        filename), input_shapes={'input': [1, 3, 224, 224]})
    backend.predict(sim_model, {'input': np.zeros((1, 3, 224, 224))})
