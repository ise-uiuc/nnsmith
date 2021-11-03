# To install tvm with pip:
# pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels

from nnsmith.backends import DiffTestBackend
import os
# TODO: Ensure XLA is enabled
os.environ['TF_XLA_FLAGS']="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" # Enable XLA JIT

import tensorflow as tf
from onnx_tf.backend import prepare


class XLAExecutor(DiffTestBackend):
    def __init__(self, device='cpu'):
        self.device = device

    def predict(self, model, inputs):
        onnx_model = self.get_onnx_proto(model)
        tf_rep = prepare(onnx_model)  # prepare tf representation

        inp_spec, out_names = self.analyze_onnx_io(onnx_model)
        shape_dict = {name: inp_spec[name].shape for name in inp_spec}
        for name in shape_dict:
            if shape_dict[name][0] == -1:  # Freeze batch size
                shape_dict[name][0] = 1
                # (JK) I think this should not occur after the concretization?
                print("Freezing batch size to 1 for {}".format(name))

        # FIXME: Enable multiple outputs
        assert len(out_names) == 1, "Only support single output at this moment"

        executor = tf_rep.run

        outputs = executor(
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
