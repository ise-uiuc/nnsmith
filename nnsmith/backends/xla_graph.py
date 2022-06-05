import os
from collections import Counter
# TODO: Ensure XLA is enabled
# Enable XLA JIT
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# See https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md
from onnx_tf.backend import prepare, supports_device
from nnsmith.backends import BackendFactory


class XLAExecutor(BackendFactory):
    def __init__(self, device: str = 'CPU'):
        """
        Args:
            device (str, optional): 'CPU' or 'CUDA'. Defaults to 'CPU'.
        """
        super().__init__()
        self.device = device

    def load_model(self, model):
        if self.cache_hit_or_install(model):
            return
        # Fix fork error
        assert supports_device(
            self.device), "Device {} not supported by ONNX-TF".format(self.device)
        onnx_model = self.get_onnx_proto(model)
        # prepare tf representation
        tf_rep = prepare(onnx_model, device=self.device)

        self.inp_spec, self.out_names = self.analyze_onnx_io(onnx_model)
        # TODO(JK): decouple concretization and this function. Ideally, we would
        # do so for every backend.
        # shape_dict = {name: inp_spec[name].shape for name in inp_spec}
        # for name in shape_dict:
        #     if shape_dict[name][0] == -1:  # Freeze batch size
        #         shape_dict[name][0] = 1
        #         print("Freezing batch size to 1 for {}".format(name))

        self.cvtd_model = tf_rep

    def predict(self, model, inputs, **kwargs):
        self.load_model(model)
        tf_rep = self.cvtd_model
        outputs = tf_rep.run(
            {iname: inputs[iname].astype(self.inp_spec[iname].dtype) for iname in inputs})
        assert Counter(self.out_names) == Counter(
            outputs._fields), "Output names don't match, analyze_onnx_io gives {} but xla returns {}".format(
                self.out_names, outputs._fields)
        return {oname: outputs[oname] for oname in self.out_names}


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
    sim_model, check = simplify(BackendFactory.get_onnx_proto(
        filename), input_shapes={'input': [1, 3, 224, 224]})
    output = backend.predict(
        sim_model, {'input': np.zeros((1, 3, 224, 224))})['output']
    assert output.shape == (1, 1000), "{} != {}".format(
        output.shape, (1, 1000))
    assert output[0, 233] - (-1.34753) < 1e-3
