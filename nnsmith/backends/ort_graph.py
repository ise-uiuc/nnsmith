# NOTE: multi-input-output is supported
from nnsmith.backends import DiffTestBackend

import onnx
import onnxruntime as ort
import numpy as np

PROVIDERS = [
    'CUDAExecutionProvider',
    # 'CPUExecutionProvider',
]
OPT_LEVELS = [
    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
]


class ORTExecutor(DiffTestBackend):
    def __init__(self, opt_level=3, providers=None):
        """opt_level ranges from 0 to 3, stands for ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED and ORT_ENABLE_ALL. 
        See https://onnxruntime.ai/docs/performance/graph-optimizations.html for detail"""
        super().__init__()
        self._opt_level = OPT_LEVELS[opt_level]
        self.providers = providers or PROVIDERS

    def get_sess_opt(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = self._opt_level
        return sess_options

    def load_model(self, model):
        if self.cache_hit_or_install(model):
            return
        onnx_model = self.get_onnx_proto(model)
        sess_options = self.get_sess_opt()
        self.sess = ort.InferenceSession(
            onnx._serialize(onnx_model), providers=self.providers, sess_options=sess_options)
        _, self.out_names = self.analyze_onnx_io(onnx_model)

    def predict(self, model, inputs, **kwargs):
        self.load_model(model)
        res = self.sess.run(self.out_names, inputs)

        return {n: r for n, r in zip(self.out_names, res)}

    @staticmethod
    def _coverage_install():
        from onnxruntime.tools import coverage
        return coverage


if __name__ == '__main__':
    import wget
    import os
    import numpy as np
    from onnxsim import simplify

    # 2-input & 2-output model.
    filename = 'yolov3-tiny.onnx'
    if not os.path.exists(filename):
        filename = wget.download(
            'https://github.com/hoaquocphan/Tiny_Yolo_v3/raw/master/yolov3-tiny.onnx', out=filename)
    backend = ORTExecutor()
    model = DiffTestBackend.get_onnx_proto(filename)
    input_spec, onames = DiffTestBackend.analyze_onnx_io(model)
    sim_model, check = simplify(
        model, input_shapes={'input_1': [1, 3, 224, 224], 'image_shape': [1, 2]})
    res = backend.predict(model, {'input_1': np.zeros(
        (1, 3, 224, 224), dtype='float32'), 'image_shape': np.array([[224, 224]], dtype='float32')})
    print(res)
