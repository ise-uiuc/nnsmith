# NOTE: multi-input-output is supported
from nnsmith.backends import BackendFactory

import onnx
import onnxruntime as ort
import numpy as np
import os

OPT_LEVELS = [
    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
]


class ORTFactory(BackendFactory):
    def __init__(self, device='cpu', optmax=True):
        """opt_level ranges from 0 to 3, stands for ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED and ORT_ENABLE_ALL.
        See https://onnxruntime.ai/docs/performance/graph-optimizations.html for detail"""
        self.name = 'ort'
        super().__init__(device, optmax)
        self.opt_level = OPT_LEVELS[-1 if optmax else 0]
        self.providers = ['CPUExecutionProvider']
        if device in ['cuda', 'gpu']:
            self.providers.append('CUDAExecutionProvider')
        elif device != 'cpu':
            raise ValueError(f'Unknown device `{device}`')

    def __repr__(self) -> str:
        return f'ort-{self.device}-O{self.opt_level}'

    def mk_backend(self, model, **kwargs):
        onnx_model = self.get_onnx_proto(model)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = self.opt_level
        # https://github.com/microsoft/onnxruntime/issues/8313
        sess_options.intra_op_num_threads = int(os.getenv('NNSMITH_CORES', 0))

        sess = ort.InferenceSession(
            onnx._serialize(onnx_model), providers=self.providers, sess_options=sess_options)
        _, out_names = self.analyze_onnx_io(onnx_model)

        def closure(inputs):
            res = sess.run(out_names, inputs)
            return {n: r for n, r in zip(out_names, res)}

        return closure

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
            'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx', out=filename)
    factory = ORTFactory()
    model = BackendFactory.get_onnx_proto(filename)
    input_spec, onames = BackendFactory.analyze_onnx_io(model)
    sim_model, check = simplify(
        model, input_shapes={'input_1': [1, 3, 224, 224], 'image_shape': [1, 2]})
    backend = factory.mk_backend(model)
    res = backend({'input_1': np.zeros(
        (1, 3, 224, 224), dtype='float32'), 'image_shape': np.array([[224, 224]], dtype='float32')})
    print(res)
