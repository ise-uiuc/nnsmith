# NOTE: multi-input-output is supported
from nnsmith.backends import DiffTestBackend

import onnx
import onnxruntime as ort
import numpy as np

providers = [
    'CUDAExecutionProvider',
    # 'CPUExecutionProvider',
]


class ORTExecutor(DiffTestBackend):
    def predict(self, model, inputs):
        onnx_model = self.get_onnx_proto(model)
        sess = ort.InferenceSession(
            onnx._serialize(onnx_model), providers=providers)

        _, out_names = self.analyze_onnx_io(onnx_model)
        res = sess.run(out_names, inputs)

        return {n: r for n, r in zip(out_names, res)}


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
    # sim_model, check = simplify(
    #     model, input_shapes={'input_1': [1, 3, 224, 224], 'image_shape': [1, 2]})
    res = backend.predict(model, {'input_1': np.zeros(
        (1, 3, 224, 224), dtype='float32'), 'image_shape': np.array([[224, 224]], dtype='float32')})
    print(res)
