from backends.ort_graph import ORTExecutor
from backends.tvm_graph import TVMExecutor
from backends.xla_graph import XLAExecutor
# from backends.trt_graph import TRTBackend
from typing import List, Union, Dict, Tuple
import onnx
import numpy as np
from numpy import testing

from nnsmith.backends import DiffTestBackend
from nnsmith.error import *

def assert_allclose(obtained: Dict[str, np.ndarray], desired: Dict[str, np.ndarray], obtained_name: str, oracle_name: str):
    try:
        index = -1
        assert set(obtained.keys()) == set(desired.keys())
        index = 0
        for key in obtained:
            testing.assert_allclose(obtained[key], desired[key], rtol=1e-02, atol=1e-05)
            index += 1
    except AssertionError as err:
        print(err)
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor:')

def difftest(
    model: Union[onnx.ModelProto, str], 
    inputs: Dict[str, np.ndarray], 
    backends: Union[List[DiffTestBackend], None] = None):

    if backends is None:
        backends = [ORTExecutor(), TVMExecutor(), XLAExecutor(), TRTBackend()]
    if isinstance(model, str):
        model = onnx.load(model)
    outputs = []
    for backend in backends:
        outputs.append(backend.predict(model, inputs))

    # for i in range(len(outputs)):
    i = 0
    for j in range(i + 1, len(outputs)):
        assert_allclose(outputs[i], outputs[j], backends[i].__class__, backends[j].__class__)

if __name__ == '__main__':
    import wget
    import os
    import numpy as np
    from onnxsim import simplify
    from tvm import relay
    import tvm
    from tvm.contrib.target.onnx import to_onnx

    # 2-input & 2-output static model.
    def get_model():
        x = relay.var("x", shape=(1, 3, 224, 224))
        y = relay.var("y", shape=(1, 2))
        mod = tvm.IRModule.from_expr(relay.Function([x, y], relay.Tuple([x, y])))
        return to_onnx(mod, {}, 'model')

    model = get_model()
    model = DiffTestBackend.get_onnx_proto(model)
    model, check = simplify(
        model, input_shapes={'x': [1, 3, 224, 224], 'y': [1, 2]})

    difftest(model, {'x': np.zeros((1, 3, 224, 224), dtype='float32'), 
        'y': np.array([[1, 2]], dtype='float32')},
        backends=[TVMExecutor(), XLAExecutor()])
    print('test1 passed')


    filename = 'mobilenetv2.onnx'
    if not os.path.exists('mobilenetv2.onnx'):
        filename = wget.download(
            'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx', out='mobilenetv2.onnx')
    sim_model, check = simplify(DiffTestBackend.get_onnx_proto(
        filename), input_shapes={'input': [1, 3, 224, 224]})
    difftest(sim_model, {'input': np.zeros((1, 3, 224, 224), dtype='float32')},
        backends=[TVMExecutor(), XLAExecutor()])
    print('test2 passed')

