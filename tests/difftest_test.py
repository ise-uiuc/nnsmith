from onnx.onnx_ml_pb2 import ModelProto
from nnsmith.difftest import *
import tvm
from tvm import relay
import wget
import os
import numpy as np
from onnxsim import simplify
from tvm.contrib.target.onnx import to_onnx
import tempfile


def dump_model_input(
    model: ModelProto, model_root: Path, model_name: str, num_inputs=2):
    model_path = model_root / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(model_path/'model.onnx'))
    inp_spec = DiffTestBackend.analyze_onnx_io(model)
    for i in range(num_inputs):
        inp = {}
        for name, shape in inp_spec[0].items():
            inp[name] = np.random.rand(*shape.shape).astype(shape.dtype)
        pickle.dump(inp, (model_path/f'input.{i}.pkl').open("wb"))

################# construct models #################
model_root = Path('./tmp/model_input')
output_dir = Path('./tmp/output')
model_root.mkdir(exist_ok=True, parents=True)
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
dump_model_input(model, model_root, 'm0')


filename = 'mobilenetv2.onnx'
if not os.path.exists('mobilenetv2.onnx'):
    filename = wget.download(
        'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx', out='mobilenetv2.onnx')
model, check = simplify(DiffTestBackend.get_onnx_proto(
    filename), input_shapes={'input': [1, 3, 224, 224]})
dump_model_input(model, model_root, 'm1')


difftest(str(model_root), [TVMExecutor(), ORTExecutor()], str(output_dir))