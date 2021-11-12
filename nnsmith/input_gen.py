import numpy as np
import os
from pathlib import Path
from nnsmith.difftest import DiffTestBackend
import pickle

def gen_inputs(model, num_inputs):
    inp_spec = DiffTestBackend.analyze_onnx_io(model)
    inps = []
    for i in range(num_inputs):
        inp = {}
        for name, shape in inp_spec[0].items():
            # random range is [-0.5, 0.5]
            inp[name] = np.random.rand(*shape.shape).astype(shape.dtype) - 0.5
        inps.append(inp)
    return inps


def gen_inputs_for_all(root, num_inputs=2):
    for model in Path(root).glob('model_input/*/*.onnx'):
        print(model)
        inps = gen_inputs(DiffTestBackend.get_onnx_proto(str(model)), num_inputs)
        for i, inp in enumerate(inps):
            model_path = Path(model).parent
            pickle.dump(inp, (model_path/f'input.{i}.pkl').open("wb"))


if __name__ == '__main__':
    import sys
    gen_inputs_for_all(sys.argv[1])
