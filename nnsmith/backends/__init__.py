from typing import List, Union, Dict, Tuple
import os

import pickle
import numpy as np

from nnsmith.util import gen_one_input

from nnsmith.backends.factory import BackendFactory


def mk_factory(name, device='cpu', optmax=True, **kwargs):
    if name == 'ort' or name == 'onnxruntime':
        from nnsmith.backends.ort_graph import ORTFactory
        return ORTFactory(device=device, optmax=optmax)
    elif name == 'tvm':
        from nnsmith.backends.tvm_graph import TVMFactory
        return TVMFactory(device=device, optmax=optmax, executor='graph')
    elif name == 'xla':
        from nnsmith.backends.xla_graph import XLAExecutor
        return XLAExecutor(device='CUDA')
    elif name == 'trt':
        from nnsmith.backends.trt_graph import TRTFactory
        return TRTFactory()
    else:
        raise ValueError(f'unknown backend: {name}')


def gen_one_input_rngs(inp_spec: Union[str, Dict], rngs: Union[str, List[Tuple[float, float]], None], seed=None) -> Dict:
    """
    Parameters
    ----------
    `inp_spec` can be either a string or a dictionary. When it's a string, it's the a path to the ONNX model.

    `rngs` can be
    - a list of tuples (low, high).
    - None, which means no valid range found, this falls back to use low=0, high=1 as a workaroun
    - a string, which is interpreted as a path to a pickled file.
    """
    if rngs is None:
        rngs = [(0, 1)]
    elif isinstance(rngs, str):
        rngs = pickle.load(open(rngs, 'rb'))
    if isinstance(inp_spec, str):  # in this case the inp_spec is a path to a the model proto
        inp_spec = BackendFactory.analyze_onnx_io(
            BackendFactory.get_onnx_proto(inp_spec))[0]
    return gen_one_input(inp_spec, *rngs[np.random.randint(len(rngs))], seed)


if __name__ == '__main__':
    import wget
    filename = 'mobilenetv2.onnx'
    if not os.path.exists('mobilenetv2.onnx'):
        filename = wget.download(
            'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx', out='mobilenetv2.onnx')
    onnx_model = BackendFactory.get_onnx_proto(filename)
    inp_dict, onames = BackendFactory.analyze_onnx_io(onnx_model)
    assert len(inp_dict) == 1
    assert 'input' in inp_dict
    assert inp_dict['input'].shape == [-1, 3, 224, 224]
    assert len(onames) == 1
    assert onames[0] == 'output'
