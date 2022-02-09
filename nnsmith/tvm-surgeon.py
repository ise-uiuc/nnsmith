from nnsmith.backends import DiffTestBackend
import tvm
from tvm import relay
import numpy as np
from tvm.relay.frontend.common import infer_type
from tvm import relay, topi
from tvm.relay import transform, analysis
from tvm.relay.testing.temp_op_attr import TempOpAttr
import onnx
from nnsmith import onnx_viz
from nnsmith.onnx_viz import select_mod


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def __init__(self, print_mod=True, show_meta_data=False) -> None:
        self.pass_cnt = 0
        self.print_mod = print_mod
        self.show_meta_data = show_meta_data

    def run_before_pass(self, mod, info):
        with tvm.transform.PassContext(instruments=[]):
            global prev_mod
            if self.print_mod:
                # print(mod.astext(show_meta_data=self.show_meta_data))
                # print(infer_type(
                #     mod['main'].body))
                print(relay.transform.InferType()(mod))
            print('>' * 40, f'Running Pass#{self.pass_cnt}:', info)

        self.pass_cnt += 1


def from_onnx(model_path, select=None, target="llvm", viz_out=None, print_pass=False):
    """Load onnx model and convert it to Relay module."""
    onnx_model = DiffTestBackend.get_onnx_proto(model_path)
    inp_spec, onames = DiffTestBackend.analyze_onnx_io(onnx_model)
    shape_dict = {name: inp_spec[name].shape for name in inp_spec}
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape_dict, freeze_params=True)
    mod = select_mod(mod, select)

    # inp = np.random.uniform(size=(1, 3, 48, 48)).astype(np.float32)
    inp = {name: np.random.uniform(size=shape_dict[name]).astype(
        inp_spec[name].dtype) for name in shape_dict}
    print('-' * 50, 'running debug')
    with tvm.transform.PassContext(opt_level=0):
        if viz_out:
            onnx_viz.visualize(mod, viz_out + '.debug.png')
        res1 = relay.build_module.create_executor(
            'debug', mod, target='llvm', device=tvm.cpu()).evaluate()(**inp)
    print('-' * 50, 'running opt')
    with tvm.transform.PassContext(opt_level=4, instruments=[PrintIR()] if print_pass else []):
        mod, _ = relay.optimize(mod, target=target)
        if viz_out:
            onnx_viz.visualize(mod, viz_out)
        res = relay.build_module.create_executor(
            'graph', mod, target=target, device=tvm.cuda() if target == 'cuda' else tvm.cpu()).evaluate()(**inp)
    assert len(res) == len(res1)
    for i in range(len(res)):
        np.testing.assert_allclose(res[i].numpy(), res1[i].numpy())

    return mod, params


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--select', type=str, nargs='+', default=None)
parser.add_argument('--viz_out', type=str)
parser.add_argument('--print_pass', action='store_true')
args = parser.parse_args()
from_onnx(args.model, select=args.select,
          viz_out=args.viz_out, print_pass=args.print_pass)
print('passed')
