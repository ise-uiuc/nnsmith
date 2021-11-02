# To install tvm with pip:
# pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels

from nnsmith.backends import DiffTestBackend

import tvm
from tvm import relay


class TVMExecutor(DiffTestBackend):
    def __init__(self, opt_level=4, target="llvm", executor="graph"):
        self.opt_level = opt_level
        self.target = tvm.target.Target(target)
        self.executor = executor

    def get_device(self):
        if self.target.export()['kind'] == 'cuda':
            return tvm.cuda()
        if self.target.export()['kind'] == 'rocm':
            return tvm.rocm()
        return tvm.cpu()

    def predict(self, model, inputs):
        onnx_model = self.get_onnx_proto(model)

        inp_spec, out_names = self.analyze_onnx_io(onnx_model)
        shape_dict = {name: inp_spec[name].shape for name in inp_spec}
        for name in shape_dict:
            if shape_dict[name][0] == -1:  # Freeze batch size
                shape_dict[name][0] = 1
                print("Freezing batch size to 1 for {}".format(name))

        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        mod = relay.transform.InferType()(mod)

        # FIXME: Enable multiple outputs
        assert len(out_names) == 1, "Only support single output at this moment"
        out_shape = tuple(mod['main'].ret_type.shape)

        with tvm.transform.PassContext(opt_level=self.opt_level):
            executor = relay.build_module.create_executor(
                self.executor, mod, self.get_device(), self.target, params
            ).evaluate()
            output = executor(
                **{iname: inputs[iname].astype(inp_spec[iname].dtype) for iname in inputs}).numpy()

        # with tvm.transform.PassContext(opt_level=self.opt_level):
        #     lib = relay.build(mod, self.target, params=params)
        #     m = graph_executor.GraphModule(lib["default"](self.get_device()))
        #     # set inputs
        #     for name in inputs:
        #         m.set_input(name, inputs[name].astype(inp_spec[name].dtype))
        #     # execute
        #     m.run()
        #     # get outputs
        #     output = m.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

        assert out_shape == output.shape, f"{out_shape} != {output.shape}"
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
    backend = TVMExecutor()
    sim_model, check = simplify(DiffTestBackend.get_onnx_proto(
        filename), input_shapes={'input': [1, 3, 224, 224]})
    backend.predict(sim_model, {'input': np.zeros((1, 3, 224, 224))})
