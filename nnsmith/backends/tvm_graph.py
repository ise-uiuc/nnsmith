# To install tvm with pip:
# pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels

from nnsmith.backends import DiffTestBackend

import tvm
from tvm import relay


def list_eq(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


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

    def cvt_result(self, output, out_shape):
        """Pack output tensor(s) into a list
        """
        # TODO(jinkun): may not work for nested list / dynamic shape
        if isinstance(output, (tvm.runtime.container.ADT, list)):
            output = [r.numpy() for r in output]
            out_shape = [tuple(r.shape) for r in out_shape.fields]
        elif output is not None:
            output = [output.numpy()]
            out_shape = [tuple(out_shape.shape)]
        else:
            assert False, "output is None"
        return output, out_shape

    def load_model(self, model):
        if self.cache_hit_or_install(model):
            return

        onnx_model = self.get_onnx_proto(model)

        inp_spec, out_names = self.analyze_onnx_io(onnx_model)
        self.inp_spec, self.out_names = inp_spec, out_names
        shape_dict = {name: inp_spec[name].shape for name in inp_spec}
        for name in shape_dict:
            if len(shape_dict[name]) > 0 and shape_dict[name][0] == -1:  # Freeze batch size
                shape_dict[name][0] = 1
                print("Freezing batch size to 1 for {}".format(name))

        mod, params = relay.frontend.from_onnx(
            onnx_model, shape_dict, freeze_params=True)
        mod = relay.transform.InferType()(mod)
        self.params = params
        self.mod = mod  # for debugging purposes

        self.out_shape = mod['main'].ret_type

        with tvm.transform.PassContext(opt_level=self.opt_level):
            executor = relay.build_module.create_executor(
                self.executor, mod, self.get_device(), self.target, params
            ).evaluate()
        self.sess = executor

    def predict(self, model, inputs, check_naming=True):
        self.load_model(model)
        with tvm.transform.PassContext(opt_level=self.opt_level):
            output = self.sess(
                **{iname: inputs[iname].astype(self.inp_spec[iname].dtype) for iname in inputs})
            output, out_shape = self.cvt_result(output, self.out_shape)

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
        output_shape = list(map(lambda x: x.shape, output))
        if check_naming:
            assert list_eq(out_shape, output_shape),\
                f"Shape mismatch between {out_shape} and {output_shape}"
        # TODO(JK): make sure the order matches (not sure how to do so with TVM)
        return dict(zip(self.out_names, output))

    @staticmethod
    def _coverage_install():
        from tvm.contrib import coverage
        return coverage


if __name__ == '__main__':
    import wget
    import os
    import numpy as np
    from onnxsim import simplify
    from tvm.contrib.target.onnx import to_onnx

    # 2-input & 2-output static model.
    def get_model():
        x = relay.var("x", shape=(1, 3, 224, 224))
        y = relay.var("y", shape=(1, 2))
        mod = tvm.IRModule.from_expr(
            relay.Function([x, y], relay.Tuple([x, y])))
        return to_onnx(mod, {}, 'model')

    backend = TVMExecutor()
    model = get_model()
    input_spec, onames = DiffTestBackend.analyze_onnx_io(model)
    sim_model, check = simplify(
        model, input_shapes={'x': [1, 3, 224, 224], 'y': [1, 2]})
    res = backend.predict(model, {'x': np.zeros(
        (1, 3, 224, 224), dtype='float32'), 'y': np.array([[1, 2]], dtype='float32')})
    print('test1 pass')

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
    output = backend.predict(
        sim_model, {'input': np.zeros((1, 3, 224, 224))})['output']
    assert output.shape == (1, 1000), "{} != {}".format(
        output.shape, (1, 1000))
    assert output[0, 233] - (-1.34753) < 1e-3
    print('test2 pass')
