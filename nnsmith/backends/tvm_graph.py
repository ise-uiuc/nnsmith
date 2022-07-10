# To install tvm with pip:
# pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels

from nnsmith.backends import BackendFactory
from nnsmith.error import NNSmithInternalError

import tvm
from tvm import relay


def list_eq(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


class TVMFactory(BackendFactory):
    def __init__(self, device="cpu", optmax=True, executor="graph") -> None:
        self.name = "tvm"
        super().__init__(device, optmax)
        self.opt_level = 4 if optmax else 0
        self.target = tvm.target.Target("llvm" if device == "cpu" else "cuda")
        self.executor_mode = executor

    def __repr__(self) -> str:
        return f"tvm-{self.target}-{self.executor_mode}-O{self.opt_level}"

    def get_device(self):
        if self.target.export()["kind"] == "cuda":
            return tvm.cuda()
        if self.target.export()["kind"] == "rocm":
            return tvm.rocm()
        return tvm.cpu()

    @staticmethod
    def cvt_result(output, out_shape):
        """Pack output tensor(s) into a list"""
        # TODO(jinkun): may not work for nested list / dynamic shape
        assert output is not None, "Output should not be None"
        if isinstance(output, (tvm.runtime.container.ADT, list)):
            output = [r.numpy() for r in output]
            if isinstance(out_shape, relay.TupleType):
                out_shapes = out_shape.fields
            else:
                out_shapes = [out_shape]
            out_shape = [tuple(r.shape) for r in out_shapes]
        elif output is not None:
            output = [output.numpy()]
            out_shape = [tuple(out_shape.shape)]
        else:
            raise NNSmithInternalError(
                f"out_shape is not tuple/list/tensorType but {type(out_shape)}"
            )
        return output, out_shape

    def mk_backend(self, model, **kwargs):
        onnx_model = self.get_onnx_proto(model)

        inp_spec, out_names = self.analyze_onnx_io(onnx_model)
        shape_dict = {name: inp_spec[name].shape for name in inp_spec}
        for name in shape_dict:
            if (
                len(shape_dict[name]) > 0 and shape_dict[name][0] == -1
            ):  # Freeze batch size
                shape_dict[name][0] = 1
                print("Freezing batch size to 1 for {}".format(name))

        mod, params = relay.frontend.from_onnx(
            onnx_model, shape_dict, freeze_params=True
        )
        mod = relay.transform.InferType()(mod)
        oshape = mod["main"].ret_type

        with tvm.transform.PassContext(opt_level=self.opt_level):
            executor = relay.build_module.create_executor(
                self.executor_mode, mod, self.get_device(), self.target, params
            ).evaluate()

        def closure(inputs):
            output = executor(
                **{
                    iname: inputs[iname].astype(inp_spec[iname].dtype)
                    for iname in inputs
                }
            )
            output, out_shape = self.cvt_result(output, oshape)
            output_shape = list(map(lambda x: x.shape, output))
            assert list_eq(
                out_shape, output_shape
            ), f"Shape mismatch between {out_shape} and {output_shape}"
            # TODO(JK): make sure the order matches (not sure how to do so with TVM)
            return dict(zip(out_names, output))

        return closure

    @staticmethod
    def _coverage_install():
        from tvm.contrib import coverage

        return coverage


if __name__ == "__main__":
    import wget
    import os
    import numpy as np
    from onnxsim import simplify
    from tvm.contrib.target.onnx import to_onnx

    # 2-input & 2-output static model.
    def get_model():
        x = relay.var("x", shape=(1, 3, 224, 224))
        y = relay.var("y", shape=(1, 2))
        mod = tvm.IRModule.from_expr(relay.Function([x, y], relay.Tuple([x, y])))
        return to_onnx(mod, {}, "model")

    factory = TVMFactory()
    model = get_model()
    input_spec, onames = BackendFactory.analyze_onnx_io(model)
    sim_model, check = simplify(
        model, input_shapes={"x": [1, 3, 224, 224], "y": [1, 2]}
    )
    backend = factory.mk_backend(model)
    res = backend(
        {
            "x": np.zeros((1, 3, 224, 224), dtype="float32"),
            "y": np.array([[1, 2]], dtype="float32"),
        }
    )
    print("test1 pass")

    import wget
    import os
    import numpy as np
    from onnxsim import simplify

    filename = "mobilenetv2.onnx"
    if not os.path.exists("mobilenetv2.onnx"):
        filename = wget.download(
            "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            out="mobilenetv2.onnx",
        )
    factory = TVMFactory()
    sim_model, check = simplify(
        BackendFactory.get_onnx_proto(filename),
        input_shapes={"input": [1, 3, 224, 224]},
    )
    backend = factory.mk_backend(sim_model)
    output = backend({"input": np.zeros((1, 3, 224, 224))})["output"]
    assert output.shape == (1, 1000), "{} != {}".format(output.shape, (1, 1000))
    assert output[0, 233] - (-1.34753) < 1e-3
    print("test2 pass")
