# FIXME: This file is going to be deprecated or fixed.

from subprocess import check_call
import time

from tqdm import tqdm
import torch
from nnsmith.backends import BackendFactory, gen_one_input_rngs


class TchExecutor(BackendFactory):
    def __init__(self, opt_level=0, dev="cpu"):
        super().__init__()
        self.opt_level = opt_level  # TODO add more opt levels
        assert dev in ["cpu", "cuda"]
        self.dev = dev

    def load_model(self, model, torch_model):
        if self.cache_hit_or_install(model):
            return
        onnx_model = self.get_onnx_proto(model)
        inp_spec, out_names = self.analyze_onnx_io(onnx_model)
        self.inp_spec, self.out_names = inp_spec, out_names
        self.torch_model = torch_model
        self.torch_model.eval()
        self.torch_model.to(self.dev)

    def predict(self, model, inputs, torch_model=None):
        assert (
            torch_model is not None
        ), "Currently we have no ways to convert onnx to torch, so original torch model is needed"
        self.load_model(model, torch_model)
        torch_out = self.torch_model.forward(
            **{k: torch.from_numpy(v).to(self.dev) for k, v in inputs.items()}
        )
        assert len(torch_out) == len(
            self.out_names
        ), "Output number mismatch, {} vs {}".format(
            len(torch_out), len(self.out_names)
        )
        return {
            self.out_names[i]: v.detach().cpu().numpy() for i, v in enumerate(torch_out)
        }


def _unittest():
    from nnsmith.backends.tvm_graph import TVMFactory
    from nnsmith.graph_gen import SymbolNet
    from nnsmith import difftest, input_gen

    def compare(tvm_exe, tch_exe):
        st = time.time()
        net: SymbolNet = torch.load("output.onnx.pt")
        print("torch model loadin time:", time.time() - st)
        onnx_model = tvm_exe.get_onnx_proto("output.onnx")

        # gen input
        inp_spec = tvm_exe.analyze_onnx_io(onnx_model)[0]
        inp = gen_one_input_rngs(inp_spec, rngs=None)

        st = time.time()
        tvm_out = tvm_exe.predict(onnx_model, inp)
        print("tvm run time:", time.time() - st)

        st = time.time()
        tch_out = tch_exe.predict(onnx_model, inp, torch_model=net)
        print("torch run time:", time.time() - st)

        difftest.assert_allclose(tch_out, tvm_out, "torch", "tvm", nan_as_err=False)

    def test():
        tvm = TVMFactory(0, "llvm")
        tch = TchExecutor(0, "cpu")
        for i in tqdm(range(50)):
            check_call("python -u ./nnsmith/graph_gen.py --max_nodes 10", shell=True)
            compare(tvm, tch)

        tvm = TVMFactory(0, "llvm")
        tch = TchExecutor(0, "cuda")
        for i in tqdm(range(50)):
            check_call("python -u ./nnsmith/graph_gen.py --max_nodes 10", shell=True)
            compare(tvm, tch)

    test()


if __name__ == "__main__":
    _unittest()
