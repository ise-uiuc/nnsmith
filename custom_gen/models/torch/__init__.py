import pickle
import os
import tempfile
import traceback
import time
import glob
import torch

from nnsmith.materialize.torch import TorchModel

class TorchModelExportable(TorchModel):
    def __init__(self) -> None:
        super().__init__()

    def export_onnx(self, result):
        dummy_inputs = [
            torch.ones(size=svar.shape).uniform_(1, 2).to(dtype=svar.dtype.torch())
            for _, svar in self.input_like.items()
        ]
        input_names = list(self.input_like.keys())
        path = "testout"
        if not os.path.exists(path):
            os.makedirs(path)
        # outFile = "/output" + str(counter)
        # with torch.no_grad():
        with tempfile.TemporaryDirectory(dir=os.getcwd(),) as tmpdir:
            try:
                start = time.time()
                torch.onnx.export(
                    self.torch_model,
                    tuple(dummy_inputs),
                    f"{tmpdir}/output.onnx",
                    opset_version=14
                )
                end   = time.time()
                result['files'] = glob.glob(f"{tmpdir}/*")
                result['time'] = end - start
            except Exception as err:
                err_string = traceback.format_exc()
                result["error_des"] = err_string
                result['files'] = glob.glob(f"{tmpdir}/*")
                if glob.glob(f"{tmpdir}/*"):
                    return result
                result["error"] = 1
                # with open(path + outFile + ".txt", "w") as f:
                    # f.write("output  " + str(counter) + " error message:\n" + str(e) + "\n" + traceback.format_exc())
        return result