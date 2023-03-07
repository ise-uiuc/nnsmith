import pickle
import os
import tempfile
import traceback
import time
import glob
import torch
import tf2onnx
import tensorflow as tf

from typing import Callable, Dict, List, Type
from nnsmith.materialize.tensorflow import TFModelCPU, TFModel
from nnsmith.gir import GraphIR


class TFModelExportable(TFModel):
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> tf.device:
        return tf.device(tf.config.list_logical_devices("CPU")[0].name)

    @classmethod
    def from_gir(cls: Type["TFModel"], ir: GraphIR, **kwargs) -> "TFModel":
        return TFModel(ir)

    def export_onnx(self, result):
        with tempfile.TemporaryDirectory(
            dir=os.getcwd(),
        ) as tmpdir:
            try:
                start = time.time()
                tf2onnx.convert.from_keras(
                    self.net, opset=14, output_path=f"{tmpdir}/output.onnx"
                )
                end = time.time()
                result["files"] = glob.glob(f"{tmpdir}/*")
                result["time"] = end - start
            except Exception as err:
                err_string = traceback.format_exc()
                result["error_des"] = err_string
                result["files"] = glob.glob(f"{tmpdir}/*")
                if glob.glob(f"{tmpdir}/*"):
                    return result
                result["error"] = 1
                # with open(path + outFile + ".txt", "w") as f:
                # f.write("output  " + str(counter) + " error message:\n" + str(e) + "\n" + traceback.format_exc())
        return result
