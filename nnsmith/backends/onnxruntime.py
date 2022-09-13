import onnx
import onnxruntime as ort
from multipledispatch import dispatch

from nnsmith.backends import BackendFactory
from nnsmith.backends.factory import BackendCallable
from nnsmith.macro import NNSMITH_ORT_INTRA_OP_THREAD
from nnsmith.materialize.onnx import ONNXModel

OPT_LEVELS = [
    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
]


class ORTFactory(BackendFactory):
    def __init__(self, target, optmax, **kwargs):
        """opt_level ranges from 0 to 3, stands for ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED and ORT_ENABLE_ALL.
        See https://onnxruntime.ai/docs/performance/graph-optimizations.html for detail"""
        super().__init__(target, optmax, **kwargs)
        self.opt_level = OPT_LEVELS[-1 if optmax else 0]

        if target == "cuda":
            self.providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]  # ordered by precedence
        elif target == "cpu":
            self.providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(
                f"Unknown target `{target}`. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "onnxruntime"

    @dispatch(ONNXModel)
    def make_backend(
        self,
        model: ONNXModel,
    ) -> BackendCallable:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = self.opt_level
        # https://github.com/microsoft/onnxruntime/issues/8313
        sess_options.intra_op_num_threads = NNSMITH_ORT_INTRA_OP_THREAD

        sess = ort.InferenceSession(
            onnx._serialize(model.native_model),
            providers=self.providers,
            sess_options=sess_options,
        )
        out_names = list(model.output_like.keys())

        def closure(inputs):
            res = sess.run(out_names, inputs)
            return {n: r for n, r in zip(out_names, res)}

        return closure
