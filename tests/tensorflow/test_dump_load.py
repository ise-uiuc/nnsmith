import numpy as np
import pytest

from nnsmith.graph_gen import model_gen
from nnsmith.materialize import TestCase
from nnsmith.materialize.tensorflow import TFModelCPU, tf_dict_from_np
from nnsmith.narrow_spec import auto_opset

TestCase.__test__ = False  # supress PyTest warning


def test_onnx_load_dump(tmp_path):
    d = tmp_path / "test_onnx_load_dump"
    d.mkdir()

    gen = model_gen(
        opset=auto_opset(TFModelCPU),
        seed=54341,
        max_nodes=5,
    )

    ir = gen.make_concrete()

    model = TFModelCPU.from_gir(ir)

    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    loaded_testcase = TestCase.load(model_type=type(model), root_folder=d)

    def compare_two_oracle(src, loaded):
        assert len(loaded.input) == len(src.input)
        assert len(loaded.output) == len(src.output)
        for k, v in loaded.input.items():
            assert np.allclose(v, src.input[k], equal_nan=True)
        for k, v in loaded.output.items():
            assert np.allclose(v, src.output[k], equal_nan=True)

    # check oracle
    compare_two_oracle(oracle, loaded_testcase.oracle)

    loaded_model = loaded_testcase.model
    rerun_oracle = loaded_model.make_oracle(tf_dict_from_np(oracle.input))
    compare_two_oracle(oracle, rerun_oracle)
