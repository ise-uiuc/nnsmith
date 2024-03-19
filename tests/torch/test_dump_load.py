import numpy as np
import pytest
import torch

from nnsmith.graph_gen import model_gen
from nnsmith.materialize import Model, Oracle, TestCase
from nnsmith.narrow_spec import auto_opset

TestCase.__test__ = False  # supress PyTest warning


def compare_two_oracle(src: Oracle, loaded: Oracle):
    assert len(loaded.input) == len(src.input)
    assert len(loaded.output) == len(src.output)
    for k, v in loaded.input.items():
        assert np.allclose(v, src.input[k], equal_nan=True)
    for k, v in loaded.output.items():
        assert np.allclose(v, src.output[k], equal_nan=True)


def test_onnx_load_dump(tmp_path):
    d = tmp_path / "test_onnx_load_dump"
    d.mkdir()

    ONNXModelCPU = Model.init("onnx")

    gen = model_gen(
        opset=auto_opset(ONNXModelCPU),
        seed=54341,
        max_nodes=5,
    )

    model = ONNXModelCPU.from_gir(gen.make_concrete())

    assert model.with_torch

    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    loaded_testcase = TestCase.load(model_type=type(model), root_folder=d)

    # check oracle
    compare_two_oracle(oracle, loaded_testcase.oracle)

    loaded_model = loaded_testcase.model
    loaded_model.sat_inputs = {k: torch.from_numpy(v) for k, v in oracle.input.items()}
    rerun_oracle = loaded_model.make_oracle()
    compare_two_oracle(oracle, rerun_oracle)


def test_bug_report_load_dump(tmp_path):
    d = tmp_path / "test_onnx_load_dump"
    d.mkdir()

    ONNXModelCPU = Model.init("onnx")
    gen = model_gen(
        opset=auto_opset(ONNXModelCPU),
        seed=5341,
        max_nodes=5,
    )

    model = ONNXModelCPU.from_gir(gen.make_concrete())

    assert model.with_torch

    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    loaded_testcase = TestCase.load(model_type=type(model), root_folder=d)

    # check oracle
    compare_two_oracle(oracle, loaded_testcase.oracle)

    loaded_model = loaded_testcase.model
    loaded_model.sat_inputs = {k: torch.from_numpy(v) for k, v in oracle.input.items()}
    rerun_oracle = loaded_model.make_oracle()
    compare_two_oracle(oracle, rerun_oracle)
