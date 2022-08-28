import pytest

from nnsmith.materialize import TestCase
from nnsmith.materialize.onnx import ONNXModel
from nnsmith.graph_gen import random_model_gen, concretize_graph, make_schedule
from nnsmith.backends.onnxruntime import ORTFactory


TestCase.__test__ = False  # supress PyTest warning


def test_synthesized_onnx_model(tmp_path):
    d = tmp_path / "test_ort_onnx"
    d.mkdir()

    gen = random_model_gen(
        opset=ONNXModel.operators(),
        init_rank=4,
        seed=23132,
        max_nodes=2,
    )

    fixed_graph, concrete_abstensors = concretize_graph(
        gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
    )

    schedule = make_schedule(fixed_graph, concrete_abstensors)

    model = ONNXModel.from_schedule(schedule)

    assert model.with_torch

    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    assert (
        ORTFactory(
            device="cpu", optmax=False, catch_process_crash=False
        ).verify_testcase(testcase)
        is None
    )
