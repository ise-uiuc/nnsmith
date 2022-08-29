import pytest

from nnsmith.graph_gen import random_model_gen
from nnsmith.materialize import TestCase, Schedule
from nnsmith.materialize.onnx import ONNXModel
from nnsmith.backends.tvm import TVMFactory
from nnsmith.graph_gen import random_model_gen, concretize_graph

import tvm

TestCase.__test__ = False  # supress PyTest warning


def test_synthesized_onnx_model(tmp_path):
    d = tmp_path / "test_tvm_onnx"
    d.mkdir()

    # TODO(@ganler): do dtype first.
    gen = random_model_gen(
        opset=ONNXModel.operators(),
        init_rank=4,
        seed=23132,
        max_nodes=1,
    )  # One op should not be easily wrong... I guess.

    fixed_graph, concrete_abstensors = concretize_graph(
        gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
    )

    schedule = Schedule.init(fixed_graph, concrete_abstensors)

    model = ONNXModel.from_schedule(schedule)

    assert model.with_torch

    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    assert (
        TVMFactory(
            device="cpu", opt_options=False, catch_process_crash=False
        ).verify_testcase(testcase)
        is None
    )

    if tvm.cuda(0).exist:
        assert (
            TVMFactory(
                device="cuda", opt_options=False, catch_process_crash=False
            ).verify_testcase(testcase)
            is None
        )
