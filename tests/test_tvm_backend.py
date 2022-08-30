import pytest

import tvm

from nnsmith.abstract.dtype import DType
from nnsmith.materialize import TestCase, Model
from nnsmith.graph_gen import random_model_gen, concretize_graph, make_schedule
from nnsmith.backends import BackendFactory
from nnsmith.narrow_spec import load_topset_from_auto_cache

TestCase.__test__ = False  # supress PyTest warning


def test_synthesized_onnx_model(tmp_path):
    d = tmp_path / "test_tvm_onnx"
    d.mkdir()

    ONNXModel = Model.init("onnx")

    gen = random_model_gen(
        opset=ONNXModel.operators(),
        init_rank=4,
        seed=23132,
        max_nodes=1,
    )  # One op should not be easily wrong... I guess.

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
        BackendFactory.init(
            "tvm", device="cpu", optmax=False, catch_process_crash=False
        ).verify_testcase(testcase)
        is None
    )

    if tvm.cuda(0).exist:
        assert (
            BackendFactory.init(
                "tvm", device="cuda", optmax=False, catch_process_crash=False
            ).verify_testcase(testcase)
            is None
        )


def test_narrow_spec_cache_make_and_reload():
    factory = BackendFactory.init("tvm", device="cpu", optmax=True)
    ONNXModel = Model.init("onnx")
    opset_lhs = load_topset_from_auto_cache(ONNXModel, factory)
    assert opset_lhs, "Should not be empty... Something must go wrong."
    opset_rhs = load_topset_from_auto_cache(ONNXModel, factory)
    assert opset_lhs == opset_rhs

    # Assert types
    assert isinstance(opset_lhs["core.ReLU"].in_dtypes[0][0], DType)

    # Assert Dictionary Type Equality
    assert type(opset_lhs) == type(opset_rhs)
    assert type(opset_lhs["core.ReLU"]) == type(opset_rhs["core.ReLU"])
    assert type(opset_lhs["core.ReLU"].in_dtypes[0][0]) == type(
        opset_rhs["core.ReLU"].in_dtypes[0][0]
    )
