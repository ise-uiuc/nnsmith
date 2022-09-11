import pytest

from nnsmith.abstract.dtype import DType
from nnsmith.backends import BackendFactory
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.materialize import Model, Schedule, TestCase
from nnsmith.materialize.tensorflow import TFModel
from nnsmith.narrow_spec import load_topset_from_auto_cache, opset_from_auto_cache

TestCase.__test__ = False  # supress PyTest warning


def test_narrow_spec_cache_make_and_reload():
    factory = BackendFactory.init("tflite", device="cpu", optmax=True)
    ModelType = Model.init("tensorflow")
    opset_lhs = load_topset_from_auto_cache(ModelType, factory)
    assert opset_lhs, "Should not be empty... Something must go wrong."
    opset_rhs = load_topset_from_auto_cache(ModelType, factory)
    assert opset_lhs == opset_rhs

    # Assert types
    assert isinstance(opset_lhs["core.ReLU"].in_dtypes[0][0], DType)

    # Assert Dictionary Type Equality
    assert type(opset_lhs) == type(opset_rhs)
    assert type(opset_lhs["core.ReLU"]) == type(opset_rhs["core.ReLU"])
    assert type(opset_lhs["core.ReLU"].in_dtypes[0][0]) == type(
        opset_rhs["core.ReLU"].in_dtypes[0][0]
    )


def test_synthesized_tf_model(tmp_path):
    d = tmp_path / "test_tflite"
    d.mkdir()

    ModelType = Model.init("tensorflow")
    factory = BackendFactory.init(
        "tflite", device="cpu", optmax=False, catch_process_crash=False
    )

    # TODO(@ganler): do dtype first.
    gen = random_model_gen(
        opset=opset_from_auto_cache(TFModel, factory),
        init_rank=4,
        seed=23132,
        max_nodes=4,
    )  # One op should not be easily wrong... I guess.

    fixed_graph, concrete_abstensors = concretize_graph(
        gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
    )

    schedule = Schedule.init(fixed_graph, concrete_abstensors)

    model = ModelType.from_schedule(schedule)

    # model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    assert factory.verify_testcase(testcase) is None
