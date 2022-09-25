import pytest

from nnsmith.abstract.dtype import DType
from nnsmith.backends import BackendFactory
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.materialize import Model, Schedule, TestCase
from nnsmith.narrow_spec import auto_opconfig, auto_opset

TestCase.__test__ = False  # supress PyTest warning


def test_narrow_spec_cache_make_and_reload():
    factory = BackendFactory.init("iree", target="cpu", optmax=True)
    ModelType = Model.init("tensorflow")
    opset_lhs = auto_opconfig(ModelType, factory)
    assert opset_lhs, "Should not be empty... Something must go wrong."
    opset_rhs = auto_opconfig(ModelType, factory)
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
    d = tmp_path / "test_iree"
    d.mkdir()

    targets = ["cpu"]

    ModelType = Model.init("tensorflow")
    for target in targets:
        factory = BackendFactory.init("iree", target=target, optmax=False)

        gen = random_model_gen(
            opset=auto_opset(ModelType, factory),
            init_rank=4,
            seed=23132,
            max_nodes=4,
        )  # One op should not be easily wrong... I guess.

        fixed_graph, concrete_abstensors = concretize_graph(
            gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
        )

        schedule = Schedule.init(fixed_graph, concrete_abstensors)

        model = ModelType.from_schedule(schedule)

        oracle = model.make_oracle()

        testcase = TestCase(model, oracle)
        assert factory.verify_testcase(testcase) is None
        testcase.dump(root_folder=d)
