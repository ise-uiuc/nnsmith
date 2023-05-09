import pytest
import torch

from nnsmith.abstract.dtype import DType
from nnsmith.backends import BackendFactory
from nnsmith.graph_gen import model_gen
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opconfig, auto_opset

TestCase.__test__ = False  # supress PyTest warning


def test_narrow_spec_cache_make_and_reload():
    factory = BackendFactory.init("torchjit", target="cpu", optmax=True)
    ModelType = Model.init("torch")
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


def test_synthesized_model(tmp_path):
    d = tmp_path / "test_torchjit"
    d.mkdir()

    targets = ["cpu"]
    if torch.cuda.is_available():
        targets.append("cuda")

    for target in targets:
        for grad in [True, False]:
            factory = BackendFactory.init("torchjit", target=target, optmax=False)

            ModelType = Model.init("torch", backend_target=target)

            gen = model_gen(
                opset=auto_opset(ModelType, factory, grad=grad),
                seed=23132,
                max_nodes=1,
            )  # One op should not be easily wrong... I guess.

            model = ModelType.from_gir(gen.make_concrete())

            model.set_grad_check(grad)
            oracle = model.make_oracle()

            testcase = TestCase(model, oracle)
            assert factory.verify_testcase(testcase) is None
            testcase.dump(root_folder=d)
