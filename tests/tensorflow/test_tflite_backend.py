import pytest
import tensorflow as tf

from nnsmith.backends.tflite import TFLiteFactory
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.materialize import Schedule, TestCase
from nnsmith.materialize.tensorflow import TFModel

TestCase.__test__ = False  # supress PyTest warning


def test_synthesized_tflite_model(tmp_path):
    d = tmp_path / "test_tflite"
    d.mkdir()

    # TODO(@ganler): do dtype first.
    gen = random_model_gen(
        opset=TFModel.operators(),
        init_rank=4,
        seed=23132,
        max_nodes=4,
    )  # One op should not be easily wrong... I guess.

    fixed_graph, concrete_abstensors = concretize_graph(
        gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
    )

    schedule = Schedule.init(fixed_graph, concrete_abstensors)

    model = TFModel.from_schedule(schedule)

    # model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    assert (
        TFLiteFactory(
            device="cpu", optmax=False, catch_process_crash=False
        ).verify_testcase(testcase)
        is None
    )
