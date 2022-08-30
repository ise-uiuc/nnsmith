import pytest

from nnsmith.materialize import TestCase, Schedule
from nnsmith.graph_gen import random_model_gen, concretize_graph
from nnsmith.backends.tflite import TFLiteFactory
from nnsmith.materialize.tensorflow import TFModel

import tensorflow as tf

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
