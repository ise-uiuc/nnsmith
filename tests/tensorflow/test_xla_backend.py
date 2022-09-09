import pytest
import tensorflow as tf

from nnsmith.backends.xla import XLAFactory
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.materialize import Schedule, TestCase
from nnsmith.materialize.tensorflow import TFModel

TestCase.__test__ = False  # supress PyTest warning


def test_synthesized_tf_model(tmp_path):
    d = tmp_path / "test_xla"
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

    devices = ["cpu"]
    if tf.config.list_logical_devices("GPU"):
        devices.append("gpu")

    for device in devices:
        assert (
            XLAFactory(
                device=device, optmax=False, catch_process_crash=False
            ).verify_testcase(testcase)
            is None
        )
