from typing import Any, Callable, Dict
import tensorflow as tf  # type: ignore

from nnsmith.graph_gen import Schedule
from nnsmith.materialize.tensorflow.tfnet import TFNet


class TFModel:
    def __init__(self, schedule: Schedule) -> None:
        self.net = TFNet(schedule)

    def save_tfnet(self, output_dir: str = "saved_tfnet") -> None:
        pass

    @staticmethod
    def load_tfnet(
        saved_dir: str = "saved_tfnet",
    ) -> Callable[[Any], Dict[str, tf.Tensor]]:
        pass
