from typing import Dict
import tensorflow as tf  # type: ignore
from tensorflow import keras

from nnsmith.graph_gen import Schedule

"""
Concrete TensorFlow Network
It is only used to do forward computation and ...
"""


class TFNet(tf.Module):
    def __init__(self, schedule: Schedule) -> None:
        super().__init__()

    @tf.function
    def __call__(self, *args, **kwargs) -> Dict[str, tf.Tensor]:

        pass
