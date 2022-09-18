from dataclasses import dataclass
from typing import Callable, Dict, List, cast

import tensorflow as tf

from nnsmith.abstract.op import AbsOpBase, Input
from nnsmith.error import SanityCheck
from nnsmith.logging import TF_LOG
from nnsmith.materialize import Schedule
from nnsmith.materialize.tensorflow.forward import forward_fn


@dataclass
class Instr:
    fwd_fn: Callable
    inp_keys: List[int]
    out_keys: List[int]


class TFNet(tf.Module):
    """
    Concrete TensorFlow Network
    It only has minimal methods to be a TF network.
    It only has minimal information "schedule" to do computation.
    """

    def __init__(
        self,
        schedule: Schedule,
    ) -> None:
        """Build a TensorFlow model from schedule

        Args:
            schedule (Schedule): minimal information for constructing a concrete graph.
        """
        super().__init__()
        self.schedule: Schedule = schedule
        self.mlist: List[Callable] = []
        self.instructions: List[Instr] = []

        for op, inp_keys, out_keys in self.schedule.instructions:
            if not isinstance(op, Input):
                op = cast(AbsOpBase, op)
                fwd_fn = forward_fn(op)
                SanityCheck.true(fwd_fn is not None, f"Bad implementation for {op}")
                if not isinstance(op, tf.Module):
                    self.mlist.append(fwd_fn)  # Add tf.Module to track its parameters
                self.instructions.append(Instr(fwd_fn, inp_keys, out_keys))

    @tf.function
    def __call__(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        return self.__forward(*args, **kwargs)

    @tf.function
    def call_by_dict(self, x: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return self.__forward(**x)

    @tf.function
    def __forward(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        mode = "Running Eagerly" if tf.executing_eagerly() else "Tracing"
        TF_LOG.debug(f"{mode} with JIT config: {tf.config.optimizer.get_jit()}")

        key2tensor: Dict[int, tf.Tensor] = {}
        if len(args) == len(self.schedule.input_keys):
            for i, key in enumerate(self.schedule.input_keys):
                key2tensor[key] = args[i]
        elif len(kwargs) == len(self.schedule.input_keys):
            for i, key in enumerate(self.schedule.input_keys):
                key2tensor[key] = kwargs[f"i{i}"]
        else:
            raise ValueError("Either user args only or kwargs only")

        for instr in self.instructions:
            # get inputs
            inp_tensors = [key2tensor[key] for key in instr.inp_keys]

            # forward
            out_tensors = instr.fwd_fn(
                *inp_tensors
            )  # TODO Colin when it can return a list?
            if not isinstance(out_tensors, list):
                out_tensors = [out_tensors]

            # store outputs
            for i_out, out_key in enumerate(instr.out_keys):
                key2tensor[out_key] = out_tensors[i_out]

        # end for instructions
        out_dict = {
            f"o{i}": key2tensor[key] for i, key in enumerate(self.schedule.leaf_keys)
        }
        return out_dict
