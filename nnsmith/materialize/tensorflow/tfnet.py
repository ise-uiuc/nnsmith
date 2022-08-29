from typing import cast, Iterator, Callable, List, Dict
from dataclasses import dataclass, astuple

import tensorflow as tf

from nnsmith.abstract.tensor import AbsTensor  # type: ignore

from nnsmith.graph_gen import Schedule
from nnsmith.abstract.op import AbsOpBase, Input
from nnsmith.materialize.tensorflow.forward import forward_fn
from nnsmith.error import ConstraintCheck, ConstraintError, SanityCheck


@dataclass
class Instr:
    fwd_fn: Callable
    inp_keys: List[int]
    out_keys: List[int]


TFNetOutDict = Dict[str, tf.Tensor]


class TFNet(tf.Module):
    """
    Concrete TensorFlow Network
    It only has minimal methods to be a TF network.
    It only has minimal information "schedule" to do computation.
    """

    def __init__(
        self,
        schedule: Schedule,
        verbose: bool = False,
    ) -> None:
        """Build a TensorFlow model from schedule

        Args:
            schedule (Schedule): minimal information for constructing a concrete graph.
        """
        super().__init__()
        self.verbose = verbose
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
        # end for

    @tf.function
    def __call__(self, *args, **kwargs) -> TFNetOutDict:
        if self.verbose:
            mode = "Running Eagerly" if tf.executing_eagerly() else "Tracing"
            print(f"{mode} with JIT config: {tf.config.optimizer.get_jit()}")

        key2tensor: Dict[int, tf.Tensor] = {}
        if len(args) == len(self.schedule.input_keys):
            for i, key in enumerate(self.schedule.input_keys):
                key2tensor[key] = args[i]
        elif len(kwargs) == len(self.schedule.input_keys):
            for i, key in enumerate(self.schedule.input_keys):
                key2tensor[key] = kwargs[f"i{i}"]
        else:
            raise ValueError("Either user args only or kwargs only")

        for i_instr, instr in enumerate(self.instructions):
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
