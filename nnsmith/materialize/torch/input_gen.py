import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch

from nnsmith.abstract.op import DType
from nnsmith.materialize.torch.symbolnet import SymbolNet, random_tensor


class InputSearchBase(ABC):
    @staticmethod
    def apply_weights(net, weight_sample):
        with torch.no_grad():
            for name, param in net.named_parameters():
                param.copy_(weight_sample[name])

    def __init__(
        self, net: SymbolNet, start_inputs=None, start_weights=None, use_cuda=False
    ):
        self.net = net
        self.start_inputs = start_inputs
        self.start_weights = start_weights
        self.use_cuda = use_cuda

    @abstractmethod
    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, torch.Tensor]:
        pass

    def search(
        self, max_time_ms: int = None, max_sample: int = 1
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        n_try = 0
        sat_inputs = None
        start_time = time.time()

        while (
            max_time_ms is None or time.time() - start_time < max_time_ms / 1000
        ) and n_try < max_sample:
            if self.start_weights is not None and n_try < len(self.start_weights):
                self.apply_weights(self.net, self.start_weights[n_try])
            else:
                weight_sample = {}
                for name, param in self.net.named_parameters():
                    weight_sample[name] = random_tensor(
                        param.shape, dtype=param.dtype, use_cuda=self.use_cuda
                    )
                self.apply_weights(self.net, weight_sample)

            if self.start_inputs is not None and n_try < len(self.start_inputs):
                cur_input = self.start_inputs[n_try]
            else:
                cur_input = self.net.get_random_inps(use_cuda=self.use_cuda)

            res = self.search_one(cur_input, max_time_ms)
            n_try += 1
            if res is not None:
                sat_inputs = res
                break

        return n_try, sat_inputs


class SamplingSearch(InputSearchBase):
    # Think about how people trivially generate inputs.
    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self.net.check_intermediate_numeric = True
            _ = self.net(**start_inp)
            if not self.net.invalid_found_last:
                return start_inp

            return None


class GradSearch(InputSearchBase):
    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, torch.Tensor]:
        timeout_s = None if timeout_ms is None else timeout_ms / 1000
        return self.net.grad_input_gen(
            init_tensors=start_inp, use_cuda=self.use_cuda, max_time=timeout_s
        )


class PracticalHybridSearch(InputSearchBase):
    def __init__(
        self, net: SymbolNet, start_inputs=None, start_weights=None, use_cuda=False
    ):
        super().__init__(net, start_inputs, start_weights, use_cuda)

        self.differentiable = None

        if all([DType.is_float(v.dtype) for _, v in self.net.input_like.items()]):
            diff_test_inp = self.net.get_random_inps(use_cuda=self.use_cuda)
            for _, item in diff_test_inp.items():
                item.requires_grad_()
            self.net.forward(**diff_test_inp)
            self.differentiable = self.net.differentiable
        else:
            self.differentiable = False

    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, torch.Tensor]:
        # if this model is purely differentiable -> GradSearch
        # otherwise                              -> SamplingSearch
        # FIXME: Estimate gradient (e.g., proxy gradient) for non-differentiable inputs.

        if self.differentiable:
            return GradSearch.search_one(self, start_inp, timeout_ms)
        else:
            return SamplingSearch.search_one(self, start_inp, timeout_ms)
