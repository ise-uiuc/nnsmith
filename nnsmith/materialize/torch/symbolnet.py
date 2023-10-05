import logging
import os
import time
import warnings
from typing import Dict, Optional

import torch
import torch.fx as fx
from torch import nn

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsOpBase, ConcreteOp, Input
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import ConstraintCheck, ConstraintError, SanityCheck
from nnsmith.gir import GraphIR
from nnsmith.logging import TORCH_LOG
from nnsmith.materialize.torch.forward import forward_fn
from nnsmith.materialize.torch.numeric import loss_fn, numeric_valid
from nnsmith.materialize.torch.proxy_grad import proxy_fn

__INPUT_FOUND_NAN_MSG__ = "[NaN] in model inputs!"
__INPUT_FOUND_INF_MSG__ = "[Inf] in model inputs!"
__ENABLE_RT_CHECK__ = os.getenv("NNSMITH_RT_CHECK", "0") == "1"


# Probabilistically, sampling at positive domain is beneficial.
def random_tensor(shape, dtype: torch.dtype, margin=4, base=5, use_cuda=False):
    # center: -margin ~ 0 ~ +margin
    dev = torch.device("cuda" if use_cuda else "cpu")
    if base == "center":
        base = -margin / 2
    else:
        assert isinstance(base, int) or isinstance(base, float)

    fp_tensor = base + (torch.rand(shape, device=dev) - 0.5) * margin

    if dtype.is_floating_point:
        return fp_tensor.to(dtype)
    else:
        return torch.round(fp_tensor).to(dtype)


class FxTracing:
    _tracing = False

    def __enter__(self):
        FxTracing._tracing = True

    def __exit__(self, exc_type, exc_value, traceback):
        FxTracing._tracing = False


def skip_on_trace(fn):
    def wrapper(*args, **kwargs):
        if FxTracing._tracing:
            return None
        else:
            return fn(*args, **kwargs)

    return wrapper


@skip_on_trace
@torch.jit.ignore
def debug_numeric(tensor_map):
    with warnings.catch_warnings():  # just shutup.
        warnings.simplefilter("ignore")
        ConstraintCheck.true(
            not any([torch.isinf(op).any() for _, op in tensor_map.items()]),
            __INPUT_FOUND_INF_MSG__,
        )
        ConstraintCheck.true(
            not any([torch.isnan(op).any() for _, op in tensor_map.items()]),
            __INPUT_FOUND_NAN_MSG__,
        )


@skip_on_trace
def check_type(op: AbsOpBase, tensors, is_input=True, msg=""):
    if __ENABLE_RT_CHECK__ and op is not None:
        like = op.input_like if is_input else op.output_like
        ioro = "input" if is_input else "output"

        assert len(like) == len(
            tensors
        ), f"{op}'s {ioro} has {len(like)} abs. input, but got {len(tensors)} real inputs."

        for i, ten in enumerate(tensors):
            ttype = AbsTensor(list(ten.shape), DType.from_torch(ten.dtype))
            assert (
                like[i] == ttype
            ), f"{msg} {ioro} abstract type != concrete {ttype} for {op} {op.input_like} -> {op.output_like}"


@skip_on_trace
@torch.jit.ignore
def debug_io(stmt_idx, itensors, otensors):
    if TORCH_LOG.isEnabledFor(logging.DEBUG):
        TORCH_LOG.debug(f">> statment {stmt_idx}")
        for inp_i, i in enumerate(itensors):
            TORCH_LOG.debug(f"  (shape={i.shape} dtype={i.dtype})")
            TORCH_LOG.debug(f"[inp]@{inp_i} :: {i.min().data:.5f} ~ {i.max().data:.5f}")
        for out_i, o in enumerate(otensors):
            TORCH_LOG.debug(f"  (shape={o.shape} dtype={o.dtype})")
            TORCH_LOG.debug(f"[out]@{out_i} :: {o.min().data:.5f} ~ {o.max().data:.5f}")


class SymbolNet(nn.Module):
    def __init__(
        self,
        ir: GraphIR,
        use_gradient=False,
    ):
        super(SymbolNet, self).__init__()
        # <TorchFunc, <keys -> inputs>, <keys -> outputs>, original op>
        self.instructions = []
        self.n_vulnerable_op = 0

        self.proxy_enabled_ = False

        self._device = None
        self.ir = ir

        for i, inst in enumerate(self.ir.insts):
            if not isinstance(inst.iexpr.op, Input):
                if isinstance(inst.iexpr.op, ConcreteOp):
                    torch_fn, target = forward_fn(inst.iexpr.op)
                else:
                    torch_fn = target = forward_fn(inst.iexpr.op)
                SanityCheck.true(torch_fn is not None, f"Bad impl for {inst.iexpr.op}")
                if isinstance(target, nn.Module):
                    self.add_module(f"m{i}", target)

                if isinstance(target, nn.Parameter):
                    setattr(self, inst.retval(), target)
                else:
                    self.instructions.append(
                        (torch_fn, inst.iexpr.args, inst.retvals(), inst.iexpr.op)
                    )

                    if loss_fn.dispatch(type(inst.iexpr.op)):
                        self.n_vulnerable_op += 1

        # the order follows `input_keys`
        self.input_map = {iname: self.ir.vars[iname] for iname in self.ir.input_var()}
        self.output_map = {oname: self.ir.vars[oname] for oname in self.ir.leaf_var()}

        self.first_run = True

        self.use_gradient = use_gradient
        if use_gradient:
            self.enable_training()
        self.check_intermediate_numeric = False
        self.invalid_found_last = None

    def to(self, device):
        self._device = device
        return super().to(device)

    @property
    def device(self):
        return self._device

    @property
    def input_like(self):
        return self.input_map

    @property
    def output_like(self):
        return self.output_map

    @property
    def proxy_enabled(self):
        return self.proxy_enabled_

    def enable_proxy_grad(self):
        for i, inst in enumerate(self.instructions):
            _, inputs, outputs, op = inst
            if proxy_fn.dispatch(type(op)):  # has proxy
                self.instructions[i] = (proxy_fn(op), inputs, outputs, op)

        self.proxy_enabled_ = True

    def disable_proxy_grad(self):
        for i, inst in enumerate(self.instructions):
            _, inputs, outputs, op = inst
            if proxy_fn.dispatch(type(op)):  # has proxy
                self.instructions[i] = (forward_fn(op), inputs, outputs, op)

        self.proxy_enabled_ = False

    def get_params(self):
        return sum([i["params"] for i in self.optimizer.param_groups], [])

    def _zero_grad(self):
        for p in self.get_params():
            p.grad = None

    def backward(self):
        if self.loss is not None:
            self._zero_grad()
            params = self.get_params()
            loss_name, l = self.loss
            l.backward()

            with torch.no_grad():
                for param in self.parameters():
                    param.copy_(
                        torch.where(
                            param.isnan().logical_or(param.isinf()),
                            random_tensor(
                                shape=param.shape, dtype=param.dtype.torch()
                            ).to(param.device),
                            param,
                        )
                    )

            if TORCH_LOG.isEnabledFor(logging.DEBUG):
                for name, i in self.interm_grad:
                    msg = (
                        f"{i.grad.min()} ~ {i.grad.max()} ~ {i.grad.mean()}"
                        if i.grad is not None
                        else "None"
                    )
                    TORCH_LOG.info(
                        f"Iter {self.n_iter} [{loss_name}] {name} grad: {msg}"
                    )

            nonzero = False
            with torch.no_grad():
                for i, p in enumerate(params):
                    if p.grad is not None and torch.any(p.grad != 0):
                        nonzero = True
                        break  # As long as there's non-zero grad.
            ConstraintCheck.true(
                nonzero, "Gradients are all zero. Cannot make progress."
            )

            torch.nn.utils.clip_grad_norm_(self.to_train, 1e-1)
            self.optimizer.step()

    def training_reset(self):
        self.loss = None
        self.stop_updating_loss = False

    def stop_training(self):
        self.use_gradient = False
        self.loss = None

    def enable_training(self, extra_trainable: Dict[str, torch.Tensor] = {}):
        self.use_gradient = True
        self.to_train = []
        for t in extra_trainable.values():
            self.to_train.append(t)
        for t in self.parameters():
            self.to_train.append(t)
        self.optimizer = torch.optim.Adam(self.to_train, lr=5e-1)
        self.training_reset()

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.to_train, lr=5e-1)

    def get_random_inps(self, **kwargs) -> Dict[str, torch.Tensor]:
        # center: -margin ~ 0 ~ +margin
        inputs = {}
        for key, abs_tensor in self.input_like.items():
            inputs[key] = random_tensor(
                abs_tensor.shape, abs_tensor.dtype.torch(), **kwargs
            )

        return inputs

    def grad_input_gen(
        self,
        init_tensors: Dict[str, torch.Tensor],
        use_cuda=False,
        max_time=None,
        **kwargs,
    ) -> Optional[Dict[str, torch.Tensor]]:
        # TODO: trim the param. max_iter is not used; remove getenv
        if max_time is None:
            max_time = float(os.getenv("NNSMITH_GRAD_TIME", 0.5))

        inputs: Dict[str, torch.Tensor] = {}
        for key, tensor in init_tensors.items():
            if tensor.data.dtype.is_floating_point:
                inputs[key] = torch.nn.Parameter(tensor.data.clone())
            else:
                inputs[key] = tensor.data

        self.enable_training(extra_trainable=inputs)

        last_check_intermediate_numeric = self.check_intermediate_numeric
        self.check_intermediate_numeric = True

        if use_cuda:
            for k in inputs:
                inputs[k] = inputs[k].cuda()
            self.use_cuda()

        sat_inputs = None
        st = time.time()
        self.n_iter = 0
        self.cur_loss_name = None
        while time.time() - st < max_time:
            self.training_reset()
            self.n_iter += 1

            try:
                _ = self.forward_grad(*inputs.values())
                if self.invalid_found_last:  # need_to_train
                    self.backward()
                else:
                    sat_inputs = {k: v.data for k, v in inputs.items()}
                    break
            except ConstraintError as e:
                if __INPUT_FOUND_INF_MSG__ in str(e) or __INPUT_FOUND_NAN_MSG__ in str(
                    e
                ):
                    # flush NaN/Inf in inputs
                    with torch.no_grad():
                        for inp in inputs.values():
                            inp.copy_(
                                torch.where(
                                    inp.isnan().logical_or(inp.isinf()),
                                    random_tensor(
                                        shape=inp.shape, dtype=inp.dtype.torch()
                                    ).to(inp.device),
                                    inp,
                                )
                            )
                    continue
                TORCH_LOG.debug(e)
                break

        self.stop_training()
        if sat_inputs is None:
            TORCH_LOG.debug("[grad] no valid range found!!!")

        self.check_intermediate_numeric = last_check_intermediate_numeric
        return sat_inputs

    def use_cuda(self):
        self.cuda()

    def make_param_map(self) -> Dict[str, torch.Tensor]:
        tensor_map: Dict[str, torch.Tensor] = {}

        for k, v in self._parameters.items():
            # Workaround: https://github.com/ise-uiuc/nnsmith/pull/122
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, (nn.Parameter, fx.Proxy)):
                    tensor_map[k] = attr
                    continue
            tensor_map[k] = v

        return tensor_map

    def forward(self, *args):
        self.differentiable = True

        tensor_map: Dict[str, torch.Tensor] = self.make_param_map()
        for i, key in enumerate(self.input_map.keys()):
            tensor_map[key] = args[i]

        debug_numeric(tensor_map)

        for stmt_idx, (inst, inps, outs, op) in enumerate(self.instructions):
            input_tensors = [tensor_map[idx] for idx in inps]

            check_type(op, input_tensors, is_input=True, msg="input")

            # REAL FORWARD.
            output_tensors = inst(*input_tensors)
            if isinstance(output_tensors, fx.proxy.Proxy):
                # TODO(@ganler, @co1lin): can we do systematic check through the output type?
                if output_tensors.node.target not in [torch.split, torch.chunk]:
                    output_tensors = [output_tensors]
            elif not isinstance(output_tensors, list):
                output_tensors = [output_tensors]

            check_type(op, output_tensors, is_input=False, msg="output")
            for i, out_key in enumerate(outs):
                # put values back to tensor_map.
                tensor_map[out_key] = output_tensors[i]
                # Check differentiability.
                self.differentiable &= output_tensors[i].grad_fn is not None

            debug_io(stmt_idx, input_tensors, output_tensors)

        self.first_run = False
        return tuple(tensor_map[key] for key in self.output_map)

    def forward_grad(self, *args):
        self.differentiable = True

        tensor_map: Dict[str, torch.Tensor] = self.make_param_map()
        for i, key in enumerate(self.input_map.keys()):
            tensor_map[key] = args[i]

        debug_numeric(tensor_map)

        self.invalid_found_last = False

        self.interm_grad = []

        # LOG.
        if TORCH_LOG.isEnabledFor(logging.DEBUG):
            for k, v in tensor_map.items():
                if v.requires_grad:
                    self.interm_grad.append((k, v))

            for i, p in enumerate(self.parameters()):
                if p.requires_grad:
                    self.interm_grad.append((f"p_{i}", p))

        for stmt_idx, (inst, inps, outs, op) in enumerate(self.instructions):
            input_tensors = [tensor_map[idx] for idx in inps]

            check_type(op, input_tensors, is_input=True, msg="input")

            # REAL FORWARD.
            output_tensors = inst(*input_tensors)
            if not isinstance(output_tensors, list):
                output_tensors = [output_tensors]

            check_type(op, output_tensors, is_input=False, msg="output")

            for i, out_key in enumerate(outs):
                # put values back to tensor_map.
                tensor_map[out_key] = output_tensors[i]
                # Check differentiability.
                self.differentiable &= output_tensors[i].grad_fn is not None
                # TODO(@ganler): optimize: unref tensors that are not going to be used anymore.

            debug_io(stmt_idx, input_tensors, output_tensors)

            if TORCH_LOG.isEnabledFor(logging.DEBUG):
                if output_tensors[0].requires_grad:
                    for i in range(len(output_tensors)):
                        output_tensors[i].retain_grad()
                        self.interm_grad.append((f"{op}{i}", output_tensors[i]))

            if self.check_intermediate_numeric or (
                self.use_gradient and not self.stop_updating_loss
            ):
                if loss_fn.dispatch(type(op)) is not None:
                    loss = loss_fn(op)(*input_tensors)
                    if not isinstance(loss, tuple):
                        loss = ("", loss)  # loss suffix, loss
                    vul_op_loss = loss
                else:
                    vul_op_loss = None
                self.invalid_found_last |= not numeric_valid(output_tensors)

                if self.invalid_found_last and (
                    self.use_gradient and not self.stop_updating_loss
                ):
                    if TORCH_LOG.isEnabledFor(logging.DEBUG):
                        for inp_i, inp in enumerate(input_tensors):
                            TORCH_LOG.info(
                                f"[inp]@{inp_i} :: {inp.min().data:.5f} ~ {inp.max().data:.5f}"
                            )

                    ConstraintCheck.true(
                        vul_op_loss is not None,
                        f"{op = } has no `torch_loss` but produces NaN or INF!",
                    )
                    # TODO: some less vulnerable ops (like Mul) may also trigger Inf and will crash the process.
                    # Given its low chance of happening, ignore it for now.
                    loss_suf, l = vul_op_loss
                    msg = f"loss_{loss_suf}: {l.min().data:.3f} ~ {l.max().data:.3f} ~ {torch.sum((l > 0) * l).item()}"
                    TORCH_LOG.debug(f"#{self.n_iter} [NaN/Inf] in output: {op} ~ {msg}")

                    ConstraintCheck.true(
                        torch.all(l > 0),
                        f"`{op}` outputs NaN/INF found non-positive loss!",
                    )
                    loss_name = f"{op}_{loss_suf}"
                    ConstraintCheck.true(self.loss is None, "Multiple loss detected!")
                    self.loss = loss_name, torch.sum((l > 0) * l)
                    if loss_name != self.cur_loss_name:
                        self.reset_optimizer()
                        self.cur_loss_name = loss_name

                    self.stop_updating_loss = True
                    return output_tensors

        self.first_run = False
        return tuple(tensor_map[key] for key in self.output_map)
