"""
Class members of AbsOpBase like in_dtypes, out_dtypes, etc. are just the superset
of the valid domain that do not take the upcoming inference engine's operator
availability into account. For example, older versions of TVM may not support trilu
and TensorRT does not accept a 2D Pool whose kernel is larger than 300. Therefore,
to narrow those specifications, we look at the following methods:
- Identifier: Model and BackendFactory
1. Single-operator specification testing: Iterate over possible operator instances
   of available `data types` (and `ranks`), and try to kick out failing ones (loggings
   will be kept to see if it is just an unimplemented feature or a bug).
2. Constraint extension: for a BackendFactory, it can add more constraints to an operator.
   This is useful for cases like TensorRT where the kernel size is limited to < 300.
- HOWTO:
1. Single-operator potential data types (and ranks) are serializable as a JSON file.
2. Constraint extensions are defined as Python codes. That said, we can just overload the
   operator specifactions in backend (see nnsmith/abstract/extension).
"""


import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from typing import Dict, List, Optional, Type

import z3
from appdirs import user_cache_dir
from omegaconf import OmegaConf

from nnsmith import __version__
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import (
    AbsOpBase,
    AbsTensor,
    Constant,
    Input,
    Placeholder,
    concretize_op,
)
from nnsmith.backends import BackendFactory
from nnsmith.gir import GraphIR, InstExpr
from nnsmith.logging import DTEST_LOG
from nnsmith.materialize import Model, TestCase

NNSMITH_CACHE_DIR = user_cache_dir(f"nnsmith-{__version__}")


def get_cache_name(model_cls: Type[Model], factory: BackendFactory, grad: bool) -> str:
    ret = model_cls.__name__
    if grad:
        ret += "_grad"
    if factory is None:
        ret += "_exportable"
    else:
        ret += f"_{factory.system_name}_{factory.version}_{factory.target}"
    return ret


@dataclass
class OpConfig:
    in_dtypes: List[List[DType]]
    out_dtypes: List[List[DType]]


def _make_single_op_irs(
    op: AbsOpBase, ishapes, available_idtypes
):  # List<Tup<DTypeComb,DTypeComb,GraphIR>>
    """Given a concretized compute op, return an GraphIR for it."""
    ir_list = []
    for idtype_group in available_idtypes:
        ir = GraphIR()
        inputs = []

        for ishape, idtype in zip(ishapes, idtype_group):
            input_op = Placeholder(AbsTensor(ishape, idtype)).input()
            inst = ir.add_inst(InstExpr(op=input_op, args=[]))
            inputs.append(inst.retval())

        this_op = deepcopy(op)
        itensors = [ir.vars[vname] for vname in inputs]
        otensors = this_op.checked_type_transfer(itensors)
        this_op.bind_input_like(itensors)
        this_op.bind_output_like(otensors)

        ir.add_inst(InstExpr(op=this_op, args=inputs))

        ir_list.append((idtype_group, tuple([out.dtype for out in otensors]), ir))
    return ir_list


def infer_topset_from_scratch(
    model_cls: Model,
    factory: Optional[BackendFactory],
    op_types=None,
    grad: bool = False,
) -> Dict[str, OpConfig]:
    if op_types is None:
        op_types = model_cls.operators()

    topset = {}

    n_ops = len(op_types)
    for idx, node_t in enumerate(op_types):
        if node_t is Input or node_t is Constant:
            continue

        DTEST_LOG.info(f"[{idx + 1} / {n_ops}] ===> Trying {node_t}")

        available_idtypes = node_t.in_dtypes

        if available_idtypes and grad:
            available_idtypes = [
                dts for dts in available_idtypes if any(dt.is_float() for dt in dts)
            ]

        if not available_idtypes:
            continue

        op_param_n = node_t.get_num_var_param()
        op_params = [z3.Int("v%s-%s" % (idx, k)) for k in range(op_param_n)]
        op = node_t(*op_params)

        solver = z3.Solver()

        inputs = []
        for i, ranks in enumerate(op.inp_ranks):
            if op.same_inp_dims and inputs:
                rank = inputs[0].ndims
            else:  # FIXME(@ganler): consider rank check over scalar & non-scalar.
                rank = min(ranks)
            shape = AbsTensor(
                shape=[z3.Int("s%s" % (k)) for k in range(rank)],
                dtype=available_idtypes[0][i],
            )
            inputs.append(shape)
            solver.add(*shape.gt_zero())
            solver.add(*[s < 64 for s in shape.shape])

        solver.add(*op.checked_requires(inputs))

        # solve
        assert solver.check() == z3.sat, f"Cannot solve the problem in {node_t}"
        m = solver.model()

        # rewrite
        concrete_op = concretize_op(op, m)
        concre_input_shapes = []
        for inp in inputs:
            shape = []
            for s in inp.shape:
                shape.append(m.eval(s).as_long())
            concre_input_shapes.append(shape)

        single_op_irs = _make_single_op_irs(
            concrete_op, concre_input_shapes, available_idtypes
        )

        # filter out unsupported dtypes by model.
        single_op_irs = [
            sset
            for sset in single_op_irs
            if set(model_cls.skip_dtypes()).isdisjoint(sset[0] + sset[1])
        ]

        if factory:
            single_op_irs = [
                sset
                for sset in single_op_irs
                if set(factory.skip_dtypes()).isdisjoint(sset[0] + sset[1])
            ]

        op_itypes = set()
        op_otypes = set()

        for itypes, otypes, sched in single_op_irs:
            model = model_cls.from_gir(sched)
            model.set_grad_check(grad=grad)
            if factory:
                # Test compilation + simple inference;
                out = factory.make_testcase(model)
                if isinstance(out, TestCase):  # Pass
                    DTEST_LOG.info(
                        f"=====> [Success] at {concrete_op}({itypes}) => {otypes}"
                    )
                    op_itypes.add(itypes)
                    op_otypes.add(otypes)
                else:  # Fail
                    DTEST_LOG.warning(
                        f"=====> [Failure] at {concrete_op}({itypes}) => {otypes}"
                    )
                    DTEST_LOG.debug(f"{out.log}")
            else:  # Test model dumping
                with tempfile.TemporaryDirectory() as tmpdirname:
                    try:
                        model.make_oracle()  # try-run.
                        model_path = os.path.join(
                            tmpdirname, model.name_prefix() + model.name_suffix()
                        )
                        model.dump(model_path)
                        DTEST_LOG.info(
                            f"=====> [Success] at {concrete_op}({itypes}) => {otypes}"
                        )
                        op_itypes.add(itypes)
                        op_otypes.add(otypes)
                    except Exception as e:
                        DTEST_LOG.warning(
                            f"=====> [Failure] at {concrete_op}({itypes}) => {otypes}"
                        )
                        DTEST_LOG.debug(f"{e}")

        if op_itypes:
            topset[op.name()] = OpConfig(
                in_dtypes=list(op_itypes), out_dtypes=list(op_otypes)
            )

    return topset


def load_topset(topset_path) -> Dict[str, OpConfig]:
    conf = OmegaConf.load(topset_path)["topset"]
    ret = {}
    # cvt str -> DType
    for k, v in conf.items():
        ret[k] = OpConfig(
            in_dtypes=[tuple([DType[t] for t in dtypes]) for dtypes in v["in_dtypes"]],
            out_dtypes=[
                tuple([DType[t] for t in dtypes]) for dtypes in v["out_dtypes"]
            ],
        )
    return ret


def dump_topset(topset: Dict[str, OpConfig], path: PathLike):
    OmegaConf.save({"topset": topset}, path)


def auto_opconfig(
    model_cls: Model, factory: Optional[BackendFactory], grad: bool = False
) -> Dict[str, OpConfig]:
    cache_path = os.path.join(
        NNSMITH_CACHE_DIR, get_cache_name(model_cls, factory, grad) + ".yaml"
    )

    # mkdir -p NNSMITH_CACHE_DIR
    if not os.path.exists(NNSMITH_CACHE_DIR):
        os.makedirs(NNSMITH_CACHE_DIR)
    if os.path.exists(cache_path):
        DTEST_LOG.info(f"Loading topset from {cache_path}.")
        DTEST_LOG.info(
            "To regenerate the topset, delete the cache file above and restart."
        )
        return load_topset(cache_path)
    else:
        DTEST_LOG.info(f"Inferring topset from scratch and cache it to {cache_path}.")
        opset = infer_topset_from_scratch(model_cls, factory, grad=grad)
        dump_topset(opset, cache_path)
        return opset


def auto_opset(
    model_cls: Type[Model],
    factory: Optional[BackendFactory] = None,
    vulops: bool = False,
    grad: bool = False,
) -> List[Type[AbsOpBase]]:
    # None means only test model exportation.
    topset_config = auto_opconfig(model_cls, factory, grad)
    opset = []
    for op in model_cls.operators():
        if op.name() not in topset_config or (vulops == False and op.limit_domain):
            continue

        op.in_dtypes = topset_config[op.name()].in_dtypes
        op.out_dtypes = topset_config[op.name()].out_dtypes
        opset.append(op)

    return opset
