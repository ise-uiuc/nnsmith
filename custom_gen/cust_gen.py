import logging
import os
import random
import time
import traceback
import json

import torch.onnx
from torch import nn

import hydra
from omegaconf import DictConfig

from nnsmith.abstract.extension import activate_ext
from nnsmith.backends.factory import BackendFactory
from nnsmith.graph_gen import SymbolicGen, model_gen, viz
from nnsmith.logging import MGEN_LOG
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import hijack_patch_requires, mkdir, op_filter
from models import ModelCust
from models.torch import TorchModelExportable

import onnx
import onnx.checker
import onnx.helper
import torch
import torch.onnx
from onnx.external_data_helper import load_external_data_for_model
from onnx.tools import update_model_dims

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import SymbolNet
from nnsmith.util import register_seed_setter

def e2o(model):
    print(model)
    model = model.torch_model
    print(model.parameters())
    print(len(list(model.parameters())))
    shape_of_first_layer = list(model.parameters())[0].shape #shape_of_first_layer
    N,C = shape_of_first_layer[:2]
    dummy_input = torch.Tensor(N,C)
    dummy_input = dummy_input[...,:, None,None] #adding the None for height and weight
    torch.onnx.export(model, dummy_input, "savehere.onnx")
@hydra.main(version_base=None, config_path="../nnsmith/config", config_name="main")
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    mgen_cfg = cfg["mgen"]

    seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]

    MGEN_LOG.info(f"Using seed {seed}")

    # TODO(@ganler): skip operators outside of model gen with `cfg[exclude]`
    results = []
    root_path = mgen_cfg['save']

    # n_nodes = 5
    # seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]
    # # mgen_cfg['max_nodes'] = n_nodes
    # mgen_cfg["save"] = root_path + f"/{n_nodes}_{seed}_pt"
    # result = {"name": mgen_cfg['save'], "error": 0, "error_des": {}, "mad": 0, "ml1": 0, "ml2": 0}

    model_cfg = cfg["model"]
    ModelType = ModelCust.init(model_cfg["type"], backend_target=cfg["backend"]["target"])
    ModelType.add_seed_setter()

    if cfg["backend"]["type"] is not None:
        factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
        )
    else:
        factory = None

    # GENERATION
    opset = auto_opset(ModelType, factory, vulops=mgen_cfg["vulops"])
    opset = op_filter(opset, mgen_cfg["include"], mgen_cfg["exclude"])
    hijack_patch_requires(mgen_cfg["patch_requires"])
    activate_ext(opset=opset, factory=factory)

    tgen_begin = time.time()
    gen = model_gen(
        opset=opset,
        method=mgen_cfg["method"],
        seed=seed,
        max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
        max_nodes=mgen_cfg["max_nodes"],
        timeout_ms=mgen_cfg["timeout_ms"],
        rank_choices=mgen_cfg["rank_choices"],
        dtype_choices=mgen_cfg["dtype_choices"],
    )
    tgen = time.time() - tgen_begin

    if isinstance(gen, SymbolicGen):
        MGEN_LOG.info(
            f"{len(gen.last_solution)} symbols and {len(gen.solver.assertions())} constraints."
        )

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug("solution:" + ", ".join(map(str, gen.last_solution)))

    # MATERIALIZATION
    tmat_begin = time.time()
    ir = gen.make_concrete()

    MGEN_LOG.info(
        f"Generated DNN has {ir.n_var()} variables and {ir.n_compute_inst()} operators."
    )


    mkdir(mgen_cfg["save"])
    if cfg["debug"]["viz"]:
        fmt = cfg["debug"]["viz_fmt"].replace(".", "")
        viz(ir, os.path.join(mgen_cfg["save"], f"graph.{fmt}"))

    model = ModelType.from_gir(ir)
    
    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()
    tmat = time.time() - tmat_begin

    tsave_begin = time.time()
    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=mgen_cfg["save"])
    print("check if this is true")
    print(type(model))
    print(isinstance(model, Model))
    #if isinstance(model, TorchModelExportable): #check if specialically torch
    if(isinstance(model, Model)):
        e2o(model)
    tsave = time.time() - tsave_begin

    MGEN_LOG.info(
        f"Time:  @Generation: {tgen:.2f}s  @Materialization: {tmat:.2f}s  @Save: {tsave:.2f}s"
    )
    # results.append(result)

    # with open('./results.json', "a") as f:
    #     for i in results:
    #         f.write(json.dumps(i))


if __name__ == "__main__":
    main()
