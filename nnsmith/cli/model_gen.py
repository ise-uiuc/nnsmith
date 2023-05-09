import logging
import os
import random
import time

import hydra
from omegaconf import DictConfig

from nnsmith.abstract.extension import activate_ext
from nnsmith.backends.factory import BackendFactory
from nnsmith.graph_gen import SymbolicGen, model_gen, viz
from nnsmith.logging import MGEN_LOG
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import hijack_patch_requires, mkdir, op_filter


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    mgen_cfg = cfg["mgen"]

    seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]

    MGEN_LOG.info(f"Using seed {seed}")

    # TODO(@ganler): skip operators outside of model gen with `cfg[exclude]`
    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"], backend_target=cfg["backend"]["target"])
    ModelType.add_seed_setter()

    if cfg["backend"]["type"] is not None:
        factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
            parse_name=True,
        )
    else:
        factory = None

    # GENERATION
    opset = auto_opset(
        ModelType,
        factory,
        vulops=mgen_cfg["vulops"],
        grad=mgen_cfg["grad_check"],
    )
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
    model.set_grad_check(mgen_cfg["grad_check"])
    oracle = model.make_oracle()
    tmat = time.time() - tmat_begin

    tsave_begin = time.time()
    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=mgen_cfg["save"])
    tsave = time.time() - tsave_begin

    MGEN_LOG.info(
        f"Time:  @Generation: {tgen:.2f}s  @Materialization: {tmat:.2f}s  @Save: {tsave:.2f}s"
    )


if __name__ == "__main__":
    main()
