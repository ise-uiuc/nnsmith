import logging
import os
import random

import hydra
from omegaconf import DictConfig

from nnsmith.graph_gen import concretize_graph, random_model_gen, viz
from nnsmith.logging import MGEN_LOG
from nnsmith.materialize import Model, Schedule, TestCase
from nnsmith.narrow_spec import opset_from_auto_cache
from nnsmith.util import mkdir


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    mgen_cfg = cfg["mgen"]

    seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]

    MGEN_LOG.info(f"Using seed {seed}")

    # TODO(@ganler): skip operators outside of model gen with `cfg[exclude]`
    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"])
    ModelType.add_seed_setter()

    gen = random_model_gen(
        opset=opset_from_auto_cache(ModelType),
        init_rank=mgen_cfg["init_rank"],
        seed=seed,
        max_nodes=mgen_cfg["max_nodes"],
        timeout_ms=mgen_cfg["timeout_ms"],
    )
    MGEN_LOG.info(
        f"{len(gen.get_solutions())} symbols and {len(gen.solver.assertions())} constraints."
    )

    if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
        MGEN_LOG.debug("solution:" + ", ".join(map(str, gen.get_solutions())))

    fixed_graph, concrete_abstensors = concretize_graph(
        gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
    )

    mkdir(mgen_cfg["save"])
    if cfg["debug"]["viz"]:
        G = fixed_graph
        fmt = cfg["debug"]["viz_fmt"].replace(".", "")
        viz(G, os.path.join(mgen_cfg["save"], f"graph.{fmt}"))

    schedule = Schedule.init(fixed_graph, concrete_abstensors)

    model = ModelType.from_schedule(schedule)
    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=mgen_cfg["save"])


if __name__ == "__main__":
    main()
