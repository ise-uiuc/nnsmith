import os
import random

import hydra
from omegaconf import DictConfig

from nnsmith.graph_gen import concretize_graph, random_model_gen, viz
from nnsmith.materialize import Model, Schedule, TestCase
from nnsmith.util import mkdir


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    seed = random.getrandbits(32) if cfg["seed"] is None else cfg["seed"]

    print(f"Using seed {seed}")

    # TODO(@ganler): skip operators outside of model gen with `cfg[exclude]`
    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"])
    ModelType.add_seed_setter()

    gen = random_model_gen(
        opset=ModelType.operators(),
        init_rank=model_cfg["init_rank"],
        seed=seed,
        max_nodes=model_cfg["max_nodes"],
        timeout_ms=model_cfg["gen_timeout_ms"],
        verbose=model_cfg["verbose"],
    )
    print(
        f"{len(gen.get_solutions())} symbols and {len(gen.solver.assertions())} constraints."
    )

    if model_cfg["verbose"]:
        print("solution:", ", ".join(map(str, gen.get_solutions())))

    fixed_graph, concrete_abstensors = concretize_graph(
        gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
    )

    schedule = Schedule.init(fixed_graph, concrete_abstensors)

    model = ModelType.from_schedule(schedule)
    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)

    mkdir(cfg["gen"]["output"])
    testcase.dump(root_folder=cfg["gen"]["output"])

    if cfg["debug"]["viz"]:
        G = fixed_graph
        fmt = cfg["debug"]["viz_fmt"].replace(".", "")
        viz(G, os.path.join(cfg["gen"]["output"], f"graph.{fmt}"))


if __name__ == "__main__":
    main()
