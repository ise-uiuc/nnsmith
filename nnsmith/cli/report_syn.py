import os

import hydra
from omegaconf import DictConfig, ListConfig

from nnsmith.backends import BackendFactory
from nnsmith.logging import RENDER_LOG
from nnsmith.materialize import Model, Render


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    RENDER_LOG.warning(
        "The duty of `nnsmith.report_syn` is to produce a BASIC but executable Python script. It may not reproduce the original bug as the report may not use the original seed, input data, and output oracles. If you want to more strictly reproduce the bug, please use `nnsmith.model_exec`."
    )

    cmp_cfg = cfg["cmp"]
    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"], cfg["backend"]["target"])

    if isinstance(model_cfg["path"], ListConfig):
        model_paths = model_cfg["path"]
    else:
        model_paths = [model_cfg["path"]]

    for model_path in model_paths:
        model = ModelType.load(model_path)

        oracle_path = None
        # Check if we can directly use oracle from `oracle.pkl`
        if "auto" == cmp_cfg["oracle"]:
            model_basename = os.path.basename(os.path.normpath(model_path))
            oracle_path = model_path.replace(model_basename, "oracle.pkl")
            if not os.path.exists(oracle_path):
                oracle_path = None
        elif cmp_cfg["oracle"] is not None:
            oracle_path = cmp_cfg["oracle"]

        if not os.path.exists(oracle_path):
            oracle_path = None

        this_fac = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
            parse_name=True,
        )

        render = Render()
        render.emit_model(model)
        render.emit_input(model, oracle_path)
        render.emit_backend(this_fac)

        print("#", "-" * 20)
        print(f"# {model_path}")
        print(render.render())
        print("#", "-" * 20)


if __name__ == "__main__":
    main()
