import hydra
from omegaconf import DictConfig

from nnsmith.backends import BackendFactory
from nnsmith.materialize import Model
from nnsmith.narrow_spec import load_topset_from_auto_cache


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    backend_cfg = cfg["backend"]
    if backend_cfg is not None:
        factory = BackendFactory.init(
            name=backend_cfg["type"],
            device=backend_cfg["device"],
            optmax=backend_cfg["optmax"],
        )
    else:
        factory = None
    model_type = Model.init(cfg["model"]["type"])
    load_topset_from_auto_cache(model_type, factory)


if __name__ == "__main__":
    main()
