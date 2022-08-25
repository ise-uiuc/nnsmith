from abc import ABC, abstractmethod
import dill as pickle
import os
import json
from typing import Dict


import numpy as np


from nnsmith.abstract.tensor import AbsTensor
from nnsmith.graph_gen import Schedule


class Oracle:
    def __init__(self, input, output):
        self.input: Dict[str, np.ndarray] = input
        self.output: Dict[str, np.ndarray] = output

    def __repr__(self) -> str:
        return f"input={self.input}, output={self.output}"

    @staticmethod
    def name() -> str:
        return "oracle.pkl"

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            to_dump = {
                "input": self.input,
                "output": self.output,
            }
            pickle.dump(to_dump, f)

    @staticmethod
    def load(path: str) -> "Oracle":
        with open(path, "rb") as f:
            to_load = pickle.load(f)
            return Oracle(to_load["input"], to_load["output"])


class Model(ABC):
    @abstractmethod
    def input_like(self) -> Dict[str, AbsTensor]:
        pass

    @abstractmethod
    def output_like(self) -> Dict[str, AbsTensor]:
        pass

    @staticmethod
    @abstractmethod
    def from_schedule(self, instructions: Schedule, **kwargs) -> "Model":
        pass

    @staticmethod
    @abstractmethod
    def load(path) -> "Model":
        pass

    @abstractmethod
    def dump(self, path):
        pass

    @staticmethod
    @abstractmethod
    def name_suffix() -> str:
        """Suffix of the model file.

        Returns:
            str: Model suffix such as ".onnx".

        Note:
            Model as a folder can be "".
        """
        pass

    @abstractmethod
    def refine_weights(self) -> None:
        pass

    @staticmethod
    def name_prefix():
        return "model"

    @property
    def type(self) -> str:
        return type(self).__name__


class TestCase:
    def __init__(self, model: Model, oracle: Oracle):
        self.oracle = oracle
        self.model = model

    @staticmethod
    def load(model_type: Model, root_folder: str) -> "TestCase":
        model_path = os.path.join(
            root_folder, model_type.name_prefix() + model_type.name_suffix()
        )
        oracle_path = os.path.join(root_folder, Oracle.name())
        return TestCase(model_type.load(model_path), Oracle.load(oracle_path))

    def dump(self, root_folder: str):
        self.model.dump(
            os.path.join(
                root_folder, self.model.name_prefix() + self.model.name_suffix()
            )
        )
        self.oracle.dump(os.path.join(root_folder, Oracle.name()))


class BugReport(ABC):
    def __init__(
        self,
        test_case: TestCase,
        symptom: str,
        system: str,
        version: str = None,
        trigger_hash: str = None,
        log: str = None,
    ):
        self.test_case = test_case
        assert symptom in ["crash", "inconsistency", "performance"]
        self.symptom = symptom
        self.system = system
        self.version = version
        self.trigger_hash = trigger_hash
        self.log = log

    @property
    def error_msg_name(self):
        return "err.log"

    @abstractmethod
    def load(self, root_folder: str) -> "BugReport":
        pass

    def dump(self, root_folder: str):
        # create folder if not exists
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        # model*
        # oracle.pkl
        self.test_case.dump(root_folder)
        # err.log
        with open(os.path.join(root_folder, self.error_msg_name), "w") as f:
            f.write(self.log)
        # meta.json
        with open(os.path.join(root_folder, "meta.json"), "w") as f:
            json.dump(
                {
                    "system": self.system,
                    "symptom": self.symptom,
                    "version": self.version,
                    "trigger_hash": self.trigger_hash,
                },
                f,
            )
