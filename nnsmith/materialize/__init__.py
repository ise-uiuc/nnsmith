import pickle
import os
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Type

import numpy as np

from nnsmith.abstract.op import AbsOpBase
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.graph_gen import Schedule


class Oracle:
    def __init__(self, input, output, provider: str = "unknown") -> None:
        self.input: Dict[str, np.ndarray] = input
        self.output: Dict[str, np.ndarray] = output
        self._provider = provider

    def __repr__(self) -> str:
        return f"input={self.input}, output={self.output}"

    @staticmethod
    def name() -> str:
        return "oracle.pkl"

    @property
    def provider(self) -> str:
        return self._provider

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            to_dump = {
                "input": self.input,
                "output": self.output,
                "provider": self.provider,
            }
            pickle.dump(to_dump, f)

    @staticmethod
    def load(path: str) -> "Oracle":
        with open(path, "rb") as f:
            to_load = pickle.load(f)
            return Oracle(to_load["input"], to_load["output"], to_load["provider"])


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

    @property
    @abstractmethod
    def native_model(self):
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

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        pass

    @staticmethod
    def add_seed_setter() -> None:
        pass


class TestCase:
    def __init__(self, model: Model, oracle: Oracle):
        self.oracle = oracle
        self.model = model

    @staticmethod
    def load(model_type: Model, root_folder: str, allow_no_oracle=False) -> "TestCase":
        model_path = os.path.join(
            root_folder, model_type.name_prefix() + model_type.name_suffix()
        )
        model = model_type.load(model_path)

        assert allow_no_oracle or os.path.exists(
            os.path.join(root_folder, Oracle.name())
        ), "Oracle is not found or auto-generated when allow_no_oracle is True."

        oracle = None
        if os.path.exists(os.path.join(root_folder, Oracle.name())):
            oracle_path = os.path.join(root_folder, Oracle.name())
            oracle = Oracle.load(oracle_path)

        return TestCase(model, oracle)

    def dump(self, root_folder: str):
        self.model.dump(
            os.path.join(
                root_folder, self.model.name_prefix() + self.model.name_suffix()
            )
        )
        self.oracle.dump(os.path.join(root_folder, Oracle.name()))


class Symptom(Enum):
    EXCEPTION = "exception"
    SEGFAULT = "segfault"
    INCONSISTENCY = "inconsistency"
    WORSE_PERF = "worse_perf"


class Stage(Enum):
    COMPILATION = "compilation"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class BugReport(ABC):
    def __init__(
        self,
        testcase: TestCase,
        symptom: Symptom,
        stage: Stage,
        system: str,
        version: str = None,
        trigger_hash: str = None,
        log: str = None,
    ):
        self.testcase = testcase
        self.symptom = symptom
        self.system = system
        self.stage = stage
        self.version = version
        self.trigger_hash = trigger_hash
        self.log = log

    @property
    @classmethod
    def error_msg_name(cls):
        return "err.log"

    def __repr__(self) -> str:
        return f"{self.system} {self.symptom.value} in {self.stage.value}\n{self.log}"

    def dump(self, root_folder: str):
        # create folder if not exists
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        # model*
        # oracle.pkl
        self.testcase.dump(root_folder)
        # err.log
        with open(os.path.join(root_folder, self.error_msg_name), "w") as f:
            f.write(self.log)
        # meta.json
        with open(os.path.join(root_folder, "meta.json"), "w") as f:
            json.dump(
                {
                    "system": self.system,
                    "symptom": self.symptom.value,
                    "stage": self.stage.value,
                    "version": self.version,
                    "trigger_hash": self.trigger_hash,
                },
                f,
            )

    @staticmethod
    def load(model_type, root_folder: str, allow_partial=False) -> "BugReport":

        symptom = None
        stage = None
        version = None
        trigger_hash = None
        system = None

        assert allow_partial or os.path.exists(
            os.path.join(root_folder, "meta.json")
        ), "meta.json must exist or allow_partial is True where the oracle will be automatically generated"

        if os.path.exists(os.path.join(root_folder, "meta.json")):
            with open(os.path.join(root_folder, "meta.json"), "r") as f:
                meta = json.load(f)

            system = meta["system"]
            symptom = Symptom(meta["symptom"])
            stage = Stage(meta["stage"])
            version = meta["version"]
            trigger_hash = meta["trigger_hash"]

        testcase = TestCase.load(model_type, root_folder, allow_partial)

        log = None
        if os.path.exists(os.path.join(root_folder, BugReport.error_msg_name)):
            with open(os.path.join(root_folder, BugReport.error_msg_name), "r") as f:
                log = f.read()

        return BugReport(testcase, symptom, stage, system, version, trigger_hash, log)
