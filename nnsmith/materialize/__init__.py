from typing import Dict, Tuple, List, Any
import pickle
from collections import namedtuple
from dataclasses import dataclass
import os
import json
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
import numpy as np

from nnsmith.abstract.op import AbsOpBase, Input
from nnsmith.abstract.tensor import AbsTensor


Instruction: Tuple[AbsOpBase, List[int], List[int]] = namedtuple(
    "Instruction", ["op", "inputs", "outputs"]
)


@dataclass
class Schedule:
    """Minimal information for constructing a graph."""

    instructions: List[Instruction]
    input_keys: List[int]
    leaf_keys: List[int]
    key2type: Dict[int, AbsTensor]

    @staticmethod
    def init(graph: nx.MultiDiGraph, key2type: Dict[int, AbsTensor]) -> "Schedule":
        # The input graph should be a concretized graph.
        instructions: List[Instruction] = []
        input_keys = []
        user_keys = set()

        # freeze node with static attributes in label;
        for node_id in nx.topological_sort(graph):
            node = graph.nodes[node_id]
            op = node["op"]

            if isinstance(op, Input):
                input_keys.append(node["otensor_idx"][0])

            for used_idx in node["itensor_idx"]:
                user_keys.add(used_idx)

            # TODO(@ganler): Better name than "otensor_idx"
            # TODO(@ganler): Add refcnt or last ref mechanism to save memory
            instructions.append(
                Instruction(
                    op=op,
                    inputs=node["itensor_idx"],
                    outputs=node["otensor_idx"],
                )
            )

        # simplify the statements above
        leaf_keys = [key for key in key2type if key not in user_keys]

        return Schedule(instructions, input_keys, leaf_keys, key2type)


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
    @property
    @abstractmethod
    def input_like(self) -> Dict[str, AbsTensor]:
        pass

    @property
    @abstractmethod
    def output_like(self) -> Dict[str, AbsTensor]:
        pass

    @staticmethod
    @abstractmethod
    def from_schedule(self, instructions: Schedule, **kwargs) -> "Model":
        pass

    @staticmethod
    @abstractmethod
    def load(path: str) -> "Model":
        pass

    @abstractmethod
    def dump(self, path: str) -> None:
        pass

    @property
    @abstractmethod
    def native_model(self) -> Any:
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
    def name_prefix() -> str:
        return "model"

    @property
    def type(self) -> str:
        return type(self).__name__


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
