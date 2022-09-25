from inspect import signature
from types import FunctionType
from typing import Set, Type

from nnsmith.materialize import BugReport, Stage, Symptom

FILTERS = {}


class filter:
    def __init__(self, name):
        self.name = name

    def __call__(self, fn_or_cls):
        assert self.name not in FILTERS, f"Filter {self.name} already exists."
        if isinstance(fn_or_cls, Type):  # Class checks
            assert not signature(
                fn_or_cls
            ).parameters, f"filter class {fn_or_cls.__name__} (aka {self.name}) should not have any parameters."
            caller_sig = signature(fn_or_cls.__call__).parameters
            assert (
                caller_sig["self"] and len(caller_sig) == 2
            ), f"filter class {fn_or_cls.__name__} (aka {self.name}) should implement __call__(self, report: BugReport)."
        elif isinstance(fn_or_cls, FunctionType):  # Function checks
            caller_sig = signature(fn_or_cls).parameters
            assert (
                len(caller_sig) == 1
            ), f"filter function {fn_or_cls.__name__} (aka {self.name}) should implement __call__(report: BugReport)."
        else:
            raise ValueError(
                f"filter {fn_or_cls} (aka {self.name}) should be a class or function."
            )

        FILTERS[self.name] = fn_or_cls
        return fn_or_cls


@filter("nan")
def filter_nan(report: BugReport) -> bool:  # True means filter;
    if report.symptom != Symptom.INCONSISTENCY or report.stage != Stage.VERIFICATION:
        return False

    # numpy.assert_allclose style.
    # TODO(ganler): can we use more well-formed checking? say directly checking the results?
    return "nan location mismatch" in report.log


@filter("inf")
def filter_inf(report: BugReport) -> bool:
    if report.symptom != Symptom.INCONSISTENCY or report.stage != Stage.VERIFICATION:
        return False

    # numpy.assert_allclose style.
    return "inf" in report.log.replace("Max relative difference: inf", "")


@filter("dup")  # duplicate
class FilterDup:
    def __init__(self):
        self.seen: Set[int] = set()

    def __call__(self, report: BugReport) -> bool:
        if (
            report.symptom != Symptom.EXCEPTION
            and report.symptom != Symptom.INCONSISTENCY
        ):
            return False  # don't filter bugs other than inconsistency/exception

        str_hash = hash(report.log)
        if str_hash in self.seen:
            return True

        self.seen.add(str_hash)
        return False


# You can patch your own filters!
