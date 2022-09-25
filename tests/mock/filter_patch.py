from nnsmith.filter import filter
from nnsmith.materialize import BugReport


@filter("test_fn")
def filter_fn(report: BugReport):
    return False  # Won't filter anything.


@filter("test_cls")
class FilterCls:
    def __call__(self, report: BugReport) -> bool:
        return False  # Won't filter anything.
