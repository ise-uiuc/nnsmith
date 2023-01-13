from typing import List, Optional, Type

REQUIRES_PATCH = {}
ACTIVATED_PATCH = {}


class patch_requires:
    def __init__(self, tag: str, opname: str):
        self.tag = tag
        self.opname = opname

    def __call__(self, f):
        REQUIRES_PATCH.setdefault(self.tag, {}).setdefault(self.opname, []).append(f)
        return f


def activate_ext(
    opset: List[Type["AbsOpBase"]], factory: Optional["BackendFactory"] = None
):
    for op in opset:
        if "global" in REQUIRES_PATCH:
            ACTIVATED_PATCH.setdefault(op.name(), []).extend(
                REQUIRES_PATCH["global"].get(op.name(), [])
            )

        if factory is not None and factory.system_name in REQUIRES_PATCH:
            ACTIVATED_PATCH.setdefault(op.name(), []).extend(
                REQUIRES_PATCH[factory.system_name].get(op.name(), [])
            )
