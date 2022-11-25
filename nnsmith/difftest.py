from typing import Dict

import numpy as np
from numpy import testing


def assert_allclose(
    actual: Dict[str, np.ndarray],
    desired: Dict[str, np.ndarray],
    actual_name: str,
    oracle_name: str,
    equal_nan=False,
    rtol=1e-2,
    atol=1e-3,
):
    akeys = set(actual.keys())
    dkeys = set(desired.keys())
    if akeys != dkeys:
        raise KeyError(f"{actual_name}: {akeys} != {oracle_name}: {dkeys}")

    for key in akeys:
        lhs = actual[key]
        rhs = desired[key]

        # check lhs & rhs same type and not None
        # 1. same type
        if type(lhs) is not type(rhs):
            raise TypeError(
                f"type({actual_name}[{key}]): {type(lhs)} != type({oracle_name}[{key}]): {type(rhs)}"
            )

        # 2. not None
        if lhs is None:
            raise TypeError(f"{actual_name}[{key}] is None")
        if rhs is None:
            raise TypeError(f"{oracle_name}[{key}] is None")

        testing.assert_allclose(
            lhs,
            rhs,
            equal_nan=equal_nan,
            rtol=rtol,
            atol=atol,
            err_msg=f"{actual_name} != {oracle_name} at {key}",
        )
