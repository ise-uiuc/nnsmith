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
        testing.assert_allclose(
            actual[key],
            desired[key],
            equal_nan=equal_nan,
            rtol=rtol,
            atol=atol,
            err_msg=f"{actual_name} != {oracle_name} at {key}",
        )
