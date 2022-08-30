from typing import Dict

import numpy as np
from numpy import testing

from nnsmith.error import *


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

    try:
        for key in akeys:
            testing.assert_allclose(
                actual[key],
                desired[key],
                equal_nan=equal_nan,
                rtol=rtol,
                atol=atol,
            )
    except AssertionError as err:
        raise AssertionError(f"{actual_name} != {oracle_name} at {key}: {str(err)}")
