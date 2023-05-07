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

        if lhs is not None and rhs is not None:
            # check if lhs is np.ndarray
            if lhs is not None and not isinstance(lhs, np.ndarray):
                raise TypeError(
                    f"{actual_name}[{key}] is not np.ndarray but {type(lhs)}"
                )

            # check if rhs is np.ndarray
            if rhs is not None and not isinstance(rhs, np.ndarray):
                raise TypeError(
                    f"{oracle_name}[{key}] is not np.ndarray but {type(rhs)}"
                )

            testing.assert_allclose(
                lhs,
                rhs,
                equal_nan=equal_nan,
                rtol=rtol,
                atol=atol,
                err_msg=f"{actual_name} != {oracle_name} at {key}",
            )
        else:
            return lhs is None and rhs is None
