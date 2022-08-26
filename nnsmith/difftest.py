from typing import List, Dict, Tuple
import numpy as np
from numpy import testing

from nnsmith.error import *
import pickle
from pathlib import Path


def assert_allclose(
    obtained: Dict[str, np.ndarray],
    desired: Dict[str, np.ndarray],
    obtained_name: str,
    oracle_name: str,
    nan_as_err=False,
    mismatch_cnt_tol=0.01,
    safe_mode=False,
    rtol=1e-2,
    atol=1e-3,
):
    # when safe_mode is turned on, it will use less memory
    err_msg = ""
    if obtained is None:
        err_msg += f"{obtained_name} crashed"
    if desired is None:
        err_msg += f"{oracle_name} crashed"
    if err_msg != "":
        raise CrashError(err_msg)

    if set(obtained.keys()) != set(desired.keys()):
        raise IncorrectResult(
            f"{obtained_name} v.s. {oracle_name} have different output tensor names: {set(obtained.keys())} ~ {set(desired.keys())}"
        )

    if nan_as_err:
        for index, key in enumerate(obtained):
            err_msg = ""
            s = {True: "has no", False: "has"}
            obtained_valid = np.isfinite(obtained[key]).all()
            oracle_valid = np.isfinite(desired[key]).all()
            err_msg += f"{obtained_name} {s[obtained_valid]} Inf/NaN, "
            err_msg += f", {oracle_name} {s[oracle_valid]} Inf/NaN"
            if not obtained_valid or not oracle_valid:
                err_msg = f"At tensor #{index} named {key}: " + err_msg
                # print(err_msg) # Mute.
                raise NumericError(err_msg)

    try:
        index = -1
        SanityCheck.eq(set(obtained.keys()), set(desired.keys()))
        index = 0
        for key in obtained:
            ac = obtained[key]
            de = desired[key]
            assert ac.shape == de.shape, f"Shape mismatch {ac.shape} vs. {de.shape}"
            assert ac.dtype == de.dtype, f"Dtype mismatch {ac.dtype} vs. {de.dtype}"
            if safe_mode:
                ac = ac.ravel()
                de = de.ravel()
                # avoid OOM by slicing
                STRIDE = 2**26
                eq = np.empty_like(ac, bool)
                for i in range(0, ac.shape[0], STRIDE):
                    eq[i : i + STRIDE] = np.isclose(
                        ac[i : i + STRIDE], de[i : i + STRIDE], rtol=rtol, atol=atol
                    )
                # allow 1% of mismatch elements
                assert (
                    1 - np.mean(eq) <= mismatch_cnt_tol
                ), f"{(1-np.mean(eq))*100}% of mismatch"
            else:
                eq = np.isclose(ac, de, rtol=rtol, atol=atol)
                if 1 - np.mean(eq) > mismatch_cnt_tol:  # allow 1% of mismatch elements
                    testing.assert_allclose(ac, de, rtol=rtol, atol=atol)
            index += 1
    except AssertionError as err:
        # print(err) # Mute.
        raise IncorrectResult(
            f"{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor named {key}: {str(err)}"
        )
