import pickle
import argparse
import os

import numpy as np

from tvm.contrib import coverage


def clipper(cov):
    # Okay, fine, due my previous stupid implementation:
    # https://github.com/ganler/tvm/blob/7e298af66ced5216846a6eb9f9bc01b677cf05d5/python/tvm/contrib/coverage.py#L19
    # the byte array is 8 times larger than the coverage (was a bit array) as I forgot to "/8"
    # but note that this won't affect the results but just waste 7x space...
    # So what I'm gonna do here is to clip it to the required size.
    cov_length = coverage.get_total()
    required_bytes = (cov_length + 7) // 8

    return np.unpackbits(cov[:required_bytes])[:cov_length]  # you got it.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tzer", type=str, required=True, help="Folder of tzer reports."
    )
    parser.add_argument(
        "--nnsmith",
        type=str,
        required=True,
        help="Folder of nnsmith evaluated in memcov.",
    )
    args = parser.parse_args()

    with open(os.path.join(args.tzer, "cov.pkl"), "rb") as fp:
        tzer_cov = pickle.load(fp)

    tzer_cov_bits = clipper(tzer_cov)

    nnsmith_cov_bits = None
    for file in os.listdir(args.nnsmith):
        if file.endswith(".memcov.pkl"):
            with open(os.path.join(args.nnsmith, file), "rb") as fp:
                nnsmith_cov = pickle.load(fp)
            tmp_nnsmith_cov_bits = clipper(nnsmith_cov)
            if nnsmith_cov_bits is None:
                nnsmith_cov_bits = tmp_nnsmith_cov_bits
            else:
                nnsmith_cov_bits = np.logical_or(nnsmith_cov_bits, tmp_nnsmith_cov_bits)

    print(f"Tzer Memcov: {np.count_nonzero(tzer_cov_bits)}")
    print(f"NNSmith Memcov: {np.count_nonzero(nnsmith_cov_bits)}")
    tzer_unique = np.count_nonzero(
        np.logical_and(
            tzer_cov_bits,
            np.logical_not(np.logical_and(tzer_cov_bits, nnsmith_cov_bits)),
        )
    )
    nnsmith_unique = np.count_nonzero(
        np.logical_and(
            nnsmith_cov_bits,
            np.logical_not(np.logical_and(tzer_cov_bits, nnsmith_cov_bits)),
        )
    )
    print(f"Tzer Unique: {tzer_unique}")
    print(f"NNSmith Unique: {nnsmith_unique}")
    print(
        f"Common: {np.count_nonzero(np.logical_and(tzer_cov_bits, nnsmith_cov_bits))}"
    )
