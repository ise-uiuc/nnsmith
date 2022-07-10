import os
import multiprocessing
import pickle
from pathlib import Path
from copy import deepcopy

import lz4.frame


def analyze_lcov(path):
    path = str(path)
    assert path.split(".")[-1] == "lz4"
    with lz4.frame.open(path, mode="rb") as fp:
        output_data = fp.read().decode("ISO-8859-1")  # LCOV issue.

    file_covs = output_data.split("end_of_record\n")[:-1]
    ret = {
        # 'filename': {
        #    'lines': [identifiers],
        #    'branches': [identifiers],
        #    'lf': # lines total,
        #    'bf': # branches total,
        # }
    }

    # SF: source file
    # sperated by end_of_record

    # FN: <line number>,<function name>

    for file_cov in file_covs:
        cov_lines = file_cov.split("\n")
        assert cov_lines[0].startswith("SF:")
        filename = cov_lines[0][3:]

        lines = set()
        branches = set()

        # LH: # line hits
        # FNH: # function hits | note we consider source-level functions. e.g., templates are one function.
        # BRH: # branch hits
        # NOTE: LH might (slightly) != len(lines). we only consider len(lines).

        n_line_total = 0  # LF: # lines
        n_branch_total = 0  # BRF: # branches

        lf = 0
        brf = 0

        for cov_line in cov_lines[1:]:
            cov_line = cov_line.rstrip("\n")
            if cov_line.startswith("DA:"):
                # DA: <line number>,<execution count> for each instrumented line
                line_number, exec_count = cov_line[3:].split(",")
                if exec_count != "0":
                    lines.add(int(line_number))
                n_line_total += 1
            elif cov_line.startswith("BRDA:"):
                # BRDA: <line number>,<block number>,<branch number>,<taken>
                line_number, block_number, branch_number, taken = cov_line[5:].split(
                    ","
                )
                if taken != "-" and taken != "0":
                    branches.add(line_number + ":" + block_number + ":" + branch_number)
                n_branch_total += 1
            # elif cov_line.startswith('LH:'):
            #     n_line_hit = int(cov_line[3:])
            elif cov_line.startswith("LF:"):
                lf = int(cov_line[3:])
            # elif cov_line.startswith('FNH:'):
            #     n_func_hit = int(cov_line[4:])
            elif cov_line.startswith("BRF:"):
                brf = int(cov_line[4:])
            else:
                pass
        # # BRF might be even smaller than branches you hit. we conservatively use n_branch_total.
        n_branch_total = max(brf, n_branch_total)
        # Similarly
        n_line_total = max(lf, n_line_total)

        assert (
            len(lines) <= n_line_total
        ), f"{len(lines)} <= {n_line_total} in {filename}"
        assert (
            len(branches) <= n_branch_total
        ), f"{len(branches)} <= {n_branch_total} in {filename}"

        ret[filename] = {
            "lines": lines,
            "branches": branches,
            "lf": n_line_total,
            "bf": n_branch_total,
        }

    return ret


def analyze_folder(folder, redo=False, max_time=None):
    return_write_name = os.path.join(folder, "merged_cov.pkl")

    if os.path.exists(return_write_name) and not redo:
        print("===> {} already exists.".format(return_write_name))
        with open(return_write_name, "rb") as fp:
            return pickle.load(fp)

    file_list = set()
    for file in Path(folder).rglob("*.lcov.lz4"):
        file_list.add(file)

    assert len(file_list) > 0, "no lcov files found in {}".format(folder)

    file_list = sorted(list(file_list), key=lambda x: int(str(x.name).split(".")[-3]))
    pool = multiprocessing.Pool(
        processes=min(len(file_list), multiprocessing.cpu_count())
    )
    results = pool.map(analyze_lcov, file_list)

    assert len(results) == len(file_list)
    stats = open(os.path.join(folder, "stats.csv"), "r").read().split("\n")
    if len(stats[-1]) == 0:
        stats = stats[:-1]
    assert len(stats) == len(results), f"{len(stats)} != {len(results)}"

    ret = {
        # 'time': time point. { # ! Not time duration.
        #   'n_model': # how many models got successfully executed?
        #   'merged_cov': coverage dictionary.
        # }
    }

    def merge_cov(current, rhs, hint=""):
        # {
        #     'lines': lines,
        #     'branches': branches,
        #     'lf': n_line_total,
        #     'bf': n_branch_total,
        # }

        for k in set(current.keys()).union(set(rhs.keys())):
            if k not in current:
                current[k] = rhs[k]
                continue
            elif k not in rhs:
                continue
            else:
                current[k]["lines"] = current[k]["lines"].union(rhs[k]["lines"])
                current[k]["branches"] = current[k]["branches"].union(
                    rhs[k]["branches"]
                )

                if current[k]["lf"] != rhs[k]["lf"]:
                    print(
                        f'[WARNING] total line {current[k]["lf"]} != {rhs[k]["lf"]} in {k} '
                        + hint
                    )
                    current[k]["lf"] = max(current[k]["lf"], rhs[k]["lf"])

                if current[k]["bf"] != rhs[k]["bf"]:
                    print(
                        f'[WARNING] total branch {current[k]["bf"]} != {rhs[k]["bf"]} in {k} '
                        + hint
                    )
                    current[k]["bf"] = max(current[k]["bf"], rhs[k]["bf"])

        return current

    current_cov = {}
    current_time = 0
    for i, r in enumerate(results):
        tokens = stats[i].split(",")
        t = tokens[0]
        n_model = tokens[2]
        current_time += float(t)
        if max_time is not None and current_time > max_time:
            break
        ret[current_time] = {}
        ret[current_time]["n_model"] = int(n_model)
        current_cov = merge_cov(current_cov, r, f"merging {file_list[i]}")
        ret[current_time]["merged_cov"] = deepcopy(current_cov)

    with open(return_write_name, "wb") as fp:
        pickle.dump(ret, fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze coverage and plot visualizations."
    )
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    parser.add_argument("--redo", action="store_true", help="redo analysis")
    parser.add_argument(
        "--max_time",
        type=int,
        default=60 * 60 * 4,
        help="max time in seconds for coverage evaluation",
    )
    args = parser.parse_args()

    for folder in args.folders:
        analyze_folder(folder, redo=args.redo, max_time=args.max_time)
