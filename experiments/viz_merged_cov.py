import matplotlib.pyplot as plt
import matplotlib
from matplotlib_venn import venn2, _venn2, venn3, _venn3
import numpy as np

import os

import pandas as pd


SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC_TWO = None
MIN_FAC = None

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{xfrac}")


class Ploter:
    def __init__(self, cov_lim=None, use_pdf=False, one_plot=False, scale=1) -> None:
        # cov / time, cov / iteration, iteration / time
        fig0, axs0 = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 3.5))
        fig1, axs1 = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 3))

        self.one_plot = one_plot
        self.fig = [fig0, fig1]
        self.axs = [axs0, axs1]
        self.cov_lim = cov_lim
        self.cov_maxes = []
        self.xspan = 0
        self.use_pdf = use_pdf
        self.scale = scale

    def add(self, data, name=None):
        df = np.array(data)
        df[:, 2] = df[:, 2] / self.scale

        LW = 2
        MARKER_SIZE = 10
        N_MARKER = 8
        MARKERS = ["d", "^", "p", "*"]
        LS = ":"
        COLORS = ["dodgerblue", "violet", "coral"]

        # make it even over time
        markevery = np.zeros_like(df[:, 0], dtype=bool)
        step = int(df[:, 0].max() / N_MARKER)
        offset = step
        for _ in range(N_MARKER):
            idx = list(map(lambda i: i >= offset, df[:, 0])).index(True)
            markevery[idx] = True
            offset += step

        # markevery = int(len(df) / N_MARKER)
        marker = MARKERS[len(self.cov_maxes) % len(MARKERS)]
        color = COLORS[len(self.cov_maxes) % len(MARKERS)]

        # linestyle=LS, marker=marker, markevery=markevery, markersize=MARKER_SIZE, alpha=ALPHA, lw=LW
        style_kw = {
            "linestyle": LS,
            "marker": marker,
            "markevery": markevery,
            "markersize": MARKER_SIZE,
            "lw": LW,
            "color": color,
            "markeredgecolor": "k",
            "markeredgewidth": 1.5,
        }

        self.axs[0].plot(df[:, 0], df[:, 2], label=name, **style_kw)  # cov / time
        print(
            f"----> max cov {df[:, 2].max() * self.scale} + max tests {int(df[:, 1].max())}"
        )

        self.axs[1].plot(df[:, 1], df[:, 2], label=name, **style_kw)  # cov / iteration

        self.xspan = max(self.xspan, df[-1, 0])

        self.cov_maxes.append(df[:, 2].max())

    def plot(self, save="cov", cov_lim=None, loc=0):
        for ax in self.axs:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], loc=loc)

        self.cov_maxes = sorted(self.cov_maxes)

        if len(self.cov_maxes) > 1:
            for rk in range(1, len(self.cov_maxes)):
                best = self.cov_maxes[-1]
                cur = self.cov_maxes[-(rk + 1)]
                print(
                    f"==> Best one is [{best} / {cur}] = **{best / cur:.2f}x** better than the NO. {rk + 1} baseline."
                )

        cov_max = max(self.cov_maxes)
        cov_min = min(self.cov_maxes)

        top_lim = cov_max + (cov_max - cov_min) * 0.5

        if cov_lim is not None:
            ann_size = BIGGER_SIZE + 4
            max_cov_unscale = int(cov_max * self.scale)
            self.axs[0].annotate(
                f"$\\sfrac{{{max_cov_unscale}_\\mathbf{{best}}}}{{{int(cov_lim)}_\\mathbf{{max}}}}$ = \\textbf{{{max_cov_unscale / cov_lim * 100 :.1f}\%}}",
                xy=(0, cov_max + (top_lim - cov_max) / 2.25),
                xycoords="data",
                va="center",
                ha="left",
                fontsize=ann_size,
                bbox=dict(boxstyle="round", fc="w"),
            )

        if self.cov_lim is not None:
            self.axs[0].set_ylim(bottom=self.cov_lim, top=top_lim)
            self.axs[1].set_ylim(bottom=self.cov_lim, top=top_lim)
        else:
            self.axs[0].set_ylim(bottom=cov_min * MIN_FAC, top=top_lim)
            self.axs[1].set_ylim(bottom=cov_min * MIN_FAC, top=top_lim)

        ylabel = "\# Coverage"
        if self.scale != 1:
            ylabel = f"\# Coverage ({self.scale} branches)"
        self.axs[0].set(xlabel="Time (Second)", ylabel=ylabel)
        self.axs[0].grid(alpha=0.5, ls=":")

        self.axs[1].set(xlabel="\# Iteration", ylabel=ylabel)
        self.axs[1].grid(alpha=0.5, ls=":")

        if self.use_pdf:
            self.fig[0].savefig(save + "-time.pdf")
            self.fig[1].savefig(save + "-iter.pdf")
        self.fig[0].savefig(save + "-time.png")
        self.fig[1].savefig(save + "-iter.png")


def cov_summerize(data, pass_filter=None, tlimit=None, branch_only=True, gen_time=None):
    model_total = 0

    branch_by_time = [[0, 0, 0]]
    final_bf = 0

    line_by_time = [[0, 0, 0]]
    final_lf = 0

    for time, value in data.items():
        lf = 0
        bf = 0

        n_model = value["n_model"]
        cov = value["merged_cov"]
        model_total += n_model
        if gen_time is not None:
            time -= gen_time[0][:model_total].sum()
        line_cov = 0
        branch_cov = 0
        for fname in cov:
            if pass_filter is not None and not pass_filter(fname):
                continue

            branch_cov += len(cov[fname]["branches"])
            bf += cov[fname]["bf"]

            if not branch_only:
                line_cov += len(cov[fname]["lines"])
                lf += cov[fname]["lf"]

        if not branch_only:
            line_by_time.append([time, model_total, line_cov])

        branch_by_time.append([time, model_total, branch_cov])

        final_lf = max(final_lf, lf)
        final_bf = max(final_bf, bf)

        if tlimit is not None and time > tlimit:
            break
    return line_by_time, branch_by_time, (final_lf, final_bf)


def tvm_pass_filter(fname):
    if "relay/transforms" in fname:
        return True
    elif "src/tir/transforms" in fname:
        return True
    elif "src/ir/transform.cc" in fname:
        return True

    return False


def ort_pass_filter(fname):
    return "onnxruntime/core/optimizer/" in fname


def tvm_arith_filter(fname):
    return "arith" in fname


def plot_one_round(
    folder,
    data,
    pass_filter=None,
    fuzz_tags=None,
    target_tag="",
    tlimit=None,
    pdf=False,
    pass_tag="",
    gen_time=None,
    venn=False,
    scale=1,
):
    branch_ploter = Ploter(use_pdf=pdf, scale=scale)

    assert fuzz_tags is not None
    if pass_filter is not None:
        assert pass_tag != ""

    # Due to lcov, diff lcov's total cov might be slightly different.
    # We took the max.
    lf = 0
    bf = 0

    for idx, (k, v) in enumerate(data.items()):
        _, branch_by_time, (lf_, bf_) = cov_summerize(
            v,
            tlimit=tlimit,
            pass_filter=pass_filter,
            gen_time=gen_time[k] if gen_time is not None else None,
        )
        branch_ploter.add(data=branch_by_time, name=fuzz_tags[idx])

        lf = max(lf, lf_)
        bf = max(bf, bf_)

    branch_ploter.plot(
        save=os.path.join(folder, target_tag + pass_tag + "branch_cov"),
        cov_lim=bf,
        loc=7 if ("ort" in target_tag) else 0,
    )

    if not venn:
        return

    # venn graph plot
    branch_cov_sets = []
    for _, v in data.items():
        last_key = sorted(list(v.keys()))[-1]
        # file -> {lines, branches}
        final_cov = v[last_key]["merged_cov"]
        branch_set = set()
        for fname in final_cov:
            if pass_filter is not None and not pass_filter(fname):
                continue
            brset = set([fname + br for br in final_cov[fname]["branches"]])
            branch_set.update(brset)
        branch_cov_sets.append(branch_set)

    if len(branch_cov_sets) != 1:
        plt.clf()
        if len(branch_cov_sets) == 2:
            plt.figure(figsize=(3.5, 2.5), constrained_layout=True)
            ks = ["10", "01", "11"]
            sets = {}
            total_covs = [len(s) for s in branch_cov_sets]
            for k, val in zip(ks, _venn2.compute_venn2_subsets(*branch_cov_sets)):
                sets[k] = val
            v = venn2(
                subsets=(5, 5, 2),
                set_labels=[f"{{{t}}}\n({c})" for t, c in zip(fuzz_tags, total_covs)],
            )

            lb = v.get_label_by_id("A")
            x, y = lb.get_position()
            lb.set_position((x + 0.07, y + 0.4))

            lb = v.get_label_by_id("B")
            x, y = lb.get_position()
            lb.set_position((x - 0.07, y + 0.4))

            for id in ["11"]:
                if v.get_label_by_id(id):
                    v.get_label_by_id(id).set_text(sets[id])
                    v.get_label_by_id(id).set_fontsize(MEDIUM_SIZE + 2)
                    v.get_patch_by_id(id).set_alpha(0.15)
                    v.get_patch_by_id(id).set_facecolor("royalblue")

            hatches = ["|", "\\"]
            # circles = ['dodgerblue', 'MediumVioletRed', 'coral', 'white'] # colorful.
            # fcolors = ['violet', 'navajowhite'] # for binning
            fcolors = ["lightgreen", "navajowhite"]
            for idx, id in enumerate(["10", "01"]):
                if sets[id] == 0:
                    continue
                cnt = sets[id]
                v.get_label_by_id(id).set_text(f"\\textbf{{{cnt}}}")
                v.get_label_by_id(id).set_fontsize(BIGGER_SIZE + 4)
                v.get_patch_by_id(id).set_edgecolor("gray")
                v.get_patch_by_id(id).set_hatch(hatches[idx])
                v.get_patch_by_id(id).set_facecolor(fcolors[idx])

        elif len(branch_cov_sets) == 3:
            plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
            ks = ["100", "010", "110", "001", "101", "011", "111"]
            sets = {}
            total_covs = [len(s) for s in branch_cov_sets]
            for k, val in zip(ks, _venn3.compute_venn3_subsets(*branch_cov_sets)):
                sets[k] = val
            v = venn3(
                subsets=(6, 6, 3, 6, 3, 3, 4.5),
                set_labels=[f"{t}\n({c})" for t, c in zip(fuzz_tags, total_covs)],
            )

            lb = v.get_label_by_id("A")
            x, y = lb.get_position()
            lb.set_position((x - 0.2, y - 0.3))

            lb = v.get_label_by_id("B")
            x, y = lb.get_position()
            lb.set_position((x + 0.2, y - 0.3))

            lb = v.get_label_by_id("C")
            x, y = lb.get_position()
            lb.set_position((x + 0.6, y + 0.3))

            for id in ["110", "101", "011"]:
                if v.get_label_by_id(id):
                    v.get_label_by_id(id).set_text(sets[id])
                    v.get_label_by_id(id).set_fontsize(MEDIUM_SIZE)
                    v.get_patch_by_id(id).set_alpha(0.15)

            hatches = ["*", ".", "\\"]
            # circles = ['dodgerblue', 'MediumVioletRed', 'coral', 'white'] # colorful.
            circles = ["k"] * 4  # chill man!
            fcolors = ["lightblue", "violet", "navajowhite"]
            for idx, id in enumerate(["100", "010", "001", "111"]):
                if sets[id] == 0:
                    continue
                cnt = sets[id]
                v.get_label_by_id(id).set_text(f"\\textbf{{{cnt}}}")
                v.get_label_by_id(id).set_fontsize(BIGGER_SIZE + 4)
                v.get_patch_by_id(id).set_edgecolor(circles[idx])
                if id != "111":
                    v.get_patch_by_id(id).set_hatch(hatches[idx])
                    v.get_patch_by_id(id).set_facecolor(fcolors[idx])
                    # v.get_patch_by_id(id).set_linewidth(2)
                else:
                    v.get_patch_by_id(id).set_alpha(0.2)

        for text in v.set_labels:
            text.set_fontsize(BIGGER_SIZE - 1)

    plt.savefig(
        f'{os.path.join(folder, target_tag + pass_tag + "br_cov_venn")}.png',
        bbox_inches="tight",
    )
    if pdf:
        plt.savefig(
            f'{os.path.join(folder, target_tag + pass_tag + "br_cov_venn")}.pdf',
            bbox_inches="tight",
        )
    plt.close()


if "__main__" == __name__:
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    parser.add_argument("--tags", type=str, nargs="+", help="tags")
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="results folder"
    )
    parser.add_argument("-t", "--tlimit", type=int, default=4 * 3600, help="time limit")
    parser.add_argument("--tvm", action="store_true", help="use tvm")
    parser.add_argument("--ort", action="store_true", help="use ort")
    parser.add_argument("--pdf", action="store_true", help="use pdf as well")
    parser.add_argument(
        "--no_count_gen", action="store_true", help="do not count generation time"
    )
    parser.add_argument("--venn", action="store_true", help="plot venn")
    args = parser.parse_args()

    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split("-")[0] for f in args.folders]
        # args.tags = [os.path.split(os.path.split(f)[-2])[-1]
        #              for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    pass_filter = None
    target_tag = ""
    if args.tvm:
        pass_filter = tvm_pass_filter
        target_tag = "tvm_"
    elif args.ort:
        pass_filter = ort_pass_filter
        target_tag = "ort_"
    else:
        print(f"[WARNING] No pass filter is used (use --tvm or --ort)")

    arith_filter = None
    if args.tvm:
        arith_filter = tvm_arith_filter
        MIN_FAC = 0.87
    elif args.ort:
        arith_filter = None
        MIN_FAC = 0.7
    else:
        print(f"[WARNING] No pass filter is used (use --tvm or --ort)")

    data = {}
    gen_time = {} if args.no_count_gen else None
    for f in args.folders:
        with open(os.path.join(f, "merged_cov.pkl"), "rb") as fp:
            data[f] = pickle.load(fp)
            if args.no_count_gen:
                gen_time[f] = pd.read_csv(
                    os.path.join(f, "../gentime.csv"), header=None
                )

    if pass_filter is not None:
        plot_one_round(
            folder=args.output,
            data=data,
            pass_filter=pass_filter,
            pass_tag="opt_",
            scale=100,
            tlimit=args.tlimit,
            fuzz_tags=args.tags,
            target_tag=target_tag,
            pdf=args.pdf,
            gen_time=gen_time,
            venn=args.venn,
        )
    plot_one_round(
        folder=args.output,
        data=data,
        pass_filter=None,
        tlimit=args.tlimit,
        scale=1000,
        fuzz_tags=args.tags,
        target_tag=target_tag,
        pdf=args.pdf,
        gen_time=gen_time,
        venn=args.venn,
    )  # no pass
