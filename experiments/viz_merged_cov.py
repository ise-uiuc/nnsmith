import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from matplotlib_venn._venn3 import compute_venn3_subsets
import numpy as np

import os


class Ploter:
    def __init__(self, cov_lim=None, use_pdf=False) -> None:
        self.legends = []  # type: ignore
        # cov / time, cov / iteration, iteration / time
        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(13, 5))
        self.fig = fig
        self.axs = axs
        self.cov_lim = cov_lim
        self.cov_max = 0
        self.cov_min = 1000000000000000000
        self.xspan = None
        self.use_pdf = use_pdf

    def add(self, data, name=None):
        df = np.array(data)

        self.axs[0].plot(df[:, 0], df[:, 2])  # cov / time
        self.axs[1].plot(df[:, 1], df[:, 2])  # cov / iteration
        self.axs[2].plot(df[:, 0], df[:, 1])  # iter / time

        self.xspan = df[-1, 0]

        self.cov_max = max(self.cov_max, df[:, 2].max())
        self.cov_min = min(self.cov_min, df[:, 2].max())

        if name:
            self.legends.append(name)
        else:
            assert not self.legends

    def plot(self, save='cov', cov_type='', cov_lim=None):
        for axs in self.axs:
            axs.legend(self.legends, loc='upper left')

        if cov_lim is not None:
            self.axs[0].annotate(f"{int(self.cov_max)}/{int(cov_lim)} ~ $\\bf{{{self.cov_max / cov_lim * 100 :.1f}\%}}$", xy=(self.xspan, self.cov_max * 1.02), xycoords="data",
                                 va="center", ha="right", fontsize=11,
                                 bbox=dict(boxstyle="sawtooth", fc="w"))
        #     self.axs[0].axhline(y=cov_lim, color='r', linestyle='dashdot')
        #     self.axs[1].axhline(y=cov_lim, color='r', linestyle='dashdot')

        if cov_type:
            cov_type += ' '

        if self.cov_lim is not None:
            self.axs[0].set_ylim(bottom=self.cov_lim)
            self.axs[1].set_ylim(bottom=self.cov_lim)
        else:
            self.axs[0].set_ylim(bottom=self.cov_min * 0.85)
            self.axs[1].set_ylim(bottom=self.cov_min * 0.85)

        self.axs[0].set(
            xlabel='Time / Second',
            ylabel=f'# {cov_type}Coverage')
        self.axs[0].set_title('Coverage $\\bf{Time}$ Efficiency')

        self.axs[1].set(
            ylabel=f'# {cov_type}Coverage',
            xlabel='# Iteration')
        self.axs[1].set_title('Coverage $\\bf{Iteration}$ Efficiency')

        self.axs[2].set(
            xlabel='Time / Second',
            ylabel='# Iteration')
        self.axs[2].set_title('Iteration Speed')

        if self.use_pdf:
            self.fig.savefig(save + '.pdf')
        self.fig.savefig(save + '.png')


def cov_summerize(data, pass_filter=None, tlimit=None):
    line_by_time = [[0, 0, 0]]
    branch_by_time = [[0, 0, 0]]
    func_by_time = [[0, 0, 0]]
    model_total = 0

    final_lf = 0
    final_bf = 0
    final_ff = 0

    for time, value in data.items():
        lf = 0
        ff = 0
        bf = 0

        n_model = value['n_model']
        cov = value['merged_cov']
        model_total += n_model
        line_cov = 0
        branch_cov = 0
        func_cov = 0
        for fname in cov:
            if pass_filter is not None and pass_filter(fname):
                continue
            line_cov += len(cov[fname]['lines'])
            branch_cov += len(cov[fname]['branches'])
            func_cov += len(cov[fname]['functions'])

            lf += cov[fname]['lf']
            ff += cov[fname]['ff']
            bf += cov[fname]['bf']
        line_by_time.append([time, model_total, line_cov])
        branch_by_time.append([time, model_total, branch_cov])
        func_by_time.append([time, model_total, func_cov])

        final_lf = max(final_lf, lf)
        final_bf = max(final_bf, bf)
        final_ff = max(final_ff, ff)

        if tlimit is not None and time > tlimit:
            break
    return line_by_time, branch_by_time, func_by_time, (final_lf, final_bf, final_ff)


def tvm_pass_filter(fname):
    if 'relay/transforms' in fname:
        return True
    elif 'src/tir/transforms' in fname:
        return True
    elif 'src/ir/transform.cc' in fname:
        return True

    return False


def ort_pass_filter(fname):
    return 'onnxruntime/core/optimizer/' in fname


def plot_one_round(folder, data, pass_filter=None, fuzz_tags=None, target_tag='', tlimit=None, pdf=False):
    line_ploter = Ploter(use_pdf=pdf)
    branch_ploter = Ploter(use_pdf=pdf)
    func_ploter = Ploter(use_pdf=pdf)

    assert fuzz_tags is not None
    pass_tag = 'opt_' if pass_filter is not None else ''

    # Due to lcov, diff lcov's total cov might be slightly different.
    # We took the max.
    lf = 0
    bf = 0
    ff = 0

    for idx, (k, v) in enumerate(data.items()):
        line_by_time, branch_by_time, func_by_time, (lf_, bf_, ff_) = cov_summerize(
            v, tlimit=tlimit, pass_filter=pass_filter)
        line_ploter.add(data=line_by_time, name=fuzz_tags[idx])
        branch_ploter.add(data=branch_by_time, name=fuzz_tags[idx])
        func_ploter.add(data=func_by_time, name=fuzz_tags[idx])

        lf = max(lf, lf_)
        bf = max(bf, bf_)
        ff = max(ff, ff_)

    line_ploter.plot(save=os.path.join(
        folder, target_tag + pass_tag + 'line_cov'), cov_type='Line', cov_lim=lf)
    branch_ploter.plot(save=os.path.join(
        folder, target_tag + pass_tag + 'branch_cov'), cov_type='Branch', cov_lim=bf)
    func_ploter.plot(save=os.path.join(
        folder, target_tag + pass_tag + 'func_cov'), cov_type='Function', cov_lim=ff)

    # venn graph plot
    branch_cov_sets = []
    for _, v in data.items():
        last_key = sorted(list(v.keys()))[-1]
        # file -> {lines, branches, functions}
        final_cov = v[last_key]['merged_cov']
        branch_set = set()
        for fname in final_cov:
            if pass_filter is not None and pass_filter(fname):
                continue
            brset = set([fname + br for br in final_cov[fname]['branches']])
            branch_set.update(brset)
        branch_cov_sets.append(branch_set)

    if len(branch_cov_sets) != 1:
        plt.clf()
        if len(branch_cov_sets) == 2:
            v = venn2(subsets=branch_cov_sets, set_labels=[
                f'$\\bf{{{t}}}$' for t in fuzz_tags], alpha=0.3)
        elif len(branch_cov_sets) == 3:
            v = venn3(subsets=branch_cov_sets, set_labels=[
                f'$\\bf{{{t}}}$' for t in fuzz_tags], alpha=0.3)

            v.get_label_by_id('110').set_text('')
            v.get_label_by_id('011').set_text('')
            v.get_label_by_id('101').set_text('')

            # v.get_label_by_id("100").set_x(v.get_label_by_id('100').get_position()[0] * 1.05)
            # v.get_label_by_id("010").set_x(v.get_label_by_id('010').get_position()[0] * 1.3)
            # v.get_label_by_id("001").set_x(v.get_label_by_id('001').get_position()[0] * 1.2)

            h, l = [],[]
            hatches = ['\\', '.', '*']
            circles = ['MediumVioletRed', 'SeaGreen', 'Lavender']
            for idx, id in enumerate(['100', '010', '001', '111']):
                cnt = int(v.get_label_by_id(id).get_text())
                v.get_label_by_id(id).set_text('')
                if id != '111':
                    v.get_patch_by_id(id).set_alpha(0.5)
                    v.get_patch_by_id(id).set_hatch(hatches[idx])
                    v.get_patch_by_id(id).set_edgecolor(circles[idx])
                    v.get_patch_by_id(id).set_linewidth(2)
                # append patch to handles list
                h.append(v.get_patch_by_id(id))
                # append count to labels list
                l.append(cnt)

            plt.legend(handles=h, labels=l, title="counts", loc='lower right')
    plt.title("Venn Diagram of Branch Coverage")
    plt.savefig(f'{os.path.join(folder, target_tag + pass_tag + "br_cov_venn")}.png',
                bbox_inches='tight')
    if pdf:
        plt.savefig(f'{os.path.join(folder, target_tag + pass_tag + "br_cov_venn")}.pdf',
                    bbox_inches='tight')
    plt.close()


if '__main__' == __name__:
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folders', type=str,
                        nargs='+', help='bug report folder')
    parser.add_argument('--tags', type=str, nargs='+', help='tags')
    parser.add_argument('-o', '--output', type=str,
                        default='results', help='results folder')
    parser.add_argument('-t', '--tlimit', type=int,
                        default=4 * 3600, help='time limit')
    parser.add_argument('--tvm', action='store_true', help='use tvm')
    parser.add_argument('--ort', action='store_true', help='use ort')
    parser.add_argument('--pdf', action='store_true', help='use pdf as well')
    args = parser.parse_args()

    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split('-')[0] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    pass_filter = None
    target_tag = ''
    if args.tvm:
        pass_filter = tvm_pass_filter
        target_tag = 'tvm_'
    elif args.ort:
        pass_filter = ort_pass_filter
        target_tag = 'ort_'
    else:
        print(f'[WARNING] No pass filter is used (use --tvm or --ort)')

    data = {}
    for f in args.folders:
        with open(os.path.join(f, 'merged_cov.pkl'), 'rb') as fp:
            data[f] = pickle.load(fp)

    if pass_filter is not None:
        plot_one_round(folder=args.output, data=data,
                       pass_filter=pass_filter, tlimit=args.tlimit, fuzz_tags=args.tags, target_tag=target_tag, pdf=args.pdf)
    plot_one_round(folder=args.output, data=data,
                   pass_filter=None, tlimit=args.tlimit, fuzz_tags=args.tags, target_tag=target_tag, pdf=args.pdf)  # no pass
