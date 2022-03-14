import matplotlib.pyplot as plt
import numpy as np
import os

class Ploter:
    def __init__(self, cov_lim=None) -> None:
        self.legends = []  # type: ignore
        # cov / time, cov / iteration, iteration / time
        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(13, 5))
        self.fig = fig
        self.axs = axs
        self.cov_lim = cov_lim
        self.cov_max = 0
        self.cov_min = 0

    def add(self, data, name=None):
        df = np.array(data)
        
        self.axs[0].plot(df[:, 0], df[:, 2])  # cov / time
        self.axs[1].plot(df[:, 1], df[:, 2])  # cov / iteration
        self.axs[2].plot(df[:, 0], df[:, 1])  # iter / time

        self.cov_max = max(self.cov_max, df[:, 2].max())
        self.cov_min = min(self.cov_min, df[:, 2].max())

        if name:
            self.legends.append(name)
        else:
            assert not self.legends

    def plot(self, save='cov', cov_type=''):
        for axs in self.axs:
            axs.legend(self.legends)
        
        if cov_type:
            cov_type += ' '
        
        if self.cov_lim is not None:
            self.axs[0].set_ylim(bottom=self.cov_lim)
            self.axs[1].set_ylim(bottom=self.cov_lim)
        else:
            self.axs[0].set_ylim(bottom=self.cov_min * 0.75)
            self.axs[1].set_ylim(bottom=self.cov_min * 0.75)

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

        self.fig.savefig(save + '.pdf')
        self.fig.savefig(save + '.png')


def cov_summerize(data, filter_only=None, tlimit=None):
    line_by_time = [[0, 0, 0]]
    branch_by_time = [[0, 0, 0]]
    func_by_time = [[0, 0, 0]]
    model_total = 0
    for time, value in data.items():
        n_model = value['n_model']
        cov = value['merged_cov']
        model_total += n_model
        line_cov = 0
        branch_cov = 0
        func_cov = 0
        for fname in cov:
            if filter_only is not None and fname not in filter_only:
                continue
            line_cov += len(cov[fname]['lines'])
            branch_cov += len(cov[fname]['branches'])
            func_cov += len(cov[fname]['functions'])
        line_by_time.append([time, model_total, line_cov])
        branch_by_time.append([time, model_total, branch_cov])
        func_by_time.append([time, model_total, func_cov])
        if tlimit is not None and time > tlimit:
            break
    return line_by_time, branch_by_time, func_by_time


if '__main__' == __name__:
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folders', type=str, nargs='+', help='bug report folder')
    parser.add_argument('--tags', type=str, nargs='+', help='tags')
    parser.add_argument('-o', '--output', type=str, default='results', help='results folder')
    parser.add_argument('-t', '--tlimit', type=int, default=4 * 3600, help='time limit')
    args = parser.parse_args()

    if args.tags is None:
        args.tags = [os.path.split(f)[-1] for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    data = {}
    for f in args.folders:
        with open(os.path.join(f, 'merged_cov.pkl'), 'rb') as fp:
            data[f] = pickle.load(fp)

    # line plot
    line_ploter = Ploter()
    branch_ploter = Ploter()
    func_ploter = Ploter()

    for k, v in data.items():
        line_by_time, branch_by_time, func_by_time = cov_summerize(v, tlimit=args.tlimit)
        line_ploter.add(data=line_by_time, name=k)
        branch_ploter.add(data=branch_by_time, name=k)
        func_ploter.add(data=func_by_time, name=k)

    line_ploter.plot(save=os.path.join(args.output, 'line_cov'), cov_type='Line')
    branch_ploter.plot(save=os.path.join(args.output, 'branch_cov'), cov_type='Branch')
    func_ploter.plot(save=os.path.join(args.output, 'func_cov'), cov_type='Function')
    
    # venn graph plot
    branch_cov_sets = []
    for _, v in data.items():
        last_key = sorted(list(v.keys()))[-1]
        final_cov = v[last_key]['merged_cov'] # file -> {lines, branches, functions}
        branch_set = set()
        for fname in final_cov:
            brset = set([fname + br for br in final_cov[fname]['branches']])
            branch_set.update(brset)
        branch_cov_sets.append(branch_set)

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    plt.clf()
    vg = venn2(subsets=branch_cov_sets, set_labels=[f'$\\bf{{{t}}}$' for t in args.tags], alpha=0.3)
    plt.title("Venn Diagram of Branch Coverage")
    plt.savefig(f'{os.path.join(args.output, "br_cov_venn")}.png', bbox_inches='tight')
    plt.savefig(f'{os.path.join(args.output, "br_cov_venn")}.pdf', bbox_inches='tight')
    plt.close()

