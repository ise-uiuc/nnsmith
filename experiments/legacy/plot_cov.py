import argparse
import matplotlib.pyplot as plt
import pandas
import os


class Ploter:
    def __init__(self, cov_lim=None) -> None:
        self.legends = []  # type: ignore
        # cov / time, cov / iteration, iteration / time
        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(16, 5))
        self.fig = fig
        self.axs = axs
        self.cov_lim = cov_lim

    def add(self, folder, name=None):
        path = os.path.join(folder, "cov_by_time.csv")
        df = pandas.read_csv(path, usecols=[0, 1], header=None).to_numpy()

        self.axs[0].plot(df[:, 0], df[:, 1])  # cov / time
        self.axs[1].plot(range(len(df[:, 0])), df[:, 1])  # cov / iteration
        self.axs[2].plot(df[:, 0], range(len(df[:, 1])))  # iter / time

        if name:
            self.legends.append(name)
        else:
            assert not self.legends

    def plot(self, save="cov"):
        for axs in self.axs:
            axs.legend(self.legends)
        # plt.legend(self.legends)

        if self.cov_lim is not None:
            self.axs[0].set_ylim(bottom=self.cov_lim)
            self.axs[1].set_ylim(bottom=self.cov_lim)

        self.axs[0].set(xlabel="Time / Second", ylabel="# Coverage")
        self.axs[0].set_title("Coverage $\\bf{Time}$ Efficiency")

        self.axs[1].set(ylabel="# Coverage", xlabel="# Iteration")
        self.axs[1].set_title("Coverage $\\bf{Iteration}$ Efficiency")

        self.axs[2].set(xlabel="Time / Second", ylabel="# Iteration")
        self.axs[2].set_title("Iteration Speed")

        plt.savefig(save + ".pdf")
        plt.savefig(save + ".png")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    parser.add_argument(
        "-cl", "--cov_lim", type=int, default=None, help="coverage starting lim"
    )
    parser.add_argument(
        "--tvmfuzz", type=str, nargs="?", help="TVMFuzz coverage by time file"
    )
    args = parser.parse_args()

    ploter = Ploter(cov_lim=args.cov_lim)

    for f in args.folders:
        ploter.add(f, f)
    ploter.plot("cov")
