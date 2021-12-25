import time
import os
import uuid
import datetime
import random
import shutil
import datetime
from typing import Dict, Iterable, Union, List

# Edge coverage. See https://github.com/ganler/tvm/tree/coverage
from tvm.contrib import coverage
import git
import rich
from rich.progress import Progress, BarColumn, ProgressColumn
from rich.panel import Panel
from rich.console import RenderableType
from rich.columns import Columns

from nnsmith.error import NNSmithInternalError, assert_gt, assert_false
from nnsmith.graph_gen import GenerationTable
from nnsmith.backends.tvm_graph import TVMExecutor
from nnsmith.backends import DiffTestBackend
from nnsmith.input_gen import gen_one_input_rngs
from nnsmith.difftest import assert_allclose
from nnsmith.graph_input_gen import forked_execution

_METADATA_NAME_ = 'meta.txt'
_COV_BY_TIME_NAME_ = 'cov_by_time.csv'

# NOTE: Currently only engineered for TVM.


class Reporter:  # From Tzer.
    def __init__(self, report_folder=None) -> None:
        # Checks
        tvm_home = os.getenv('TVM_HOME')

        assert_false(not tvm_home or not os.path.exists(tvm_home),
                     f'got incorrect env var `TVM_HOME`: "{tvm_home}"')

        self.start_time = time.perf_counter()
        self.report_folder = report_folder

        if report_folder is None:
            self.report_folder = f'fuzzing-report-{uuid.uuid4()}'

        if os.path.exists(self.report_folder):
            # TODO: Allow continous fuzzing...
            decision = ''
            while decision.lower() not in ['y', 'n']:
                decision = input(
                    'Report folder already exists. Press [Y/N] to continue or exit...')
            if decision.lower() == 'n':
                raise NNSmithInternalError(
                    f'{self.report_folder} already exist... We want an empty folder to report...')
            else:
                shutil.rmtree(self.report_folder)

        os.mkdir(self.report_folder)
        print(f'Create report folder: {self.report_folder}')

        print(f'Using `{self.report_folder}` as the fuzzing report folder')
        with open(os.path.join(self.report_folder, _METADATA_NAME_), 'w') as f:
            fuzz_repo = git.Repo(search_parent_directories=True)
            tvm_repo = git.Repo(search_parent_directories=True)

            def _log_repo(f, tag, repo: git.Repo):
                f.write(f'{tag} GIT HASH: {repo.head.object.hexsha}\n')
                f.write(f'{tag} GIT STATUS: ')
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
                f.write(repo.git.status())
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')

            f.write(f'START TIME: {datetime.datetime.now()}')
            _log_repo(f, 'Fuzzer', fuzz_repo)
            _log_repo(f, 'TVM', tvm_repo)

        self.n_bug = 0

    def report_bug(self, err_type: Exception, buggy_onnx_path: str, message: str):
        dir = f'{type(err_type).__name__}__{self.n_bug}'
        os.mkdir(os.path.join(self.report_folder, dir))

        shutil.move(buggy_onnx_path, os.path.join(
            self.report_folder, dir, 'model.onnx'))
        with open(os.path.join(self.report_folder, dir, 'err.txt'), 'w') as f:
            f.write(message)
        self.n_bug += 1

    def record_coverage(self):
        with open(os.path.join(self.report_folder, _COV_BY_TIME_NAME_), 'a') as f:
            f.write(
                f'{time.perf_counter() - self.start_time :.2f},{coverage.get_now()}\n')


class CustomProgress(Progress):
    def __init__(self, fuzz_status, columns: List[Union[str, ProgressColumn]]):
        self.fuzz_status = fuzz_status
        super().__init__(*columns)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        yield self.fuzz_status()
        table = self.make_tasks_table(self.tasks)
        yield table


class FuzzingLoop:  # TODO: Support multiple backends.
    def __init__(self, backends: Dict[str, DiffTestBackend], mode='table', root=None, time_budget=60 * 60 * 4, max_nodes=32):
        self.root = root
        self.reporter = Reporter(report_folder=root)
        self.mode = mode  # `random` or `table`
        self.table = GenerationTable() if mode == 'table' else None

        assert_gt(len(backends), 0, "Empty backends are not allowed!")
        self.backends = backends

        self.time_budget = time_budget
        self.max_nodes = max_nodes

        self.cur_model_gen_t = float('nan')
        self.slowest_model_gen_t = -float("inf")
        self.fastest_model_gen_t = float("inf")

        self.cur_model_eval_t = float('nan')
        self.slowest_model_eval_t = -float("inf")
        self.fastest_model_eval_t = float("inf")

        rich.print(
            f'[bold yellow]To exit the program: `kill {os.getpid()}`[/bold yellow]')
        rich.print(
            '[grey]This is because we use z3 written in C++ w/ Python wrappers. Ctrl+C may not stop it.')

    def rich(self):
        return Columns([
            Panel.fit(
                f'{datetime.timedelta(seconds=round(time.time()-self.start_time))} ~ '
                f'{datetime.timedelta(seconds=self.time_budget)}',
                title="Time Left ~ Total Time"),
            Panel.fit(f'{self.reporter.n_bug}',
                      title="#Bugs", style="magenta"),
            Panel.fit(f'[green]Fast: {self.fastest_model_gen_t:.3f}s[/green]|'
                      f'[bold]Cur: {self.cur_model_gen_t:.3f}s[/bold]|'
                      f'[red]Slow: {self.slowest_model_gen_t:.3f}s',
                      title="Model Generation Time"),
            Panel.fit(f'[green]Fast: {self.fastest_model_eval_t:.3f}s[/green]|'
                      f'[bold]Cur: {self.cur_model_eval_t:.3f}s[/bold]|'
                      f'[red]Slow: {self.slowest_model_eval_t:.3f}s',
                      title="Model Evaluation Time"),
        ])

    def fuzz(self):
        _TMP_ONNX_FILE_ = f'tmp_{uuid.uuid4()}.onnx'
        _PER_MODEL_TIMEOUT_ = 1000  # milliseconds
        self.start_time = time.time()

        last_cov = 0

        try:
            with CustomProgress(
                fuzz_status=self.rich,
                columns=[
                    "[progress.description]{task.description}",
                    BarColumn(),
                    '[progress.percentage]{task.completed:>3.0f}/{task.total}',
                    '[progress.percentage]{task.percentage:>3.0f}%'],
            ) as progress:
                task_fuzz = progress.add_task(
                    '[green]Fuzzing time.', total=self.time_budget)
                task_coverage = progress.add_task(
                    '[green]Edge coverage.', total=coverage.get_total())

                while True:
                    self.reporter.record_coverage()

                    gen_t_s = time.time()
                    # gen = PureSymbolGen()
                    # gen.abstract_gen(max_node_size=random.randint(1, self.max_nodes),
                    #                  max_gen_millisec=_PER_MODEL_TIMEOUT_)
                    # solution = gen.get_symbol_solutions()
                    # # input_shape = gen.concretize_input_shape(solution)
                    # net = SymbolNet(gen.abstract_graph, solution)
                    # net.eval()
                    # # net.set_input_spec(input_shape)
                    # torch2onnx(model=net, filename=_TMP_ONNX_FILE_)

                    rngs, state, edge_set = forked_execution(self.mode,
                                                             _TMP_ONNX_FILE_,
                                                             table=self.table,
                                                             max_node_size=random.randint(
                                                                 1, self.max_nodes),
                                                             max_gen_millisec=_PER_MODEL_TIMEOUT_)

                    # Generation time logging.
                    self.cur_model_gen_t = time.time() - gen_t_s
                    self.fastest_model_gen_t = min(
                        self.fastest_model_gen_t, self.cur_model_gen_t)
                    self.slowest_model_gen_t = max(
                        self.slowest_model_gen_t, self.cur_model_gen_t)

                    try:  # TODO: multi-process support for isolation.
                        eval_t_s = time.time()

                        onnx_model = DiffTestBackend.get_onnx_proto(
                            _TMP_ONNX_FILE_)
                        input_spec, onames = DiffTestBackend.analyze_onnx_io(
                            onnx_model)
                        inp = gen_one_input_rngs(input_spec, rngs)

                        difftest_pool = {}
                        for bname in self.backends:
                            difftest_pool[bname] = self.backends[bname].predict(
                                onnx_model, inp)

                        keys = list(difftest_pool.keys())
                        for idx in range(1, len(keys)):
                            assert_allclose(
                                difftest_pool[keys[0]],
                                difftest_pool[keys[idx]],
                                keys[0], keys[idx],
                                nan_as_err=False)

                        # Evaluation time logging.
                        self.cur_model_eval_t = time.time() - eval_t_s
                        self.fastest_model_eval_t = min(
                            self.fastest_model_eval_t, self.cur_model_eval_t)
                        self.slowest_model_eval_t = max(self.slowest_model_eval_t,
                                                        self.cur_model_eval_t)

                        cur_cov = coverage.get_now()
                        if edge_set:
                            for src, dst in edge_set:
                                if cur_cov == last_cov:
                                    self.table.on_no_cov(src, dst)
                                else:
                                    self.table.on_new_cov(src, dst)

                        # for the whole graph
                        # self.table.on_no_cov()
                    except Exception as e:
                        self.reporter.report_bug(e, _TMP_ONNX_FILE_, str(e))

                    cur_time = time.time()
                    progress.update(
                        task_fuzz, completed=cur_time - self.start_time)
                    progress.update(
                        task_coverage, completed=coverage.get_now())

                    if cur_time - self.start_time > self.time_budget:
                        break
        finally:  # cleanup
            if os.path.exists(_TMP_ONNX_FILE_):
                os.remove(_TMP_ONNX_FILE_)
            last_cov = coverage.get_now()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./fuzz_report')
    parser.add_argument('--time_budget', type=int, default=60 * 60 * 4)
    args = parser.parse_args()

    fuzzing_loop = FuzzingLoop(
        root=args.root,
        backends={'tvm-opt': TVMExecutor(opt_level=4),
                  'tvm-debug': TVMExecutor(opt_level=0)},
        time_budget=args.time_budget)
    fuzzing_loop.fuzz()
