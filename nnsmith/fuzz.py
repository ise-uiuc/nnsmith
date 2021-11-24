import time
import os
import uuid
import datetime
import random
import shutil
import datetime
from typing import Dict, Iterable

# Edge coverage. See https://github.com/ganler/tvm/tree/coverage
from tvm.contrib import coverage
import git
from rich.progress import Progress, BarColumn
from rich.panel import Panel
from rich.console import RenderableType
from rich.columns import Columns

from nnsmith.error import NNSmithInternalError
from nnsmith.graph_gen import PureSymbolGen, SymbolNet
from nnsmith.export import torch2onnx
from nnsmith.backends.tvm_graph import TVMExecutor
from nnsmith.backends import DiffTestBackend
from nnsmith.input_gen import InputGenBase
from nnsmith.difftest import assert_allclose

_METADATA_NAME_ = 'meta.txt'
_COV_BY_TIME_NAME_ = 'cov_by_time.txt'

# NOTE: Currently only engineered for TVM.


class Reporter:  # From Tzer.
    def __init__(self, report_folder=None) -> None:
        # Checks
        tvm_home = os.getenv('TVM_HOME')
        if not tvm_home or not os.path.exists(tvm_home):
            raise NNSmithInternalError(
                'got incorrect env var `TVM_HOME`: "{tvm_home}"')

        self.start_time = time.perf_counter()
        self.report_folder = report_folder

        if report_folder is None:
            self.report_folder = f'fuzzing-report-{uuid.uuid4()}'

        if os.path.exists(self.report_folder):
            # TODO: Allow continous fuzzing...
            raise NNSmithInternalError(
                f'{self.report_folder} already exist... We want an empty folder to report...')

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
        bug_prefix = f'{type(err_type).__name__}__{uuid.uuid4()}'
        shutil.move(buggy_onnx_path, os.path.join(
            self.report_folder, f'{bug_prefix}.onnx'))
        with open(os.path.join(self.report_folder, f'{bug_prefix}.error_message.txt'), 'w') as f:
            f.write(message)
        self.n_bug += 1

    def record_coverage(self):
        with open(os.path.join(self.report_folder, _COV_BY_TIME_NAME_), 'a') as f:
            f.write(
                f'{time.perf_counter() - self.start_time :.2f},{coverage.get_now()}\n')


class CustomProgress(Progress):
    def __init__(self, fuzz_status, *args, **kwargs):
        self.fuzz_status = fuzz_status
        super().__init__(*args, **kwargs)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        yield self.fuzz_status()
        table = self.make_tasks_table(self.tasks)
        yield table


class FuzzingLoop:  # TODO: Support multiple backends.
    def __init__(self, backends: Dict[str, DiffTestBackend], root=None, time_budget=60 * 60 * 4, max_nodes=32):
        self.root = root
        self.reporter = Reporter(report_folder=root)

        assert len(backends) > 0, "Empty backends are not allowed!"
        self.backends = backends

        self.time_budget = time_budget
        self.max_nodes = max_nodes

    def rich(self):
        return Columns([
            Panel.fit(
                f'{datetime.timedelta(seconds=time.time())} ~ {datetime.timedelta(seconds=self.start_time)}',
                title="Time Left ~ Total Time"),
            Panel.fit('sdasd', title="#Bugs"),
            Panel.fit('asdasda', title="Gen Speed"),
        ])

    def fuzz(self):
        _TMP_ONNX_FILE_ = f'tmp_{uuid.uuid4()}.onnx'
        _PER_MODEL_TIMEOUT_ = 1000  # milliseconds
        self.start_time = time.time()

        with CustomProgress(
            "[progress.description]{task.description}",
            BarColumn(),
            '[progress.percentage]{task.completed}/{task.total}',
            fuzz_status=self.rich,
        ) as progress:
            task_fuzz = progress.add_task(
                '[cyan]Fuzzing time.', total=self.time_budget)
            task_coverage = progress.add_task(
                '[green]Edge coverage.', total=coverage.get_total())

            while True:
                self.reporter.record_coverage()

                gen = PureSymbolGen()
                gen.abstract_gen(max_node_size=random.randint(1, self.max_nodes),
                                 max_gen_millisec=_PER_MODEL_TIMEOUT_)
                solution = gen.get_symbol_solutions()
                input_shape = gen.concretize_input_shape(solution)
                net = SymbolNet(gen.abstract_graph, solution)
                net.eval()
                net.set_input_spec(input_shape)
                torch2onnx(model=net, filename=_TMP_ONNX_FILE_)

                try:  # TODO: multi-process support for isolation.
                    onnx_model = DiffTestBackend.get_onnx_proto(
                        _TMP_ONNX_FILE_)
                    input_spec, onames = DiffTestBackend.analyze_onnx_io(
                        onnx_model)
                    inp = InputGenBase.gen_one_input(input_spec, 0, 1)

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
                except Exception as e:
                    self.reporter.report_bug(e, _TMP_ONNX_FILE_, str(e))

                cur_time = time.time()
                progress.update(
                    task_fuzz, completed=cur_time - self.start_time)
                progress.update(task_coverage, completed=coverage.get_now())

                if cur_time - self.start_time > self.time_budget:
                    break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./tmp')
    parser.add_argument('--time_budget', type=int, default=60 * 60 * 4)
    args = parser.parse_args()

    fuzzing_loop = FuzzingLoop(
        root=args.root,
        backends={'tvm-opt': TVMExecutor(opt_level=4),
                  'tvm-debug': TVMExecutor(opt_level=0)},
        time_budget=args.time_budget)
    fuzzing_loop.fuzz()
