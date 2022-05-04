from multiprocessing import Process
from pathlib import Path
import pickle
import sys
import time
import os
import uuid
import datetime
import random
import shutil
from typing import Dict, Iterable, Union, List
import socket
import warnings

# Edge coverage. See https://github.com/ganler/tvm/tree/coverage
import git
import rich
import numpy as np
from rich.progress import Progress, BarColumn, ProgressColumn
from rich.panel import Panel
from rich.console import RenderableType
from rich.columns import Columns
import networkx as nx
import torch

from backend_executor import DummyExecutor
from nnsmith.abstract.op import ALL_OP_TYPES, AbsOpBase, auto_infer_in_dtypes, config_skip_op
from nnsmith.error import NNSmithInternalError, SanityCheck
from nnsmith.backends import DiffTestBackend
from nnsmith.difftest import assert_allclose
from nnsmith.graph_gen import random_model_gen, SymbolNet, random_tensor
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.export import torch2onnx

__COV_DRIVER__ = None

_METADATA_NAME_ = 'meta.txt'
_COV_BY_TIME_NAME_ = 'cov_by_time.csv'

# NOTE: Currently only engineered for TVM.


class Reporter:  # From Tzer.
    def __init__(self, report_folder=None, name_hint='', yes=False) -> None:
        # Checks
        self.start_time = time.perf_counter()
        self.report_folder = report_folder

        if report_folder is None:
            self.report_folder = f'fuzzing-report-{uuid.uuid4()}'

        if os.path.exists(self.report_folder):
            # TODO: Allow continous fuzzing...
            decision = '' if not yes else 'y'
            while decision.lower() not in ['y', 'n']:
                decision = input(
                    'Report folder already exists. Press [Y/N] to continue or exit...')
            if decision.lower() == 'n':
                raise NNSmithInternalError(
                    f'{self.report_folder} already exist... We want an empty folder to report...')
            else:
                for f in Path(self.report_folder).glob('*'):
                    if str(f.name) not in ['stdout.log', 'stderr.log'] and not f.name.endswith('.profraw'):
                        os.system(f'rm -rf {f}')

        Path(self.report_folder).mkdir(parents=True, exist_ok=True)
        print(f'Create report folder: {self.report_folder}')

        print(f'Using `{self.report_folder}` as the fuzzing report folder')
        with open(os.path.join(self.report_folder, _METADATA_NAME_), 'w') as f:
            fuzz_repo = git.Repo(search_parent_directories=True)

            def _log_repo(f, tag, repo: git.Repo):
                f.write(f'{tag} GIT HASH: {repo.head.object.hexsha}\n')
                f.write(f'{tag} GIT STATUS: ')
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
                f.write(repo.git.status())
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')
                f.write(repo.git.diff())

            f.write(f'START TIME: {datetime.datetime.now()}\n')
            f.write(f'COMMAND: {sys.argv}\n')
            f.write(f'NNSMITH ENVIRONMENT:\n')
            for k, v in os.environ.items():
                if k.startswith('NNSMITH_'):
                    f.write(f'\t{k}={v}\n')
            f.write(f'HOSTNAME: {socket.gethostname()}\n')
            _log_repo(f, 'Fuzzer', fuzz_repo)
            if 'tvm' in name_hint and os.getenv('TVM_HOME'):
                _log_repo(f, 'TVM', git.Repo(os.getenv('TVM_HOME')))

        self.n_bug = 0
        self.record_coverage_cnt = 0

    def report_bug(self, err_type: Exception, buggy_onnx_path: str, buggy_torch_path: str, message: str, stdout: str, stderr: str, graph_path: str, sat_inputs=None):
        dir = f'{type(err_type).__name__}__{self.n_bug}'
        os.mkdir(os.path.join(self.report_folder, dir))
        G = pickle.load(open(graph_path, 'rb'))
        nx.drawing.nx_pydot.to_pydot(G).write_png(os.path.join(
            self.report_folder, dir, 'graph.png'))
        shutil.move(buggy_onnx_path, os.path.join(
            self.report_folder, dir, 'model.onnx'))
        if buggy_torch_path is not None:
            shutil.move(buggy_torch_path, os.path.join(
                self.report_folder, dir, 'model.pt'))
        shutil.move(stdout, os.path.join(
            self.report_folder, dir, 'stdout.log'))
        shutil.move(stderr, os.path.join(
            self.report_folder, dir, 'stderr.log'))
        for i in Path('.').glob('mlist.*'):
            shutil.move(str(i), os.path.join(
                self.report_folder, dir, i.stem))

        with open(os.path.join(self.report_folder, dir, 'err.txt'), 'w') as f:
            f.write(message)

        if sat_inputs is not None:
            pickle.dump(sat_inputs, open(os.path.join(
                self.report_folder, dir, 'sat_inputs.pkl'), 'wb'))

        self.n_bug += 1

    def simple_bug_report(self, buggy_onnx_path, oracle_path=None, message=''):
        dir = os.path.join(self.report_folder, f'{self.n_bug}-simple')
        os.mkdir(dir)
        shutil.move(buggy_onnx_path, os.path.join(dir, 'model.onnx'))
        if oracle_path is not None:
            shutil.move(oracle_path, os.path.join(dir, 'oracle.pkl'))

        if message:
            with open(os.path.join(dir, 'err.txt'), 'w') as f:
                f.write(message)

        self.n_bug += 1

    def record_coverage(self, fuzz):
        self.record_coverage_cnt += 1
        with open(os.path.join(self.report_folder, _COV_BY_TIME_NAME_), 'a') as f:
            f.write(
                f'{time.perf_counter() - self.start_time :.2f},{__COV_DRIVER__.get_now()}\n')


class CustomProgress(Progress):
    def __init__(self, fuzz_status, columns: List[Union[str, ProgressColumn]], disable=False):
        self.fuzz_status = fuzz_status
        super().__init__(*columns, disable=disable)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        for i in self.fuzz_status():
            yield i
        table = self.make_tasks_table(self.tasks)
        yield table


class FuzzingLoop:  # TODO: Support multiple backends.
    def __init__(self, backends: Dict[str, DiffTestBackend], mode='random', root=None, time_budget=60 * 60 * 4, max_nodes=10, inp_gen='sampling',
                 _PER_MODEL_TIMEOUT_=1000, use_bitvec=False, limnf=True, use_cuda=False, yes=False):
        self.root = root
        self.reporter = Reporter(
            report_folder=root, name_hint=list(backends.keys())[0], yes=yes)
        self.mode = mode  # `random` or `guided` or 'hybrid'
        self.inp_gen = inp_gen  # `random` or `grad`

        SanityCheck.gt(len(backends), 0, "Empty backends are not allowed!")
        self.backends = backends

        self.time_budget = time_budget
        self.max_nodes = max_nodes

        self.start_time = time.time()

        self.cur_seed = 'N/A'
        self.cur_node_size = 'N/A'

        self._PER_MODEL_TIMEOUT_ = _PER_MODEL_TIMEOUT_  # milliseconds

        self.rich_profile = {
            'succ_gen': np.array([]),
            'bad_gen': np.array([]),
            'eval': np.array([]),
            'inp_gen': np.array([]),
        }

        self.use_bitvec = use_bitvec
        self.limnf = limnf
        self.use_cuda = use_cuda

        rich.print(
            f'[bold yellow]To exit the program: `kill {os.getpid()}`[/bold yellow]')
        rich.print(
            '[grey]This is because we use z3 written in C++ w/ Python wrappers. Ctrl+C may not stop it.')

    def rich(self):
        def make_panel(title, time_arr: np.ndarray):
            if len(time_arr) == 0:
                return None

            title_map = {
                'succ_gen': 'Successful NN Generation Time',
                'bad_gen': 'Failed NN Generation Time',
                'eval': 'Evaluation Time',
                'inp_gen': 'Input Generation Time',
            }

            return Panel.fit(f'[green]Fast: {time_arr.min():.3f}s[/green]|'
                             f'[bold]Last: {time_arr[-1]:.3f}s[/bold]\n'
                             f'[red]Slow: {time_arr.max():.3f}s[/red]|'
                             f'[red]Avg: {time_arr.mean():.3f}s',
                             title=title_map[title] if title in title_map else title)

        panels = [
            Panel.fit(
                f'{datetime.timedelta(seconds=round(time.time()-self.start_time))} ~ '
                f'{datetime.timedelta(seconds=self.time_budget)}'
                f'\ncur seed: {self.cur_seed}'
                f'\ncur node size: {self.cur_node_size}'
                f'\nmax node size: {self.max_nodes}',
                title="Time Left ~ Total Time"),
            Panel.fit(f'{self.reporter.n_bug}/{len(self.rich_profile["succ_gen"])}',
                      title="Bug/Iter", style="magenta", width=16)
        ]

        for k in self.rich_profile:
            p = make_panel(k, self.rich_profile[k])
            if p is not None:
                panels.append(p)

        if len(self.rich_profile["succ_gen"]) + len(self.rich_profile["bad_gen"]) == 0:
            gen_succ_rate = 0
        else:
            gen_succ_rate = len(self.rich_profile["succ_gen"]) / (
                len(self.rich_profile["succ_gen"]) + len(self.rich_profile["bad_gen"]))
        panels.append(
            f'[green]Generation Succ. Rate = {gen_succ_rate * 100 :.1f}%[/green]')

        # split panels by 3
        cols = [Columns(panels[i:i + 3]) for i in range(0, len(panels), 3)]

        return cols

    def testcase_gen(self, path, seed):
        if self.mode == 'hybrid':
            mode = random.choice(['random', 'guided'])
        else:
            mode = self.mode

        torch.manual_seed(seed)
        gen, solution = random_model_gen(
            # Only rank useful. Dim sizes means nothing.
            min_dims=[1, 3, 48, 48],
            seed=seed, max_nodes=self.max_nodes,
            mode=mode)
        self.cur_node_size = gen.num_op()

        net = SymbolNet(gen.abstract_graph, solution,
                        verbose=False, alive_shapes=gen.alive_shapes)

        with torch.no_grad():
            net.eval()
            if self.use_cuda:
                net.cuda()

            # export inputs & outputs.
            test_inputs = net.get_random_inps(use_cuda=self.use_cuda)
            outputs = net(*test_inputs)

            if not AbsOpBase.numeric_valid(outputs) and self.inp_gen is not None:
                # use sampling.
                inpgen_tstart = time.time()
                net.check_intermediate_numeric = True
                for _ in range(3):
                    # renew weights.
                    for name, param in net.named_parameters():
                        param.copy_(random_tensor(
                            param.shape, dtype=param.dtype, use_cuda=self.use_cuda))
                    # renew inputs
                    test_inputs = net.get_random_inps(
                        use_cuda=self.use_cuda)
                    outputs = net(*test_inputs)

                    if AbsOpBase.numeric_valid(outputs):
                        break
                self.rich_profile['inp_gen'] = np.append(
                    self.rich_profile['inp_gen'], [time.time() - inpgen_tstart])

            inames, onames = torch2onnx(
                net, path, verbose=False, use_cuda=self.use_cuda)

            inputs = [t.cpu().numpy() for t in test_inputs]
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs.cpu().numpy()]
            else:
                outputs = [o.cpu().numpy() for o in outputs]

            return {ina: inp for ina, inp in zip(inames, inputs)}, {ona: out for ona, out in zip(onames, outputs)}

    def difftest(self, onnx_model, oracle_path, redirect_log=None):
        if redirect_log is not None:
            sys.stdout = open(redirect_log, "w")
            sys.stderr = open(redirect_log, "w")

        # get test case (pickle)
        with open(oracle_path, 'rb') as f:
            inputs, outputs = pickle.load(f)

        for bname in self.backends:
            results = self.backends[bname].predict(onnx_model, inputs)
            assert_allclose(
                outputs,
                results,
                'torch',
                bname,
                nan_as_err=False,
                safe_mode=True
            )

    def fuzz(self):
        start_time = time.time()

        onnx_path = f'tmp_{uuid.uuid4()}.onnx'
        oracle_path = f'tmp_{uuid.uuid4()}.oracle.pkl'
        log_path = f'tmp_{uuid.uuid4()}.log'

        try:
            with CustomProgress(
                fuzz_status=self.rich,
                columns=[
                    "[progress.description]{task.description}",
                    BarColumn(),
                    '[progress.percentage]{task.completed:>3.0f}/{task.total}',
                    '[progress.percentage]{task.percentage:>3.0f}%']
            ) as progress:
                task_fuzz = progress.add_task(
                    '[green]Fuzzing time.', total=self.time_budget)
                all_tstart = time.time()
                while time.time() - start_time < self.time_budget:
                    # =================================
                    # Testcase generation phase
                    gen_tstart = time.time()
                    try:
                        with warnings.catch_warnings():  # just shutup.
                            warnings.simplefilter("ignore")
                            self.cur_seed = seed = random.getrandbits(32)
                            inps, outs = self.testcase_gen(onnx_path, seed)
                            with open(oracle_path, 'wb') as f:
                                pickle.dump((inps, outs), f)  # store oracle.
                            self.rich_profile['succ_gen'] = np.append(
                                self.rich_profile['succ_gen'], [time.time() - gen_tstart])
                    except Exception as e:
                        print(f'Fail when seed={seed}')
                        print(e)  # Skip a few errors.
                        self.rich_profile['bad_gen'] = np.append(
                            self.rich_profile['bad_gen'], [time.time() - gen_tstart])
                        progress.update(
                            task_fuzz, completed=time.time() - all_tstart)
                        continue
                    # =================================

                    # =================================
                    # Model evaluation phase
                    eval_tstart = time.time()
                    p = Process(target=self.difftest,
                                args=(onnx_path, oracle_path, log_path))
                    p.start()
                    p.join()
                    self.rich_profile['eval'] = np.append(
                        self.rich_profile['eval'], [time.time() - eval_tstart])
                    # =================================

                    if p.exitcode != 0:
                        # failed... report this.
                        self.reporter.simple_bug_report(
                            buggy_onnx_path=onnx_path,
                            oracle_path=oracle_path,
                            message=open(log_path).read(),
                        )

                    progress.update(
                        task_fuzz, completed=time.time() - all_tstart)
        finally:
            # clean up.
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                os.remove(oracle_path)
                os.remove(log_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./fuzz_report')
    parser.add_argument('--time_budget', type=int, default=60 * 60 * 4)
    parser.add_argument('--backend', type=str, default='tvm')
    parser.add_argument('--mode', type=str, default='random')
    parser.add_argument(
        '--skip', help='Node types to skip. Split by `,`. By default a blacklist for each backend is also appended.', type=str)
    parser.add_argument('--inp_gen', type=str,
                        help='default is None. Can be sampling.')
    parser.add_argument('--gen_timeout', type=int,
                        default=1000, help='in milliseconds')
    parser.add_argument('--use_bitvec', action='store_true')
    parser.set_defaults(limnf=True)
    parser.add_argument('--no_limnf', dest='limnf', action='store_false',
                        help='Disable the limit on the number of floats')
    parser.add_argument('--limnf', dest='limnf', action='store_true',
                        help='Enable the limit on the number of floats')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('-y', action='store_true', help='Yes to all')
    parser.add_argument('--max_nodes', type=int, default=10)
    args = parser.parse_args()

    backends = None
    if args.backend == 'tvm':
        from nnsmith.backends.tvm_graph import TVMExecutor
        backends = {'tvm-opt': TVMExecutor(opt_level=4),
                    'tvm-debug': TVMExecutor(opt_level=0)}
        __COV_DRIVER__ = TVMExecutor.coverage_install()
    elif args.backend == 'tvm-cuda':
        from nnsmith.backends.tvm_graph import TVMExecutor
        backends = {'tvm-opt': TVMExecutor(opt_level=4, target="cuda"),
                    'tvm-debug': TVMExecutor(opt_level=0)}
        __COV_DRIVER__ = TVMExecutor.coverage_install()
    elif args.backend == 'ort':
        from nnsmith.backends.ort_graph import ORTExecutor
        backends = {'ort-opt': ORTExecutor(opt_level=3),
                    'ort-debug': ORTExecutor(opt_level=0)}
        __COV_DRIVER__ = ORTExecutor.coverage_install()
    elif args.backend == 'trt':
        from nnsmith.backends.trt_graph import TRTBackend
        from nnsmith.backends.tch_graph import TchExecutor
        backends = {'trt-opt': TRTBackend(),
                    'tch-debug': TchExecutor(opt_level=0, dev='cpu')}
        __COV_DRIVER__ = TRTBackend.coverage_install()
    elif args.backend == 'tch':
        from nnsmith.backends.tch_graph import TchExecutor
        backends = {'tch-opt': TchExecutor(dev='cuda'),
                    'tch-debug': TchExecutor(opt_level=0, dev='cpu')}
        __COV_DRIVER__ = TchExecutor.coverage_install()
    elif args.backend == 'dummy':  # for debugging
        backends = {'dummy': DummyExecutor()}
        __COV_DRIVER__ = DummyExecutor.coverage_install()
    else:
        raise NotImplementedError("Other backends not supported yet.")
    skip = 'backend:' + args.backend
    if args.skip is not None:
        skip += ',' + args.skip
    auto_infer_in_dtypes()  # TODO: remove this someday
    if not args.backend.startswith('tvm'):
        cache_file = f'config/fuzz_{list(backends.keys())[0]}_op_dtype.json'

        def run():
            rewrite_op_dtype(
                ALL_OP_TYPES,
                backend=list(backends.values())[0],
                cache=cache_file,
                print_failures=True)
        if not Path(cache_file).exists():
            Path('config').mkdir(exist_ok=True)
            print('Warning: Op dtypes config file does not exist. '
                  'Inferring op dtype for the first run...')
            p = Process(target=run)
            p.start()
            p.join()
            assert p.exitcode == 0, 'Failed to infer op dtypes'
        print('Reading cache config file:', cache_file)
        run()
    config_skip_op(skip)
    fuzzing_loop = FuzzingLoop(
        root=args.root,
        backends=backends,
        mode=args.mode,
        time_budget=args.time_budget,
        inp_gen=args.inp_gen,
        _PER_MODEL_TIMEOUT_=args.gen_timeout,
        use_bitvec=args.use_bitvec,
        limnf=args.limnf,
        use_cuda=args.use_cuda,
        yes=args.y,
        max_nodes=args.max_nodes,
    )
    fuzzing_loop.fuzz()
