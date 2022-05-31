import glob
from multiprocessing import Process
import subprocess
from pathlib import Path
import pickle
import cloudpickle
import sys
import time
import os
import uuid
import datetime
import random
import shutil
from typing import Iterable, Union, List
import socket
import warnings
import traceback

import git
import torch
import numpy as np
import networkx as nx

import rich
from rich.progress import Progress, BarColumn, ProgressColumn
from rich.panel import Panel
from rich.console import RenderableType
from rich.columns import Columns

from nnsmith.abstract.op import ALL_OP_TYPES, auto_infer_in_dtypes, config_skip_op
from nnsmith.error import NNSmithInternalError
from nnsmith.difftest import assert_allclose
from nnsmith.graph_gen import random_model_gen, SymbolNet
from nnsmith.dtype_test import rewrite_op_dtype
from nnsmith.input_gen import PracticalHybridSearch
from nnsmith.export import torch2onnx


_METADATA_NAME_ = 'meta.txt'

# NOTE: Currently only engineered for TVM.


def locate_crash_testcase(batch_path):
    os.path.basename
    idx = [int(Path(i).stem)
           for i in glob.glob(os.path.join(batch_path, '*.onnx'))]
    idx = sorted(idx)[0]  # get the first one
    return (os.path.join(batch_path, f'{idx}.onnx'), os.path.join(batch_path, f'{idx}.pkl'))


def simple_bug_report(report_folder, buggy_onnx_path, oracle_path=None, message='', bug_type='unknown'):
    n_bug = len(glob.glob(os.path.join(report_folder, 'bug-*')))
    dir = os.path.join(report_folder, f'bug-{bug_type}-#{n_bug}')
    os.mkdir(dir)
    shutil.move(buggy_onnx_path, os.path.join(dir, 'model.onnx'))
    if oracle_path is not None:
        shutil.move(oracle_path, os.path.join(dir, 'oracle.pkl'))
    if message:
        with open(os.path.join(dir, 'err.txt'), 'w') as f:
            f.write(message)
    graph_path = buggy_onnx_path[:-len('.onnx')] + '-graph.pkl'
    G = pickle.load(open(graph_path, 'rb'))
    nx.drawing.nx_pydot.to_pydot(G).write_png(os.path.join(
        dir, 'graph.png'))

# TODO: simplify or delete Reporter. Currently using the above funtcton for reporting bugs.


class Reporter:  # From Tzer.
    def __init__(self, report_folder=None, lib_path=None, name_hint='', yes=False) -> None:
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

        self.lib_path = os.environ.get('LIB_PATH', lib_path)
        if self.lib_path is not None:  # Enabling coverage tracing
            print(f'Using `{self.lib_path}` as the lib path')
            self.lib_expr = ''
            for lib in self.lib_path.split():
                assert os.path.exists(lib), f'{lib} does not exist!'
                self.lib_expr += f' -object {os.path.realpath(lib)} '

            LLVM_MIN = 9
            LLVM_MAX = 14
            self.llvm_version = None
            for i in range(LLVM_MAX, LLVM_MIN - 1, -1):
                if 0 == os.system(f'which llvm-cov-{i}'):
                    self.llvm_version = i
                    break
            if self.llvm_version is None:
                assert 0 == os.system(f'which llvm-cov')
            self.lcov_config = open(os.path.join(
                self.report_folder, 'stats.csv'), 'w')

            print(f'LLVM VERSION: {self.llvm_version}')

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

    def handle_profraw(self, profraw_path, n_models, time_spent):
        # Use env var: `LIB_PATH` to indicate the lib paths for tracing.
        if self.lib_path is None:
            return

        # write
        lcov_path = profraw_path.replace('.profraw', '.lcov')
        profdata_path = profraw_path.replace('.profraw', '.profdata')

        if os.path.exists(profraw_path):
            llvm_profdata = 'llvm-profdata'
            llvm_cov = 'llvm-cov'
            if self.llvm_version:
                llvm_profdata += f'-{self.llvm_version}'
                llvm_cov += f'-{self.llvm_version}'

            # summary might be useless as it does not consider prior runs.
            if 0 != os.system(f'{llvm_profdata} merge -sparse {profraw_path} -o {profdata_path}') or \
                    0 != os.system(f'{llvm_cov} export -instr-profile={profdata_path} -format=lcov {self.lib_expr} > {lcov_path}'):
                print(f'Getting coverage failed!!', file=sys.stderr)
            else:  # clean temporary files
                assert 0 == os.system(
                    f'lz4 {lcov_path} {lcov_path}.lz4')

                self.lcov_config.write(
                    f'{time_spent},{-1},{n_models},{os.path.basename(lcov_path)}\n')
                self.lcov_config.flush()

                os.remove(profraw_path)
                os.remove(profdata_path)
                os.remove(lcov_path)
        else:
            print(f'{profraw_path} does not exist...', file=sys.stderr)


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
    def __init__(self, backend: str, mode='random', inp_gen='sampling', root=None,
                 time_budget=60 * 60 * 4, max_nodes=10, eval_freq=1, use_bitvec=False,
                 limnf=True, use_cuda=False, yes=False, no_progress=False):
        self.root = root
        self.reporter = Reporter(
            report_folder=root, name_hint=backend, yes=yes)
        self.mode = mode  # `random` or `guided` or 'hybrid'
        self.inp_gen = inp_gen  # `random` or `grad`

        self.backend = backend

        self.time_budget = time_budget
        self.max_nodes = max_nodes

        if eval_freq > 1:
            self.eval_batch = []  # max_size <- eval_freq
            self.batch_path = os.path.join(root, 'batch')
            os.mkdir(self.batch_path)
            self.executed_batches = 0

        self.eval_freq = eval_freq

        self.start_time = time.time()

        self.cur_seed = 'N/A'
        self.cur_node_size = 'N/A'

        self.rich_profile = {
            'succ_gen': np.array([]),
            'bad_gen': np.array([]),
            'eval': np.array([]),
            'inp_gen': np.array([]),
            'inp_sat': np.array([], dtype=np.bool8),
        }

        self.use_bitvec = use_bitvec
        self.limnf = limnf
        self.use_cuda = use_cuda
        self.no_progress = no_progress
        self.n_bug = 0

        rich.print(
            f'[bold yellow]To exit the program: `kill {os.getpid()}`[/bold yellow]')
        rich.print(
            '[grey]This is because we use z3 written in C++ w/ Python wrappers. Ctrl+C may not stop it.')

    def rich(self):
        def make_panel(title, time_arr: np.ndarray):
            if len(time_arr) == 0:
                return None

            time_title_map = {
                'succ_gen': 'Successful NN Generation Time',
                'bad_gen': 'Failed NN Generation Time',
                'eval': 'Evaluation Time',
                'inp_gen': 'Input Generation Time',
            }

            if title not in time_title_map:
                return None

            return Panel.fit(f'[green]Fast: {time_arr.min():.3f}s[/green]|'
                             f'[bold]Last: {time_arr[-1]:.3f}s[/bold]\n'
                             f'[red]Slow: {time_arr.max():.3f}s[/red]|'
                             f'[red]Avg: {time_arr.mean():.3f}s',
                             title=time_title_map[title] if title in time_title_map else title)

        panels = [
            Panel.fit(
                f'{datetime.timedelta(seconds=round(time.time()-self.start_time))} ~ '
                f'{datetime.timedelta(seconds=self.time_budget)}'
                f'\ncur seed: {self.cur_seed}'
                f'\ncur node size: {self.cur_node_size}'
                f'\n#gen {len(self.rich_profile["succ_gen"]) + len(self.rich_profile["bad_gen"])}'
                f'\nmax node size: {self.max_nodes}',
                title="Time Left ~ Total Time"),
            Panel.fit(f'{self.n_bug}/{len(self.rich_profile["succ_gen"])}',
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
            f'[green]NN Gen. Succ. Rate = {gen_succ_rate * 100 :.1f}%[/green]' + '\n' +
            f'[green]Input/Weight Gen. Succ. Rate = {self.rich_profile["inp_sat"].mean() * 100 :.1f}%[/green]')

        # split panels by 3
        cols = [Columns(panels[i:i + 3]) for i in range(0, len(panels), 3)]

        return cols

    def testcase_gen(self, path, seed):
        gen_tstart = time.time()

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

        gen_time = time.time() - gen_tstart

        # Generate Inputs.
        if self.use_cuda:
            net.cuda()

        net.enable_proxy_grad()
        net.eval()  # otherwise BN wants batch > 1
        searcher = PracticalHybridSearch(net)
        n_try, sat_inputs = searcher.search(
            max_time_ms=gen_time * 0.02 * 1000, max_sample=2, return_list=True)
        net.disable_proxy_grad()
        self.rich_profile['inp_gen'] = np.append(self.rich_profile['inp_gen'], [
            time.time() - gen_tstart - gen_time])
        self.rich_profile['inp_sat'] = np.append(
            self.rich_profile['inp_sat'], [sat_inputs is not None])
        # ----------------

        with torch.no_grad():
            net.eval()

            test_inputs = sat_inputs if sat_inputs else net.get_random_inps(
                use_cuda=self.use_cuda)

            outputs = net.forward(*test_inputs)

            inames, onames, oidx = torch2onnx(
                net, path, verbose=False, use_cuda=self.use_cuda, dummy_inputs=test_inputs)

            inputs = [t.cpu().numpy() for t in test_inputs]

            if isinstance(outputs, torch.Tensor):
                outputs = [outputs.cpu().numpy()]
            else:
                outputs = [o.cpu().numpy() for o in outputs]
        net.to_picklable()
        cloudpickle.dump(net.concrete_graph, open(
            path + '-graph.pkl', 'wb'), protocol=4)
        self.rich_profile['succ_gen'] = np.append(
            self.rich_profile['succ_gen'], [gen_time])
        return {ina: inp for ina, inp in zip(inames, inputs)}, {onames[i]: outputs[i] for i in oidx}

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
                results,
                outputs,
                bname,
                'torch',
                safe_mode=True
            )

    def batch_add(self, onnx_path, oracle_path, force=False):
        target_onnx = os.path.join(
            self.batch_path, f'{len(self.eval_batch)}.onnx')
        target_oracle = os.path.join(
            self.batch_path, f'{len(self.eval_batch)}.pkl')
        target_graph = os.path.join(
            self.batch_path, f'{len(self.eval_batch)}-graph.pkl')

        graph_path = onnx_path + '-graph.pkl'
        shutil.move(onnx_path, target_onnx)
        shutil.move(oracle_path, target_oracle)
        shutil.move(graph_path, target_graph)
        # TODO: consider adding mlist.*.param (they will be generated for large models)
        self.eval_batch.append(target_onnx)

        if (len(self.eval_batch) == self.eval_freq or force) and len(self.eval_batch) > 0:
            # Execute batch evaluation
            copied_env = os.environ.copy()
            # Path to store llvm profile.
            profraw_path = os.path.join(
                self.root, f'{self.executed_batches}.profraw')
            copied_env["LLVM_PROFILE_FILE"] = str(profraw_path)

            arguments = [
                'python', 'experiments/batch_eval.py',
                '--models', *self.eval_batch,
                '--backend', self.backend,
                '--fuzz_max_nodes', str(self.max_nodes),
                '--fuzz_seed', str(self.cur_seed),
                '--fuzz_report_folder', self.root,
                '--clean_after_eval',  # remove tested files in batch folder.
            ]
            print(f'Starting batch evaluation: {arguments}')
            p = subprocess.Popen(
                arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=copied_env)
            _, errs = p.communicate()  # wait for completion
            if p.returncode != 0:
                print(
                    f'A batch process exits with non-zero error code: {errs.decode("utf-8")}')
                onnx_path, oracle_path = locate_crash_testcase(self.batch_path)
                msg = "===== stdout =====\n" + \
                    _.decode("utf-8") + "\n\n===== stderr =====\n" + \
                    errs.decode("utf-8")
                simple_bug_report(
                    self.root, onnx_path, oracle_path, message=msg, bug_type='Crash')
            self.n_bug = len(glob.glob(os.path.join(
                self.reporter.report_folder, 'bug-*')))

            self.reporter.handle_profraw(
                profraw_path=profraw_path,
                n_models=len(self.eval_batch),
                time_spent=time.time() - self.last_eval_time)

            # clean folder.
            self.eval_batch = []

            self.executed_batches += 1
            self.last_eval_time = time.time()

    def fuzz(self):
        start_time = time.time()
        self.last_eval_time = time.time()

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
                    '[progress.percentage]{task.percentage:>3.0f}%'],
                disable=self.no_progress
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
                    except Exception as e:
                        print(f'Fail when seed={seed}')
                        print(e)  # Skip a few errors.
                        traceback.print_exc()
                        self.rich_profile['bad_gen'] = np.append(
                            self.rich_profile['bad_gen'], [time.time() - gen_tstart])
                        progress.update(
                            task_fuzz, completed=time.time() - all_tstart)
                        continue
                    # =================================

                    # =================================
                    # Model evaluation phase
                    if self.eval_freq == 1:
                        raise NotImplementedError(
                            'For now use --eval_freq with a value > 1')
                        eval_tstart = time.time()
                        p = Process(target=self.difftest,
                                    args=(onnx_path, oracle_path, log_path))
                        p.start()
                        p.join()
                        self.rich_profile['eval'] = np.append(
                            self.rich_profile['eval'], [time.time() - eval_tstart])

                        if p.exitcode != 0:
                            # failed... report this.
                            to_repro = f'python nnsmith/graph_gen.py --max_nodes {self.max_nodes} --seed {seed} --viz_graph'
                            self.reporter.simple_bug_report(
                                buggy_onnx_path=onnx_path,
                                oracle_path=oracle_path,
                                message=to_repro + '\n' +
                                open(log_path).read(),
                            )
                    else:  # batched execution
                        self.batch_add(onnx_path, oracle_path)

                    # =================================

                    progress.update(
                        task_fuzz, completed=time.time() - all_tstart)
        finally:
            # clean up.
            if self.eval_freq > 1:  # last execution
                self.batch_add(onnx_path, oracle_path, force=True)

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
    parser.add_argument('--eval_freq', type=int,
                        default=1, help='(EVALUATION ONLY) for batch processing')
    parser.add_argument('--use_bitvec', action='store_true')
    parser.set_defaults(limnf=True)
    parser.add_argument('--no_limnf', dest='limnf', action='store_false',
                        help='Disable the limit on the number of floats')
    parser.add_argument('--limnf', dest='limnf', action='store_true',
                        help='Enable the limit on the number of floats')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('-y', action='store_true', help='Yes to all')
    parser.add_argument('--max_nodes', type=int, default=10)
    parser.add_argument('--no_progress', action='store_true')
    args = parser.parse_args()

    skip = 'backend:' + args.backend
    if args.skip is not None:
        skip += ',' + args.skip
    auto_infer_in_dtypes()  # TODO: remove this someday
    if not args.backend.startswith('tvm'):
        cache_file = f'config/fuzz_{args.backend}_op_dtype.pkl'

        def run():
            rewrite_op_dtype(
                ALL_OP_TYPES,
                backend=args.backend,
                cache=cache_file,
                print_failures=True)
        if not Path(cache_file).exists():
            Path('config').mkdir(exist_ok=True)
            print(f'Warning: Op dtypes config file `{cache_file}` does not exist. '
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
        backend=args.backend,
        mode=args.mode,
        time_budget=args.time_budget,
        inp_gen=args.inp_gen,
        use_bitvec=args.use_bitvec,
        limnf=args.limnf,
        eval_freq=args.eval_freq,
        use_cuda=args.use_cuda,
        yes=args.y,
        max_nodes=args.max_nodes,
        no_progress=args.no_progress
    )
    fuzzing_loop.fuzz()
