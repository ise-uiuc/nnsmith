from typing import List, Union, Dict, Tuple
import numpy as np
from numpy import testing

from nnsmith.error import *
import pickle
from pathlib import Path
from nnsmith.backend_executor import BackendCreator


def assert_allclose(obtained: Dict[str, np.ndarray], desired: Dict[str, np.ndarray], obtained_name: str, oracle_name: str, nan_as_err=True):
    err_msg = ''
    if obtained is None:
        err_msg += f'{obtained_name} crashed'
    if desired is None:
        err_msg += f'{oracle_name} crashed'
    if err_msg != '':
        raise CrashError(err_msg)

    if set(obtained.keys()) != set(desired.keys()):
        print('Key sets differ')
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} have different output tensor names')

    if nan_as_err:
        for index, key in enumerate(obtained):
            err_msg = ''
            if np.isnan(obtained[key]).any():
                err_msg += f'{obtained_name} has NaN, '
            if np.isnan(desired[key]).any():
                err_msg += f'{oracle_name} has NaN'
            if err_msg != '':
                err_msg = f'At tensor #{index} named {key}: ' + err_msg
                # print(err_msg) # Mute.
                raise NaNError(err_msg)

    try:
        index = -1
        SanityCheck.eq(set(obtained.keys()), set(desired.keys()))
        index = 0
        for key in obtained:
            testing.assert_allclose(
                obtained[key], desired[key], rtol=1e-02, atol=1e-03)
            index += 1
    except AssertionError as err:
        # print(err) # Mute.
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor named {key}: {str(err)}')


def known_bug(report: dict, db: List[dict]):
    def same_model_same_stderr(report: dict):
        for last in reversed(db):
            if (report['backend'] == last['backend'] and report['model_idx'] == last['model_idx'] and
                    report['stderr'] == last['stderr']):
                return True
            # NOTE: when the input is deliverately skipped (because it already crashed on the same model but different input), I didn't create stderr/stdout.
            if report['stderr'] == 'file not found' and report['stdout'] == 'file not found':
                return True
        return False

    def trt_clip_int32(report: dict):
        return (report['backend'] == BackendCreator.NAME_MAP['trt'] and
                'INVALID_NODE: Invalid Node - Clip' in report['stdout'])

    def tvm_int_mismatch(report: dict):
        return (report['backend'].startwith('tvm') and
                'TypeError: mismatched types. int64 vs. int32' in report['stdout'])

    def tvm_layout_argmin(report: dict):
        return (report['backend'].startwith('tvm') and
                'Invalid layout' in report['stdout'] and '(exist_axis[axis' in report['stdout'])

    filters = [same_model_same_stderr, trt_clip_int32,
               tvm_int_mismatch, tvm_layout_argmin]
    for f in filters:
        if f(report):
            return True
    return False


def unsupported_feature(report: dict, db: List[dict]):
    def trt_round(report: dict):
        return (report['backend'] == BackendCreator.NAME_MAP['trt'] and
                'getPluginCreator could not find plugin: Round version: 1' in report['stderr'])

    def trt_general(report: dict):
        return (report['backend'] == BackendCreator.NAME_MAP['trt'] and
                'UNSUPPORTED_NODE' in report['stdout'])

    def xla_squeeze(report: dict):
        return (report['backend'] == BackendCreator.NAME_MAP['xla'] and
                'BackendIsNotSupposedToImplementIt: Squeeze version 13 is not implemented' in report['stderr'])

    filters = [trt_round, trt_general, xla_squeeze]
    for f in filters:
        if f(report):
            return True
    return False


def difftest(root: str):
    import pandas as pd
    """
    This function compares the outputs of each backend and generate bug reports.
    Note: `run_backend` must run before this function.
    Args:
        output_dir: path to the directory storing the outputs.
    """
    # file structure:
    # root: /path/to/root
    # model_root: ${root}/model_and_input/
    # - all models: ${model_root}/${model_name}/model.onnx
    # - i-th input: ${model_root}/${model_name}/input.${i}.pkl
    # output_dir: ${root}/output
    # - i-th output: ${output_dir}/${model_name}/output.${i}.pkl

    # input and output pickle format:
    # inputs.pkl: Dict[str, np.ndarray]
    # outputs.pkl: {'exit_code': 0, 'outputs': outputs}, where outputs is of type Dict[str, np.ndarray]

    root = Path(root)
    output_dir = root / 'output'
    # numerical consistency check
    report = []
    for model_folder in sorted(output_dir.glob('*/')):  # reproducibility
        model_name = model_folder.name

        def get_meta_info():
            a = list(model_folder.glob(f'*.output.*.pkl'))
            bknd_names = set(map(lambda x: x.name.split('.')[0], a))
            num_out = len(a) // len(bknd_names)
            num_in = len(list(
                (output_dir.parent / 'model_input' /
                 model_name).glob(f'input.*.pkl')))

            SanityCheck.true(num_out == num_in or num_in == 0, 'inputs and outputs are not matched. Do you forget to run_backends?\n'
                             'model_folder: {}\nbknd_names: {}\nnum_out: {}\nlen(a): {}'.format(
                                 model_folder, bknd_names, num_out, len(a)))
            SanityCheck.eq(len(a) % len(bknd_names), 0,
                           f'{model_name} has {len(a)} outputs, but {len(bknd_names)} backends which cannot divide')
            return num_out, bknd_names

        def get_output(backend_name: str, idx: str) -> Tuple[Dict[str, np.ndarray], str]:
            out_path = output_dir / \
                f'{model_name}/{backend_name}.output.{idx}.pkl'
            return pickle.load(out_path.open('rb')), str(out_path)

        # TODO(JK): use more advanced oracle (e.g., clustering?) if this does not work well
        num_out, bknd_names = get_meta_info()
        bknd_names = sorted(bknd_names)  # sort for reproducibility
        for i in range(num_out):
            oracle_path = None
            for backend in bknd_names:
                output_json, out_path = get_output(backend, i)
                output, infer_succ = output_json['outputs'], output_json.get(
                    'infer_succ', None)
                if oracle_path is None:
                    # read oracle's data (assume first backend as oracle)
                    oracle = output
                    oracle_infer_succ = infer_succ
                    oracle_path = out_path
                    oracle_name = Path(out_path).name.split('.')[0]
                    continue
                try:
                    assert_allclose(output, oracle, out_path, oracle_path)
                except ModeledError as err:
                    if isinstance(err, (NaNError, IncorrectResult)) and (infer_succ is False or oracle_infer_succ is False):
                        err = NaNError('Infer domain failed.')
                    item = {
                        'model_idx': model_folder.name,
                        'model_path': str(model_folder / 'model.onnx'),
                        'input_path': str(output_dir.parent / 'model_input' / model_name / f'input.{i}.pkl'),
                        'backend': backend,
                        'oracle': oracle_name,
                        'input_idx': i,
                        'output_backend': out_path,
                        'output_oracle': oracle_path,
                        'error': err,
                        'stdout': open(out_path + '.stdout').read() if Path(out_path + '.stdout').exists() else 'file not found',
                        'stderr': open(out_path + '.stderr').read() if Path(out_path + '.stderr').exists() else 'file not found',
                    }
                    item['known'] = known_bug(item, report)
                    item['unsupported'] = unsupported_feature(item, report)
                    report.append(item)
                    print(err)
    import json
    df = pd.DataFrame(report)
    df.to_pickle(str(root / 'report.pkl'))
    # pickle.dump(report, open(root / 'report.pkl', 'wb'))
    for i in report:
        i['error'] = str(i['error'])
    json.dump(report, open(root / 'report.json', 'w'), indent=2)

    def _cond(): return ((~df.known) & (~df.unsupported) &
                         (~df.error.map(lambda x: isinstance(x, NaNError))))
    if len(df) > 0 and len(df[_cond()]) > 0:
        print(
            f'{len(df[_cond()])} unknown non-unsupported unique differences found!!!')
    else:
        print('No differences found!')


if __name__ == '__main__':  # generate bug reports.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./tmp')
    args = parser.parse_args()
    difftest(args.root)
