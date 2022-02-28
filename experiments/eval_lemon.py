"""This file is used to mock the execution of LEMON and record its coverage.
To evaluate LEMON's coverage efficiency, there are several steps here:
1. Run LEMON to generate models (https://github.com/ganler/LEMON);
2. Run this script by setting `model_dir` to the output model folder of LEMON;

In step 2, we record the file creation time of each model and compute the diff
of ranked creation time to mimic the generation time. Then, we run it on TVM and
further apply execution time on top of the generation time to get the evaluation
results.
"""

import os
import shutil
from time import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import tvm
from tvm.contrib import coverage
import tvm.relay.testing.tf as tf_testing
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay.build_module import bind_params_by_name


tf.executing_eagerly()


def keras2tf(model):
    full_model = tf.function(lambda x: model(x))
    freeze_shape = model.inputs[0].shape

    shape_list = []
    for v in freeze_shape:
        try:
            shape_list.append(int(v))
        except TypeError as e:
            shape_list.append(1)

    full_model = full_model.get_concrete_function(
        tf.TensorSpec(tf.TensorShape(shape_list), model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)

    # print(frozen_func.graph.as_graph_def())
    return frozen_func.graph.as_graph_def()


def run_model(model_name):
    model = keras.models.load_model(model_name)
    shape_list = []
    for v in model.inputs[0].shape:
        try:
            shape_list.append(int(v))
        except TypeError as e:
            shape_list.append(1)

    print(model.inputs[0].name)
    print(model.inputs[0].shape)

    # TVM:
    graph_def = keras2tf(model)
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # convert relay ir to tir
    def run_tvm(opt_level):
        module, params = relay.frontend.from_tensorflow(graph_def)

        # compile the model
        target = tvm.target.Target("llvm")

        if params is not None:
            module = IRModule.from_expr(
                bind_params_by_name(module["main"], params))

        with tvm.transform.PassContext(opt_level=opt_level):
            module, params = relay.optimize(
                module, target=target, params=params)

            module = relay.transform.InferType()(module)
            module = relay.transform.DynamicToStatic()(module)

            executor = relay.build_module.create_executor(
                'graph', module, tvm.cpu(), 'llvm', params
            ).evaluate()

            return executor(
                **{'x': np.zeros(shape_list, dtype='float32')})

    run_tvm(0)
    run_tvm(4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the folder.')
    parser.add_argument('--report_folder', type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.report_folder):
        # TODO: Allow continous fuzzing...
        decision = ''
        while decision.lower() not in ['y', 'n']:
            decision = input(
                'Report folder already exists. Press [Y/N] to continue or exit...')
        if decision.lower() == 'n':
            raise RuntimeError(
                f'{args.report_folder} already exist... We want an empty folder to report...')
        else:
            shutil.rmtree(args.report_folder)

    os.mkdir(args.report_folder)

    cov_file = open(args.out_dir + '/cov_by_time.csv', 'w')

    time_list = []
    file_list = []
    for file in Path(args.model_dir).rglob('mut_model/*.h5'):
        file_list.append(file)
        time_list.append(file.stat().st_mtime)

    time_arr = np.array(time_list)
    time_arr -= time_arr.min()
    idx = time_arr.argsort()

    print(time_arr[idx])

    time_stamp = 0.
    cov_file.write(f'{time_stamp},{coverage.get_now()}\n')
    for i in tqdm(range(len(file_list))):
        if i != 0:
            diff = (time_arr[idx[i]] - time_arr[idx[i - 1]])
            assert diff > 0, f'{diff} <= 0 in {file_list[idx[i]]}'
            time_stamp += diff

        file = file_list[idx[i]]

        time_begin = time()
        try:
            run_model(file)
        except Exception as e:
            print(e)
        time_end = time()

        time_stamp += time_end - time_begin

        cov_file.write(f'{time_stamp},{coverage.get_now()}\n')
        cov_file.flush()
        print(f'{time_stamp},{coverage.get_now()}\n')
