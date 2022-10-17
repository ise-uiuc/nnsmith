#!/bin/bash
export NNSMITH_DCE=0.1
# TVM.
export LIB_PATH='../tvm/build/libtvm.so ../tvm/build/libtvm_runtime.so'
start_time=`date +%s`
python nnsmith/fuzz.py --time 14400 --max_nodes 10 --eval_freq 256 \
                --mode random --backend tvm --root nnsmith-tvm-base
exp0_t=$(expr `date +%s` - $start_time)

start_time=`date +%s`
python nnsmith/fuzz.py --time 14400 --max_nodes 10 --eval_freq 256 \
                --mode guided --backend tvm --root nnsmith-tvm-guided
exp1_t=$(expr `date +%s` - $start_time)

# ONNXRuntime.
export LIB_PATH='../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime_providers_shared.so ../onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so'
start_time=`date +%s`
python nnsmith/fuzz.py --time 14400 --max_nodes 10 --eval_freq 256 \
                --mode random --backend ort --root nnsmith-ort-base
exp2_t=$(expr `date +%s` - $start_time)

start_time=`date +%s`
python nnsmith/fuzz.py --time 14400 --max_nodes 10 --eval_freq 256 \
                --mode guided --backend ort --root nnsmith-ort-guided
exp3_t=$(expr `date +%s` - $start_time)

echo "Experiment time of last 4 runs: '$exp0_t','$exp1_t','$exp2_t','$exp3_t' seconds."
