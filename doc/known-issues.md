## Incompatibility of TensorFlow-GPU over fork-based crash safty

`fuzz.crash_safe=true` allows running compilation & execution in a forked process as a sandbox to catch crash and timeout. However, CUDA runtime is not compatible with fork. In tensorflow, the symptom is crash in forked subprocess:

```txt
F tensorflow/stream_executor/cuda/cuda_driver.cc:219] Failed setting context: CUDA_ERROR_NOT_INITIALIZED: initialization error
```

- For `tflite` it's okay as it does not require GPU and `nnsmith.fuzz` will directly set `CUDA_VISIBLE_DEVICES=-1` in the beginning;
- For `xla` it's a bit headache, currently we need to manually specify `fuzz.crash_safe=false` for fuzzing and allow it to crash;
- We are tracking this [issue](https://github.com/tensorflow/tensorflow/issues/57877) in TensorFlow. We are likely to fix this by executing a TensorFlow model in a seperated process if it cannot be resolved in the near future.
