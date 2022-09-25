## TensorFlow Lite CUDA Init Error in Crash-safe Mode

If we run `tflite` in the fuzzing loop with `fuzz.crash_safe=true`, you may encounter tons of crashes of:

```txt
F tensorflow/stream_executor/cuda/cuda_driver.cc:219] Failed setting context: CUDA_ERROR_NOT_INITIALIZED: initialization error
```

It is temporarily "fixed" by setting environment variable `CUDA_VISIBLE_DEVICES=-1` if we found cuda is not used in the fuzzing loop. Nevertheless, this should be a TensorFlow bug that needs to be fixed.
