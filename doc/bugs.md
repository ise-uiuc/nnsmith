## Bugs found with NNSmith

> **Annotation**: ✅ means fixed; 🚨 means this bug has been marked with a `high-priority` label (PyTorch)

### Table of Contents

* [**PyTorch**](#pytorch)
* [**PyTorch-ONNX Converter**](#pytorch-onnx-converter)
* [**ONNX**](#onnx)
* [**ONNXRuntime**](#onnxruntime)
* [**TVM**](#tvm)
* [**TensorRT**](#tensorrt)
* [**TensorFlow**](#tensorflow)
* [**Methodology**](#methodology)

### PyTorch

01. ✅ 🚨 [SIGIOT when running model with conv2d and avgpool2d after `optimize_for_inference` · Issue #86535 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86535)

02. ✅ [`optimize_for_inference` leads to wrong results for model with conv2d, max and clip · Issue #86556 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86556)

03. ✅ 🚨 [RuntimeError: could not construct a memory descriptor using a format tag · Issue #86664 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86664)

04. ✅ [[NNPack] Runtime error with padded `Conv1d` and `>=16` batch size · Issue #90142 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90142)

05. ✅ 🚨 [[pt2] `torch.where` gives wrong results with `torch.compile` · Issue #93374 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93374)

06. 🚨 [[pt2] compiled function with cat and mul gives wrong results · Issue #93365 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93365)

07. ✅ [[pt2] cannot compile model with linear layer when the input has rank 1 · Issue #93372 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93372)

08. ✅ [[pt2] `torch._inductor.exc.CppCompileError: C++ compile error` when compiling `neg` and `max` · Issue #93380 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93380)

09. ✅ [[pt2] `torch.compile` produces wrong results for function with `neg` on `uint8` tensor · Issue #93829 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93829)

10. ✅ [[pt2] Cannot compile model with `neg` and `linear` · Issue #93836 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93836)

11. ✅ [`pad` + `gt` produce wrong results in compile mode · Issue #93351 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93351)

12. ✅ [[pt2] `torch._inductor.exc.CppCompileError: C++ compile error` when compiling `argmax` and `min` · Issue #94055 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94055)

13. ✅ [`torch.compile` fails when using `torch.sub` with python constant · Issue #95181 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95181)

14. ✅ [`torch.ge` produces wrong results in compile mode when given int tensors · Issue #95695 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95695)

15. [[JIT] Zero-channel conv2d cannot be applied with `optimize_for_inference` · Issue #91396 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91396)

16. [`min` reduction on float16 tensor failed on certain shapes · Issue #93249 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93249)

17. [`torch.compile` produce wrong result in `interpolate` when `mode=bilinear` · Issue #93262 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93262)

18. [[pt2] compiled model with cat and expand gives wrong results · Issue #93357 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93357)

19. [Adding a linear layer leads to failure of `optimize_for_mobile` · Issue #86667 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86667)

### PyTorch-ONNX Converter

01. ✅ [[ONNX] `f64 * LeakyReLU(f64)` mistakingly returns f32 · Issue #85316 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/85316)

02. ✅ [[ONNX] Converter did not consider the implicit casting specifically for `Max` · Issue #87609 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/87609)

03. ✅ [fix: onnx PReLU unidirectional broadcasting by ganler · Pull Request #70571 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/70571)

04. ✅ [Clip] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)

05. ✅ [Min] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)

06. ✅ [Max] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)

07. ✅ [ReLU] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)

08. ✅ [Pad] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)

09. ✅ [[onnx export] Add broadcast to matmul shape inference by lazycal · Pull Request #70534 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/70534)

10. ✅ [[Bug][ONNX] Specification Inconsistency in Flatten · Issue #74142 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/74142)

11. ✅ [[ONNX] Fix shape inconsistency when exporting scalar log2 by lazycal · Pull Request #78701 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/78701)

12. ✅ [[ONNX Export] Interpolation likely should be exported with `half_pixel` instead of `pytorch_half_pixel` · Issue #79361 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/79361)

### ONNX

01. ✅ [[Bug] Checker misses data type mismatch for Max · Issue #4619 · onnx/onnx · GitHub](https://github.com/onnx/onnx/issues/4619)

### ONNXRuntime

01. ✅ [Crashes when relu is followed by a clip · Issue #9753 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/9753)

02. ✅ [MatMul fusion failed at scalar input · Issue #10950 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/10950)

03. ✅ [GemmTransposeFusion error when C is transposed (`Gemm(A,B,Transpose(C)`), complained with confusing name `_transformed_transformed_transformed...` · Issue #12071 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12071)

04. [[Bug] Mixing negative and positive paddings causes segfault/uninitialized memory values produced in reflected pad · Issue #11828 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/11828)

05. [Runtime Exception when relu is followed by a clip · Issue #10936 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/10936)

06. [Inconsistent result to NumPy and PyTorch when consecutively casting a float tensor to int32 and then to bool · Issue #11994 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/11994)

07. [Wrong output shape due to MergeShape failure · Issue #11870 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/11870)

08. [Wrong Floor output on very large input · Issue #12076 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12076)

09. [Resize with mode linear always produces 0.5 on GPU regardless of the input · Issue #12091 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12091)

10. [Resize with `nearest` mode have inconsistent results compared to PyTorch and TVM · Issue #12098 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12098)

11. [Parameters are optimized out even if it is a needed return value · Issue #13425 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/13425)

### TVM

01. ✅ [[Bug] shape int32-int64 check error in `trilu`'s `te.compute` · Issue #13029 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13029)

02. ✅ [[Bug] `trilu` not tagged with `injective` and thus miss reduce schedule · Issue #13030 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13030)

03. ✅ [[Bug] Wrong results of `cast<int32>( cast<bool>(-1i64) )` · Issue #13048 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13048)

04. ✅ [[BugFix] resolve integer 32. ~ 64. mismatch by casting by ganler · Pull Request #9582 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/9582)

05. ✅ [[onnx] fix onnx where broadcast by lazycal · Pull Request #10106 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10106)

06. ✅ [Fix broadcast InferCorrectLayout by lazycal · Pull Request #10156 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10156)

07. ✅ [[BUGFIX][ARITH] Fix FloorMod Simplifier by lazycal · Pull Request #10336 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10336)

08. ✅ [[BugFix]: select node type error in NarrowDataType pass by ganler · Pull Request #10519 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10519)

09. [[Bug] GetStoreRule failure at simple Conv2d + Squeeze model · Issue #10528 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/10528)

10. ✅ [[Relay][ONNX][Fix] Flatten in OnnxConverter by ganler · Pull Request #10593 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10593)

11. ✅ [NarrowDataType] [[TIR] Fix Ramp int32~64 mismatch in VectorizeLoop and NarrowDataType passes by lazycal · Pull Request #10172 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10172)

12. ✅ [VectorizeLoop] [[TIR] Fix Ramp int32~64 mismatch in VectorizeLoop and NarrowDataType passes by lazycal · Pull Request #10172 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10172)

13. ✅ [[Bug][TE Schedule] Unsupported nested parallel created by Softmax TE schedule · Issue #12001 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/12001)

14. ✅ [[fix] vec * mat in matmul in onnx converter by ganler · Pull Request #11174 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11174)

15. ✅ [fix vec*mat in PyTorch converter by ganler · Pull Request #11347 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11347)

16. ✅ [[TIR] Fix int32 vs int64 mismatch in For construct. by lazycal · Pull Request #10595 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10595)

17. ✅ [Add missing Slice layout fallback check of `stride=1` . by lazycal · Pull Request #10690 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10690)

18. ✅ [Onnx squeeze enabled with auto axis handling. by ganler · Pull Request #10742 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10742)

19. ✅ [Reduce] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)

20. ✅ [ReduceSumSquare] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)

21. ✅ [ReduceL1] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)

22. ✅ [ReduceL2] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)

23. ✅ [ReduceLogSum][[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)

24. ✅ [[FIX] resolve int64/32 for AttrStmtNode by ganler · Pull Request #10983 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10983)

25. ✅ [Fix onnx round import with float64 inputs. by lazycal · Pull Request #11685 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11685)

26. ✅ [Fix 1d-softmax schedule. by lazycal · Pull Request #11719 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11719)

27. ✅ [[Fix] int32/64 mismatch of buffer elem_offset at HandleBufferBindScope by ganler · Pull Request #11755 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11755)

28. ✅ [[Bug] Int64 BroadCast-ArgMax triggers assertion error at graph runtime · Issue #11794 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/11794)

29. ✅ [[TE Schedule] Fix broken 2D softmax TE schedules when axis=0 by lazycal · Pull Request #11803 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11803)

30. ✅ [[Bug] `concat([x], axis=1)` return random results · Issue #11895 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/11895)

31. ✅ [Fix infercorrect layout in Layoutrewrite and improve naming. by lazycal · Pull Request #12007 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/12007/files)

32. ✅ [Several type mismatch fixes and checks by kparzysz-quic · Pull Request #12041 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/12041)

33. ✅ [[FIX][ONNX][Relay] onnx converter on matmul with scalar; bring back nn.matmul check by ganler · Pull Request #13448 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/13448)

34. ✅ [[Bug] Layout Error when Putting `argmin` after `conv2d` · Issue #9813 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/9813)

35. ✅ [Fix LayoutRewriter by lazycal · Pull Request #10118 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10118)

36. [[Bug] concatenating strided slice and negative padding causes wrong buffer binding · Issue #11897 · apache/tvm](https://github.com/apache/tvm/issues/11897)

37. [[Bug] GPU `lower_thread_allreduce` is_zero(index) check false · Issue #11898 · apache/tvm](https://github.com/apache/tvm/issues/11898)

38. [Resize does not reject unsupported layout during AlterOpLayout · Issue #12008 · apache/tvm](https://github.com/apache/tvm/issues/12008)

39. [[Bug] Compilation failure for `broadcast-argmax` in internal type inference · Issue #13031 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13031)

40. [[Bug] Compiled `squeeze-broadcast_to-argmin` fails at graph runtime · Issue #13045 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13045)

### TensorRT

01. ✅ [Segfault on const+prelu+reduce_mean+comparison_op · Issue #1738 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/1738#issuecomment-1019633288)

02. ✅ [Gemm conversion error, seem to be caused by squeeze · Issue #824 · onnx/onnx-tensorrt · GitHub](https://github.com/onnx/onnx-tensorrt/issues/824)

03. ✅ [[Bug] --loadInputs not working: input name mismatch when Flatten is the input node · Issue #1990 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/1990)

04. ✅ [Cuda OutOfMemory when creating tensor with 2^29 (~0.5 G) elements - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/cuda-outofmemory-when-creating-tensor-with-2-29-0-5-g-elements/203009)

05. ✅ [Myelin error on onnx model: Assertion `i < crds_.size() < failed · Issue #1781 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/1781)

06. [Segmentation fault when using TensorRT to compile a model - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/segmentation-fault-when-using-tensorrt-to-compile-a-model/218872)

07. [Internal Error: GPU error during getBestTactic: PWN(LeakyRelu_4) : misaligned address - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/internal-error-gpu-error-during-getbesttactic-pwn-leakyrelu-4-misaligned-address/218832)

08. [Duplicated reshapes triggers "[graphOptimizer.cpp::findOne::510] Error Code 2: Internal Error (Assertion it != v.end() failed. )" - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/duplicated-reshapes-triggers-graphoptimizer-cpp-510-error-code-2-internal-error-assertion-it-v-end-failed/203540)

09. [Incorrect slicing of boolean constant tensor with step size > 1 - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/incorrect-slicing-of-boolean-constant-tensor-with-step-size-1/215793)

### TensorFlow

01. [Inconsistent behavior of Conv2D between eager mode and tracing · Issue #57664 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57664)

02. [TFLite fails to run a model with a dense layer following an Add operator · Issue #57697 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57697)

03. [TFLite throws an error with certain tensor value · Issue #57708 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57708)

04. [TFLite's max operator has wrong broadcasting behavior · Issue #57759 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57759)

05. [[TFLite] Slice-Conv2d Crash · Issue #58035 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/58035)

06. [pow operation gives valid output even the input is invalid · Issue #57757 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57757)

07. [TFLite produce wrong results when add follows a leakyrelu · Issue #57818 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57818)

08. [TFLite runner crashes with XOR and squeeze in the model · Issue #57882 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57882)

09. [Conv2D with XLA jit_compile=True fails to run · Issue #57748 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57748)

10. [log operator outputs wrong results with XLA compilation · Issue #57744 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57744)

11. [Inconsistent behavior of TF eager and XLA in int64 casting · Issue #57883 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57883)

12. [LRN operator outputs wrong results with `jit_compile=True` · Issue #57746 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57746)

13. [Conv2D layer fails to run with XLA on CUDA · Issue #57838 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57838)

### Methodology

* Though most bugs are identified via individual reports, there are cases where multiple **similar-looking** bugs are merged into one report to avoid potential duplication. Nonetheless, they might be counted for multiple times according to the actual required different fixes.
* "won't fix" bugs are omitted.
* Part of the bugs are found by experimental repositories of NNSmith (e.g., PT2 bugs) but the features will be eventually upstreamed.
