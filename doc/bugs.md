# Bugs uncovered by the NNSmith project

> [!IMPORTANT]
>
> **Summary of Bugs**
>
> | System | #Fixed | #Confirmed | #Pending | #Total |
> |-----|-----|-----|-----|-----|
> | PyTorch | 60 | 9 | 15 | 84 |
> | PyTorch-ONNX Converter | 12 | 0 | 0 | 12 |
> | ONNX | 1 | 0 | 0 | 1 |
> | ONNXRuntime | 3 | 4 | 4 | 11 |
> | TVM | 34 | 0 | 6 | 40 |
> | TensorRT | 6 | 2 | 2 | 10 |
> | TensorFlow | 1 | 13 | 0 | 14 |
> | Hidet | 13 | 0 | 0 | 13 |
> | Sum | 130 | 28 | 27 | 185 |

> [!NOTE]
>
> - **Status**: ✅ means fixed; 🔵 means confirmed; 🚨 means this bug has been marked with a `high-priority` label (PyTorch)
> - **Symptom**: 💥 Crash or exception; 🧮 Result inconsistency (silent semantic bug); 🧴 Sanitizers

> [!NOTE]
>
> **Table of Content**
>
> * [**PyTorch**](#pytorch)
> * [**PyTorch-ONNX Converter**](#pytorch-onnx-converter)
> * [**ONNX**](#onnx)
> * [**ONNXRuntime**](#onnxruntime)
> * [**TVM**](#tvm)
> * [**TensorRT**](#tensorrt)
> * [**TensorFlow**](#tensorflow)
> * [**Hidet**](#hidet)
> * [**Methodology**](#methodology)

## [PyTorch](https://github.com/pytorch/pytorch)

* ✅🧮🚨 [`torch.compile` produce wrong result in backward ad with `conv2d + interpolate` when interpolate`mode=nearest/bilinear/bicubic` · Issue #100794 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/100794)
* ✅💥🚨 [SIGIOT when running model with conv2d and avgpool2d after `optimize_for_inference` · Issue #86535 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86535)
* ✅🧮 [`optimize_for_inference` leads to wrong results for model with conv2d, max and clip · Issue #86556 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86556)
* ✅💥🚨 [RuntimeError: could not construct a memory descriptor using a format tag · Issue #86664 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86664)
* ✅💥 [[NNPack] Runtime error with padded `Conv1d` and `&gt;=16` batch size · Issue #90142 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90142)
* ✅💥 [stable `torch.sort` crash with expanded tensor · Issue #91420 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91420)
* ✅💥 [[Crash] `torch.searchsorted` with out-of-bound sorter · Issue #91606 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91606)
* ✅🧮 [`index_select` with scalar input and 0-dimed vector leads to undeterministic output · Issue #94340 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94340)
* ✅🧮 [`index_select` with scalar input and 0-dimed vector leads to undeterministic output · Issue #94340 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94340)
* ✅💥 [`torch.compile` failed on `torch.add` with a constant python number · Issue #92324 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92324)
* ✅💥 [`torch.compile` generates wrong profiling program for `randn_like` · Issue #92368 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92368)
* ✅💥 [`torch.compile` generates wrong profiling program for function having `transpose` and `lerp` · Issue #93229 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93229)
* ✅💥 [`torch.compile` triggers assertion error when explicitly provide `out=None` · Issue #92814 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92814)
* ✅💥 [INTERNAL ASSERT FAILED in `torch.compile` when the input tensor of `torch.clamp` has `requires_grad=True` · Issue #93225 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93225)
* ✅💥 [`torch.compile` failed to run in-place operation `unsqueeze_(0)` · Issue #93259 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93259)
* ✅🧮 [`stack` + inplace operator produce wrong results in `torch.compile` · Issue #93283 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93283)
* ✅🧮 [[pt2] compiled model with cat and expand gives wrong results · Issue #93357 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93357)
* ✅🧮🚨 [[pt2] compiled function with cat and mul gives wrong results · Issue #93365 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93365)
* ✅🧮 [[pt2] cannot compile model with linear layer when the input has rank 1 · Issue #93372 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93372)
* ✅💥 [`softmax` + `transpose` + `div_` triggers assertion fail in compile mode · Issue #93371 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93371)
* ✅🧮🚨 [[pt2] `torch.where` gives wrong results with `torch.compile` · Issue #93374 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93374)
* ✅💥 [`torch.rsub` with `alpha=xxx` triggers assertion fail in compile mode · Issue #93376 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93376)
* ✅🧮 [[pt2] compile gives wrong result for function having `expand` and `div_` · Issue #93377 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93377)
* ✅💥 [[pt2] `torch._inductor.exc.CppCompileError: C++ compile error` when compiling `neg` and `max` · Issue #93380 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93380)
* ✅💥 [[pt2] exception when compiling `max_pool2d_with_indices` · Issue #93384 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93384)
* ✅💥 [[pt2] cannot compile function having `gt`, `expand` and `add_` · Issue #93386 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93386)
* ✅💥🚨 [`torch.compile` trigger assertion error when executing `histogramdd` · Issue #93274 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93274)
* ✅🧮 [[pt2] `torch.compile` produces wrong results for `masked_fill` · Issue #93823 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93823)
* ✅🧮 [[pt2] `torch.compile` produces wrong results for function with `reciprocal_` · Issue #93824 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93824)
* ✅🧮 [[pt2] `torch.compile` produces wrong results for function with `neg` on `uint8` tensor · Issue #93829 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93829)
* ✅💥 [`log_softmax` + `pad` triggers assertion fail in compile mode · Issue #93819 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93819)
* ✅💥 [[pt2] Cannot compile model with `neg` and `linear` · Issue #93836 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93836)
* ✅🧮 [`pad` + `gt` produce wrong results in compile mode · Issue #93351 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93351)
* ✅💥 [[pt2] (`interpolate` with `mode=nearest`) + `kthvalue` triggers assertion fail in compile mode · Issue #93849 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93849)
* ✅💥 [[pt2] `torch._inductor.exc.CppCompileError: C++ compile error` when compiling `argmax` and `min` · Issue #94055 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94055)
* ✅💥 [`Tensor.select` + `add_` triggers C++ Compile Error · Issue #94960 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/94960)
* ✅💥 [`torch.compile` fails when using `torch.sub` with python constant · Issue #95181 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95181)
* ✅💥 [`Tensor.copy_` + `moveaxis` Trigger Exception in Compile Mode · Issue #95262 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95262)
* ✅🧮 [`torch.ge` produces wrong results in compile mode when given int tensors · Issue #95695 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95695)
* ✅💥 [[pt2] `bitwise_and` + `clamp_max` Triggers Compilation Error · Issue #97968 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/97968)
* ✅🧮 [[pt2] `add` + `unfold` + `abs_` produces wrong results · Issue #98143 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/98143)
* ✅🧮 [[pt2] `pow` + `cos` produces wrong result · Issue #98149 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/98149)
* ✅💥 [`torch._C._nn.fractional_max_pool3d` Trigger Segmentation Fault · Issue #89648 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89648)
* ✅💥🚨 [`torch.nn.functional.embedding_bag` Trigger &quot;IOT instruction&quot; Failure · Issue #89677 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89677)
* ✅🧴 [`torch.Tensor.index_select` Trigger heap-buffer-overflow with AddressSanitizer · Issue #88940 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88940)
* ✅🧴 [`nn.utils.rnn.pack_sequence` Trigger heap-buffer-overflow with AddressSanitizer · Issue #88334 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88334)
* ✅🚨🧴 [`MultiMarginLoss` Trigger out-of-bound Read under Compute Sanitizer · Issue #88724 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88724)
* ✅🧴 [`nn.functional.max_unpool3d` Trigger heap-buffer-overflow with AddressSanitizer · Issue #88032 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88032)
* ✅🧴 [`torch.nn.functional.interpolate` Trigger heap-buffer-overflow with AddressSanitizer  · Issue #88939 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88939)
* ✅🧴 [`torch.fft.hfft` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88985 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88985)
* ✅🧴 [`torch.nn.functional.interpolate` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88951 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88951)
* ✅💥🚨 [`torch.compile` failed on `torch.bitwise_xor` with a constant python number · Issue #93224 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93224)
* ✅💥 [[CPU Inductor] Compile error when passing float16 tensors to `vector_norm` + `remainder` · Issue #97758 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/97758)
* ✅💥 [[pt2] `movedim` + `add_` + `cat` triggers exception · Issue #98122 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/98122)
* ✅🧮 [`dstack` + `reciprocal` produce wrong result in compile mode · Issue #93078 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93078)
* ✅💥 [`min` reduction on float16 tensor failed on certain shapes · Issue #93249 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93249)
* ✅💥 [`argmin` + `view` Trigger Exception in compile mode · Issue #95370 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/95370)
* 🔵💥 [[JIT] Zero-channel conv2d cannot be applied with `optimize_for_inference` · Issue #91396 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/91396)
* 🔵💥 [[JIT] Applying `conv2d` over Constants Leads to Exception · Issue #92740 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92740)
* ✅🧮🚨 [`torch.compile` produce wrong result in `interpolate` when `mode=bilinear` · Issue #93262 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/93262)
* 🔵🧮 [`torch.fmod` produces inconsistent results in eager and compile mode · Issue #97333 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/97333)
* 🔵💥 [`torch.Tensor.flatten` Trigger Segmentation Fault when trying to provide and output named dim · Issue #89718 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89718)
* 🔵🧴 [`nn.functional.embedding_bag` Trigger out-of-bound Read under Compute Sanitizer · Issue #88563 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88563)
* ✅🔵🧴 [`torch.nn.CTCLoss` Trigger heap-buffer-overflow under AddressSanitizer · Issue #88047 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88047)
* 🔵🧴 [`torch.nn.ReplicationPad2D` Report &quot;invalid configuration argument&quot; Error under Compute Sanitizer · Issue #89254 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89254)
* 🔵🧴 [`torch.nn.LayerNorm` Abort with &quot;invalid device ordinal&quot; Error · Issue #89218 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89218)
* 🔵🧴 [`torch.svd_lowrank` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88942 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88942)
* 🔵🧴 [`torch.linalg.lstsq` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88941 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88941)
* 💥 [Adding a linear layer leads to failure of `optimize_for_mobile` · Issue #86667 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/86667)
* 💥 [[JIT] INTERNAL ASSERT FAILED when dispatching for `torch.Tensor.view` · Issue #90365 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90365)
* 💥 [[JIT] INTERNAL ASSERT FAILED `torch.add` with boolean primitive constant · Issue #90367 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90367)
* 💥 [[JIT] INTERNAL ASSERT FAILED `torch.mul` with boolean primitive constant · Issue #90366 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90366)
* 💥 [[JIT] Wrong type inference leads to misleading error message · Issue #90369 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/90369)
* 💥 [[JIT] INTERNAL ASSERT FAILED when `Conv2d` and `clamp` used together · Issue #92563 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92563)
* 💥 [[JIT] Inconsistency  in tensor shape between eager mode and JIT · Issue #92548 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92548)
* 💥 [[JIT][TracingCheckError] inplace ops incompatible with `contiguous(.., channels_last)` · Issue #92558 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92558)
* 💥 [[JIT] Consecutive use of `addmm` Leads to Exception · Issue #92742 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/92742)
* 🧴 [`torch.topk` Trigger RuntimError under UndefinedBehaviorSanitizer · Issue #88944 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88944)
* 🧴 [`torch.vander` Trigger RuntimeError with UndefinedBehaviorSanitizer · Issue #88943 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88943)
* ✅🧴 [`torch.nn.CTCLoss` Trigger out-of-bound Read under Compute Sanitizer · Issue #89208 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/89208)
* 🧴 [`torch.nn.functional.embedding_bag` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88950 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88950)
* 🧴 [`torch.set_rng_state` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88949 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88949)
* 🧴 [`torch.Tensor.msort` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88947 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88947)
* 🧴 [`torch.linalg.eigvals` Trigger RuntimeError under UndefinedBehaviorSanitizer · Issue #88945 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/88945)

## [PyTorch-ONNX Converter](https://pytorch.org/docs/stable/onnx.html)

* ✅ [[ONNX] `f64 * LeakyReLU(f64)` mistakingly returns f32 · Issue #85316 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/85316)
* ✅ [[ONNX] Converter did not consider the implicit casting specifically for `Max` · Issue #87609 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/87609)
* ✅ [fix: onnx PReLU unidirectional broadcasting by ganler · Pull Request #70571 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/70571)
* ✅ [Clip] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)
* ✅ [Min] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)
* ✅ [Max] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)
* ✅ [ReLU] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)
* ✅ [Pad] [[ONNX] Make Non-Float Op Exportation Compatible to Avoid Invalid ONNX Models by ganler · Pull Request #72401 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/72401)
* ✅ [[onnx export] Add broadcast to matmul shape inference by lazycal · Pull Request #70534 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/70534)
* ✅ [[Bug][ONNX] Specification Inconsistency in Flatten · Issue #74142 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/74142)
* ✅ [[ONNX] Fix shape inconsistency when exporting scalar log2 by lazycal · Pull Request #78701 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/78701)
* ✅ [[ONNX Export] Interpolation likely should be exported with `half_pixel` instead of `pytorch_half_pixel` · Issue #79361 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/79361)

## [ONNX](https://github.com/onnx/onnx)

* ✅ [[Bug] Checker misses data type mismatch for Max · Issue #4619 · onnx/onnx · GitHub](https://github.com/onnx/onnx/issues/4619)

## [ONNXRuntime](https://github.com/microsoft/onnxruntime)

* ✅ [Crashes when relu is followed by a clip · Issue #9753 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/9753)
* ✅ [MatMul fusion failed at scalar input · Issue #10950 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/10950)
* ✅ [GemmTransposeFusion error when C is transposed (`Gemm(A,B,Transpose(C)`), complained with confusing name `_transformed_transformed_transformed...` · Issue #12071 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12071)
* 🔵 [[Bug] Mixing negative and positive paddings causes segfault/uninitialized memory values produced in reflected pad · Issue #11828 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/11828)
* 🔵 [Runtime Exception when relu is followed by a clip · Issue #10936 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/10936)
* 🔵 [Inconsistent result to NumPy and PyTorch when consecutively casting a float tensor to int32 and then to bool · Issue #11994 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/11994)
* 🔵 [Wrong output shape due to MergeShape failure · Issue #11870 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/11870)
* [Wrong Floor output on very large input · Issue #12076 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12076)
* [Resize with mode linear always produces 0.5 on GPU regardless of the input · Issue #12091 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12091)
* [Resize with `nearest` mode have inconsistent results compared to PyTorch and TVM · Issue #12098 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/12098)
* [Parameters are optimized out even if it is a needed return value · Issue #13425 · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/issues/13425)

## [TVM](https://github.com/apache/tvm)

* ✅ [[Bug] shape int32-int64 check error in `trilu`'s `te.compute` · Issue #13029 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13029)
* ✅ [[Bug] `trilu` not tagged with `injective` and thus miss reduce schedule · Issue #13030 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13030)
* ✅ [[Bug] Wrong results of `cast<int32>( cast<bool>(-1i64) )` · Issue #13048 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13048)
* ✅ [[BugFix] resolve integer 32. ~ 64. mismatch by casting by ganler · Pull Request #9582 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/9582)
* ✅ [[onnx] fix onnx where broadcast by lazycal · Pull Request #10106 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10106)
* ✅ [Fix broadcast InferCorrectLayout by lazycal · Pull Request #10156 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10156)
* ✅ [[BUGFIX][ARITH] Fix FloorMod Simplifier by lazycal · Pull Request #10336 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10336)
* ✅ [[BugFix]: select node type error in NarrowDataType pass by ganler · Pull Request #10519 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10519)
* [[Bug] GetStoreRule failure at simple Conv2d + Squeeze model · Issue #10528 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/10528)
* ✅ [[Relay][ONNX][Fix] Flatten in OnnxConverter by ganler · Pull Request #10593 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10593)
* ✅ [NarrowDataType] [[TIR] Fix Ramp int32~64 mismatch in VectorizeLoop and NarrowDataType passes by lazycal · Pull Request #10172 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10172)
* ✅ [VectorizeLoop] [[TIR] Fix Ramp int32~64 mismatch in VectorizeLoop and NarrowDataType passes by lazycal · Pull Request #10172 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10172)
* ✅ [[Bug][TE Schedule] Unsupported nested parallel created by Softmax TE schedule · Issue #12001 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/12001)
* ✅ [[fix] vec * mat in matmul in onnx converter by ganler · Pull Request #11174 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11174)
* ✅ [fix vec*mat in PyTorch converter by ganler · Pull Request #11347 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11347)
* ✅ [[TIR] Fix int32 vs int64 mismatch in For construct. by lazycal · Pull Request #10595 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10595)
* ✅ [Add missing Slice layout fallback check of `stride=1` . by lazycal · Pull Request #10690 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10690)
* ✅ [Onnx squeeze enabled with auto axis handling. by ganler · Pull Request #10742 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10742)
* ✅ [Reduce] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)
* ✅ [ReduceSumSquare] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)
* ✅ [ReduceL1] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)
* ✅ [ReduceL2] [[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)
* ✅ [ReduceLogSum][[ONNX] fix reduce crash on scalar inputs by ganler · Pull Request #10780 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10780)
* ✅ [[FIX] resolve int64/32 for AttrStmtNode by ganler · Pull Request #10983 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10983)
* ✅ [Fix onnx round import with float64 inputs. by lazycal · Pull Request #11685 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11685)
* ✅ [Fix 1d-softmax schedule. by lazycal · Pull Request #11719 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11719)
* ✅ [[Fix] int32/64 mismatch of buffer elem_offset at HandleBufferBindScope by ganler · Pull Request #11755 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11755)
* ✅ [[Bug] Int64 BroadCast-ArgMax triggers assertion error at graph runtime · Issue #11794 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/11794)
* ✅ [[TE Schedule] Fix broken 2D softmax TE schedules when axis=0 by lazycal · Pull Request #11803 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/11803)
* ✅ [[Bug] `concat([x], axis=1)` return random results · Issue #11895 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/11895)
* ✅ [Fix infercorrect layout in Layoutrewrite and improve naming. by lazycal · Pull Request #12007 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/12007/files)
* ✅ [Several type mismatch fixes and checks by kparzysz-quic · Pull Request #12041 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/12041)
* ✅ [[FIX][ONNX][Relay] onnx converter on matmul with scalar; bring back nn.matmul check by ganler · Pull Request #13448 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/13448)
* ✅ [[Bug] Layout Error when Putting `argmin` after `conv2d` · Issue #9813 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/9813)
* ✅ [Fix LayoutRewriter by lazycal · Pull Request #10118 · apache/tvm · GitHub](https://github.com/apache/tvm/pull/10118)
* [[Bug] concatenating strided slice and negative padding causes wrong buffer binding · Issue #11897 · apache/tvm](https://github.com/apache/tvm/issues/11897)
* [[Bug] GPU `lower_thread_allreduce` is_zero(index) check false · Issue #11898 · apache/tvm](https://github.com/apache/tvm/issues/11898)
* [Resize does not reject unsupported layout during AlterOpLayout · Issue #12008 · apache/tvm](https://github.com/apache/tvm/issues/12008)
* [[Bug] Compilation failure for `broadcast-argmax` in internal type inference · Issue #13031 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13031)
* [[Bug] Compiled `squeeze-broadcast_to-argmin` fails at graph runtime · Issue #13045 · apache/tvm · GitHub](https://github.com/apache/tvm/issues/13045)

## [TensorRT](https://developer.nvidia.com/tensorrt)

* ✅ [Segfault on const+prelu+reduce_mean+comparison_op · Issue #1738 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/1738#issuecomment-1019633288)
* ✅ [Gemm conversion error, seem to be caused by squeeze · Issue #824 · onnx/onnx-tensorrt · GitHub](https://github.com/onnx/onnx-tensorrt/issues/824)
* ✅ [[Bug] crash on poolings with larger-than-317 pool sizes · Issue #2094 · NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT/issues/2094)
* ✅ [[Bug] --loadInputs not working: input name mismatch when Flatten is the input node · Issue #1990 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/1990)
* ✅ [Cuda OutOfMemory when creating tensor with 2^29 (~0.5 G) elements - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/cuda-outofmemory-when-creating-tensor-with-2-29-0-5-g-elements/203009)
* ✅ [Myelin error on onnx model: Assertion `i < crds_.size() < failed · Issue #1781 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/1781)
* [Segmentation fault when using TensorRT to compile a model - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/segmentation-fault-when-using-tensorrt-to-compile-a-model/218872)
* [Internal Error: GPU error during getBestTactic: PWN(LeakyRelu_4) : misaligned address - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/internal-error-gpu-error-during-getbesttactic-pwn-leakyrelu-4-misaligned-address/218832)
* 🔵 [Duplicated reshapes triggers "[graphOptimizer.cpp::findOne::510] Error Code 2: Internal Error (Assertion it != v.end() failed. )" - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/duplicated-reshapes-triggers-graphoptimizer-cpp-510-error-code-2-internal-error-assertion-it-v-end-failed/203540)
* 🔵 [Incorrect slicing of boolean constant tensor with step size > 1 - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/incorrect-slicing-of-boolean-constant-tensor-with-step-size-1/215793)

## [TensorFlow](https://github.com/tensorflow/tensorflow)

* 🔵💥 [Inconsistant behavior of Conv2D between eager mode and tracing · Issue #57664 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57664)
* 🔵💥 [TFLite fails to run a model with a dense layer following an Add operator · Issue #57697 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57697)
* 🔵💥 [TFLite throws an error with certain tensor value · Issue #57708 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57708)
* 🔵🧮 [TFLite&#39;s max operator has wrong broadcasting behavior · Issue #57759 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57759)
* 🔵💥 [Issues · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/58035 )
* 🔵🧮 [pow operation gives valid output even the input is invalid · Issue #57757 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57757)
* 🔵🧮 [TFLite produce wrong results when add follows a leakyrelu · Issue #57818 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57818)
* 🔵💥 [TFLite runner crashes with XOR and squeeze in the model · Issue #57882 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57882)
* ✅💥 [ Conv2D with XLA jit_compile=True fails to run · Issue #57748 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57748)
* 🔵🧮 [log operator outputs wrong results with XLA compilation · Issue #57744 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57744)
* 🔵🧮 [pow operator output nan for valid inputs · Issue #57747 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57747)
* 🔵🧮 [LRN operator outputs wrong results with `jit_compile=True` · Issue #57746 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57746)
* 🔵💥 [Conv2D layer fails to run with XLA on CUDA · Issue #57838 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/57838)
* 🔵🧴 [`tf.raw_ops.SegmentMax` Behaves Differently Under CPU and GPU · Issue #58469 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/issues/58469)

## [Hidet](https://github.com/hidet-org/hidet)

Based on NNSmith, [@soodoshll](https://github.com/soodoshll) found a number of bugs for Hidet, including:

* ✅ [[Bug] Use int64 in argmax · Issue #103 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/103)
* ✅ [[Bug] broadcast_shape parameter type error · Issue #85 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/85)
* ✅ [[Bug] Data type casting from onnx · Issue #87 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/87)
* ✅ [[Bug] cuda code compilation error · Issue #89 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/89)
* ✅ [[Bug] MinOp generates max code · Issue #90 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/90)
* ✅ [[Bug] FP64 reduce · Issue #91 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/91)
* ✅ [[Bug] Inconsistent definition of the inputs parameter of operators · Issue #93 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/93)
* ✅ [[Bug] Slice indexing in ONNX · Issue #94 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/94)
* ✅ [[Bug] binary arithmetic with CUDA scalar · Issue #95 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/95)
* ✅ [[Bug] Unexpected behavior when inputs and outputs overlap · Issue #96 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/96)
* ✅ [Followup][[Bug] Unexpected behavior when inputs and outputs overlap · Issue #96 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/96)
* ✅ [[Bug] arguments of clip drop after fusion · Issue #97 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/97)
* ✅ [[Bug] fusion rewrite fails · Issue #99 · hidet-org/hidet](https://github.com/hidet-org/hidet/issues/99)

> [!NOTE]
>
> **Methodology**
>
> * Though most bugs are identified via individual reports, there are cases where multiple **similar-looking** bugs are merged into one report to avoid potential duplication. Nonetheless, they might be counted for multiple times according to the actual required different fixes.
> * "won't fix" bugs are omitted.
> * Detected bugs come from the [ASPLOS'23 (NNSmith)](https://docs.google.com/spreadsheets/d/1gzMPlY0sOfyVBGhq9CPkynDnuVpiGm7JpFQ-CPoLStc/edit#gid=0) and [FSE'23 (NeuRI)](https://github.com/ise-uiuc/neuri-artifact/blob/main/docs/rq3-bug-reports.md) projects.
