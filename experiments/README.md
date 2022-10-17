# Experiments

## Evaluating source-level coverage of fuzzer-generated test

1. "Link" the non-instrumented SUT and run the fuzzer with `fuzz.save_test=/path/to/tests` to save all intermediate test cases;
2. Compile an instrumented SUT;
3. Replay the tests on instrumented SUT: `python experiments/evaluate_profraws.py --root /path/to/tests --model_type ?? --backend_type ?? --backend_target ??` (`??` needs to be filled);
4. Analyze the raw coverage profiles: `python experiments/process_profraws.py --root /path/to/tests --llvm-config-path ?? --instrumented-libs ??`;

We only support LLVM's coverage profile for now. See [here](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html) for how to compile the SUT with LLVM coverage instrumentation.
