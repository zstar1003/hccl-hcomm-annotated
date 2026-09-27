# Algorithm Analyzer Tool Usage Guide

## Introduction

This document is only used to guide users in compiling and running the algorithm analyzer test cases in this directory. For detailed tool principles, test case writing, parameter settings, result analysis, and issue location, please refer to the [Algorithm Analyzer Tool User Guide](../../../st/algorithm/README_en.md).

## Environment Setup

Refer to the environment setup in [Source Build](../../../../docs/en/build/build.md), install the CANN Toolkit development package, and prepare the prerequisites for compiling the algorithm analyzer.

## Test Execution

Run the following commands from the hcomm source code root directory to compile and execute algorithm analyzer test cases:

```bash
# Compile all test suite cases and execute automatically
bash build.sh --legacy_all_testcase

# Compile individual test suite cases and execute automatically
bash build.sh --legacy_aicpu_2d_testcase
bash build.sh --legacy_ccu_2d_testcase
bash build.sh --legacy_ccu_1d_hf16p_testcase
bash build.sh --legacy_ccu_1d_testcase_part1
bash build.sh --legacy_ccu_1d_testcase_part2
bash build.sh --legacy_alg_ccu_reduce
bash build.sh --legacy_function_ut_testcase
bash build.sh --legacy_alg_testcase

# Manually re-execute test cases
./build/test/legacy/st/algorithm/testcase/aicpu_2d_testcase/legacy_alg_aicpu_2d_testcase
./build/test/legacy/st/algorithm/testcase/ccu_2d_testcase/legacy_alg_ccu_2d_testcase
./build/test/legacy/st/algorithm/testcase/ccu_1d_hf16p_testcase/legacy_alg_ccu_1d_hf16p_testcase
./build/test/legacy/st/algorithm/testcase/ccu_1d_testcase_part1/legacy_alg_ccu_1d_testcase_part1
./build/test/legacy/st/algorithm/testcase/ccu_1d_testcase_part2/legacy_alg_ccu_1d_testcase_part2
./build/test/legacy/st/algorithm/testcase/ccu_reduce_testcase/legacy_alg_ccu_reduce
./build/test/legacy/st/algorithm/testcase/function_ut_testcase/legacy_alg_function_ut_testcase
./build/test/legacy/st/algorithm/testcase/legacy_alg_testcase/legacy_alg_testcase
```
