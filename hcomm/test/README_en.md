# HCCL LLT

English | [简体中文](./README.md)

## Overview

HCCL LLT (Low Level Test) is the test framework for HCCL, designed to systematically verify the functional completeness and performance stability of HCCL components at all levels. LLT covers multiple parts of HCCL including the algorithm layer, framework layer, platform layer, and external interface layer. Through comprehensive test cases, it ensures the reliability and efficiency of HCCL in various business scenarios.

## Directory Structure

```text
test/
├── legacy                                          # Historical version compatibility test framework
│   ├── common                                      # Common utilities
│   ├── depends                                     # Test dependencies on other component headers
│   ├── st                                          # ST integration test cases
│   │   ├── algorithm                               # Communication algorithm test cases
│   │   ├── fwk                                     # Communication framework test cases
│   │   ├── service                                 # Service layer test cases
│   │   └── test_case                               # Test cases
│   └── ut                                          # UT integration test cases
│       ├── aicpu                                   # AICPU-specific test cases
│       ├── all_source_code                         # Source code file paths
│       ├── common                                  # Common utilities
│       ├── framework                               # Communication framework test cases
│       ├── service                                 # Service layer test cases
│       └── unified_platform                        # Unified platform layer test cases
├── st/algorithm                                    # ST integration test cases (algorithm analyzer)
│   ├── testcase                                    # Test cases
│   └── utils                                       # Common utilities
└── ut                                              # UT unit test cases
    ├── aicpu_kfc                                   # MC2-related tests
    ├── common                                      # Common utilities
    ├── depends                                     # Test dependencies on other component headers
    ├── device                                      # Device tests
    ├── framework                                   # Communication framework test cases
    ├── impl                                        # Communication algorithm implementation test cases
    ├── inter                                       # Interface adaptation layer test cases
    ├── misc                                        # Miscellaneous test cases
    ├── platform                                    # Communication platform implementation test cases
    └── stub                                        # Test stub functions
```

## Build and Run

Execute the following commands from the repository root directory:

```bash
# Build and run all unit test cases
bash build.sh --ut

# Build and run all integration test cases
bash build.sh --st
# Build and run individual test suite cases
bash build.sh --open_hccl_test
bash build.sh --executor_hccl_test
bash build.sh --executor_reduce_hccl_test
bash build.sh --executor_pipeline_hccl_test
# Manually execute test cases
./build/test/st/algorithm/testcase/testcase/open_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_reduce_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_pipeline_hccl_test
```

## Executable Output Location

All executables are output to the `build/test` directory by default. The path can be adjusted by modifying `CMakeLists.txt`:

```cmake
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${HCCL_OPEN_CODE_ROOT}/build/test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HCCL_OPEN_CODE_ROOT}/build/test)
```

## How to Write Test Cases

> HCCL LLT test cases are implemented using the Google Test framework. For detailed writing guidelines, refer to the [Google Test User Guide](https://google.github.io/googletest/).

1. Select the appropriate directory based on the test target, e.g., `test/algorithm` or `test/framework`
2. Create a new test class based on Google Test
3. Update the corresponding directory's `CMakeLists.txt` to add the test entry

Example test code:

```c++
#include "gtest/gtest.h"

TEST(MyTestClass, MyTestCase) {
    // Implement assertion logic
    EXPECT_EQ(actual_value, expected_value);
}
```

## How to Run Specific Test Cases

To run specific test cases individually, refer to the [Google Test User Guide](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests) and add the `--gtest_filter` parameter when executing the test.

Using `hccl_utest_framework_op_base_api` as an example:

```bash
# Run only test cases of the HcclCommInitRootInfoTest test class
./build/test/hccl_utest_framework_op_base_api --gtest_filter=HcclCommInitRootInfoTest.*
```
