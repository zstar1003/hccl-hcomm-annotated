# HCCL_VM UT Test Execution Script

## Overview

`run_ut.sh` is the unit test execution script in the HCCL_VM project. It automates compilation and execution of test cases.

## Three-Step Workflow

The script automatically completes the following steps during execution:

| Step | Description | Log Output |
|------|-------------|------------|
| Step 1 | CMake configuration + make compilation | `build.log` |
| Step 2 | Generate executable files | List all binaries with size and time |
| Step 3 | Execute test cases + display results | `run.log` + `summary.log` |
| Step 4 | Generate gcov/lcov coverage HTML report (requires `--cov`) | `coverage.log` |

## Usage

```bash
cd {HCCL_VM path}/test

# Basic usage
./run_ut.sh                           # Compile and execute all tests
./run_ut.sh --cov                     # Compile, execute, and generate gcov/lcov coverage HTML report
./run_ut.sh <directory>               # Compile and execute all tests in the specified directory (recursive)
./run_ut.sh <binary name>             # Compile and execute the specified binary test
./run_ut.sh <test file name>          # Compile and execute the binary corresponding to the specified test file
./run_ut.sh <file> <test case name>   # Compile and execute a single test case in the specified file

# Other commands
./run_ut.sh -l, --list                # List all available tests
./run_ut.sh -h, --help                # Display help information
```

## Command Details

### 1. Full Execution

```bash
./run_ut.sh
```

- Execute all tests under the `test` directory.
- Complete three-step workflow: compile → generate executables → execute all tests.
- Suitable for full regression testing.

### 2. Coverage Report

```bash
./run_ut.sh --cov
```

- Compile all tests, execute all tests, and generate gcov/lcov coverage HTML report.
- Automatically enable the `--coverage` compilation option and generate `.gcno`/`.gcda` files.
- Four-step workflow: compile (with coverage instrumentation) → execute → collect coverage data → generate HTML report.
- Report output path: `$CODE_DIR/coverage_report/html/index.html`.
- Automatically filter system headers, third-party libraries, stub files, and other non-business code.

### 3. Directory Execution

```bash
./run_ut.sh plugin/checker
./run_ut.sh plugin/ccu_executor
./run_ut.sh store
```

- Recursively find all `*_test.cc` files in the specified directory.
- Compile and execute all tests in that directory.
- Suitable for module-level testing.

### 4. Binary Execution

```bash
./run_ut.sh test_checker
./run_ut.sh test_allgather_semantics_checker
```

- Compile and execute the specified test binary.
- Binary names start with `test_`.

### 5. File Execution

```bash
./run_ut.sh checker_test.cc
./run_ut.sh allgather_semantics_checker_test.cc
```

- Automatically match the corresponding binary based on the test file name.
- Compile and execute.

### 6. Single Test Case Execution

```bash
./run_ut.sh checker_test.cc CheckerTest.GenAndCheckGraph_EmptyQueues
./run_ut.sh allgather_semantics_checker_test.cc AllgatherSemanticsCheckerTest.CheckBasic
```

- Execute a single test case in the specified test file.
- Test case name format: `TestSuiteName.TestCaseName`.

### 7. List Tests

```bash
./run_ut.sh -l
./run_ut.sh --list
```

- List all available test files and their status.
- Display test case count and corresponding binary name.

## Samples

```bash
# Sample 1: Full test
./run_ut.sh

# Sample 2: Full test + coverage report
./run_ut.sh --cov

# Sample 3: Execute all tests in the plugin/checker directory
./run_ut.sh plugin/checker

# Sample 4: Compile and execute test_checker
./run_ut.sh test_checker

# Sample 5: Compile and execute the binary corresponding to checker_test.cc
./run_ut.sh checker_test.cc

# Sample 6: Execute a single test case
./run_ut.sh checker_test.cc CheckerTest.GenAndCheckGraph_EmptyQueues

# Sample 7: List all tests
./run_ut.sh -l
```

## Log Directory

Each execution generates log files under `ut_logs/<timestamp>/`:

```text
{HCCL_VM path}/ut_logs/20260425_142048/
├── build.log      # Detailed compilation log (cmake + make output)
├── run.log        # Detailed execution log (complete output of each test)
└── summary.log    # Summary log (execution result of each test)
```

### Log File Description

| File | Content |
|------|---------|
| `build.log` | CMake configuration output, make compilation output, list of generated executables |
| `run.log` | Complete output of each test (including detailed gtest information) |
| `summary.log` | Execution status, pass/fail count, and duration summary for each test |

## Execution Results

The script displays summary information after execution completes:

```text
========================================
  Directory Test Result Summary: plugin/checker
========================================
  Executed:   16
  PASSED:  185
  FAILED:  8
  CRASHED: 0
  TIMEOUT: 0
```

### Status Description

| Status | Description |
|--------|-------------|
| `PASS` | Test passed |
| `FAIL` | Test failed (assertion failure) |
| `CRASH` | Test crashed (core dump) |
| `TIMEOUT` | Test timed out (default 60 seconds) |

## Directory Structure

```text
test/
├── run_ut.sh              # This script
├── cmd/                   # Command-related tests
│   ├── base/
│   ├── subcmds/
│   └── utils/
├── device_arm/            # Device-related tests
├── device_vir/
├── ipc/                   # IPC-related tests
│   └── shm/
├── log/                   # Log-related tests
├── plugin/                # Plugin-related tests
│   ├── ccu_executor/
│   │   ├── control_type/
│   │   ├── load_type/
│   │   ├── reduce_type/
│   │   └── trans_type/
│   └── checker/
│       └── framework/
│           ├── mem_conflict_check/
│           ├── semantics_check/
│           └── singletask_check/
├── proxy/                 # Proxy-related tests
│   ├── level1/
│   └── level2/
├── runnerdb/              # Database-related tests
├── store/                 # Storage-related tests
│   └── hccl_shm/
└── src_root/              # Source root directory tests
```

## Environment Requirements

**Before executing the script, you must modify the following environment variables in `run_ut.sh` and replace them with actual paths:**

```bash
# The following are sample paths. Modify them according to your actual environment.
export HCOMM_CODE_HOME=/home/q30033976/checker/hcomm      # hcomm source code path
export HCCL_CODE_HOME=/home/q30033976/checker/hccl        # hccl source code path (required for AIV/AICPU mode)
source /home/q30033976/checker/Ascend/cann/set_env.sh     # Environment script under the CANN installation path
```

**Sample**: Assume your working directory is `/home/workspace`. Modify the paths as follows:

```bash
export HCOMM_CODE_HOME=/home/workspace/hcomm
export HCCL_CODE_HOME=/home/workspace/hccl
source /home/workspace/Ascend/cann/set_env.sh
```

The script automatically loads these environment variables. Failure to modify them causes compilation failure.

## Configuration Parameters

The script includes the following built-in configurations (modifiable at the beginning of the script):

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | CMake build type |
| `MAKE_JOBS` | `8` | Number of parallel make jobs |
| `LOG_DIR` | `$CODE_DIR/ut_logs` | Log output directory |

## Precautions

1. **First execution**: The first execution performs a full CMake configuration and takes longer.
2. **Compilation failure**: If compilation fails, check `build.log` for detailed error information.
3. **Test failure**: If tests fail, check `run.log` for specific failed test cases.
4. **Log cleanup**: Log directories are named by timestamp. Clean up old logs regularly to save space.

## Frequently Asked Questions

### Q: How do I compile without executing?

A: The script currently integrates compilation and execution. To compile only, use the cmake and make commands directly:

```bash
cd {HCCL_VM path}/build
cmake .. && make -j8 test_checker
```

### Q: How do I view detailed output of a specific test?

A: Check the `run.log` file. It contains complete gtest output for each test.

### Q: What should I do if a test times out?

A: The default timeout is 60 seconds. You can modify the `timeout_sec` parameter in the script.

### Q: How do I add a new test?

A: Create a `*_test.cc` file in the corresponding directory and add a compilation target in the corresponding CMakeLists.txt.

## Viewing the Coverage Report

After generating the HTML coverage report, start an HTTP server on Linux to view it:

```bash
cd <path to coverage_report/html>
python3 -m http.server 8080
```

- Access from a local browser: `http://localhost:8080`.
- For remote servers: use SSH port forwarding and access locally.

```bash
ssh -L 8080:localhost:8080 <user>@<server_ip>
# Then access http://localhost:8080 from your local browser
```

Press the `^C` key combination to stop the server.

## Version History

- v1.0 - Initial release. Supports full, directory, file, and test case level execution.
- v2.0 - Optimized command-line arguments. Added automatic path resolution and three-step workflow logging.
- v2.1 - Added the `--cov` parameter. Supports gcov/lcov coverage HTML report generation.
