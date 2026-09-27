# HCCL_VM  UT 测试执行脚本

## 概述

`run_ut.sh` 是 HCCL_VM 项目中的单元测试执行脚本，用于自动化编译和执行测试用例。

## 三步流程

脚本执行时自动完成以下三步：

| 步骤 | 说明 | 日志输出 |
|------|------|----------|
| 第一步 | CMake 配置 + make 编译 | `build.log` |
| 第二步 | 生成可执行文件 | 列出所有二进制及大小/时间 |
| 第三步 | 执行用例 + 显示结果 | `run.log` + `summary.log` |
| 第四步 | 生成 gcov/lcov 覆盖率 HTML 报告（需 `--cov`） | `coverage.log` |

## 用法

```bash
cd {HCCL_VM路径}/test

# 基本用法
./run_ut.sh                           # 全量编译+执行所有测试
./run_ut.sh --cov                     # 全量编译+执行+生成gcov/lcov覆盖率HTML报告
./run_ut.sh <目录>                    # 编译+执行指定目录下所有测试(递归)
./run_ut.sh <二进制名>                 # 编译+执行指定二进制测试
./run_ut.sh <测试文件名>               # 编译+执行指定测试文件对应的二进制
./run_ut.sh <文件> <测试用例名>         # 编译+执行指定文件中的单个测试用例

# 其他命令
./run_ut.sh -l, --list                # 列出所有可用测试
./run_ut.sh -h, --help                # 显示帮助信息
```

## 命令详解

### 1. 全量执行

```bash
./run_ut.sh
```

- 执行所有 `test` 目录下的测试。
- 完整三步流程：编译 → 生成可执行文件 → 执行所有测试。
- 适用于完整回归测试。

### 2. 覆盖率报告

```bash
./run_ut.sh --cov
```

- 全量编译 + 执行所有测试 + 生成 gcov/lcov 覆盖率 HTML 报告。
- 自动启用 `--coverage` 编译选项，生成 `.gcno`/`.gcda` 文件。
- 四步流程：编译（含覆盖率插桩）→ 执行 → 收集覆盖率数据 → 生成 HTML 报告。
- 报告输出路径：`$CODE_DIR/coverage_report/html/index.html`。
- 会自动过滤系统头文件、第三方库、stub 文件等非业务代码。

### 3. 目录执行

```bash
./run_ut.sh plugin/checker
./run_ut.sh plugin/ccu_executor
./run_ut.sh store
```

- 递归查找指定目录下所有 `*_test.cc` 文件。
- 编译并执行该目录下所有测试。
- 适用于模块级测试。

### 4. 二进制执行

```bash
./run_ut.sh test_checker
./run_ut.sh test_allgather_semantics_checker
```

- 编译并执行指定的测试二进制。
- 二进制名以 `test_` 开头。

### 5. 文件执行

```bash
./run_ut.sh checker_test.cc
./run_ut.sh allgather_semantics_checker_test.cc
```

- 根据测试文件名自动匹配对应二进制。
- 编译并执行。

### 6. 单个用例执行

```bash
./run_ut.sh checker_test.cc CheckerTest.GenAndCheckGraph_EmptyQueues
./run_ut.sh allgather_semantics_checker_test.cc AllgatherSemanticsCheckerTest.CheckBasic
```

- 执行指定测试文件中的单个测试用例。
- 用例名格式：`TestSuiteName.TestCaseName`。

### 7. 列出测试

```bash
./run_ut.sh -l
./run_ut.sh --list
```

- 列出所有可用测试文件及其状态。
- 显示用例数量和对应二进制名。

## 示例

```bash
# 示例1: 全量测试
./run_ut.sh

# 示例2: 全量测试 + 覆盖率报告
./run_ut.sh --cov

# 示例3: 执行 plugin/checker 目录下所有测试
./run_ut.sh plugin/checker

# 示例4: 编译并执行 test_checker
./run_ut.sh test_checker

# 示例5: 编译并执行 checker_test.cc 对应的二进制
./run_ut.sh checker_test.cc

# 示例6: 执行单个测试用例
./run_ut.sh checker_test.cc CheckerTest.GenAndCheckGraph_EmptyQueues

# 示例7: 列出所有测试
./run_ut.sh -l
```

## 日志目录

每次执行都会在 `ut_logs/<时间戳>/` 下生成日志文件：

```text
{HCCL_VM路径}/ut_logs/20260425_142048/
├── build.log      # 编译详细日志 (cmake + make 输出)
├── run.log        # 执行详细日志 (每个测试的完整输出)
└── summary.log    # 汇总日志 (每个测试的执行结果)
```

### 日志文件说明

| 文件 | 内容 |
|------|------|
| `build.log` | CMake 配置输出、make 编译输出、生成的可执行文件列表 |
| `run.log` | 每个测试的完整输出（包括 gtest 详细信息） |
| `summary.log` | 每个测试的执行状态、通过/失败数量、耗时汇总 |

## 执行结果

脚本执行完成后显示汇总信息：

```text
========================================
  目录测试结果汇总: plugin/checker
========================================
  执行数量:   16
  PASSED:  185
  FAILED:  8
  CRASHED: 0
  TIMEOUT: 0
```

### 状态说明

| 状态 | 说明 |
|------|------|
| `PASS` | 测试通过 |
| `FAIL` | 测试失败（有断言失败） |
| `CRASH` | 测试崩溃（core dump） |
| `TIMEOUT` | 测试超时（默认 60 秒） |

## 目录结构

```text
test/
├── run_ut.sh              # 本脚本
├── cmd/                   # 命令相关测试
│   ├── base/
│   ├── subcmds/
│   └── utils/
├── device_arm/            # 设备相关测试
├── device_vir/
├── ipc/                   # IPC 相关测试
│   └── shm/
├── log/                   # 日志相关测试
├── plugin/                # 插件相关测试
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
├── proxy/                 # 代理相关测试
│   ├── level1/
│   └── level2/
├── runnerdb/              # 数据库相关测试
├── store/                 # 存储相关测试
│   └── hccl_shm/
└── src_root/              # 源码根目录测试
```

## 环境要求

**执行脚本前，必须先修改 `run_ut.sh` 中的以下环境变量，将其替换为实际路径：**

```bash
# 以下为示例路径，请根据实际环境修改
export HCOMM_CODE_HOME=/home/q30033976/checker/hcomm      # hcomm源码路径
export HCCL_CODE_HOME=/home/q30033976/checker/hccl        # hccl源码路径（AIV/AICPU模式需要）
source /home/q30033976/checker/Ascend/cann/set_env.sh     # CANN安装路径下的环境脚本
```

**示例**：假设您的工作目录为 `/home/workspace`，则修改为：

```bash
export HCOMM_CODE_HOME=/home/workspace/hcomm
export HCCL_CODE_HOME=/home/workspace/hccl
source /home/workspace/Ascend/cann/set_env.sh
```

脚本会自动加载这些环境变量，未修改将导致编译失败。

## 配置参数

脚本内置以下配置（可在脚本开头修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | `Debug` | CMake 构建类型 |
| `MAKE_JOBS` | `8` | make 并行编译数 |
| `LOG_DIR` | `$CODE_DIR/ut_logs` | 日志输出目录 |

## 注意事项

1. **首次执行**：首次执行会进行完整的 CMake 配置，耗时较长。
2. **编译失败**：如果编译失败，请检查 `build.log` 查看详细错误信息。
3. **测试失败**：如果测试失败，请检查 `run.log` 查看具体失败的用例。
4. **日志清理**：日志目录按时间戳命名，定期清理旧日志以节省空间。

## 常见问题

### Q: 如何只编译不执行？

A: 目前脚本设计为编译+执行一体化，如需只编译，请直接使用 cmake 和 make 命令：

```bash
cd {HCCL_VM路径}/build
cmake .. && make -j8 test_checker
```

### Q: 如何查看某个测试的详细输出？

A: 查看 `run.log` 文件，包含每个测试的完整 gtest 输出。

### Q: 测试超时怎么办？

A: 默认超时时间为 60 秒，可在脚本中修改 `timeout_sec` 参数。

### Q: 如何添加新的测试？

A: 在对应目录下创建 `*_test.cc` 文件，并在对应 CMakeLists.txt 中添加编译目标。

## 查看覆盖率报告

生成 HTML 覆盖率报告后，在 Linux 下可通过启动 HTTP 服务器查看：

```bash
cd <coverage_report/html所在路径>
python3 -m http.server 8080
```

- 本机浏览器访问：`http://localhost:8080`。
- 远程服务器：通过 SSH 端口转发后本地访问。

```bash
ssh -L 8080:localhost:8080 <user>@<server_ip>
# 然后本地浏览器访问 http://localhost:8080
```

按 `^C` 组合键停止服务器。

## 版本历史

- v1.0 - 初始版本，支持全量/目录/文件/用例级测试执行。
- v2.0 - 优化命令行参数，支持自动路径推导，三步流程日志记录。
- v2.1 - 新增 `--cov` 参数，支持 gcov/lcov 覆盖率 HTML 报告生成。
