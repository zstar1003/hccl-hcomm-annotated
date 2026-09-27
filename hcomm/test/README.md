# HCCL LLT

简体中文 | [English](./README_en.md)

## 概述

HCCL LLT（Low Level Test）是 HCCL 的测试框架，旨在系统化验证 HCCL 各层级组件的功能完整性与性能稳定性。LLT 涵盖了 HCCL 的算法层、框架层、平台层、对外接口层多个部分，通过全面的测试用例，保障 HCCL 在各种业务场景下的可靠性和高效性。

## 目录结构

```text
test/
├── legacy                                          # 历史版本兼容测试框架
│   ├── common                                      # 公共处理
│   ├── depends                                     # 测试依赖其他组件头文件
│   ├── st                                          # ST 集成测试用例
│   │   ├── algorithm                               # 通信算法测试用例
│   │   ├── fwk                                     # 通信框架测试用例
│   │   ├── service                                 # 服务层测试用例
│   │   └── test_case                               # 测试用例
│   └── ut                                          # UT 集成测试用例
│       ├── aicpu                                   # AICPU 专用测试用例
│       ├── all_source_code                         # 源码文件路径
│       ├── common                                  # 公共处理
│       ├── framework                               # 通信框架测试用例
│       ├── service                                 # 服务层测试用例
│       └── unified_platform                        # 统一平台层测试用例
├── st/algorithm                                    # ST 集成测试用例 (算法分析器)
│   ├── testcase                                    # 测试用例
│   └── utils                                       # 通用处理
└── ut                                              # UT 单元测试用例
    ├── aicpu_kfc                                   # MC2相关测试
    ├── common                                      # 公共处理
    ├── depends                                     # 测试依赖其他组件头文件
    ├── device                                      # device测试
    ├── framework                                   # 通信框架测试用例
    ├── impl                                        # 通信算法实现测试用例
    ├── inter                                       # 接口适配层测试用例
    ├── misc                                        # 其他测试用例
    ├── platform                                    # 通信平台实现测试用例
    └── stub                                        # 测试桩函数
```

## 编译与运行

在仓库根目录下执行如下命令：

```bash
# 编译并运行所有单元测试用例
bash build.sh --ut

# 编译并运行所有集成测试用例
bash build.sh --st
# 编译并运行单个测试套用例
bash build.sh --open_hccl_test
bash build.sh --executor_hccl_test
bash build.sh --executor_reduce_hccl_test
bash build.sh --executor_pipeline_hccl_test
# 手动执行测试用例
./build/test/st/algorithm/testcase/testcase/open_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_reduce_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_pipeline_hccl_test
```

## 可执行文件位置

所有可执行文件默认输出至 `build/test` 目录，可通过修改 `CMakeLists.txt` 调整路径：

```cmake
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${HCCL_OPEN_CODE_ROOT}/build/test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HCCL_OPEN_CODE_ROOT}/build/test)
```

## 如何编写测试用例

> HCCL LLT 测试用例采用 Google Test 框架实现，详细编写规范请参阅 [Google Test 用户指南](https://google.github.io/googletest/)。

1. 根据测试对象选择对应目录，如 `test/algorithm` 或 `test/framework`
2. 基于 Google Test 创建新测试类
3. 更新对应目录的 `CMakeLists.txt`，添加测试入口

测试代码示例：

```c++
#include "gtest/gtest.h"

TEST(MyTestClass, MyTestCase) {
    // 实现断言逻辑
    EXPECT_EQ(actual_value, expected_value);
}
```

## 如何运行特定的测试用例

若想要单独运行某些测试用例，可参考 [Google Test 用户指南](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)，执行用例时添加 `--gtest_filter` 参数即可。

以 `hccl_utest_framework_op_base_api` 为例：

```bash
# 只执行 HcclCommInitRootInfoTest 测试类的测试用例
./build/test/hccl_utest_framework_op_base_api --gtest_filter=HcclCommInitRootInfoTest.*
```
