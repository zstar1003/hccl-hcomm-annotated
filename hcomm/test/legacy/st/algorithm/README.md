# 算法分析器工具使用指南

## 工具简介

本文档仅用于指导用户编译、运行本目录下的算法分析器用例。详细的工具原理、用例编写、参数设置、结果解析和问题定位等内容请参照[算法分析器工具用户指南](../../../st/algorithm/README.md)。


## 环境准备

参照[源码构建](../../../../docs/zh/build/build.md)中的环境准备，安装CANN Toolkit开发套件包，进行算法分析器编译前的准备工作。

## 用例执行

在hcomm源码仓根目录下，按如下命令，编译并执行算法分析器用例：

```bash
# 编译所有测试套用例，并自动执行
bash build.sh --legacy_all_testcase

# 编译单个测试套用例，并自动执行
bash build.sh --legacy_aicpu_2d_testcase
bash build.sh --legacy_ccu_2d_testcase
bash build.sh --legacy_ccu_1d_hf16p_testcase
bash build.sh --legacy_ccu_1d_testcase_part1
bash build.sh --legacy_ccu_1d_testcase_part2
bash build.sh --legacy_alg_ccu_reduce
bash build.sh --legacy_function_ut_testcase
bash build.sh --legacy_alg_testcase

# 手动重新执行测试用例
./build/test/legacy/st/algorithm/testcase/aicpu_2d_testcase/legacy_alg_aicpu_2d_testcase
./build/test/legacy/st/algorithm/testcase/ccu_2d_testcase/legacy_alg_ccu_2d_testcase
./build/test/legacy/st/algorithm/testcase/ccu_1d_hf16p_testcase/legacy_alg_ccu_1d_hf16p_testcase
./build/test/legacy/st/algorithm/testcase/ccu_1d_testcase_part1/legacy_alg_ccu_1d_testcase_part1
./build/test/legacy/st/algorithm/testcase/ccu_1d_testcase_part2/legacy_alg_ccu_1d_testcase_part2
./build/test/legacy/st/algorithm/testcase/ccu_reduce_testcase/legacy_alg_ccu_reduce
./build/test/legacy/st/algorithm/testcase/function_ut_testcase/legacy_alg_function_ut_testcase
./build/test/legacy/st/algorithm/testcase/legacy_alg_testcase/legacy_alg_testcase
```
