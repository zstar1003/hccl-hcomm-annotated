/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <vector>
#include <tuple>
#include <iostream>

#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "coll_native_executor_base.h"

#include "checker.h"
using namespace checker;

// 本测试用例用于看护HCCL_ALGO算法配置，全量覆盖level1和level2所有可配置项
constexpr u32 RANK_NUM_LIMIT = 32;
class ExecutorTest :
    public ::testing::TestWithParam<std::tuple<
        CheckerOpType,           // 0. 算子类型
        CheckerDataType,         // 1. 数据类型
        CheckerReduceOp,         // 2. 规约类型
        uint64_t,                // 3. 数据量
        int,                     // 4. 根节点rank号
        int,                     // 5. 对称拓扑，pod数（作为GenTopoMeta的输入）
        int,                     // 6. 对称拓扑，server数（作为GenTopoMeta的输入）
        int,                     // 7. 对称拓扑，die数（作为GenTopoMeta的输入）
        TopoMeta,                // 8. 自定义拓扑（上一个参数为空时，本参数才有效）
        std::string,             // 9. level1算法，用于配置HCCL_ALGO
        std::string,             // 10. level2算法，用于配置HCCL_ALGO
        CheckerOpMode,           // 11. 算子模式（图模式OFFLOAD，单算子OPBASE）
        bool,                    // 12. 支持零拷贝
        CheckerDevType,          // 13. 设备类型
        std::map<string, string> // 14. 环境变量设置
        >> {
public:
    static void SetUpTestCase() { std::cout << "ExecutorTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ExecutorTest tear down." << std::endl; }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName
            = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
    }

    virtual void TearDown()
    {
        Checker::SetDumpFileName("analysis_result");
        // GlobalMockObject::verify();
        ClearHcclEnv();
    }
};

TEST_P(ExecutorTest, Test_ExecutorTest)
{
    // 收集入参
    const auto& settingTuple = GetParam();
    CheckerOpType opType = std::get<0>(settingTuple);
    CheckerDataType dataType = std::get<1>(settingTuple);
    CheckerReduceOp reduceType = std::get<2>(settingTuple);
    uint64_t dataSize = std::get<3>(settingTuple);
    int root = std::get<4>(settingTuple);
    int uniformTopoPodNum = std::get<5>(settingTuple);
    int uniformTopoServerNum = std::get<6>(settingTuple);
    int uniformTopoDieNum = std::get<7>(settingTuple);
    TopoMeta userDefinedTopo = std::get<8>(settingTuple);
    std::string level1AlgoStr = std::get<9>(settingTuple);
    std::string level2AlgoStr = std::get<10>(settingTuple);
    CheckerOpMode opMode = std::get<11>(settingTuple);
    bool supportZeroCopy = std::get<12>(settingTuple);
    CheckerDevType devType = std::get<13>(settingTuple);
    std::map<string, string> envSettings = std::get<14>(settingTuple);

    // // 打印基本参数
    // std::cout << "OpType: " << opType << std::endl;
    // std::cout << "DataType: " << dataType << std::endl;
    // std::cout << "ReduceType: " << reduceType << std::endl;
    // std::cout << "Level1AlgoStr: " << level1AlgoStr << std::endl;
    // std::cout << "Level2AlgoStr: " << level2AlgoStr << std::endl;
    // std::cout << "OpMode: " << opMode << std::endl;
    // std::cout << "SupportZeroCopy: " << supportZeroCopy << std::endl;
    // std::cout << "DeviceType: " << devType << std::endl;

    // 构造topo并打印，优先使用自定义拓扑
    TopoMeta topoMeta;
    if (userDefinedTopo == TopoMeta()) {
        // std::cout << "UniformTopo: " << uniformTopoPodNum << ", " << uniformTopoServerNum << ", " <<
        // uniformTopoDieNum << std::endl;
        RankTable_For_LLT gen;
        gen.GenTopoMeta(topoMeta, uniformTopoPodNum, uniformTopoServerNum, uniformTopoDieNum);
    } else {
        topoMeta = userDefinedTopo;
    }
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);

    // 检查总卡数是否超规格，避免用例执行时间过长
    if (rankNum > RANK_NUM_LIMIT) {
        std::cout << "RankNum[" << rankNum << "] > RANK_NUM_LIMIT[" << RANK_NUM_LIMIT << "], skip testcase"
                  << std::endl;
        return;
    }

    // 检查和修正root参数
    // std::cout << "Root: " << root << std::endl;
    switch (opType) {
        case CheckerOpType::REDUCE:    // fall through
        case CheckerOpType::BROADCAST: // fall through
        case CheckerOpType::SCATTER:
            for (; root < 0; root += rankNum)
                ;
            root %= rankNum;
            break;
        default:
            if (root != -1) {
                root = -1;
            }
            break;
    }
    // std::cout << "RealRoot: " << root << std::endl;

    // 检查规约类型
    switch (opType) {
        // Reduce类算子忽略REDUCE_RESERVED
        case CheckerOpType::REDUCE:         // fall through
        case CheckerOpType::REDUCE_SCATTER: // fall through
        case CheckerOpType::ALLREDUCE:
            if (reduceType == CheckerReduceOp::REDUCE_RESERVED) {
                std::cout << "reduceType[" << reduceType << "] not support for op[" << opType << "], skip testcase"
                          << std::endl;
                return;
            }
            break;
        // 非Reduce类算子只处理REDUCE_RESERVED
        default:
            if (reduceType != CheckerReduceOp::REDUCE_RESERVED) {
                reduceType = CheckerReduceOp::REDUCE_RESERVED;
            }
            break;
    }

    // 计算count
    // std::cout << "DataSize: " << dataSize << std::endl;
    switch (opType) {
        case CheckerOpType::ALLGATHER:      // fall through
        case CheckerOpType::REDUCE_SCATTER: // fall through
        case CheckerOpType::GATHER:         // fall through
        case CheckerOpType::SCATTER:
            dataSize /= rankNum;
            break;
        default:
            break;
    }
    u64 count = dataSize / SIZE_TABLE[dataType];
    // std::cout << "Count: " << count << std::endl;

    // 算法选择
    std::string hcclAlgoStr = "";
    if (!level1AlgoStr.empty()) {
        hcclAlgoStr += "level1:" + level1AlgoStr;
    }
    if (!level2AlgoStr.empty()) {
        if (!hcclAlgoStr.empty()) {
            hcclAlgoStr += ";";
        }
        hcclAlgoStr += "level2:" + level2AlgoStr;
    }
    // std::cout << "Set env HCCL_ALGO=" << hcclAlgoStr << std::endl;
    setenv("HCCL_ALGO", hcclAlgoStr.c_str(), 1);

    // 设置环境变量
    for (auto envSet : envSettings) {
        std::cout << "Set env " << envSet.first << "=" << envSet.second << std::endl;
        setenv(envSet.first.c_str(), envSet.second.c_str(), 1);
    }

    // 零拷贝与图模式互斥，跳过
    if (supportZeroCopy && opMode == OFFLOAD) {
        std::cout << "supportZeroCopy[" << supportZeroCopy << "] not support for opMode[" << opMode
                  << "], skip testcase" << std::endl;
        return;
    }

    // 执行checker
    CheckerOpParam checkerOpParam;
    checkerOpParam.tag = std::to_string(opType);
    checkerOpParam.opType = opType;
    checkerOpParam.DataDes.dataType = dataType;
    checkerOpParam.DataDes.count = count;
    checkerOpParam.reduceType = reduceType;
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = devType;
    checkerOpParam.supportZeroCopy = supportZeroCopy;
    checkerOpParam.root = root;
    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(
    HcclAlgoEnvTest, ExecutorTest,
    testing::Combine(
        // 0. 算子类型
        testing::Values(
            CheckerOpType::ALLGATHER, CheckerOpType::REDUCE_SCATTER, CheckerOpType::ALLREDUCE, CheckerOpType::BROADCAST,
            CheckerOpType::REDUCE, CheckerOpType::SCATTER),
        // 1. 数据类型
        testing::Values(CheckerDataType::DATA_TYPE_FP32),
        // 2. 规约类型
        testing::Values(CheckerReduceOp::REDUCE_SUM),
        // 3. 数据量(Bytes)
        testing::Values(128 * 1024 * 1024), // 较大数据量避免进入小数据量算法
        // 4. 根节点
        testing::Values(0), // 当非root类算子时会自动置为-1
        // 5. 对称拓扑Pod数
        testing::Values(2),
        // 6. 对称拓扑Server数
        testing::Values(2),
        // 7. 对称拓扑Die数
        testing::Values(1), // 每机只出一die，只看护level1和level2的算法配置
        // 8. 自定义拓扑（自定义拓扑优先级高于对称拓扑，如果传入了有效的自定义拓扑，将忽略对称拓扑入参）
        testing::Values(TopoMeta()),
        // 9. level1算法，用于HCCL_ALGO环境变量设置
        testing::Values("ring", "H-D_R", "NHR", "NHR_V1", "NB", "pipeline", "pairwise", "AHC"),
        // 10. level2算法，用于HCCL_ALGO环境变量设置
        testing::Values("ring", "H-D_R", "NHR", "NHR_V1", "NB", "pipeline", "pairwise", "AHC"),
        // 11. 算子模式
        testing::Values(OPBASE, OFFLOAD),
        // 12. 支持零拷贝
        testing::Values(true, false),
        // 13. 设备类型
        testing::Values(CheckerDevType::DEV_TYPE_910_93),
        // 14. 其他环境变量
        testing::Values(std::map<string, string>())));
