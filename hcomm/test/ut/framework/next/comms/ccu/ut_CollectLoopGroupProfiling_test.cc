/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#define private public
#define protected public
#include "ccu_kernel.h"
#include "ccu_rep_loopgroup_bundle_v1.h"
#include "ccu_rep_base_v1.h"
#undef private
#undef protected

using namespace hcomm;
using namespace hcomm::CcuRep;

namespace {
// 构造一个 CcuRepLoopGroupBundle，instrId 可由测试设定。
// Variable(nullptr) 是合法构造（ccu_datatype_v1.h:61，context 默认 nullptr），不会访问外部资源。
std::shared_ptr<CcuRepLoopGroupBundle> MakeLoopGroupBundle(uint16_t instrId)
{
    auto bundle = std::make_shared<CcuRepLoopGroupBundle>(nullptr, Variable(nullptr), Variable(nullptr));
    bundle->instrId = instrId;
    return bundle;
}

// 构造 taskArgs：varId -> argIndex -> taskArgs 下标 -> 值
struct ArgLayout {
    uint16_t loopParamId;
    uint16_t parallelParamId;
    uint16_t residualId;
    uint64_t loopParamVal;
    uint64_t parallelParamVal;
    uint64_t residualVal;
};

// 准备 varId2ArgIndexMap（三个 varId 均映射到各自 taskArgs 下标）和 taskArgs 数组。
// loopParamId -> idx0, parallelParamId -> idx1, residualId -> idx2
void BuildArgMapsAndArgs(
    const ArgLayout& layout, std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, std::vector<uint64_t>& taskArgs)
{
    varId2ArgIndexMap.clear();
    varId2ArgIndexMap[layout.loopParamId] = 0;
    varId2ArgIndexMap[layout.parallelParamId] = 1;
    varId2ArgIndexMap[layout.residualId] = 2;
    taskArgs = {layout.loopParamVal, layout.parallelParamVal, layout.residualVal};
}
} // namespace

class CollectLoopGroupProfilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 每个用例独立构造 kernel，避免状态污染
    }
    void TearDown() override {}

    // 填充三容器，使其长度分别为 repN / goN / profN，push 的 GroupInfo/Profiling 内容由调用方给出。
    void FillContainers(
        CcuKernel& kernel, size_t repN, size_t goN, size_t profN, const GroupInfo& infoTemplate = GroupInfo{})
    {
        auto& lgProfInfo = kernel.GetLGProfilingInfo();
        lgProfInfo.lgProfilingReps.assign(repN, nullptr);
        lgProfInfo.ccuProfilingInfos.assign(profN, CcuProfilingInfo{});
        kernel.groupOpSizeInfo_.assign(goN, infoTemplate);
    }

    // 在 lgProfilingReps 指定下标放入真实 CcuRepLoopGroupBundle（供 dynamic_cast 分支使用）。
    void SetRep(CcuKernel& kernel, size_t idx, uint16_t instrId)
    {
        kernel.GetLGProfilingInfo().lgProfilingReps[idx] = MakeLoopGroupBundle(instrId);
    }

    // 构造 ArgLayout 入参并调用 CollectLoopGroupProfilingInfo，期望 SUCCESS。
    void BuildArgsAndCallKernel(
        CcuKernel& kernel, uint16_t loopParamId, uint16_t parallelParamId, uint16_t residualId, uint64_t loopParamVal,
        uint64_t parallelParamVal, uint64_t residualVal)
    {
        ArgLayout layout{loopParamId, parallelParamId, residualId, loopParamVal, parallelParamVal, residualVal};
        std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
        std::vector<uint64_t> taskArgsVec;
        BuildArgMapsAndArgs(layout, varId2ArgIndexMap, taskArgsVec);
        std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
        EXPECT_EQ(
            kernel.CollectLoopGroupProfilingInfo(
                taskArgsVec.data(), taskArgsVec.size(), varId2ArgIndexMap, varId2VarIdMap),
            HcclResult::HCCL_SUCCESS);
    }

    // 构造 parallelParam(repeatNum=3, residual=7) 入参并调用，期望 SUCCESS。
    // parallelParam 构造：repeatNum 占 [61:55] 共 7 bit，设 repeatNum=3 → parallelParam = 3<<55
    void
    SetupParallelParamAndCall(CcuKernel& kernel, uint16_t loopParamId, uint16_t parallelParamId, uint16_t residualId)
    {
        kernel.moConfig_.memSlice = 4;
        uint64_t parallelParam = static_cast<uint64_t>(3) << 55;
        BuildArgsAndCallKernel(kernel, loopParamId, parallelParamId, residualId, 0, parallelParam, 7);
    }
};

// 决策点#2(空): safeSize=0，循环不进入
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_AllContainersEmpty_Expect_SuccessNoOutput)
{
    CcuKernel kernel;
    FillContainers(kernel, 0, 0, 0);
    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    uint64_t taskArgs[1] = {0};

    EXPECT_EQ(
        kernel.CollectLoopGroupProfilingInfo(taskArgs, 1, varId2ArgIndexMap, varId2VarIdMap), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.allCcuProfilingInfos_.empty());
}

// 决策点#1(mismatch): repSize=4 但 goSizeNum=2、profSize=2 → safeSize=2 != repSize=4，触发 warning 分支
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_SizeMismatch_Expect_SafeSizeTruncate)
{
    CcuKernel kernel;
    FillContainers(kernel, 4, 2, 2);
    // argSize=0 使循环体内提前返回，聚焦验证 safeSize 截断不越界
    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    uint64_t taskArgs[1] = {0};

    EXPECT_EQ(
        kernel.CollectLoopGroupProfilingInfo(taskArgs, 0, varId2ArgIndexMap, varId2VarIdMap), HcclResult::HCCL_SUCCESS);
    // safeSize=2 → 循环 1 轮(i=0)，但 argSize=0 体内直接返回，无 push
    EXPECT_TRUE(kernel.allCcuProfilingInfos_.empty());
}

// 决策点#1(match): 三容器等长，safeSize==repSize，不触发 warning
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_SizesEqual_Expect_NoTruncate)
{
    CcuKernel kernel;
    FillContainers(kernel, 2, 2, 2);
    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    uint64_t taskArgs[1] = {0};

    // argSize=0 让循环体提前返回，聚焦验证等长路径不异常
    EXPECT_EQ(
        kernel.CollectLoopGroupProfilingInfo(taskArgs, 0, varId2ArgIndexMap, varId2VarIdMap), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.allCcuProfilingInfos_.empty());
}

// 决策点#3(argSize=0): 已由 Ut_CollectLGProf_When_SizeMismatch 和 Ut_CollectLGProf_When_SizesEqual 覆盖
// （两者均 argSize=0 触发早返回），此处不再重复。

// 决策点#3(map空): varId2ArgIndexMap 为空，循环体提前返回
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_VarIdMapEmpty_Expect_SkipAndSuccess)
{
    CcuKernel kernel;
    FillContainers(kernel, 2, 2, 2);
    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap; // 空
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    uint64_t taskArgs[1] = {0};

    EXPECT_EQ(
        kernel.CollectLoopGroupProfilingInfo(taskArgs, 1, varId2ArgIndexMap, varId2VarIdMap), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(kernel.allCcuProfilingInfos_.empty());
}

// 决策点#4(true): loopParam!=0 → dataSize 赋值 + push，需要 lgProfilingReps[i] 为真实 bundle
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_LoopParamNonZero_Expect_DataSizePushed)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 0, 0x1234);

    // moConfig_ 默认 {0xFF.., 0xFF.., 0xFF..}，为可控断言设小值
    kernel.moConfig_.loopCount = 2;
    kernel.moConfig_.memSlice = 3;

    BuildArgsAndCallKernel(kernel, loopParamId, parallelParamId, residualId, 5, 0, 0); // loopParam=5, parallelParam=0

    ASSERT_EQ(kernel.allCcuProfilingInfos_.size(), 1U);
    // dataSize = loopParam(5) * loopCount(2) * memSlice(3) = 30
    EXPECT_EQ(kernel.allCcuProfilingInfos_[0].dataSize, 5ULL * 2ULL * 3ULL);
    EXPECT_EQ(kernel.allCcuProfilingInfos_[0].instrId, 0x1234U);
}

// 决策点#4(false): loopParam=0 → 不进入赋值分支，不 push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_LoopParamZero_Expect_NoPush)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 0, 0x1234);

    BuildArgsAndCallKernel(kernel, loopParamId, parallelParamId, residualId, 0, 0, 0); // 全 0
    EXPECT_TRUE(kernel.allCcuProfilingInfos_.empty());
}

// 决策点#5(true)+#6(false): parallelParam!=0 且 i+1<repSize(配对 rep 存在) → push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_ParallelNonZero_PairedRep_Expect_Push)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    FillContainers(kernel, 2, 2, 2, info);
    SetRep(kernel, 1, 0x5678); // i+1=1 处的配对 rep

    SetupParallelParamAndCall(kernel, loopParamId, parallelParamId, residualId);

    ASSERT_EQ(kernel.allCcuProfilingInfos_.size(), 1U);
    // dataSize = repeatNum(3) * memSlice(4) + residual(7) = 19
    EXPECT_EQ(kernel.allCcuProfilingInfos_[0].dataSize, 3ULL * 4ULL + 7ULL);
    EXPECT_EQ(kernel.allCcuProfilingInfos_[0].instrId, 0x5678U);
}

// 决策点#5(true)+#6(true): parallelParam!=0 但 repSize 为奇数 → i+1>=repSize → 跳过不 push
TEST_F(CollectLoopGroupProfilingTest, Ut_CollectLGProf_When_ParallelNonZero_NoPairedRep_Expect_Skip)
{
    CcuKernel kernel;
    constexpr uint16_t loopParamId = 100;
    constexpr uint16_t parallelParamId = 101;
    constexpr uint16_t residualId = 102;
    GroupInfo info{loopParamId, parallelParamId, residualId};
    // repSize=1(奇数), goSize=1, prof=1 → safeSize=1，循环 i=0 进入
    FillContainers(kernel, 1, 1, 1, info);

    SetupParallelParamAndCall(kernel, loopParamId, parallelParamId, residualId);
    // parallelParam!=0 但 i+1=1 >= repSize=1 → 跳过，无 push；loopParam=0 也不 push
    EXPECT_TRUE(kernel.allCcuProfilingInfos_.empty());
}
