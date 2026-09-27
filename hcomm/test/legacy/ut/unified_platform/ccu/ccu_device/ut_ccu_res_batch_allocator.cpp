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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>

#define private public
#define protected public
#include "ccu_res_batch_allocator_legacy.h"

#include "hccl_common_v2.h"
#include "ccu_component.h"

#undef private
#undef protected

using namespace Hccl;

/*
 * 重要事项请注意：
 * 因单例不容易实现完全打桩，ccu_component用例运行后
 * 部分单例已在内存中，故以下用例调整需注意设备号
 */
extern void MockCcuResources(const int32_t devLogicId, const CcuVersion ccuVersion);
extern void MockCcuNetworkDevice(const int32_t devLogicId);

class CcuResBatchAllocatorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResBatchAllocatorTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResBatchAllocatorTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResBatchAllocatorTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResBatchAllocatorTest TearDown" << std::endl;
    }
};

void CheckRes(CcuResRepository& ccuResRepo)
{
    std::cout << "------------------------" << std::endl;
    for (int i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        std::cout << "BlockMS: ";
        auto blockMsInfos = ccuResRepo.blockMs[i];
        for (int j = 0; j < blockMsInfos.size(); j++) {
            std::cout << blockMsInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "BlockLoop: ";
        auto blockLoopInfos = ccuResRepo.blockLoopEngine[i];
        for (int j = 0; j < blockLoopInfos.size(); j++) {
            std::cout << blockLoopInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "BlockCke: ";
        auto blockCkeInfos = ccuResRepo.blockCke[i];
        for (int j = 0; j < blockCkeInfos.size(); j++) {
            std::cout << blockCkeInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "MS: ";
        auto msInfos = ccuResRepo.ms[i];
        for (int j = 0; j < msInfos.size(); j++) {
            std::cout << msInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Loop: ";
        auto loopInfos = ccuResRepo.loopEngine[i];
        for (int j = 0; j < loopInfos.size(); j++) {
            std::cout << loopInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Cke: ";
        auto ckeInfos = ccuResRepo.cke[i];
        for (int j = 0; j < ckeInfos.size(); j++) {
            std::cout << ckeInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Xn: ";
        auto xnInfos = ccuResRepo.xn[i];
        for (int j = 0; j < xnInfos.size(); j++) {
            std::cout << xnInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Gsa: ";
        auto gsaInfos = ccuResRepo.gsa[i];
        for (int j = 0; j < gsaInfos.size(); j++) {
            std::cout << gsaInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Mission: ReqType: " << (int)ccuResRepo.mission.reqType << " ";
        auto missionInfos = ccuResRepo.mission.mission[i];
        for (int j = 0; j < missionInfos.size(); j++) {
            std::cout << missionInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------" << std::endl;
}

void DumpBlockResInfo(ResType resType, const std::vector<BlockInfo>& blocks)
{
    HCCL_INFO("Dump ResType[%s] block resources info: ", resType.Describe().c_str());
    uint32_t blockNum = blocks.size();
    for (size_t k = 0; k < blockNum; k++) {
        HCCL_INFO(
            "Block[id[%u], startId[%u], num[%u], handle(uintptr_t)[%llu], allocated[%d]]", blocks[k].id,
            blocks[k].startId, blocks[k].num, blocks[k].handle, static_cast<int>(blocks[k].allocated));
    }
}

void MockerCcuComponent(const int32_t devLogicId, const CcuVersion ccuVersion)
{
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);
    EXPECT_NO_THROW(CcuComponent::GetInstance(devLogicId).Init());
}

TEST_F(CcuResBatchAllocatorTest, Ut_Init_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());
}

TEST_F(CcuResBatchAllocatorTest, Ut_Init_When_AX_Mainboard_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 3; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);
    auto& ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    ccuResSpecs.isAX = true;

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());
    constexpr uint32_t BLOCK_SIZE_MS_AX_DIE0 = 128;
    EXPECT_EQ(allocater.resStrategies[0].msNum, BLOCK_SIZE_MS_AX_DIE0);
    ASSERT_GE(allocater.resBlocks[0].size(), 3u);
    DumpBlockResInfo(ResType::LOOP, allocater.resBlocks[0][ResType::LOOP]);
    DumpBlockResInfo(ResType::MS, allocater.resBlocks[0][ResType::MS]);
    DumpBlockResInfo(ResType::CKE, allocater.resBlocks[0][ResType::CKE]);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq;
    resReq.blockLoopEngineReq[0] = 1;
    resReq.loopEngineReq[0] = 2;

    resReq.blockCkeReq[0] = 65;
    resReq.ckeReq[0] = 128;

    resReq.blockMsReq[0] = 512;

    resReq.blockGsaReq[0] = 512;
    resReq.gsaReq[0] = 3 * 16 * 2; // RESERVED_DISCRETE_GSA_NUM = 3*16*2

    resReq.missionReq.req[0] = {3};

    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    CcuResRepository ccuResRepo;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    CheckRes(ccuResRepo);

    // 释放其他资源避免影响其他用例
    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1AndResNumIsEmpty_Expect_Return_ErrorPara)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq; // 检查空申请
    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1AndResNumIsMaxNum_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq;
    resReq.blockCkeReq[1] = 0;
    resReq.blockLoopEngineReq[0] = 192;
    resReq.loopEngineReq[0] = 8;                                 // {8, 8};
    resReq.blockMsReq[0] = 1536;                                 // {1536, 1536};
    resReq.blockCkeReq[0] = 128;                                 // {128, 128};
    resReq.ckeReq[0] = 4 * 128 + 2 * 16 * 2;                     // RESERVED_DISCRETE_CKE_NUM = 4*128+2*16*2
    resReq.gsaReq[0] = 3 * 16 * 2;                               // RESERVED_DISCRETE_GSA_NUM = 3*16*2
    resReq.xnReq[0] = 4 * 128 + 36 * 16 + 5 * 16 + (4 * 16) * 2; // RESERVED_DISCRETE_XN_NUM = 4*128+36*16+5*16+(4*16)*2
    resReq.missionReq.req[0] = 16;                               // {16, 16};

    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    CcuResRepository ccuResRepo;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    CheckRes(ccuResRepo);

    // 释放其他资源避免影响其他用例
    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1AndResNumExceedsLeftNum_Expect_Return_ErrorUnavalible)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq;
    resReq.blockLoopEngineReq[0] = 1; // {1, 0};
    resReq.loopEngineReq[0] = 2;      // {2, 3};

    resReq.blockLoopEngineReq[0] = 1;
    resReq.loopEngineReq[0] = 2;
    resReq.missionReq.req[0] = 2;

    resReq.blockCkeReq[0] = 65; // {65, 0};
    resReq.ckeReq[0] = 129;     // {129, 0};

    // 1. 资源申请超过了一半，故第二次申请资源会不足
    resReq.blockMsReq[0] = 64 * 13; // {64 * 13, 0};

    resReq.missionReq.req[0] = 3; // {3, 3};
    resReq.missionReq.req[1] = 2; // 会选用较多的，即 3

    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    CcuResHandle errorHandle;
    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    // 2. 申请超过剩余资源的loop
    resReq = {};                  // 重置错误的请求
    resReq.loopEngineReq[0] = 50; // 申请超过剩余资源
    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    // 3. 申请超过mission规格的mission
    resReq = {}; // 重置错误的请求
    resReq.missionReq.req[0] = 17;
    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    // 释放其他资源避免影响其他用例
    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 4. mission资源申请超过一半，故第二次申请资源会不足
    resReq = {};
    resReq.missionReq.req[0] = 9;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetResourceAndReleaseResHandle_When_resHandleIsInvalid_Expect_Return_ErrorPara)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator& allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResHandle handle = nullptr;
    CcuResRepository ccuResRepo;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = allocater.ReleaseResHandle(handle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    handle = (CcuResHandle)0x89674878;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocConsecutiveRes_When_AllocRes_fail_Expect_HCCL_E_PARA)
{
    // 前置条件
    MOCKER_CPP(&CcuComponent::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));
    CcuResReq resReq;
    resReq.xnReq[0] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();
    resRepoPtr->xn[0].push_back(ResInfo(5, 3));
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.AllocConsecutiveRes(resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CcuResBatchAllocatorTest, Ut_TryAllocResHandle_When_AllocContinuousRes_fail_Expect_HCCL_E_PARA)
{
    // 前置条件
    MOCKER_CPP(&CcuResBatchAllocator::AllocBlockRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuResBatchAllocator::CcuMissionMgr::Alloc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));
    std::unique_ptr<CcuResRepository> resRepoPtr = std::make_unique<CcuResRepository>();
    uintptr_t handleKey = reinterpret_cast<uintptr_t>(resRepoPtr.get());
    CcuResReq resReq;
    resReq.xnReq[0] = 1;
    resRepoPtr->xn[0].push_back(ResInfo(5, 3));
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.TryAllocResHandle(handleKey, resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocContinuousRes_When_AllocRes_success_Expect_HCCL_SUCCESS)
{
    // 前置条件
    MOCKER_CPP(&CcuComponent::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CcuResReq resReq;
    auto resRepoPtr = std::make_unique<CcuResRepository>();
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.AllocConsecutiveRes(resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_TryAllocResHandle_When_AllocContinuousRes_success_Expect_HCCL_SUCCESS)
{
    // 前置条件
    MOCKER_CPP(&CcuResBatchAllocator::AllocBlockRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuResBatchAllocator::CcuMissionMgr::Alloc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<CcuResRepository> resRepoPtr = std::make_unique<CcuResRepository>();
    uintptr_t handleKey = reinterpret_cast<uintptr_t>(resRepoPtr.get());
    CcuResReq resReq;
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.TryAllocResHandle(handleKey, resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_NoAllocations_Expect_MaxGapEqualsTotalCapacity)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    // 池 [0,64), 4 blocks × 16 (strategy 为 16)
    allocator.resStrategies[0].loopNum = 16;
    auto trategy = allocator.resStrategies[0].loopNum;
    allocator.maxResBlockNums.loopNum = 4; // poolSize = 4 * 16 = 64
    uint32_t totalSize = trategy * allocator.maxResBlockNums.loopNum;
    allocator.resBlocks[0][ResType::LOOP] = {
        {0, 0, trategy, 0, false},
        {1, trategy, trategy, 0, false},
        {2, trategy * 2, trategy, 0, false},
        {3, trategy * 3, trategy, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remainNum, totalSize) << "no allocations -> max gap = total capacity 64";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_WithAllocations_Expect_MaxContiguousGap)
{
    // 池 8 blocks × 8 = 64, 标记部分 block 已分配
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    allocator.maxResBlockNums.loopNum = 8; // poolSize = 8 * 8 = 64
    allocator.resBlocks[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},   // 已分配
        {1, 8, 8, 0, true},   // 已分配
        {2, 16, 8, 0, false}, // 空闲
        {3, 24, 8, 0, true},  // 已分配
        {4, 32, 8, 0, false}, // 空闲
        {5, 40, 8, 0, false}, // 空闲
        {6, 48, 8, 0, false}, // 空闲
        {7, 56, 8, 0, false}, // 空闲
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // blocks 3-7 连续空闲: 4 × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_AllAllocated_Expect_Zero)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    allocator.maxResBlockNums.loopNum = 4; // poolSize = 4 * 8 = 32
    allocator.resBlocks[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},
        {1, 8, 8, 0, true},
        {2, 16, 8, 0, true},
        {3, 24, 8, 0, true},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remainNum, 0u) << "all allocated -> zero remaining";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_HoleAtStart_Expect_InitialGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    allocator.maxResBlockNums.loopNum = 6; // poolSize = 6 * 8 = 48
    allocator.resBlocks[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},   // allocated
        {1, 8, 8, 0, true},   // allocated
        {2, 16, 8, 0, false}, // free
        {3, 24, 8, 0, false}, // free
        {4, 32, 8, 0, false}, // free
        {5, 40, 8, 0, false}, // free
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 free blocks × 8 = 32
    EXPECT_EQ(remainNum, 32u) << "max gap after first two allocated blocks";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_HoleAtEnd_Expect_TailGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    allocator.maxResBlockNums.loopNum = 6;
    allocator.resBlocks[0][ResType::LOOP] = {
        {0, 0, 8, 0, false},  {1, 8, 8, 0, false}, {2, 16, 8, 0, false},
        {3, 24, 8, 0, false}, {4, 32, 8, 0, true}, {5, 40, 8, 0, true},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 free blocks × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

// ── 预预留Block (A+X CCUA0 MS blocks, allocated=true, handle=0) ──

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_PreReservedMSBlocks_Expect_MaxGapReducedByReserved)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    allocator.resStrategies[0].msNum = 64;
    allocator.maxResBlockNums.msNum = 4; // poolSize = 4 * 64 = 256

    // Block0: 预预留(allocated), Block1: 空闲, Block2: 已分配, Block3: 空闲
    allocator.resBlocks[0][ResType::MS] = {
        {0, 0, 64, 0, true},
        {1, 64, 64, 0, false},
        {2, 128, 64, 0, true},
        {3, 192, 64, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::MS, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 已分配: Block0(allocated), Block2(allocated)
    // 最大连续空闲: 单 block = 64
    EXPECT_EQ(remainNum, 64u) << "pre-reserved + allocated block leave max gap 64";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_PoolStartsAtNonZero_Expect_GapWithinPoolOnly)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags[0] = true;
    allocator.devLogicId = 0;

    // 模拟 mission 预分配池 [100, 356), 4 blocks of 64
    allocator.missionMgr.blocks = {
        {0, 100, 64, 0, true},  // 已分配
        {1, 164, 64, 0, false}, // 空闲
        {2, 228, 64, 0, true},  // 已分配
        {3, 292, 64, 0, false}, // 空闲
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::MISSION, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 最大连续空闲: 单 block = 64
    EXPECT_EQ(remainNum, 64u) << "max gap 64 within pool only";
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetConsecutiveRemainSize_When_PartialAlloc_Expect_MaxGap)
{
    CcuResIdAllocator allocator(100);
    allocator.resInfos = {{0, 20}, {70, 20}};

    EXPECT_EQ(allocator.GetConsecutiveRemainSize(), 50u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetConsecutiveRemainSize_When_Full_Expect_Zero)
{
    CcuResIdAllocator allocator(50);
    std::vector<ResInfo> resInfos;
    allocator.Alloc(50, true, resInfos);

    EXPECT_EQ(allocator.GetConsecutiveRemainSize(), 0u);
}
