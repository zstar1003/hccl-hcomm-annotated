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

#include "ccu_jetty_mgr.h"

#include "hccl_common_v2.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"

#undef private
#undef protected

using namespace Hccl;

class CcuJettyMgrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuJettyMgrTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuJettyMgrTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuJettyMgrTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuJettyMgrTest TearDown" << std::endl;
    }
};

constexpr uint32_t CCU_CHANNLE_GOURP_SIZE = 8; // 真实环境，ccu v1 为 1，ccu v2 根据配比关系确定
constexpr uint32_t CCU_JETTY_GOURP_SIZE = 2; // 真实环境，ccu v1 默认为 1，可变，ccu v2 根据配比关系确定

HcclResult CcuAllocChannelsStub(
    const int32_t deviceLogicId, const CcuChannelPara& channelPara, std::vector<CcuChannelInfo>& channelInfos)
{
    static uint32_t channelCnt = 0;
    static uint32_t jettyCnt = 0;
    constexpr uint64_t fakeMemAddr = 0x12345678;

    const uint32_t startTaJettyId = 1024;
    const uint64_t fakeSqBufVa = fakeMemAddr;
    const uint32_t fakeSqBufSize = 1024;
    const uint32_t fakeSqDepth = 4;
    const IpAddress locAddr{"1.1.1.1"};
    const IpAddress rmtAddr{"2.2.2.2"};

    for (uint32_t i = 0; i < CCU_CHANNLE_GOURP_SIZE; i++) { // 数量不得少于用例需要
        CcuChannelInfo channelInfo;
        channelInfo.channelId = 0 + channelCnt;
        channelCnt += 1; // 保证 channel id 申请总是不同
        channelInfo.dieId = 1;

        vector<unique_ptr<CcuJetty>> ccuJettys;
        vector<CcuJetty*> ccuJettyPtrs;
        for (uint32_t i = 0; i < CCU_JETTY_GOURP_SIZE; i++) {
            CcuJettyInfo jettyInfo;
            jettyInfo.jettyCtxId = 0 + jettyCnt + i; // 保证同一组channel jetty编号一致
            jettyInfo.taJettyId = 0 + jettyCnt + startTaJettyId + i;
            jettyInfo.sqDepth = fakeSqDepth;
            jettyInfo.wqeBBStartId = 16;
            jettyInfo.sqBufVa = fakeSqBufVa + i;
            jettyInfo.sqBufSize = fakeSqBufSize + i;
            channelInfo.jettyInfos.push_back(jettyInfo);
            auto ccuJetty = make_unique<CcuJetty>(locAddr, jettyInfo);
            ccuJettyPtrs.emplace_back(ccuJetty.get());
            ccuJettys.emplace_back(std::move(ccuJetty));
        }

        channelInfos.emplace_back(channelInfo);
    }

    jettyCnt += CCU_JETTY_GOURP_SIZE; // 保证每次调用接口 jettyId id 申请总是不同

    return HcclResult::HCCL_SUCCESS;
}

void MockPlatformDeps(uint32_t ccuVersion)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue((DevId)(MAX_MODULE_DEVICE_NUM)));
    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue((void*)0x12345678));
    pair<uint32_t, uint32_t> fakeDieFuncPair = make_pair(1, 4);
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(fakeDieFuncPair));
    pair<TokenIdHandle, uint32_t> fakeTokenInfo = make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));

    MOCKER(CcuAllocChannels).stubs().will(invoke(CcuAllocChannelsStub));
    HcclResult OkResult = HcclResult::HCCL_SUCCESS;
    MOCKER(CcuReleaseChannel).stubs().will(returnValue(OkResult));
    MOCKER(CcuCreateJetty).stubs().will(returnValue(OkResult));
}

vector<LinkData> MockMultiLinkData(uint32_t baseIpAddrInt, uint32_t num)
{
    vector<LinkData> links;
    for (uint32_t i = 0; i < num; i++) {
        BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
        LinkData link(portType, 0, 1, 0, 1);
        link.localAddr_ = IpAddress(baseIpAddrInt + i);

        for (uint32_t j = 0; j < num; j++) {
            link.remoteAddr_ = IpAddress(baseIpAddrInt + j);
            links.push_back(link);
        }
    }

    return links;
}

TEST_F(CcuJettyMgrTest, Ut_GetChannelJettys_When_InterfaceOk_Expect_Return_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 打桩的channel分配函数，每个本端分配8个channel
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), linkNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), rankNum); // rankNum个远端
    ccuJettyMgr.Confirm();

    // 模拟新一轮算子下发，链路与之前重叠
    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // channel全部复用
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), 0);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), 0);
    ccuJettyMgr.Confirm();

    // 模拟新一轮算子下发，有部分链路与之前重叠
    uint32_t newAddrNum = 2;
    auto newLinks = MockMultiLinkData(baseIpAddrInt - newAddrNum, rankNum + newAddrNum);
    EXPECT_EQ(ccuJettyMgr.PrepareCreate(newLinks), HcclResult::HCCL_SUCCESS);
    for (const auto& link : newLinks) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 新增两条两条连续需要申请
    uint32_t newRankNum = rankNum + newAddrNum;
    uint32_t newLinkNum = newRankNum * newRankNum - (rankNum * rankNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), newLinkNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), newAddrNum);
}

TEST_F(CcuJettyMgrTest, Ut_GetChannelJettysAndNeedAllocResOverOneTime_When_InterfaceOk_Expect_Return_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 10;                                   // 模拟数量超过单轮申请数量
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum); // rank * rank 个链路
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 打桩的channel分配函数，每个本端分配8个channel
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), linkNum);
    // 每个ip需要申请多轮，远端
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), rankNum * (rankNum / CCU_CHANNLE_GOURP_SIZE + 1));
    ccuJettyMgr.Confirm();
}

static void CheckCcuJettyMgrDataEmpty(const CcuJettyMgr& ccuJettyMgr)
{
    EXPECT_EQ(ccuJettyMgr.batchMap_.empty(), true);

    EXPECT_EQ(ccuJettyMgr.allocatedChannelIdMap_.empty(), true);
    EXPECT_EQ(ccuJettyMgr.channelJettyInfoMap_.empty(), true);

    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.empty(), true);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.empty(), true);
}

TEST_F(CcuJettyMgrTest, Ut_AllocResUnavailable_When_InterfaceRetUnavailable_Expect_ResIsEmpty)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);

    MOCKER(CcuAllocChannels).stubs().will(returnValue(HcclResult::HCCL_E_UNAVAIL));

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_E_UNAVAIL);
    CheckCcuJettyMgrDataEmpty(ccuJettyMgr);
}

TEST_F(CcuJettyMgrTest, Ut_ReleaseTempRes_When_CatchUnexpectedThrow_Expect_ResIsEmpty)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    MOCKER_CPP(&CcuJettyMgr::CreateAndSaveNewBatch).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_E_INTERNAL);
    CheckCcuJettyMgrDataEmpty(ccuJettyMgr);
}

TEST_F(CcuJettyMgrTest, Ut_Fallback_When_InterfaceOk_Expect_NoThrow)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 打桩的channel分配函数，每个本端分配8个channel
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), linkNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), rankNum); // rankNum个远端

    EXPECT_NO_THROW(ccuJettyMgr.Fallback());
    CheckCcuJettyMgrDataEmpty(ccuJettyMgr); // 因为是首轮申请所以全空
}

TEST_F(CcuJettyMgrTest, Ut_FallbackAndGetChannelJettys_When_InterfaceOk_Expect_Return_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 打桩的channel分配函数，每个本端分配8个channel
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), linkNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), rankNum); // rankNum个远端
    ccuJettyMgr.Confirm();

    // 以某个ipAddr为例，找到其申请资源的最后批次的剩余资源
    const uint32_t leftResourceNum
        = ccuJettyMgr.batchMap_[links[0].GetLocalAddr()].back()->availableChannelIdKeys.size();
    EXPECT_NE(leftResourceNum, 0); // 当前剩余资源应该不为0

    // 模拟新一轮算子下发，链路与之前重叠
    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // channel全部复用
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), 0);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), 0);

    const uint32_t newLeftResourceNum
        = ccuJettyMgr.batchMap_[links[0].GetLocalAddr()].back()->availableChannelIdKeys.size();
    EXPECT_NE(newLeftResourceNum, 0);               // 当前剩余资源应该不为0
    EXPECT_EQ(newLeftResourceNum, leftResourceNum); // 因全部复用，资源与之前一致

    ccuJettyMgr.Fallback(); // fallback，但实际无需释放资源，但需要回退分配请求

    const uint32_t fallbackLeftResourceNum
        = ccuJettyMgr.batchMap_[links[0].GetLocalAddr()].back()->availableChannelIdKeys.size();
    EXPECT_EQ(leftResourceNum, fallbackLeftResourceNum); // 资源回退到上一轮
}

TEST_F(CcuJettyMgrTest, Ut_CleanAndResume_When_InterfaceOk_Expect_NoThrow)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 打桩的channel分配函数，每个本端分配8个channel
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), linkNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), rankNum); // rankNum个远端
    ccuJettyMgr.Confirm();

    EXPECT_NO_THROW(ccuJettyMgr.Clean());
    EXPECT_NE(ccuJettyMgr.allocatedChannelIdMap_.empty(), true); // linkData 不清理

    EXPECT_EQ(ccuJettyMgr.batchMap_.empty(), true);
    EXPECT_EQ(ccuJettyMgr.channelJettyInfoMap_.empty(), true);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.empty(), true);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.empty(), true);

    EXPECT_NO_THROW(ccuJettyMgr.Resume());
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
}

TEST_F(CcuJettyMgrTest, Ut_GetUsedChannelCount_When_InterfaceOk_Expect_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t rankNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, rankNum);
    const uint32_t linkNum = links.size();
    MockPlatformDeps(0);

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuJettyMgr ccuJettyMgr(devLogicId);

    EXPECT_EQ(ccuJettyMgr.PrepareCreate(links), HcclResult::HCCL_SUCCESS);
    for (const auto& link : links) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> channelJettyPair;
        EXPECT_NO_THROW(channelJettyPair = ccuJettyMgr.GetChannelJettys(link));
    }
    // 打桩的channel分配函数，每个本端分配8个channel
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.allocations.size(), linkNum);
    EXPECT_EQ(ccuJettyMgr.unconfirmedRecord_.newBatchSet.size(), rankNum); // rankNum个远端
    ccuJettyMgr.Confirm();

    const uint8_t dieId = 1;
    EXPECT_EQ(ccuJettyMgr.GetUsedChannelCount(1), rankNum * CCU_CHANNLE_GOURP_SIZE);

    const uint8_t unUsedDieId = 0;
    EXPECT_EQ(ccuJettyMgr.GetUsedChannelCount(0), 0);
}
