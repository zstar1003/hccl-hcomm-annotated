/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/MockObject.h>
#include "communicator_impl.h"
#include "rdma_handle_manager.h"
#include "dev_rdma_connection.h"
#include "ccu_transport_manager.h"
#include "ccu_jetty_mgr.h"
#include "coll_service_device_mode.h"
#include "ccu_component.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_context_mgr_imp.h"
#include "coll_operator_check.h"
#include "hccl_common_v2.h"
#include "recover_info.h"
#include "hccp_tlv_hdc_manager.h"
#include "ccu_driver_handle.h"

#undef private

using namespace Hccl;

class CcuTransportMgrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuTransportMgrTest tests set up." << std::endl;
    }

    static void TearDownTestCase() { std::cout << "CcuTransportMgrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuTransportMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify(); // 避免用例之间的联系，防止上一个用例的打桩函数在本用例生效
        std::cout << "A Test case in CcuTransportMgrTest TearDown" << std::endl;
    }
};

const char filePath[] = "ranktable.json";
const char ranktable4pPath[] = "ranktable.json";
const char topoPath[] = "topo.json";
const std::string RankTable1Ser8Dev = R"(
    {
    "server_count":"1",
    "server_list":
    [
        {
            "device":[
                        {
                        "device_id":"0",
                        "rank_id":"0"
                        },
                        {
                        "device_id":"1",
                        "rank_id":"1"
                        },
                        {
                        "device_id":"2",
                        "rank_id":"2"
                        },
                        {
                        "device_id":"3",
                        "rank_id":"3"
                        },
                        {
                        "device_id":"4",
                        "rank_id":"4"
                        },
                        {
                        "device_id":"5",
                        "rank_id":"5"
                        },
                        {
                        "device_id":"6",
                        "rank_id":"6"
                        },
                        {
                        "device_id":"7",
                        "rank_id":"7"
                        }
                    ],
            "server_id":"1"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    )";

static void GenRankTableFile1Ser8Dev()
{
    try {
        nlohmann::json rankTableJson = nlohmann::json::parse(RankTable1Ser8Dev);
        std::ofstream out(filePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        std::cout << filePath << " generate failed!" << std::endl;
        return;
    }
    std::cout << filePath << " generated." << std::endl;
}

extern vector<LinkData> MockMultiLinkData(uint32_t baseIpAddrInt, uint32_t num);
extern HcclResult
AllocCcuResStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& resInfos);

std::unique_ptr<CommunicatorImpl> MockCommImpl()
{
    u32 localRank = 0;
    u32 remoteRank = 1;
    auto impl = std::make_unique<CommunicatorImpl>();
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl->rankGraph = make_unique<RankGraph>(0);
    impl->rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs();
    MOCKER_CPP(&CtxMgrImp::Init).stubs();
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl->Init(commParams, "ranktable.json", config);
    impl->SelectCollService();
    impl->currentCollOperator = std::make_unique<CollOperator>();
    impl->currentCollOperator->opMode = OpMode::OPBASE;
    impl->currentCollOperator->opType = OpType::DEBUGCASE;
    impl->currentCollOperator->inputMem = std::make_shared<DevBuffer>(0x100, 10);
    impl->currentCollOperator->outputMem = std::make_shared<DevBuffer>(0x100, 10);
    impl->currentCollOperator->scratchMem = std::make_shared<DevBuffer>(0x100, 10);
    impl->cclBuffer = std::make_shared<DevBuffer>(0x100, 10);
    return std::move(impl);
}

HcclResult CcuJettyMgrPrepareCreateStub(CcuJettyMgr* self, const std::vector<LinkData>& links)
{
    const pair<uint8_t, uint32_t> fakeChannelId = {2, 3}; // 故意打桩die是2
    CcuJettyMgr::CcuChannelJettyInfo fakeChannelJettyInfo;
    fakeChannelJettyInfo.first = CcuChannelInfo{};
    fakeChannelJettyInfo.second = {};
    for (const auto& link : links) {
        self->allocatedChannelIdMap_[link] = fakeChannelId;
        self->channelJettyInfoMap_[fakeChannelId] = fakeChannelJettyInfo;
    }
    return HcclResult::HCCL_SUCCESS;
}

void MockCcuTransportMgrDevs()
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM - 1));

    HcclResult OkResult = HcclResult::HCCL_SUCCESS;
    HcclResult AgainResult = HcclResult::HCCL_E_AGAIN;
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(AllocCcuResStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(AllocCcuResStub));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::ReleaseXn).stubs().will(returnValue(OkResult));

    MOCKER(CcuDeviceManager::GetCcuResourceSpaceBufInfo).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::GetCcuResourceSpaceTokenInfo).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::ConfigChannel).stubs().will(returnValue(OkResult));

    MOCKER_CPP(&CcuJetty::CreateJetty).stubs().will(returnValue(AgainResult)).then(returnValue(OkResult));
    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue((void*)0x12345678));
    std::pair<uint32_t, uint32_t> fakeDieFuncPair = std::make_pair(1, 4);
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(fakeDieFuncPair));
    std::pair<TokenIdHandle, uint32_t> fakeTokenInfo = std::make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));

    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> ccuJettys;
    MOCKER_CPP(&CcuJettyMgr::PrepareCreate).stubs().will(invoke(CcuJettyMgrPrepareCreateStub));
    MOCKER(CcuReleaseChannel).stubs().will(returnValue(OkResult));
    MOCKER(GetUbToken).stubs().will(returnValue(1));

    MOCKER_CPP(&CcuTransport::Init).stubs().will(returnValue(OkResult));
    MOCKER_CPP(&CcuConnection::Init).stubs().will(returnValue(OkResult));
}

TEST_F(CcuTransportMgrTest, St_PrepareCreate_When_InterfaceOk_Expect_Return_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t linkNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, linkNum);
    const auto& link = links[0];
    auto commImpl = MockCommImpl();
    MockCcuTransportMgrDevs();

    std::string socketTag = commImpl->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(1, link, socketTag);
    commImpl->GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    unique_ptr<LocalRmaBuffer> fakeBuffer = make_unique<LocalUbRmaBuffer>(commImpl->cclBuffer);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const string& opTag, const PortData& portData, BufferType bufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeBuffer.get()));

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(commImpl->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    (void)ccuJettyMgr->PrepareCreate(links); // GetChannelJettys是const不能打桩

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuTransportMgr transportMgr(*commImpl, devLogicId);

    CcuTransport* transport;
    EXPECT_EQ(transportMgr.PrepareCreate(link, transport), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(transport, nullptr);
    EXPECT_EQ(transportMgr.tempTransport.empty(), false);

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::READY));

    commImpl->currentCollOperator = make_unique<CollOperator>();
    commImpl->currentCollOperator->opType = OpType::ALLREDUCE;
    MOCKER(CheckCollOperator).stubs();
    transport->rmtHandshakeMsg = std::vector<char>(2000);

    EXPECT_NO_THROW(transportMgr.Confirm());
    EXPECT_EQ(transportMgr.tempTransport.empty(), true);

    EXPECT_NE(transportMgr.Get(links[0]), nullptr);

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData fakeLinkData(portType, 0, 1, 0, 1);
    EXPECT_NE(transportMgr.Get(links[0]), nullptr);
    EXPECT_EQ(transportMgr.Get(fakeLinkData), nullptr);

    RankId rankId = links[0].GetRemoteRankId();
    EXPECT_EQ(transportMgr.Get(rankId).empty(), false);
    EXPECT_EQ(transportMgr.Get(0xffff).empty(), true);
}

TEST_F(CcuTransportMgrTest, Ut_PrepareCreateFailAndFallback_When_InterfaceUnavailable_Expect_Return_Unavailable)
{
    MOCKER(CcuCreateTransport).stubs().will(returnValue(HcclResult::HCCL_E_UNAVAIL));

    const uint32_t baseIpAddrInt = 100;
    const uint32_t linkNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, linkNum);
    const auto& link = links[0];
    auto commImpl = MockCommImpl();
    MockCcuTransportMgrDevs();

    std::string socketTag = commImpl->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(1, link, socketTag);
    commImpl->GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    unique_ptr<LocalRmaBuffer> fakeBuffer = make_unique<LocalUbRmaBuffer>(commImpl->cclBuffer);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const string& opTag, const PortData& portData, BufferType bufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeBuffer.get()));

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(commImpl->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    (void)ccuJettyMgr->PrepareCreate(links); // GetChannelJettys是const不能打桩

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuTransportMgr transportMgr(*commImpl, devLogicId);

    CcuTransport* transport = nullptr; // 设空指针，避免随机初始化
    EXPECT_EQ(transportMgr.PrepareCreate(link, transport), HcclResult::HCCL_E_UNAVAIL);
    EXPECT_EQ(transport, nullptr);

    EXPECT_NO_THROW(transportMgr.Fallback());
    EXPECT_EQ(transportMgr.ccuLink2TransportMap.empty(), true);

    for (const auto& iter : transportMgr.ccuRank2TransportsMap) {
        EXPECT_EQ(iter.second.empty(), true);
    }
}

TEST_F(CcuTransportMgrTest, Ut_CleanAndResume_When_InterfaceOk_Expect_Return_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t linkNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, linkNum);
    const auto& link = links[0];
    auto commImpl = MockCommImpl();
    MockCcuTransportMgrDevs();

    std::string socketTag = commImpl->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(1, link, socketTag);
    commImpl->GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    unique_ptr<LocalRmaBuffer> fakeBuffer = make_unique<LocalUbRmaBuffer>(commImpl->cclBuffer);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const string& opTag, const PortData& portData, BufferType bufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeBuffer.get()));

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(commImpl->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    (void)ccuJettyMgr->PrepareCreate(links); // GetChannelJettys是const不能打桩

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuTransportMgr transportMgr(*commImpl, devLogicId);

    CcuTransport* transport = nullptr; // 设空指针，避免随机初始化
    EXPECT_EQ(transportMgr.PrepareCreate(link, transport), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(transport, nullptr);
    EXPECT_EQ(transportMgr.tempTransport.empty(), false);

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::READY));

    commImpl->currentCollOperator = make_unique<CollOperator>();
    commImpl->currentCollOperator->opType = OpType::ALLREDUCE;
    MOCKER(CheckCollOperator).stubs();
    transport->rmtHandshakeMsg = std::vector<char>(2000);

    EXPECT_NO_THROW(transportMgr.Confirm());
    EXPECT_EQ(transportMgr.tempTransport.empty(), true);

    const uint32_t linkSize = transportMgr.ccuLink2TransportMap.size();
    EXPECT_NO_THROW(transportMgr.Clean());
    EXPECT_NE(0, linkSize);
    EXPECT_EQ(transportMgr.ccuLink2TransportMap.size(), linkSize);

    EXPECT_NO_THROW(transportMgr.Resume());
    EXPECT_EQ(transportMgr.ccuLink2TransportMap.size(), linkSize);
    EXPECT_EQ(transportMgr.tempTransport.size(), linkSize);
}

TEST_F(CcuTransportMgrTest, Ut_CleanAndResumeFailed_When_InterfaceError_Expect_Return_Throw)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t linkNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, linkNum);
    const auto& link = links[0];
    auto commImpl = MockCommImpl();
    MockCcuTransportMgrDevs();

    std::string socketTag = commImpl->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(1, link, socketTag);
    commImpl->GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    unique_ptr<LocalRmaBuffer> fakeBuffer = make_unique<LocalUbRmaBuffer>(commImpl->cclBuffer);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const string& opTag, const PortData& portData, BufferType bufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeBuffer.get()));

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(commImpl->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    (void)ccuJettyMgr->PrepareCreate(links); // GetChannelJettys是const不能打桩

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuTransportMgr transportMgr(*commImpl, devLogicId);

    CcuTransport* transport = nullptr; // 设空指针，避免随机初始化
    EXPECT_EQ(transportMgr.PrepareCreate(link, transport), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(transport, nullptr);
    EXPECT_EQ(transportMgr.tempTransport.empty(), false);

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::READY));

    commImpl->currentCollOperator = make_unique<CollOperator>();
    commImpl->currentCollOperator->opType = OpType::ALLREDUCE;
    MOCKER(CheckCollOperator).stubs();
    transport->rmtHandshakeMsg = std::vector<char>(2000);

    EXPECT_NO_THROW(transportMgr.Confirm());
    EXPECT_EQ(transportMgr.tempTransport.empty(), true);

    const uint32_t linkSize = transportMgr.ccuLink2TransportMap.size();
    EXPECT_NO_THROW(transportMgr.Clean());
    EXPECT_NE(0, linkSize);
    EXPECT_EQ(transportMgr.ccuLink2TransportMap.size(), linkSize);

    MOCKER(CcuCreateTransport).stubs().will(returnValue(HcclResult::HCCL_E_UNAVAIL));
}

TEST_F(CcuTransportMgrTest, Ut_RecoverTransports_When_InterfaceOk_Expect_Return_Ok)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t linkNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, linkNum);
    const auto& link = links[0];
    auto commImpl = MockCommImpl();
    MockCcuTransportMgrDevs();

    std::string socketTag = commImpl->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(1, link, socketTag);
    commImpl->GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    unique_ptr<LocalRmaBuffer> fakeBuffer = make_unique<LocalUbRmaBuffer>(commImpl->cclBuffer);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const string& opTag, const PortData& portData, BufferType bufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeBuffer.get()));

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(commImpl->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    (void)ccuJettyMgr->PrepareCreate(links); // GetChannelJettys是const不能打桩

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuTransportMgr transportMgr(*commImpl, devLogicId);

    CcuTransport* transport;
    EXPECT_EQ(transportMgr.PrepareCreate(link, transport), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(transport, nullptr);
    EXPECT_EQ(transportMgr.tempTransport.empty(), false);

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::READY));

    RecoverInfoData recoverInfoData;
    recoverInfoData.collOpIndex = commImpl->GetCollOpIndex();
    recoverInfoData.crcValue = 0;
    recoverInfoData.step = commImpl->GetStep();
    RecoverInfo recoverInfo(recoverInfoData, commImpl->GetMyRank());
    transport->rmtHandshakeMsg = recoverInfo.GetUniqueId();
    commImpl->isWorldGroup = false;

    EXPECT_NO_THROW(transportMgr.RecoverConfirm());
    EXPECT_EQ(transportMgr.tempTransport.empty(), true);

    EXPECT_NE(transportMgr.Get(links[0]), nullptr);

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData fakeLinkData(portType, 0, 1, 0, 1);
    EXPECT_NE(transportMgr.Get(links[0]), nullptr);
    EXPECT_EQ(transportMgr.Get(fakeLinkData), nullptr);

    RankId rankId = links[0].GetRemoteRankId();
    EXPECT_EQ(transportMgr.Get(rankId).empty(), false);
    EXPECT_EQ(transportMgr.Get(0xffff).empty(), true);
}

TEST_F(CcuTransportMgrTest, Ut_RecoverTransportsFailed_When_RecoverMsgError_Expect_Throw)
{
    const uint32_t baseIpAddrInt = 100;
    const uint32_t linkNum = 4;
    const auto& links = MockMultiLinkData(baseIpAddrInt, linkNum);
    const auto& link = links[0];
    auto commImpl = MockCommImpl();
    MockCcuTransportMgrDevs();

    std::string socketTag = commImpl->GetEstablishLinkSocketTag();
    SocketConfig socketConfig(1, link, socketTag);
    commImpl->GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        nullptr, IpAddress(), 0, IpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    unique_ptr<LocalRmaBuffer> fakeBuffer = make_unique<LocalUbRmaBuffer>(commImpl->cclBuffer);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const string& opTag, const PortData& portData, BufferType bufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeBuffer.get()));

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(commImpl->GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    (void)ccuJettyMgr->PrepareCreate(links); // GetChannelJettys是const不能打桩

    int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1;
    CcuTransportMgr transportMgr(*commImpl, devLogicId);

    CcuTransport* transport;
    EXPECT_EQ(transportMgr.PrepareCreate(link, transport), HcclResult::HCCL_SUCCESS);
    EXPECT_NE(transport, nullptr);
    EXPECT_EQ(transportMgr.tempTransport.empty(), false);

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::CONNECT_FAILED));

    EXPECT_THROW(transportMgr.RecoverConfirm(), InternalException);
}
