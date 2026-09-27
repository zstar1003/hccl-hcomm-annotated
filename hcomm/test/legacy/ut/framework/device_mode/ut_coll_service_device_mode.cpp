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
#include "test_mock_setup.h"
#include <hccl/hccl_types.h>
#define private public
#define protected public
#include "coll_service_device_mode.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"
#include "communicator_callback.h"
#include "stream_utils.h"
#include "rdma_handle_manager.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "internal_exception.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

#define FUSION_SUB_TASK_MAX_CPU_NUM (1U)

typedef struct tagRtAicpuArgs {
    uint16_t kfcArgsFmtOffset;     // default value is 0xffff
    uint16_t soNameAddrOffset;     // just for CCE Kernel, default value is 0xffff for FWK kernel
    uint16_t kernelNameAddrOffset; // just for CCE Kernel, default value is 0xffff for FWK kernel
    uint16_t rev;
} rtAicpuArgs_t;

typedef struct tagRtFusionArgsEx {
    void* args;                                           // args host mem addr
    rtHostInputInfo_t* hostInputInfoPtr;                  // nullptr means no host mem input
    uint32_t argsSize;                                    // input + output + host mem
    uint16_t hostInputInfoNum;                            // hostInputInfo num
    uint8_t aicpuNum;                                     // aicpu task num
    uint8_t isNoNeedH2DCopy;                              // is no need host to device copy: 0 means need H2D copy,
                                                          // others means doesn't need H2D copy.
    rtAicpuArgs_t aicpuArgs[FUSION_SUB_TASK_MAX_CPU_NUM]; // aicpuArgsInfo
} rtFusionArgsEx_t;

class CollServiceDeviceModeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollServiceDeviceModeTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollServiceDeviceModeTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        u32 fakeDevPhyId = 1;
        u64 fakeNotifyHandleAddr = 100;
        u32 fakeNotifyId = 1;
        u64 fakeOffset = 200;
        char fakeName[65] = "testRtsNotify";
        SETUP_COMM_NOTIFY_MOCKS();

        // 资源初始化
        MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());

        Buffer* buf = nullptr;
        LocalRmaBuffer* rmaBuf = nullptr;
        MOCKER_CPP(&DataBufManager::Get)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(buf));
        MOCKER_CPP(
            &LocalRmaBufManager::Reg, LocalRmaBuffer
                                          * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                    const PortData&, LinkProtocol))
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(rmaBuf));
        RtsNotify notify(false);
        RtsNotify notify1(false);
        MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
        MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
        MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(std::vector<char>{'1', '2'}));
        void* ptr1 = (void*)1;
        MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr1));
        MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));

        fakeComm.cclBuffer = DevBuffer::Create(0x100, 0x100);
        fakeComm.SetCommStatus(CommStatus::COMM_READY);
        fakeComm.InitNotifyManager();
        fakeComm.InitSocketManager();
        fakeComm.InitRmaConnManager();
        fakeComm.InitStreamManager();
        fakeComm.InitMemTransportManager();
        fakeComm.InitMirrorTaskManager();
        fakeComm.InitProfilingReporter();
        fakeComm.myRank = 0;
        fakeComm.id = "testTag";
        fakeComm.streamManager->opbase = make_unique<OpbaseStreamManager>(&fakeComm);
        std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
        fakeComm.dataBufferManager = std::make_unique<DataBufManager>();
        fakeComm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
        fakeComm.rankGraph = std::make_unique<RankGraph>(0);
        fakeComm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&fakeComm);
        fakeComm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&fakeComm);
        fakeComm.rmaConnectionManager = std::make_unique<RmaConnManager>(fakeComm);
        fakeComm.currentCollOperator = std::make_unique<CollOperator>();
        fakeComm.currentCollOperator->opMode = OpMode::OPBASE;
        fakeComm.currentCollOperator->opType = OpType::DEBUGCASE;
        fakeComm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
        fakeComm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
        fakeComm.currentCollOperator->opTag = "test";
        fakeComm.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
        fakeComm.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
        fakeComm.hostDeviceSyncNotifyManager = std::make_unique<HostDeviceSyncNotifyManager>();
        fakeComm.memTransportManager = make_unique<MemTransportManager>(fakeComm);

        s32 rankId = 0;
        s32 localId = 0;
        DeviceId deviceId = 0;
        IpAddress inputAddr(0);
        std::set<std::string> ports = {"0/1"};
        std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
        shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
        shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
            inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
        peer0->AddConnInterface(0, connInterface);
        fakeComm.rankGraph->AddPeer(peer0);
        fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
        fakeComm.trace = std::make_unique<Trace>();

        fakeComm.InitCollService();
        fakeComm.CollAlgComponentInit();
        MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        fakeComm.RegisterAcceStateCallBack(CommunicatorCallback());
        OpExecuteConfig opConfig; // ccu 展开
        opConfig.accState = AcceleratorState::CCU_MS;
        fakeComm.opExecuteConfig = opConfig;
        fakeComm.SelectCollService();

        // 算法组件初始化
        CollAlgOpReq collAlgOpReq;
        collAlgOpReq.algName = "testAlg";
        std::vector<std::pair<u32, RankId>> levelRankPairs;
        levelRankPairs.push_back({1, 1});
        collAlgOpReq.resReq.levelRankPairs = levelRankPairs;

        CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
        MOCKER_CPP_VIRTUAL(
            collAlgComponent, &CollAlgComponent::Orchestrate,
            HcclResult(CollAlgComponent::*)(
                const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER_CPP_VIRTUAL(
            collAlgComponent, &CollAlgComponent::CalcResOffload,
            HcclResult(CollAlgComponent::*)(
                const OpType& opType, const u64& dataSize, const HcclDataType& dataType,
                const OpExecuteConfig& opConfig, CollOffloadOpResReq& resReq))
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(collAlgOpReq));
        MOCKER_CPP(&Trace::Save).stubs();
        MOCKER_CPP(&CollServiceAiCpuImpl::AllocOpMem).stubs();
        MOCKER_CPP(&Stream::InitDevPhyId).stubs();
        MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();
        MOCKER_CPP(&CollServiceAiCpuImpl::AddPostToUserStream).stubs().with(mockcpp::any());
        MOCKER_CPP(&CollServiceAiCpuImpl::AddWaitToUserStream).stubs().with(mockcpp::any());
        MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
        std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }

    CommunicatorImpl fakeComm;
};

TEST_F(CollServiceDeviceModeTest, test_init_LoadWithOpBasedMode)
{
    auto service = dynamic_cast<CollServiceDeviceMode*>(fakeComm.collService);
    auto stream = std::make_unique<Stream>();
    service->LoadWithOpBasedMode(*fakeComm.currentCollOperator, std::move(stream));
    EXPECT_NO_THROW(service->GetAicpuInsPreprocessor());
    EXPECT_NO_THROW(service->GetCcuInsPreprocessor());
    EXPECT_NO_THROW(fakeComm.GetCollAlgComponent());

    MOCKER_CPP(&AicpuInsPreprocessor::IsAicpuResExisted).stubs().with().will(returnValue(true));
    EXPECT_NO_THROW(service->IsAicpuResExisted("test"));

    DevBuffer* devbuf;
    MOCKER_CPP(&AicpuInsPreprocessor::GetAicpuResBuffer).stubs().with().will(returnValue(devbuf));
    EXPECT_NO_THROW(service->GetAicpuResBuffer("test"));
}

TEST_F(CollServiceDeviceModeTest, test_init_LoadWithOffloadMode)
{
    void* ptr1 = (void*)1;
    MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr1));
    MOCKER(HrtGetStreamId).stubs().will(returnValue(0));
    MOCKER_CPP(&CollServiceDefaultImpl::AddCountTask).stubs().will(ignoreReturnValue());
    auto service = dynamic_cast<CollServiceDeviceMode*>(fakeComm.collService);
    OpType opType = OpType::ALLREDUCE;
    auto stream = std::make_unique<Stream>();
    fakeComm.currentCollOperator->opMode = OpMode::OFFLOAD;
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    fakeComm.currentCollOperator->scratchMem = buffer;
    EXPECT_NO_THROW(service->LoadWithOffloadMode(*fakeComm.currentCollOperator, std::move(stream)));

    CollOffloadOpResReq resReq;
    EXPECT_NO_THROW(fakeComm.CalcCollOffloadOpRes(opType, 4, HCCL_DATA_TYPE_INT8, resReq));
    fakeComm.currentCollOperator->opMode = OpMode::OPBASE;
}

TEST_F(CollServiceDeviceModeTest, should_return_success_when_calling_GetUniqueLinks)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    CollServiceDeviceMode collServiceDeviceMode(comm.get());
    CollAlgResReq collAlgResReq;
    TemplateInfo tempInfo;
    auto insQueue = make_shared<InsQueue>();
    insQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    insQueue->Append(std::move(std::make_unique<AicpuInstruction>("test", collAlgResReq, tempInfo)));
    auto subInsQueue = insQueue->Fork();
    subInsQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    subInsQueue->Append(std::move(std::make_unique<AicpuInstruction>("test", collAlgResReq, tempInfo)));

    EXPECT_NO_THROW(collServiceDeviceMode.GetUniqueLinks(insQueue));
}

TEST_F(CollServiceDeviceModeTest, test_alloc_comm_resource_by_tiling_success)
{
    CommunicatorImpl comm;
    comm.rankSize = 2;
    comm.myRank = 0;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;

    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    comm.socketManager = std::make_unique<SocketManager>(comm, 0, 0, 6000);

    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);

    comm.rankGraph = std::make_unique<RankGraph>(0);
    auto peer0 = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto peer1 = std::make_shared<NetInstance::Peer>(1, 1, 0, 0);
    IpAddress inputAddr(0);
    std::set<std::string> ports1 = {"0/1"};
    std::set<std::string> ports2 = {"0/2"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    LinkDirection direction = LinkDirection::BOTH;
    u32 hop = 1;
    shared_ptr<NetInstance::ConnInterface> sourceIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports1, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::ConnInterface> targetIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports2, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::Link> link = std::make_shared<NetInstance::Link>(
        peer0, peer1, sourceIface, targetIface, LinkType::PEER2PEER, protocols, direction, hop);
    peer0->AddConnInterface(0, sourceIface);
    peer1->AddConnInterface(0, targetIface);
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddPeer(peer1);

    MOCKER_CPP(&Mc2Compont::FillCollOperator).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&Mc2Compont::AllocCommResource).stubs().with().will(ignoreReturnValue());
    std::vector<CcuTaskParam> vec;
    MOCKER_CPP(&Mc2Compont::GetCcuTaskInfo).stubs().with(mockcpp::any()).will(returnValue(vec));

    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&)).stubs();

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs();
    std::vector<Hccl::LinkData> linkVec;
    char* buf = new char[16 * 1024 * 1024];
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(buf)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x100000));

    rtFusionArgsEx_t fusionArgs;
    rtCcuTaskGroup_t ccuTaskGroup;
    fusionArgs.args = malloc(sizeof(void*) * 7 + sizeof(Mc2Tiling) + sizeof(HcclCommParamDesc));
    fusionArgs.aicpuNum = 1;
    fusionArgs.aicpuArgs[0].kfcArgsFmtOffset = (sizeof(void*) * 7 + sizeof(Mc2Tiling)) / sizeof(void*);
    Mc2Tiling* tilingData
        = reinterpret_cast<Mc2Tiling*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7);
    *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 5)
        = reinterpret_cast<uint64_t>(tilingData);
    HcclCommParamDesc* commParamDesc = reinterpret_cast<HcclCommParamDesc*>(
        reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7 + sizeof(Mc2Tiling));
    commParamDesc->groupNum = 1;
    commParamDesc->hasFfts = 0;
    commParamDesc->tilingDataPtrOff = 5;

    tilingData->version = 3;
    tilingData->commConfigNum = 1;
    tilingData->serverCfg = {0};
    tilingData->commConfig.opType = 6; // Allgahter
    tilingData->commConfig.reduceType = 0;
    tilingData->commConfig.dataType = 3; // FP16
    tilingData->commConfig.outputDataType = 3;

    void* commContext = nullptr;

    CollServiceDeviceMode service(&comm);
    EXPECT_NO_THROW(service.AllocCommResource(tilingData, &commContext, Hccl::AcceleratorState::CCU_MS));
    EXPECT_NO_THROW(service.GetCcuTaskInfo(&fusionArgs, &ccuTaskGroup));

    free(fusionArgs.args);
    delete[] buf;
}

TEST_F(CollServiceDeviceModeTest, test_ccu_RecoverTransport)
{
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().will(returnValue(0));
    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&ConnLocalCntNotifyManager::ApplyFor).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&MemTransportManager::BatchRecoverOpbasedTransports).stubs().will(returnValue(0));
    MOCKER_CPP(&CcuInsPreprocessor::RecoverCcuTransportCtx).stubs().will(returnValue(HCCL_E_PARA));

    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.myRank = 0;
    comm.id = "testTag";
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.GetCurrentCollOperator()->opMode = OpMode::OPBASE;
    CollServiceDeviceMode service(&comm);

    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(make_pair<uint32_t, uint32_t>(0, 0)));
    vector<LinkData> links;
    vector<std::pair<LinkGroup, u32>> linkGroupPair;
    LinkData linkData(
        PortDeploymentType::P2P, LinkProtocol::UB_CTP, 0, 1, IpAddress{"10.0.0.1"}, IpAddress{"10.0.0.2"});
    links.push_back(linkData);
    LinkGroup linkGroup{};
    linkGroup.AddLink({linkData});
    LinkData otherLinkData(
        PortDeploymentType::P2P, LinkProtocol::UB_CTP, 1, 1, IpAddress{"10.0.0.3"}, IpAddress{"10.0.0.4"});
    ;
    linkGroup.AddLink({otherLinkData});
    linkGroupPair.push_back(make_pair(linkGroup, 0));
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    EXPECT_THROW(service.RecoverTransport(links, linkGroupPair), InternalException);
}

TEST_F(CollServiceDeviceModeTest, test_GetSnapShotDynamicBuf)
{
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOffloadTransports).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(true));

    CommunicatorImpl comm;
    comm.InitMemTransportManager();
    comm.InitRmaConnManager();
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "test";
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    CcuInsPreprocessor ccuInsPreprocessor(&comm);
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    std::vector<std::pair<u32, RankId>> levelRankPairs;
    levelRankPairs.push_back({1, 1});
    collAlgOpReq.resReq.levelRankPairs = levelRankPairs;

    LinkInfo linkInfo{1, 0, IpAddress{"10.0.0.1"}, IpAddress{"10.0.0.2"}};
    LinkGroup utLinkGroup{vector<LinkInfo>{linkInfo}};
    vector<LinkGroup> utLinkGroups{utLinkGroup};

    u32 utCntCke = 3;
    vector<CcuTransport*> utCcuTransportVec;
    MOCKER_CPP(&CcuTransportGroupMgr::GetAllTransportGroups).stubs().with().will(returnValue(utLinkGroups));
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    comm.collAlgComponent = make_shared<CollAlgComponent>(nullptr, DevType::DEV_TYPE_950, 0, 1);
    CollOperator op;
    BinaryStream bs{};

    u32 opAccState{0};
    u32 commAccState{0};
    bool isLoadOp = true;
    u32 submittedOpCnt = 2;
    u32 opMode{0};
    bs << opAccState << commAccState << isLoadOp << submittedOpCnt << opMode;
    EXPECT_EQ(collService.GetSnapShotDynamicBuf(op, bs), HcclResult::HCCL_SUCCESS);

    SnapShotParser parse;
    SnapShotDynamic snapShotDynamicBuf;
    EXPECT_EQ(parse.DeSnapShotDynamicBuf(bs, snapShotDynamicBuf), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(opAccState, snapShotDynamicBuf.opExecuteConfig.accState);
    EXPECT_EQ(commAccState, snapShotDynamicBuf.commExecuteConfig.accState);
    EXPECT_EQ(isLoadOp, snapShotDynamicBuf.isLoadOp);
    EXPECT_EQ(submittedOpCnt, snapShotDynamicBuf.submittedOpCnt);
    EXPECT_EQ(opMode, snapShotDynamicBuf.opMode);
}

TEST_F(CollServiceDeviceModeTest, test_IsAllTransportRecoveredReady)
{
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOffloadTransports).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(true));

    CommunicatorImpl comm;
    comm.InitMemTransportManager();
    comm.InitRmaConnManager();
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "test";
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    CcuInsPreprocessor ccuInsPreprocessor(&comm);
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    std::vector<std::pair<u32, RankId>> levelRankPairs;
    levelRankPairs.push_back({1, 1});
    collAlgOpReq.resReq.levelRankPairs = levelRankPairs;

    vector<RankId> utRanks = {0};
    RankGroup utRankGroup(utRanks);
    vector<RankGroup> utRankGroups = {utRankGroup};

    u32 utCntCke = 3;
    vector<CcuTransport*> utCcuTransportVec;
    MOCKER_CPP(&CcuTransportGroupMgr::GetAllTransportGroups).stubs().with().will(returnValue(utRankGroups));
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    comm.collAlgComponent = make_shared<CollAlgComponent>(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP(&CcuInsPreprocessor::RecoverCcuTransportConfirm)
        .stubs()
        .with()
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    EXPECT_EQ(collService.IsAllTransportRecoveredReady("test"), true);
}

TEST_F(CollServiceDeviceModeTest, test_IsAllTransportRecoveredReady_false)
{
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOffloadTransports).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(true));

    CommunicatorImpl comm;
    comm.InitMemTransportManager();
    comm.InitRmaConnManager();
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "test";
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    CcuInsPreprocessor ccuInsPreprocessor(&comm);
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    std::vector<std::pair<u32, RankId>> levelRankPairs;
    levelRankPairs.push_back({1, 1});
    collAlgOpReq.resReq.levelRankPairs = levelRankPairs;

    vector<RankId> utRanks = {0};
    RankGroup utRankGroup(utRanks);
    vector<RankGroup> utRankGroups = {utRankGroup};

    u32 utCntCke = 3;
    vector<CcuTransport*> utCcuTransportVec;
    MOCKER_CPP(&CcuTransportGroupMgr::GetAllTransportGroups).stubs().with().will(returnValue(utRankGroups));
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    comm.collAlgComponent = make_shared<CollAlgComponent>(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP(&CcuInsPreprocessor::RecoverCcuTransportConfirm)
        .stubs()
        .with()
        .will(returnValue(HcclResult::HCCL_E_PARA));
    EXPECT_EQ(collService.IsAllTransportRecoveredReady("test"), false);
}

TEST_F(CollServiceDeviceModeTest, test_coll_service_device_mode_resume)
{
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::ALLREDUCE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(0, connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.trace = std::make_unique<Trace>();
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;

    CollServiceDeviceMode service(&comm);

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData{PortDeploymentType::P2P, LinkProtocol::UB_CTP, 0, 1,
                      IpAddress{"10.0.0.1"},   IpAddress{"10.0.0.2"}};
    std::vector<LinkData> links;
    links.push_back(linkData);
    vector<LinkInfo> infos{
        {LinkData{PortDeploymentType::P2P, LinkProtocol::UB_CTP, 1, 1, IpAddress{"10.0.0.3"}, IpAddress{"10.0.0.4"}}}};
    LinkGroup linkGroup{infos};
    service.ccuInsPreprocessor.ccuComm.ccuTransportMgr.ccuLink2TransportMap[linkData] = nullptr;
    service.ccuInsPreprocessor.ccuComm.ccuTransportGroupMgr.linkGrp2TransportGrpMap[linkGroup] = nullptr;

    MOCKER_CPP(&CcuTransportGroupMgr::ResumeAll).stubs();
    MOCKER_CPP(&CcuTransportMgr::Confirm).stubs();
    MOCKER_CPP(&CcuJettyMgr::Confirm).stubs();
    MOCKER_CPP(&CcuTransportGroupMgr::Confirm).stubs();
    MOCKER(&CcuCleanDieCkes).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_NO_THROW(service.Resume());
}

TEST_F(CollServiceDeviceModeTest, test_coll_service_device_mode_resume_when_links_is_empty)
{
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::ALLREDUCE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(0, connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.trace = std::make_unique<Trace>();
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;

    CollServiceDeviceMode service(&comm);

    std::vector<LinkData> links;
    service.ccuInsPreprocessor.ccuComm.ccuTransportMgr.ccuLink2TransportMap.clear();
    service.ccuInsPreprocessor.ccuComm.ccuTransportGroupMgr.linkGrp2TransportGrpMap.clear();

    MOCKER_CPP(&CcuTransportGroupMgr::ResumeAll).stubs();
    MOCKER_CPP(&CcuTransportMgr::Confirm).stubs();
    MOCKER_CPP(&CcuJettyMgr::Confirm).stubs();
    MOCKER_CPP(&CcuTransportGroupMgr::Confirm).stubs();
    MOCKER(&CcuCleanDieCkes).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_NO_THROW(service.Resume());
}

TEST_F(CollServiceDeviceModeTest, test_init_ReLoadWithOpBasedMode_Opbased)
{
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    auto service = dynamic_cast<CollServiceDeviceMode*>(fakeComm.collService);
    auto stream = std::make_unique<Stream>();

    service->ccuInsPreprocessor.resAllocSuccess = false;

    EXPECT_ANY_THROW(service->LoadWithOpBasedMode(*fakeComm.currentCollOperator, std::move(stream)));
}

TEST_F(CollServiceDeviceModeTest, test_init_ReLoadWithOpBasedMode_Offload)
{
    fakeComm.currentCollOperator->opMode = OpMode::OFFLOAD;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    auto service = dynamic_cast<CollServiceDeviceMode*>(fakeComm.collService);
    auto stream = std::make_unique<Stream>();

    service->ccuInsPreprocessor.resAllocSuccess = false;
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    fakeComm.currentCollOperator->scratchMem = buffer;
    EXPECT_THROW(service->LoadWithOffloadMode(*fakeComm.currentCollOperator, std::move(stream)), NotSupportException);
}

TEST_F(CollServiceDeviceModeTest, Ut_GetCcuMc2ServerNum_Expect_equality)
{
    // 前置条件
    auto service = dynamic_cast<CollServiceDeviceMode*>(fakeComm.collService);
    u32 ccuMc2ServerNum = 0;

    // 执行步骤
    auto ret = service->GetCcuMc2ServerNum();

    // 后置验证
    EXPECT_EQ(ret, ccuMc2ServerNum);
}

TEST_F(CollServiceDeviceModeTest, should_success_when_AllocCommResource_aiv)
{
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));

    CommunicatorImpl comm;
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.rankSize = 2;
    comm.myRank = 0;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);

    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    comm.socketManager = std::make_unique<SocketManager>(comm, 0, 0, 6000);
    comm.ubMemoryTransportMgr = std::make_unique<UbMemoryTransportMgr>(comm);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);

    comm.rankGraph = std::make_unique<RankGraph>(0);
    auto peer0 = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto peer1 = std::make_shared<NetInstance::Peer>(1, 1, 0, 0);
    IpAddress inputAddr(0);
    std::set<std::string> ports1 = {"0/1"};
    std::set<std::string> ports2 = {"0/2"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    u32 hop = 1;
    LinkDirection direction = LinkDirection::BOTH;
    shared_ptr<NetInstance::ConnInterface> sourceIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports1, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::ConnInterface> targetIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports2, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::Link> link = std::make_shared<NetInstance::Link>(
        peer0, peer1, sourceIface, targetIface, LinkType::PEER2PEER, protocols, direction, hop);
    peer0->AddConnInterface(0, sourceIface);
    peer1->AddConnInterface(0, targetIface);
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddPeer(peer1);

    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&)).stubs();

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs();
    std::vector<Hccl::LinkData> linkVec;
    char* buf = new char[16 * 1024 * 1024];
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(buf)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x100000));

    rtFusionArgsEx_t fusionArgs;
    rtCcuTaskGroup_t ccuTaskGroup;
    fusionArgs.args = malloc(sizeof(void*) * 7 + sizeof(Mc2Tiling) + sizeof(HcclCommParamDesc));
    fusionArgs.aicpuNum = 1;
    fusionArgs.aicpuArgs[0].kfcArgsFmtOffset = (sizeof(void*) * 7 + sizeof(Mc2Tiling)) / sizeof(void*);
    Mc2Tiling* tilingData
        = reinterpret_cast<Mc2Tiling*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7);
    *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 5)
        = reinterpret_cast<uint64_t>(tilingData);
    HcclCommParamDesc* commParamDesc = reinterpret_cast<HcclCommParamDesc*>(
        reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7 + sizeof(Mc2Tiling));
    commParamDesc->groupNum = 1;
    commParamDesc->hasFfts = 0;
    commParamDesc->tilingDataPtrOff = 5;

    tilingData->version = 3;
    tilingData->commConfigNum = 1;
    tilingData->serverCfg = {0};
    tilingData->commConfig.opType = 6; // Allgahter
    tilingData->commConfig.reduceType = 0;
    tilingData->commConfig.dataType = 3; // FP16
    tilingData->commConfig.outputDataType = 3;
    tilingData->commConfig.communicationEngine = 3; // aiv

    void* commContext = nullptr;

    CollServiceDeviceMode service(&comm);
    comm.collService = &service;
    MOCKER_CPP(&UbMemoryTransportMgr::TransportsConnect).stubs();
    EXPECT_NO_THROW(service.AllocCommResource(tilingData, &commContext, Hccl::AcceleratorState::AIV));

    free(fusionArgs.args);
    delete[] buf;
}

TEST_F(CollServiceDeviceModeTest, Ut_AllocCommResource_When_versionIs0_Expect_THROW)
{
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));

    CommunicatorImpl comm;
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.rankSize = 2;
    comm.myRank = 0;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);

    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    comm.socketManager = std::make_unique<SocketManager>(comm, 0, 0, 6000);
    comm.ubMemoryTransportMgr = std::make_unique<UbMemoryTransportMgr>(comm);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);

    comm.rankGraph = std::make_unique<RankGraph>(0);
    auto peer0 = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto peer1 = std::make_shared<NetInstance::Peer>(1, 1, 1, 0);
    IpAddress inputAddr(0);
    std::set<std::string> ports1 = {"0/1"};
    std::set<std::string> ports2 = {"0/2"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    u32 hop = 1;
    LinkDirection direction = LinkDirection::BOTH;
    shared_ptr<NetInstance::ConnInterface> sourceIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports1, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::ConnInterface> targetIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports2, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::Link> link = std::make_shared<NetInstance::Link>(
        peer0, peer1, sourceIface, targetIface, LinkType::PEER2PEER, protocols, direction, hop);
    peer0->AddConnInterface(0, sourceIface);
    peer1->AddConnInterface(0, targetIface);
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddPeer(peer1);

    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&)).stubs();

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs();
    std::vector<Hccl::LinkData> linkVec;
    char* buf = new char[16 * 1024 * 1024];
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(buf)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x100000));

    rtFusionArgsEx_t fusionArgs;
    rtCcuTaskGroup_t ccuTaskGroup;
    fusionArgs.args = malloc(sizeof(void*) * 7 + sizeof(Mc2Tiling) + sizeof(HcclCommParamDesc));
    fusionArgs.aicpuNum = 1;
    fusionArgs.aicpuArgs[0].kfcArgsFmtOffset = (sizeof(void*) * 7 + sizeof(Mc2Tiling)) / sizeof(void*);
    Mc2Tiling* tilingData
        = reinterpret_cast<Mc2Tiling*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7);
    *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 5)
        = reinterpret_cast<uint64_t>(tilingData);
    HcclCommParamDesc* commParamDesc = reinterpret_cast<HcclCommParamDesc*>(
        reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7 + sizeof(Mc2Tiling));
    commParamDesc->groupNum = 1;
    commParamDesc->hasFfts = 0;
    commParamDesc->tilingDataPtrOff = 5;

    tilingData->version = 0;
    tilingData->commConfigNum = 1;
    tilingData->serverCfg = {0};
    tilingData->commConfig.opType = 6; // Allgahter
    tilingData->commConfig.reduceType = 0;
    tilingData->commConfig.dataType = 3; // FP16
    tilingData->commConfig.outputDataType = 3;
    tilingData->commConfig.communicationEngine = 2; // aiv

    void* commContext = nullptr;

    CollServiceDeviceMode service(&comm);
    comm.collService = &service;
    MOCKER_CPP(&UbMemoryTransportMgr::TransportsConnect).stubs();
    EXPECT_THROW(service.AllocCommResource(tilingData, &commContext, Hccl::AcceleratorState::AIV), NotSupportException);

    free(fusionArgs.args);
    delete[] buf;
}

TEST_F(CollServiceDeviceModeTest, Ut_HandleAclGraphFirstOpAivBuff_When_InputValue_Expect_HCCL_SUCCESS)
{
    auto service = dynamic_cast<CollServiceDeviceMode*>(fakeComm.collService);
    char* ptr = "test";
    void* voidPtr = ptr;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), outBound(voidPtr), outBound(true))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_SUCCESS));

    rtStream_t stream;
    // 执行步骤
    auto ret = service->HandleAclGraphFirstOpAivBuff(stream);

    // 后置验证
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CollServiceDeviceModeTest, Ut_AllocCommResource_When_versionIs100_Expect_NOTHROW)
{
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));

    CommunicatorImpl comm;
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.rankSize = 2;
    comm.myRank = 0;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);

    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    comm.socketManager = std::make_unique<SocketManager>(comm, 0, 0, 6000);
    comm.ubMemoryTransportMgr = std::make_unique<UbMemoryTransportMgr>(comm);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);

    comm.rankGraph = std::make_unique<RankGraph>(0);
    auto peer0 = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto peer1 = std::make_shared<NetInstance::Peer>(1, 1, 1, 0);
    IpAddress inputAddr(0);
    std::set<std::string> ports1 = {"0/1"};
    std::set<std::string> ports2 = {"0/2"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    u32 hop = 1;
    LinkDirection direction = LinkDirection::BOTH;
    shared_ptr<NetInstance::ConnInterface> sourceIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports1, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::ConnInterface> targetIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports2, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::Link> link = std::make_shared<NetInstance::Link>(
        peer0, peer1, sourceIface, targetIface, LinkType::PEER2PEER, protocols, direction, hop);
    peer0->AddConnInterface(0, sourceIface);
    peer1->AddConnInterface(0, targetIface);
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddPeer(peer1);

    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&)).stubs();

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs();
    std::vector<Hccl::LinkData> linkVec;
    char* buf = new char[16 * 1024 * 1024];
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(buf)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x100000));

    void* mem = malloc(sizeof(Mc2InitTilingInner) + sizeof(Mc2CcTilingInner));
    Mc2InitTilingInner* mc2TilingPtr = reinterpret_cast<Mc2InitTilingInner*>(mem);
    mc2TilingPtr->version = 100;
    mc2TilingPtr->mc2HcommCnt = 1;
    mc2TilingPtr->offset[0] = sizeof(Mc2InitTilingInner);
    Mc2CcTilingInner* commConfigPtr
        = reinterpret_cast<Mc2CcTilingInner*>(reinterpret_cast<uint8_t*>(mc2TilingPtr) + mc2TilingPtr->offset[0]);
    commConfigPtr->opType = AicpuComType::HCCL_CMD_ALLTOALLV;
    commConfigPtr->reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    commConfigPtr->srcDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    commConfigPtr->dstDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    commConfigPtr->communicationEngine = 3; // aiv
    commConfigPtr->protocol = 0;            // aiv ubmemory

    void* commContext = nullptr;

    CollServiceDeviceMode service(&comm);
    comm.collService = &service;
    MOCKER_CPP(&UbMemoryTransportMgr::TransportsConnect).stubs();
    EXPECT_NO_THROW(service.AllocCommResource(mc2TilingPtr, &commContext, Hccl::AcceleratorState::AIV));

    delete[] buf;
    free(mc2TilingPtr);
}

TEST_F(CollServiceDeviceModeTest, ut_alloc_cnt_notify_for_single_queue_with_local_post_to)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));

    CommunicatorImpl comm;
    auto queueNotifyManager = std::make_unique<QueueNotifyManager>(comm);
    CollServiceDeviceMode service(&comm);
    comm.ccuQueueNotifyManager_ = std::move(queueNotifyManager);

    InsQueue insQueue;
    auto insLocalPostTo = std::make_unique<InsLocalPostTo>(1);
    insLocalPostTo->SetPostQid(99);
    insQueue.Append(std::move(insLocalPostTo));
    service.AllocQNotifyForSingleQ(insQueue);
}

TEST_F(CollServiceDeviceModeTest, ut_alloc_cnt_notify_for_single_queue_with_local_wait_from)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));

    CommunicatorImpl comm;
    auto queueNotifyManager = std::make_unique<QueueNotifyManager>(comm);
    CollServiceDeviceMode service(&comm);
    comm.ccuQueueNotifyManager_ = std::move(queueNotifyManager);

    InsQueue insQueue;
    auto insLocalWaitFrom = std::make_unique<InsLocalWaitFrom>(1);
    insLocalWaitFrom->SetWaitQid(99);
    insQueue.Append(std::move(insLocalWaitFrom));
    service.AllocQNotifyForSingleQ(insQueue);
}
