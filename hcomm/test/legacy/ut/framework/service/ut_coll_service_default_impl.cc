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
#include <hccl/hccl_types.h>
#define private public
#define protected public
#include "coll_service_default_impl.h"
#include "communicator_impl.h"
#include "stream_manager.h"
#include "base_config_legacy.h"
#include "cfg_field.h"
#include "env_config_v2.h"
#include "rank_table.h"
#include "virtual_topo.h"
#include "json_parser.h"
#include "internal_exception.h"
#include "dev_ub_connection.h"
#include "virtual_topo_stub.h"
#include "coll_alg_component_builder.h"
#include "local_ipc_rma_buffer_v2.h"
#include "not_support_exception.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_component.h"
#include "rank_gph.h"
#include "rdma_handle_manager.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#undef protected
#undef private
#include "hccl_comm.h"
#include "coll_operator.h"
#include "coll_service_base.h"

using namespace Hccl;
using namespace std;

class CollServiceDefaultImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollServiceDefaultImpl SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollServiceDefaultImpl TearDown" << std::endl; }

    virtual void SetUp()
    {
        devBuf = DevBuffer::Create(0x100, 0x100);
        MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
        MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
        MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
        std::cout << "A Test case in CollServiceDefaultImpl SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in CollServiceDefaultImpl TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    std::shared_ptr<DevBuffer> devBuf;
};

class FakeCollAlgComponent : public CollAlgComponent {
public:
    FakeCollAlgComponent() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {};
    HcclResult Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr queue, string& algName)
    {
        return HCCL_SUCCESS;
    }

    HcclResult Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr queue, string& algName)
    {
        return HCCL_SUCCESS;
    }
};

class FakeCollAlgComponentWithError : public CollAlgComponent {
public:
    FakeCollAlgComponentWithError() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {}
    HcclResult Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr queue, string& algName)
    {
        return HCCL_E_INTERNAL;
    }
};

#define SETUP_COLL_SERVICE_DEFAULT_MOCKS()                                                     \
    do {                                                                                       \
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                     \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1))); \
        DevType devType = DevType::DEV_TYPE_950;                                               \
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));                           \
        MOCKER(HrtIpcSetMemoryName).stubs();                                                   \
        MOCKER(HrtDevMemAlignWithPage).stubs();                                                \
        MOCKER(HrtIpcDestroyMemoryName).stubs();                                               \
        std::unique_ptr<RdmaHandle> handle = std::make_unique<RdmaHandle>();                   \
        RdmaHandle handlePtr = handle.get();                                                   \
        MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any()).will(returnValue(handlePtr));      \
    } while (0)

TEST_F(CollServiceDefaultImplTest, test_base_register_op_base_buf)
{
    CommunicatorImpl comm;
    comm.id = "test";
    comm.dataBufferManager = make_unique<DataBufManager>();
    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    EXPECT_NO_THROW(service.RegisterOpBufToBufMgr(op));
}

TEST_F(CollServiceDefaultImplTest, test_orchestrate_with_ins)
{
    CommunicatorImpl comm;
    comm.id = "test";
    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    op.opMode = OpMode::OPBASE;
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 4;
    op.scratchMem = DevBuffer::Create(0x100, 1);
    op.root = 0;
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    EXPECT_NO_THROW(service.OrchestrateWithIns(op));
}

TEST_F(CollServiceDefaultImplTest, test_orchestrate_with_prim)
{
    CommunicatorImpl comm;
    comm.id = "test";
    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    op.opMode = OpMode::OPBASE;
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 4;
    op.root = 0;
    op.scratchMem = DevBuffer::Create(0x100, 1);
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, PrimQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    EXPECT_NO_THROW(service.OrchestrateWithPrim(op));
}

TEST_F(CollServiceDefaultImplTest, test_orchestrate_with_ins_throw_exception)
{
    CommunicatorImpl comm;
    comm.id = "test";
    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    op.opMode = OpMode::OPBASE;
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.scratchMem = DevBuffer::Create(0x100, 1);
    op.dataCount = 4;
    op.root = 0;
    comm.collAlgComponent = std::make_shared<FakeCollAlgComponentWithError>();
    EXPECT_THROW(service.OrchestrateWithIns(op), InternalException);
}

TEST_F(CollServiceDefaultImplTest, test_base_register_op_base_stream)
{
    CommunicatorImpl comm;
    comm.id = "test";
    comm.streamManager = make_unique<StreamManager>(&comm);
    CollServiceDefaultImpl service(&comm);
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    EXPECT_NO_THROW(service.RegisterOpbasedStream(move(make_unique<Stream>((void*)1))));
}

TEST_F(CollServiceDefaultImplTest, test_alloc_queue_notify)
{
    CommunicatorImpl comm;
    CollServiceDefaultImpl service(&comm);
    InsQueue queue;
    service.AllocQueueNotify(queue);
}

TEST_F(CollServiceDefaultImplTest, alloc_queue_notify_for_single_queue)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));

    CommunicatorImpl comm;
    auto queueNotifyManager = std::make_unique<QueueNotifyManager>(comm);
    auto queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
    CollServiceDefaultImpl service(&comm);
    comm.ccuQueueNotifyManager_ = std::move(queueNotifyManager);
    comm.queueWaitGroupCntNotifyManager = std::move(queueWaitGroupCntNotifyManager);

    InsQueue insQueue;
    auto insLocalWaitGroup = std::make_unique<InsLocalWaitGroup>(0);
    insQueue.Append(std::move(insLocalWaitGroup));
    service.AllocQNotifyForSingleQ(insQueue);
}

TEST_F(CollServiceDefaultImplTest, alloc_cnt_notify_for_single_queue)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));

    CommunicatorImpl comm;
    auto queueNotifyManager = std::make_unique<QueueNotifyManager>(comm);
    auto queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
    CollServiceDefaultImpl service(&comm);
    comm.ccuQueueNotifyManager_ = std::move(queueNotifyManager);
    comm.queueBcastPostCntNotifyManager = std::move(queueBcastPostCntNotifyManager);

    InsQueue insQueue;
    auto insLocalBcastPost = std::make_unique<InsLocalBcastPost>(0);
    insQueue.Append(std::move(insLocalBcastPost));
    service.AllocQNotifyForSingleQ(insQueue);
}

TEST_F(CollServiceDefaultImplTest, col_service_default_impl_load_with_op_based_mode_success)
{
    u32 remoteRank = 1;
    CommunicatorImpl comm;
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    std::string name = "test";
    comm.cclBuffer = DevBuffer::Create(0x1, 200);
    comm.CovertToCurrentCollOperator(name, collOpParams, OpMode::OPBASE);
    comm.id = "test";
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.streamManager = make_unique<StreamManager>(&comm);
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.socketManager = std::make_unique<SocketManager>(comm, 1, 1, 1);
    comm.memTransportManager = make_unique<MemTransportManager>(comm);

    CollServiceDefaultImpl service(&comm);
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    service.connectionsBuilders[comm.id] = std::make_unique<ConnectionsBuilder>(comm);
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    MOCKER_CPP(&CollServiceDefaultImpl::RegisterOpBufToBufMgr).stubs();
    MOCKER_CPP(&CollServiceDefaultImpl::RegisterOpbasedStream).stubs();

    MOCKER_CPP(&Trace::Save).stubs();

    vector<LinkData> links;
    MOCKER_CPP(&InsQueue::GetUniqueLinks).stubs().will(returnValue(links));
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();

    shared_ptr<InsQueue> insQueue = make_shared<InsQueue>();
    MOCKER_CPP(&PrimTranslator::Translate).stubs().will(returnValue(insQueue));

    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&)).stubs();

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs();

    MOCKER_CPP(&Interpreter::Submit).stubs();

    CollOperator op;
    op.opType = OpType::BARRIER;
    op.opMode = OpMode::OPBASE;
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    op.scratchMem = DevBuffer::Create(0x100, 1);
    auto stream = std::make_unique<Stream>(nullptr);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.rankGraph = std::make_unique<RankGraph>(comm.GetMyRank());
    comm.rankGraph->peers_[comm.GetMyRank()] = std::make_shared<NetInstance::Peer>(comm.GetMyRank(), 0, 0, 0);

    comm.streamManager->opbase->slaves.push_back(std::make_unique<Stream>(nullptr));
    MOCKER(HcclStreamSynchronize).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    EXPECT_NO_THROW(service.LoadWithOpBasedMode(op, std::move(stream)));
}

TEST_F(CollServiceDefaultImplTest, coll_service_default_impl_orchestrate_with_ins_success)
{
    CommunicatorImpl comm;
    comm.id = "test";
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.streamManager = make_unique<StreamManager>(&comm);
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.socketManager = std::make_unique<SocketManager>(comm, 1, 1, 1);
    comm.trace = std::make_unique<Trace>();
    comm.memTransportManager = make_unique<MemTransportManager>(comm);
    comm.cclBuffer = DevBuffer::Create(0x100, 200);

    u32 remoteRank = 1;
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    std::string name = "test";
    comm.CovertToCurrentCollOperator(name, collOpParams, OpMode::OPBASE);

    CollServiceDefaultImpl service(&comm);
    std::vector<Stream*> slaveVec;
    MOCKER_CPP(&Interpreter::Submit).stubs();
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    service.connectionsBuilders[comm.id] = std::make_unique<ConnectionsBuilder>(comm);

    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER_CPP(&CollServiceDefaultImpl::RegisterOpBufToBufMgr).stubs();
    MOCKER_CPP(&CollServiceDefaultImpl::RegisterOpbasedStream).stubs();
    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&)).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();

    vector<LinkData> links;
    MOCKER_CPP(&InsQueue::GetUniqueLinks).stubs().will(returnValue(links));

    shared_ptr<InsQueue> insQueue = make_shared<InsQueue>();
    MOCKER_CPP(&PrimTranslator::Translate).stubs().will(returnValue(insQueue));

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs();

    CollOperator op;
    op.opMode = OpMode::OPBASE;
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    op.scratchMem = DevBuffer::Create(0x100, 1);
    auto stream = std::make_unique<Stream>(nullptr);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.rankGraph = std::make_unique<RankGraph>(comm.GetMyRank());
    comm.rankGraph->peers_[comm.GetMyRank()] = std::make_shared<NetInstance::Peer>(comm.GetMyRank(), 0, 0, 0);

    EnvAlgoConfig fakeAlgoCfg;
    EnvAlgoConfig& algoCfg = fakeAlgoCfg;
    algoCfg.bufferSize = CfgField<u64>(
        "HCCL_BUFFSIZE", 200 * 1024 * 1024, Str2T<u64>, CHK_RANGE_CLOSED<u64>(1, ULLONG_MAX), [](u64& i) {
            i *= 1024 * 1024;
        });
    algoCfg.bufferSize.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetAlgoConfig).stubs().will(returnValue(algoCfg));
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    EXPECT_NO_THROW(service.LoadWithOpBasedMode(op, std::move(stream)));
}

std::string topoInfoPath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/service/topo.json"};
TEST_F(CollServiceDefaultImplTest, test_base_register_offload_buf)
{
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath).stubs().will(returnValue(topoInfoPath));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER(HrtIpcSetMemoryName).stubs();
    MOCKER(HrtDevMemAlignWithPage).stubs();
    MOCKER(HrtIpcDestroyMemoryName).stubs();
    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));

    GenRankTableFile4p();
    GenTopoFile();

    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HCCL_SUCCESS));

    LocalIpcRmaBuffer localRmaBuf(devBuf);
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .will(returnValue(dynamic_cast<LocalRmaBuffer*>(&localRmaBuf)));

    CommunicatorImpl comm;
    comm.devLogicId = 1;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = 1;
    commParams.rankSize = 4;
    commParams.devType = DevType::DEV_TYPE_950;
    comm.InitDataBufferManager();
    comm.Init(commParams, RankTable4p, config);

    DelRankTableFile4p();
    DelTopoFile();

    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    op.opTag = "test_opTag";
    EXPECT_NO_THROW(service.RegisterOpBufToBufMgr(op));
    EXPECT_NO_THROW(service.RegisterOffloadLocalRmaBuf(op.opTag));
}

TEST_F(CollServiceDefaultImplTest, test_base_register_offload_stream)
{
    CommunicatorImpl comm;
    comm.id = "commId";
    comm.streamManager = make_unique<StreamManager>(&comm);
    CollServiceDefaultImpl service(&comm);
    std::string opTag = "opTag";
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    EXPECT_NO_THROW(service.RegisterOffloadMasterStream(opTag, move(make_unique<Stream>())));
}
#if 0
TEST_F(CollServiceDefaultImplTest, test_calc_coll_offload_op_res_with_hccl_success_returned)
{
    setenv("PRIM_QUEUE_GEN_NAME", "CcuAllReduceMesh1D", 1);
    CommunicatorImpl comm;
    comm.id = "commId";
    comm.streamManager = make_unique<StreamManager>(&comm);
    CollServiceDefaultImpl service(&comm);

    OpType opType = OpType::ALLREDUCE;
    u64 dataSize = 100;
    CollOffloadOpResReq resReq1;
    resReq1.requiredSubQueNum = 2;
    resReq1.requiredScratchMemSize = 0;

    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);

    RankId myRank = 0;
    u32 rankSize = 4;

    CollAlgComponentBuilder collAlgComponentBuilder;
    std::shared_ptr<CollAlgComponent> collAlgComponent = collAlgComponentBuilder.SetRankGraph(&virtTopo)
                                                             .SetDevType(DevType::DEV_TYPE_950)
                                                             .SetMyRank(myRank)
                                                             .SetRankSize(rankSize)
                                                             .Build();
    comm.collAlgComponent = collAlgComponent;
    OpExecuteConfig opConfig;  // host 展开，图模式使用
    opConfig.accState = AcceleratorState::HOSTCPU_TS;
    comm.opExecuteConfig = opConfig;
    EXPECT_NO_THROW(comm.CalcCollOffloadOpRes(opType, dataSize, HCCL_DATA_TYPE_INT8, resReq1));
    EXPECT_EQ(16, resReq1.requiredSubQueNum);
    EXPECT_EQ(256 * 1024 * 1024, resReq1.requiredScratchMemSize);
}
#endif
TEST_F(CollServiceDefaultImplTest, test_init)
{
    std::pair<TokenIdHandle, uint32_t> retPair = {1, 1};
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(retPair));
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath).stubs().will(returnValue(topoInfoPath));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    std::unique_ptr<RdmaHandle> handle = std::make_unique<RdmaHandle>();
    RdmaHandle handlePtr = handle.get();
    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any()).will(returnValue(handlePtr));

    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER(HrtIpcSetMemoryName).stubs();
    MOCKER(HrtDevMemAlignWithPage).stubs();
    MOCKER(HrtIpcDestroyMemoryName).stubs();

    GenRankTableFile4p();
    GenTopoFile();

    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HCCL_SUCCESS));

    LocalIpcRmaBuffer localRmaBuf(devBuf);

    CommunicatorImpl comm;
    comm.devLogicId = 1;
    CommParams commParams;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    commParams.commId = "commId";
    commParams.myRank = 1;
    commParams.rankSize = 4;
    commParams.devType = DevType::DEV_TYPE_950;
    comm.Init(commParams, RankTable4p, config);

    DelRankTableFile4p();
    DelTopoFile();

    CollServiceDefaultImpl service(&comm);
    service.Init();
}

TEST_F(CollServiceDefaultImplTest, test_load_with_op_based_mode)
{
    SETUP_COLL_SERVICE_DEFAULT_MOCKS();

    GenRankTableFile4p();
    GenTopoFile();

    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HCCL_SUCCESS));

    LocalIpcRmaBuffer localRmaBuf(devBuf);
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .will(returnValue(dynamic_cast<LocalRmaBuffer*>(&localRmaBuf)));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath).stubs().will(returnValue(topoInfoPath));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    CommunicatorImpl comm;
    comm.devLogicId = 1;
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    CommParams commParams;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    commParams.commId = "commId";
    commParams.myRank = 1;
    commParams.rankSize = 4;
    commParams.devType = DevType::DEV_TYPE_950;
    comm.devLogicId = 0;
    comm.Init(commParams, RankTable4p, config);
    u32 remoteRank = 1;
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    std::string name = "test";
    comm.cclBuffer = DevBuffer::Create(0x100, 10);
    comm.opExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
    comm.CovertToCurrentCollOperator(name, collOpParams, OpMode::OPBASE);
    comm.ExecAlgSelect(collOpParams, OpMode::OPBASE);

    DelRankTableFile4p();
    DelTopoFile();

    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    service.AddOpCounterMems();
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    op.scratchMem = DevBuffer::Create(0x100, 1);
    op.opTag = "test_opTag";

    StreamManager streamManager(&comm);

    for (auto opType : {OpType::ALLREDUCE, OpType::SEND}) {
        service.LoadWithOpBasedMode(op, std::move(make_unique<Stream>((void*)1)));
        CollOpParams opAdaptor;
        opAdaptor.opType = opType;
        comm.ExecAlgSelect(opAdaptor, OpMode::OPBASE);
    }
    service.LoadWithOpBasedMode(op, std::move(make_unique<Stream>((void*)1)));
}

TEST_F(CollServiceDefaultImplTest, test_load_with_offload_mode)
{
    SETUP_COLL_SERVICE_DEFAULT_MOCKS();

    GenRankTableFile4p();
    GenTopoFile();
    const string rankTablePath = "ranktable.json";

    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollServiceDefaultImpl::AddCountTask).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&Interpreter::Submit).stubs().will(ignoreReturnValue());

    LocalIpcRmaBuffer localRmaBuf(devBuf);
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .will(returnValue(dynamic_cast<LocalRmaBuffer*>(&localRmaBuf)));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath).stubs().will(returnValue(topoInfoPath));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    CommunicatorImpl comm;
    comm.devLogicId = 1;
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    CommParams commParams;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    commParams.commId = "commId";
    commParams.myRank = 1;
    commParams.rankSize = 4;
    commParams.devType = DevType::DEV_TYPE_950;
    comm.Init(commParams, RankTable4p, config);
    u32 remoteRank = 1;
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    std::string name = "test";
    comm.cclBuffer = DevBuffer::Create(0x100, 10);
    comm.opExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
    comm.CovertToCurrentCollOperator(name, collOpParams, OpMode::OPBASE);
    comm.ExecAlgSelect(collOpParams, OpMode::OPBASE);
    DelRankTableFile4p();
    DelTopoFile();

    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    service.AddOpCounterMems();
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    op.scratchMem = DevBuffer::Create(0x100, 1);
    op.opTag = "test_opTag";

    comm.streamManager = make_unique<StreamManager>(&comm);
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;

    service.LoadWithOffloadMode(op, std::move(make_unique<Stream>()));
}

TEST_F(CollServiceDefaultImplTest, test_add_nop)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    CommunicatorImpl comm;
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    std::string opTag = "test";
    comm.streamManager->offload->RegisterMaster(opTag, std::move(make_unique<Stream>()));
    CollServiceDefaultImpl service(&comm);

    CollOperator op;
    op.opTag = "test";

    IpAddress ipAddress("1.0.0.0");
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OFFLOAD);
    MOCKER_CPP(&RmaConnManager::Get).stubs().will(returnValue(dynamic_cast<RmaConnection*>(&devUbConnection)));

    std::vector<LinkData> links = {linkData};

    MOCKER(HrtUbDbSend).stubs().with(mockcpp::any(), mockcpp::any());
    service.AddNop(op.opTag, links);
}

TEST_F(CollServiceDefaultImplTest, col_service_default_impl_update_ub_ci_if_need_success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    u32 remoteRank = 1;
    CommunicatorImpl comm;
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    std::string name = "test";
    comm.cclBuffer = DevBuffer::Create(0x100, 200);
    comm.CovertToCurrentCollOperator(name, collOpParams, OpMode::OPBASE);
    comm.id = "test";
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.streamManager = make_unique<StreamManager>(&comm);
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    void* ptr;
    unique_ptr<Stream> master = make_unique<Stream>((ptr));
    comm.streamManager->opbase->master = std::move(master);
    comm.socketManager = std::make_unique<SocketManager>(comm, 1, 1, 1);
    comm.memTransportManager = make_unique<MemTransportManager>(comm);

    CollServiceDefaultImpl service(&comm);
    service.updatingUbCiEvent = nullptr;
    MOCKER(IfNeedUpdatingUbCi).stubs().will(returnValue(true));
    MOCKER_CPP(&UbCiUpdaterManager::SaveConnsCi).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    CollOperator op;
    op.opTag = "test";
    service.UpdateUbCiIfNeed(op.opTag);
    service.updatingUbCiEvent = make_unique<MaskEvent>();
    RtEvent_t fakePtr = nullptr;
    aclrtEventWaitStatus status = ACL_EVENT_WAIT_STATUS_COMPLETE;
    MOCKER(aclrtQueryEventWaitStatus)
        .stubs()
        .with(mockcpp::any(), outBoundP(&status, sizeof(status)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(&UbCiUpdaterManager::UpdateConnsCi).stubs().with(mockcpp::any());
    service.UpdateUbCiIfNeed(op.opTag);
}

TEST_F(CollServiceDefaultImplTest, AddCountTask)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER(HrtReduceAsync).stubs().with(mockcpp::any());
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));

    CommunicatorImpl comm;
    comm.streamManager = make_unique<StreamManager>(&comm);
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    CollServiceDefaultImpl collServiceDefaultImpl(&comm);
    EXPECT_NO_THROW(collServiceDefaultImpl.AddCountTask(true));
}

TEST_F(CollServiceDefaultImplTest, test_load_with_offload_mode_with_task)
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER(HrtIpcSetMemoryName).stubs();
    MOCKER(HrtDevMemAlignWithPage).stubs();
    MOCKER(HrtIpcDestroyMemoryName).stubs();
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath).stubs().will(returnValue(topoInfoPath));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    std::unique_ptr<RdmaHandle> handle = std::make_unique<RdmaHandle>();
    RdmaHandle handlePtr = handle.get();
    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any()).will(returnValue(handlePtr));

    GenRankTableFile4p();
    GenTopoFile();
    const string rankTablePath = "ranktable.json";

    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HCCL_SUCCESS));

    LocalIpcRmaBuffer localRmaBuf(devBuf);
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .will(returnValue(dynamic_cast<LocalRmaBuffer*>(&localRmaBuf)));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect).stubs().will(returnValue(HCCL_SUCCESS));

    CommunicatorImpl comm;
    comm.devLogicId = 1;
    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    CommParams commParams;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    commParams.commId = "commId";
    commParams.myRank = 1;
    commParams.rankSize = 4;
    commParams.devType = DevType::DEV_TYPE_950;
    comm.Init(commParams, RankTable4p, config);
    u32 remoteRank = 1;
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    std::string name = "test";
    comm.cclBuffer = DevBuffer::Create(0x100, 10);
    comm.opExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
    comm.CovertToCurrentCollOperator(name, collOpParams, OpMode::OPBASE);
    comm.ExecAlgSelect(collOpParams, OpMode::OPBASE);
    DelRankTableFile4p();
    DelTopoFile();

    CollServiceDefaultImpl service(&comm);
    CollOperator op;
    service.AddOpCounterMems();
    op.inputMem = DevBuffer::Create(0x100, 1);
    op.outputMem = DevBuffer::Create(0x100, 1);
    op.scratchMem = DevBuffer::Create(0x100, 1);
    op.opTag = "test_opTag";

    comm.streamManager = make_unique<StreamManager>(&comm);
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    service.LoadWithOffloadMode(op, std::move(make_unique<Stream>()));
}

TEST_F(CollServiceDefaultImplTest, Test_RecoverTransport)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    CollServiceDefaultImpl collServiceDefaultImpl(comm.get());

    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(make_pair<uint32_t, uint32_t>(0, 0)));
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

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

    EXPECT_THROW(collServiceDefaultImpl.RecoverTransport(links, linkGroupPair), NotSupportException);
}
