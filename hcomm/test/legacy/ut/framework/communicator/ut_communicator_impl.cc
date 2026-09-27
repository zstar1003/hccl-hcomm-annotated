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

#define private public
#define protected public
#include "communicator_impl.h"
#include "internal_exception.h"
#include "null_ptr_exception.h"
#include "invalid_params_exception.h"
#include "orion_adapter_tsd.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "op_base_v2.h"
#include "hccp_ctx.h"
#include "hccp_common.h"

#include <stdexcept>
#include <string>
#include <exception>
#include <thread>
#include "coll_service_default_impl.h"
#include "rank_table.h"
#include "json_parser.h"
#include "rank_table.h"
#include "net_instance.h"
#include "rank_graph_builder.h"
#include "phy_topo_builder.h"
#include "detour_service.h"
#include "rank_table.h"
#include "sal.h"
#include "rank_gph.h"
#include "base_config_legacy.h"
#include "env_config_v2.h"
#include "coll_service_device_mode.h"
#include "communicator_callback.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_component.h"
#include "task_abort_handler_v2.h"
#include "hccp_tlv_hdc_manager.h"
#include "ccu_driver_handle.h"
#include "comm_manager.h"

#include "op_params_checker.h"
#include "hdc_lite.h"
#include "kfc.h"
#include "stream_manager.h"
#include "virtual_topo_stub.h"
#include "ins_all_reduce_sole_executor.h"
#include "ins_v2_all_reduce_sole_executor.h"
#include "topo_match_mesh.h"
#include "ccu_temp_all_reduce_mesh_1D_one_shot.h"
#include "ccu_context_all_reduce_mesh1d_one_shot.h"
#include "ccu_instruction_all_reduce_mesh1d_one_shot.h"

#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#include "ccu_context_all_reduce_mesh1d_mem2mem.h"
#include "ccu_instruction_all_reduce_mesh1d_mem2mem.h"

#include "ccu_ins_group.h"
#include "env_config_stub.h"
#include "coll_alg_component.h"
#include "hccl_communicator.h"
#include "op_base_v2.h"
#include "hccl_comm.h"
#include "ranktable_stub_clos.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "topo_addr_info.h"
#include "hostdpu/dpu_kernel_entrance.h"
#include "hal.h"
#undef private
#undef protected

using namespace Hccl;
std::map<std::string, std::string> envCommCfgMap = defaultEnvCfgMap;
constexpr u32 TEMP_UES_CNTCKE_NUM = 16;

#define HCCL_HDC_TYPE_D2H 0
#define HCCL_HDC_TYPE_H2D 1

char* commGetenv_stub(const char* __name)
{
    char* ret = const_cast<char*>(envCommCfgMap[std::string(__name)].c_str());
    return ret;
}

class CommunicatorImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        TaskAbortHandler::GetInstance();
        u32 fakeDevPhyId = 1;
        u64 fakeNotifyHandleAddr = 100;
        u32 fakeNotifyId = 1;
        u64 fakeOffset = 200;
        char fakeName[65] = "testRtsNotify";
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtStreamGetMode).stubs().will(returnValue((u64)1));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtNotifyCreateWithFlag).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId)));
        MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));
        MOCKER(HrtMemAsyncCopy).stubs();

        // 资源初始化
        MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
        MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
        MOCKER_CPP(&TaskAbortHandler::Register).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER_CPP(&TaskAbortHandler::UnRegister).stubs().with(mockcpp::any()).will(ignoreReturnValue());

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

        fakeComm.RegisterAcceStateCallBack(CommunicatorCallback());
        fakeComm.cclBuffer = DevBuffer::Create(0x100, 0x100);
        fakeComm.aivTagBuffer = DevBuffer::Create(0x100, 10);
        fakeComm.aivOffloadTagBuffer = DevBuffer::Create(0x100, 10);
        fakeComm.SetCommStatus(CommStatus::COMM_READY);
        fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
        fakeComm.InitNotifyManager();
        fakeComm.InitSocketManager();
        fakeComm.InitRmaConnManager();
        fakeComm.InitStreamManager();
        fakeComm.InitMemTransportManager();
        fakeComm.InitMirrorTaskManager();
        fakeComm.InitProfilingReporter();
        fakeComm.InitUbMemoryTransportMgr();
        MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
        MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
        MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
        fakeComm.devLogicId = 0;
        MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        fakeComm.myRank = 0;
        fakeComm.id = "testTag";
        fakeComm.streamManager->opbase = make_unique<OpbaseStreamManager>(&fakeComm);
        fakeComm.streamManager->opbase->master = make_unique<Stream>(&fakeComm);
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
            .defaults()
            .with(mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
        OpExecuteConfig opConfig; // aicpu 展开
        opConfig.accState = AcceleratorState::AICPU_TS;
        fakeComm.opExecuteConfig = opConfig;

        // 算法组件初始化
        CollAlgOpReq collAlgOpReq;
        collAlgOpReq.algName = "testAlg";
        collAlgOpReq.resReq.primQueueNum = 1;
        CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
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

void CommImplSendStub1() { THROW<InternalException>("HcclException &e"); }

void MockCommunicatorImpl()
{
    SETUP_COMM_INIT_MOCKS();
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RmaConnManager::Clear).stubs();

    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
}

class LoadOffloadCollOpTest : public CommunicatorImplTest {
protected:
    void SetUp() override
    {
        CommunicatorImplTest::SetUp();
        // 初始化测试环境
        // rankSize 未在 InitCommonData 中初始化, 显式设置为多卡, 避免走 rankSize==1 的
        // SingleRankProc 快速路径导致返回 HCCL_SUCCESS 而非预期错误码
        fakeComm.rankSize = 2;
        fakeComm.SetCommStatus(CommStatus::COMM_READY);
        fakeComm.commExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
        fakeComm.opExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
    }
};

TEST_F(CommunicatorImplTest, should_return_success_when_calling_init_with_all_process_finished)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 1, 1, 0);
    EXPECT_EQ(0, comm.GetIdIndex());
}

TEST_F(CommunicatorImplTest, should_return_failed_when_calling_create_subcomm_with_comm_uninitialized)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    CommParams params;
    std::vector<u32> rankIds;
    CommunicatorImpl subCommImpl;
    HcclCommConfig subConfig;
    EXPECT_EQ(HcclResult::HCCL_E_INTERNAL, comm.CreateSubComm(params, rankIds, &subCommImpl, subConfig));
}

TEST_F(CommunicatorImplTest, should_return_failed_when_calling_create_subcomm_with_comm_uninitialized_1)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    CommParams params;
    std::vector<u32> rankIds;
    CommunicatorImpl subCommImpl;
    EXPECT_EQ(HcclResult::HCCL_E_INTERNAL, comm.CreateSubComm(params, rankIds, &subCommImpl));
}

TEST_F(CommunicatorImplTest, should_return_success_when_calling_suspend_without_aicpu_kernel_launched)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitRmaConnManager();
    EXPECT_EQ(false, comm.isSuspended);
    EXPECT_EQ(false, comm.isAicpuKernelLaunched);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, comm.Suspend());
    EXPECT_EQ(true, comm.isSuspended);

    // 已经处于isSuspended == true，重复Suspend()无效
    comm.isSuspended = true;
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, comm.Suspend());
}

TEST_F(CommunicatorImplTest, should_return_success_when_calling_clean_without_aicpu_kernel_launched)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitRmaConnManager();
    comm.InitMemTransportManager();
    comm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    // isSuspended == false，不能调用Clean
    comm.isSuspended = false;
    EXPECT_EQ(HcclResult::HCCL_E_NOT_SUPPORT, comm.Clean());

    // isSuspended == true，但是未下发kernel
    comm.isSuspended = true;
    comm.isAicpuKernelLaunched = false;
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, comm.Clean());
}

constexpr u32 h2dBufferSize = sizeof(KfcCommand);
constexpr u32 d2hBufferSize = sizeof(KfcExecStatus);

static HcclResult HrtDrvMemCpyStub(void* dst, uint64_t destMax, const void* src, uint64_t count)
{
    memcpy(dst, src, count);
    return HCCL_SUCCESS;
}

class HdcBufMocker {
public:
    HdcBufMocker()
    {
        memset_s(hostBufH2d, sizeof(hostBufH2d), 0, sizeof(hostBufH2d));
        memset_s(hostBufD2h, sizeof(hostBufD2h), 0, sizeof(hostBufD2h));
        memset_s(hostCacheD2h, sizeof(hostCacheD2h), 0, sizeof(hostCacheD2h));
        MOCKER(HrtMallocHost)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<void*>(hostBufH2d)))
            .then(returnValue(static_cast<void*>(hostBufD2h)))
            .then(returnValue(static_cast<void*>(hostCacheD2h)));

        memset_s(devBufH2d, sizeof(devBufH2d), 0, sizeof(devBufH2d));
        memset_s(devCacheH2d, sizeof(devCacheH2d), 0, sizeof(devCacheH2d));
        memset_s(devBufD2h, sizeof(devBufD2h), 0, sizeof(devBufD2h));
        MOCKER(HrtMalloc)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(static_cast<void*>(devBufH2d)))
            .then(returnValue(static_cast<void*>(devCacheH2d)))
            .then(returnValue(static_cast<void*>(devBufD2h)));
        MOCKER(HrtDrvMemCpy).stubs().with().will(invoke(HrtDrvMemCpyStub));
    }

private:
    char hostBufH2d[4 * 1024];
    char hostBufD2h[4 * 1024];
    char hostCacheD2h[4 * 1024];

    char devBufH2d[4 * 1024];
    char devCacheH2d[4 * 1024];
    char devBufD2h[4 * 1024];
};

TEST_F(CommunicatorImplTest, should_return_success_when_calling_suspend_with_aicpu_kernel_launched)
{
    HdcBufMocker hdcBufMocker;

    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    // 实现host侧对应的内容
    comm.kfcControlTransferH2D = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_H2D, h2dBufferSize);
    comm.kfcStatusTransferD2H = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_D2H, d2hBufferSize);
    comm.kfcControlTransferH2D->Init();
    comm.kfcStatusTransferD2H->Init();
    auto kfcControlTransferH2DParams = comm.kfcControlTransferH2D->GetCommunicateParams();
    auto kfcStatusTransferD2HParams = comm.kfcStatusTransferD2H->GetCommunicateParams();
    // 实现device侧对应的内容->保证device侧共享内存和host侧共享内存是一个
    std::unique_ptr<HDCommunicateLite> h2dTransfer = std::make_unique<HDCommunicateLite>();
    std::unique_ptr<HDCommunicateLite> d2hTransfer = std::make_unique<HDCommunicateLite>();
    h2dTransfer->Init(kfcControlTransferH2DParams);
    d2hTransfer->Init(kfcStatusTransferD2HParams);
    KfcCommand cmd = KfcCommand::NONE;
    memset_s(&cmd, sizeof(KfcCommand), 0, sizeof(KfcCommand));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    // 这里是模拟device背景线程的行为
    thread threadHandle([&] {
        auto timeout = std::chrono::milliseconds(100);
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            h2dTransfer->Get(0, sizeof(KfcCommand), (u8*)&cmd); // 从host侧拿到NS_STOP_LAUNCH的命令字
            if (cmd != KfcCommand::NONE) {
                break;
            }
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                break;
            }
        }
        response.kfcStatus = KfcStatus::STOP_LAUNCH_DONE;
        d2hTransfer->Put(0, sizeof(KfcExecStatus), (u8*)&response); // device就会把状态改为STOP_LAUNCH_DONE
        EXPECT_EQ(cmd, KfcCommand ::NS_STOP_LAUNCH); // 这个时候就是希望从host侧拿到的命令字NS_STOP_LAUNCH
    });
    usleep(1000);

    comm.isAicpuKernelLaunched = true;
    auto ret = comm.Suspend();
    threadHandle.join();
    EXPECT_EQ(HCCL_E_SUSPENDING, ret);
}

TEST_F(CommunicatorImplTest, should_return_success_when_calling_clean_with_aicpu_kernel_launched)
{
    HdcBufMocker hdcBufMocker;

    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.InitRmaConnManager();
    comm.InitMemTransportManager();
    // 实现host侧对应的内容
    comm.kfcControlTransferH2D = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_H2D, h2dBufferSize);
    comm.kfcStatusTransferD2H = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_D2H, d2hBufferSize);
    comm.kfcControlTransferH2D->Init();
    comm.kfcStatusTransferD2H->Init();
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    auto kfcControlTransferH2DParams = comm.kfcControlTransferH2D->GetCommunicateParams();
    auto kfcStatusTransferD2HParams = comm.kfcStatusTransferD2H->GetCommunicateParams();
    // 实现device侧对应的内容->保证device侧共享内存和host侧共享内存是一个
    std::unique_ptr<HDCommunicateLite> h2dTransfer = std::make_unique<HDCommunicateLite>();
    std::unique_ptr<HDCommunicateLite> d2hTransfer = std::make_unique<HDCommunicateLite>();
    h2dTransfer->Init(kfcControlTransferH2DParams);
    d2hTransfer->Init(kfcStatusTransferD2HParams);
    KfcCommand cmd = KfcCommand::NONE;
    memset_s(&cmd, sizeof(KfcCommand), 0, sizeof(KfcCommand));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    // 这里是模拟device背景线程的行为
    thread threadHandle([&] {
        response.kfcStatus = KfcStatus::STOP_LAUNCH_DONE;
        d2hTransfer->Put(0, sizeof(KfcExecStatus), (u8*)&response); // device先把状态改为STOP_LAUNCH_DONE
        auto timeout = std::chrono::milliseconds(100);
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            h2dTransfer->Get(0, sizeof(KfcCommand), (u8*)&cmd); // 从host侧拿到NS_CLEAN的命令字
            if (cmd != KfcCommand::NONE) {
                break;
            }
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                break;
            }
        }
        response.kfcStatus = KfcStatus::CLEAN_DONE;
        d2hTransfer->Put(0, sizeof(KfcExecStatus), (u8*)&response); // device就会把状态改为CLEAN_DONE
        EXPECT_EQ(cmd, KfcCommand ::NS_CLEAN); // 这个时候就是希望从host侧拿到的命令字NS_CLEAN
    });
    usleep(1000);

    comm.isSuspended = true;
    comm.isAicpuKernelLaunched = true;
    auto ret = comm.Clean();
    threadHandle.join();
    EXPECT_EQ(HCCL_E_SUSPENDING, ret);
}

TEST_F(CommunicatorImplTest, Ut_NsRecovery_Clean_When_Not_Load_Op_Expect_Success)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.isSuspended = true;
    comm.isCleaned = false;
    comm.commExecuteConfig.accState = AcceleratorState::CCU_MS;

    EXPECT_EQ(comm.Clean(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(comm.isCleaned, true);
}

TEST_F(CommunicatorImplTest, Ut_NsRecovery_Resume_When_Not_Load_Op_Expect_Success)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.isSuspended = true;
    comm.isCleaned = true;
    comm.commExecuteConfig.accState = AcceleratorState::CCU_MS;

    EXPECT_EQ(comm.Resume(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(comm.isSuspended, false);
    EXPECT_EQ(comm.isCleaned, false);
}

TEST_F(CommunicatorImplTest, initvittualtopo_check_fail)
{
    MockCommunicatorImpl();
    s32 myRank = 0;
    std::unique_ptr<RankGraph> inputVirtualTopo = std::make_unique<RankGraph>(myRank);
    CommunicatorImpl comm;
    EXPECT_THROW(comm.InitRankGraph(inputVirtualTopo), InvalidParamsException);
}

TEST_F(CommunicatorImplTest, should_return_success_when_normal_calling_new_init_with_two_parameters_new)
{
    SETUP_COMM_INIT_MOCKS();
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    TaskAbortHandler::GetInstance();

    void* mockTlvHandle = reinterpret_cast<void*>(0x1234);
    MOCKER_CPP(&HccpTlvHdcManager::GetTlvHandle).stubs().will(returnValue(mockTlvHandle));
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    comm.ccuDrvHandle = std::make_shared<Hccl::CcuDriverHandle>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 1, 1, 0);
    const string rankTablePath = "ranktable.json";
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    EXPECT_EQ(comm.Init(params, rankTablePath, config), HcclResult::HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, should_return_success_when_normal_calling_new_init_with_two_parameters_new_return_error)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    comm.initFlag = true;
    const string rankTablePath = "ranktable.json";
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    EXPECT_EQ(comm.Init(params, rankTablePath, config), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, init_with_two_parameters)
{
    SETUP_COMM_INIT_MOCKS();
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;
    comm.initFlag = false;
    s32 myRank = 2;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(myRank);
    comm.rankGraph = make_unique<RankGraph>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 1, 1, 0);
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(std::unique_ptr<RankGraph> & virtualTopo))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    EXPECT_EQ(comm.Init(params, virtualTopo, 0), HcclResult::HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, should_return_rank_size_when_calling_get_rank_size)
{
    CommunicatorImpl comm;
    comm.rankSize = 10;
    EXPECT_EQ(comm.GetRankSize(), 10);
}

TEST_F(CommunicatorImplTest, should_return_rank_id_when_calling_get_my_rank)
{
    CommunicatorImpl comm;
    comm.myRank = 10;
    EXPECT_EQ(comm.GetMyRank(), 10);
}

TEST_F(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperator)
{
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 2;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    dfxOpInfo->op_ = op;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST_F(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperator_allgather)
{
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLGATHER;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 2;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;
    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST_F(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperator_allgather2)
{
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLGATHER;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 2;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;
    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST_F(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperator_REDUCESCATTER)
{
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::REDUCESCATTER;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 2;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::REDUCESCATTER;
    opParams.count = 1;
    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST_F(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperatorA2A)
{
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 2;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLTOALL;
    opParams.all2AllDataDes.sendType = DataType::FP32;
    opParams.all2AllDataDes.recvType = DataType::FP32;
    opParams.all2AllDataDes.sendCount = 4;
    opParams.all2AllDataDes.recvCount = 4;
    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST_F(CommunicatorImplTest, CalcA2ASendRecvMem_test)
{
    CollOpParams opParams;
    opParams.opType = OpType::ALLTOALLV;
    int* sendCounts = new int[4]{1, 2, 3, 4};
    int* recvCounts = new int[4]{4, 3, 2, 1};
    int* sdispls = new int[4]{0, 1, 3, 6};
    int* rdispls = new int[4]{7, 6, 5, 4};
    opParams.all2AllVDataDes.sendCounts = sendCounts;
    opParams.all2AllVDataDes.recvCounts = recvCounts;
    opParams.all2AllVDataDes.sdispls = sdispls;
    opParams.all2AllVDataDes.rdispls = rdispls;
    CommunicatorImpl comm;
    comm.rankSize = 1;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.ConvertCollOperatorA2A(opParams);
    delete[] sendCounts;
    delete[] recvCounts;
    delete[] sdispls;
    delete[] rdispls;
}

TEST_F(CommunicatorImplTest, CalcA2ASendRecvMem_test_2)
{
    CollOpParams opParams;
    opParams.opType = OpType::ALLTOALLVC;
    int* sendCountMatrix = new int[4]{1, 2, 3, 4};
    opParams.all2AllVCDataDes.sendCountMatrix = sendCountMatrix;
    CommunicatorImpl comm;
    comm.rankSize = 1;
    comm.myRank = 0;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.ConvertCollOperatorA2A(opParams);
    delete[] sendCountMatrix;
}

TEST_F(CommunicatorImplTest, should_fail_when_LoadOpbasedCollOp_catch_HcclException)
{
    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    fakeComm.devLogicId = 0;
    fakeComm.rankSize = 2;

    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(invoke(CommImplSendStub1));

    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_fail_when_LoadOpbasedCollOp_catch_unknown_exception)
{
    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;
    fakeComm.rankSize = 2;
    fakeComm.devLogicId = 0;

    std::string str("...");
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(str));

    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_fail_when_LoadOpbasedCollOp_CollServiceDefaultImpl)
{
    CommunicatorImpl comm;
    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitStreamManager();
    CollServiceDefaultImpl collService{&comm};
    comm.collService = &collService;
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, LoadOpbasedCollOp_rankSize_1_test)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.rankSize = 1;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitStreamManager();
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    comm.SetCommStatus(CommStatus::COMM_READY);
    MOCKER(HrtMemAsyncCopy).stubs();

    // allreduce sendBuf和recvBuf地址相同
    {
        CollOpParams opParams;
        u32 buffer = 10;
        opParams.sendBuf = static_cast<void*>(&buffer);
        opParams.recvBuf = static_cast<void*>(&buffer);
        opParams.count = 1;
        opParams.dataType = DataType::FP32;
        opParams.opType = OpType::ALLREDUCE;

        EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);
        std::string opTag = "";
        EXPECT_EQ(comm.LoadOffloadCollOp(opTag, opParams, nullptr), HcclResult::HCCL_SUCCESS);
    }

    // allreduce sendBuf和recvBuf地址不同
    {
        CollOpParams opParams;
        u32 sendBuffer = 10;
        opParams.sendBuf = static_cast<void*>(&sendBuffer);
        u32 recvBuffer = 20;
        opParams.recvBuf = static_cast<void*>(&recvBuffer);
        opParams.count = 1;
        opParams.dataType = DataType::FP32;
        opParams.opType = OpType::ALLREDUCE;

        EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);
    }

    // alltoall sendBuf和recvBuf地址不同
    {
        CollOpParams opParams;
        u32 sendBuffer = 10;
        opParams.sendBuf = static_cast<void*>(&sendBuffer);
        u32 recvBuffer = 20;
        opParams.recvBuf = static_cast<void*>(&recvBuffer);
        opParams.all2AllDataDes.sendCount = 1;
        opParams.all2AllDataDes.recvCount = 1;
        opParams.all2AllDataDes.sendType = DataType::FP32;
        opParams.all2AllDataDes.recvType = DataType::FP32;
        opParams.opType = OpType::ALLTOALL;

        EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);
    }

    // alltoallv sendBuf和recvBuf地址不同
    {
        CollOpParams opParams;
        u32 sendBuffer = 10;
        opParams.sendBuf = static_cast<void*>(&sendBuffer);
        u32 recvBuffer = 20;
        opParams.recvBuf = static_cast<void*>(&recvBuffer);
        u64 count = 1;
        opParams.all2AllVDataDes.sendCounts = static_cast<void*>(&count);
        opParams.all2AllVDataDes.recvCounts = static_cast<void*>(&count);
        opParams.all2AllVDataDes.sendType = DataType::FP32;
        opParams.all2AllVDataDes.recvType = DataType::FP32;
        opParams.opType = OpType::ALLTOALLV;

        EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_SUCCESS);
    }
}

// ok
TEST_F(CommunicatorImplTest, RecoverComm_NormalCase)
{
    // 打桩所有初始化函数，使其返回成功
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtOpenTsdProcess).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(1)));
    MOCKER(RaInit).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&CommunicatorImpl::RecoverRankGraphData).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitNotifyManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitStreamManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitSocketManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRmaConnManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitDataBufferManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitHostDeviceSyncNotifyManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::RecoverTransportData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    CommunicatorImpl comm;
    comm.RegisterAcceStateCallBack(CommunicatorCallback());
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    SnapShotComm snapShotComm;
    u32 step = 1;

    CommParams commParams("test_comm_id", 0, 4, 0, DevType::DEV_TYPE_950, false, true);
    HcclCommConfig config;
    strcpy(config.reserved, "test_reserved");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "test_comm_name");
    strcpy(config.hcclUdi, "test_udi");

    RankTableInfo ranktableInfo;
    ranktableInfo.version = "2.0";
    ranktableInfo.rankCount = 0;

    TopoInfo topoInfo;
    topoInfo.version = "2.0";
    topoInfo.peerCount = 0;
    topoInfo.edgeCount = 0;

    snapShotComm.rankTableInfo = ranktableInfo;
    snapShotComm.topoInfo = topoInfo;
    comm.myRank = 0;
    comm.rankGraph = make_unique<RankGraph>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);

    const char* filePath = "test";
    HcclResult result = comm.RecoverComm(snapShotComm, step, filePath);

    // 检查结果
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(comm.GetCommStatus(), CommStatus::COMM_RESUMING);
    EXPECT_TRUE(comm.initFlag);
}

// ok
TEST_F(CommunicatorImplTest, RecoverComm_StdException)
{
    fakeComm.initFlag = false;
    fakeComm.SetCommStatus(CommStatus::COMM_IDLE);
    SnapShotComm snapShotComm;
    u32 step = 0;
    const char* filePath = "test";
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    HcclResult result = fakeComm.RecoverComm(snapShotComm, step, filePath);
    EXPECT_NE(result, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.GetCommStatus(), CommStatus::COMM_IDLE);
}

// //OK
TEST_F(CommunicatorImplTest, RecoverSubComm_InitFlagTrue)
{
    CommunicatorImpl comm;
    comm.initFlag = true;
    SnapShotComm snapShotComm;
    u32 step = 0;

    const char* filePath = "test";
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    HcclResult result = comm.RecoverComm(snapShotComm, step, filePath);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
    EXPECT_TRUE(comm.initFlag);
}

TEST_F(CommunicatorImplTest, RecoverComm_SubCommNormalCase)
{
    // 打桩所有初始化函数，使其返回成功
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtOpenTsdProcess).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(1)));
    MOCKER(RaInit).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(std::unique_ptr<RankGraph>&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::RecoverRankGraphData).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitNotifyManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitStreamManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitSocketManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRmaConnManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitDataBufferManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitHostDeviceSyncNotifyManager).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::RecoverTransportData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    s32 myRank = 0;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(myRank);

    CommunicatorImpl comm;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    SnapShotSubComm snapShotComm;
    u32 step = 1;

    comm.rankGraph = make_unique<RankGraph>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);

    HcclResult result = comm.RecoverComm(snapShotComm, virtualTopo, step);

    // 检查结果
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(comm.GetCommStatus(), CommStatus::COMM_RESUMING);
    EXPECT_TRUE(comm.initFlag);
}

TEST_F(CommunicatorImplTest, RecoverComm_SubComStdException)
{
    s32 myRank = 0;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(myRank);

    SnapShotSubComm snapShotComm;
    u32 step = 1;

    fakeComm.initFlag = false;
    fakeComm.SetCommStatus(CommStatus::COMM_IDLE);
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    HcclResult result = fakeComm.RecoverComm(snapShotComm, virtualTopo, step);

    // 检查结果
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
    EXPECT_EQ(fakeComm.GetCommStatus(), CommStatus::COMM_IDLE);
    EXPECT_TRUE(fakeComm.initFlag);
}

// ok
TEST_F(CommunicatorImplTest, RecoverComm_SubComInitFlagTrue)
{
    CommunicatorImpl comm;
    comm.initFlag = true;
    SnapShotSubComm snapShotComm;
    s32 myRank = 0;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(myRank);
    u32 step = 0;

    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    HcclResult result = comm.RecoverComm(snapShotComm, virtualTopo, step);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(comm.GetCommStatus(), CommStatus::COMM_IDLE);
}

// ok
TEST_F(CommunicatorImplTest, RecoverSubComm_InitFlagFalse)
{
    unique_ptr<CommunicatorImpl> subCommImpl;
    CommunicatorImpl comm;
    SnapShotSubComm snapShotSubComm;
    snapShotSubComm.rankIds = {0, 1, 2};
    u32 step = 10;

    comm.initFlag = false;
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    HcclResult result = comm.RecoverSubComm(snapShotSubComm, subCommImpl.get(), step);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(comm.GetCommStatus(), CommStatus::COMM_IDLE);
}

TEST_F(CommunicatorImplTest, should_no_throw_exception_when_only_ccu_enabled)
{
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
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
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    DevBuffer dataBuffer(8);
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&DataBufManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(&dataBuffer));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
    void* ptr1 = (void*)0x123;
    MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr1));
    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));

    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.myRank = 0;
    comm.id = "testTag";
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    void* ptr;
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
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
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

    EXPECT_NO_THROW(comm.InitCollService());
}

TEST_F(CommunicatorImplTest, RecoverRankGraphData_ShouldReturnSuccess_WhenInputInValid)
{
    SnapShotComm snapShotComm;
    CommunicatorImpl commImpl;
    const char* filePath = "test_legacy";
    EXPECT_THROW(commImpl.RecoverRankGraphData(snapShotComm, filePath), InternalException);
}

TEST_F(CommunicatorImplTest, should_throw_exception_when_mirrorTaskManager_is_nullptr)
{
    CommunicatorImpl comm;
    EXPECT_THROW(comm.GetMirrorTaskManager(), NullPtrException);
}

TEST_F(CommunicatorImplTest, should_fail_when_comm_status_error)
{
    CommunicatorImpl comm;
    comm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param = {};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    comm.InitStreamManager();
    auto res = comm.LoadOpbasedCollOp(param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
}

void CommConfigCondition(CommunicatorImpl& comm, CollOpParams& param)
{
    comm.SetCommStatus(CommStatus::COMM_READY);
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::UINT8;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS; // aicpu 展开
}

TEST_F(CommunicatorImplTest, Ut_LoadOpbasedCollOp_When_Datatype_Not_Support_Expect_Return_HCCL_E_PARA)
{
    // 前置条件
    CommunicatorImpl comm;
    CollOpParams param = {};
    CollServiceDeviceMode collService{&comm};
    CommConfigCondition(comm, param);
    comm.collService = &collService;
    // 执行步骤
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(ignoreReturnValue());
    comm.InitStreamManager();
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    auto res = comm.LoadOpbasedCollOp(param, nullptr);
    // 后置验证
    EXPECT_EQ(res, HcclResult::HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_LoadOffloadCollOp_When_Datatype_Not_Support_Expect_Return_HCCL_E_PARA)
{
    // 前置条件
    CommunicatorImpl comm;
    CollOpParams param = {};
    CollServiceDeviceMode collService{&comm};
    std::string opTag = "";
    CommConfigCondition(comm, param);
    comm.collService = &collService;
    // 执行步骤
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::UpdateProfStat).stubs().will(ignoreReturnValue());
    auto res = comm.LoadOffloadCollOp(opTag, param, nullptr);
    // 后置验证
    EXPECT_EQ(res, HcclResult::HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, should_fail_when_comm_status_error2)
{
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(InternalException("")));
    fakeComm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param = {};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOpbasedCollOp(param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_fail_when_comm_status_error3)
{
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(1));
    fakeComm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param = {};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOpbasedCollOp(param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_fail_when_comm_status_error4)
{
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(InternalException("")));
    fakeComm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param = {};
    std::string opTag = "";
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOffloadCollOp(opTag, param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_fail_when_comm_status_error5)
{
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(1));
    fakeComm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param = {};
    std::string opTag = "";
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOffloadCollOp(opTag, param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_fail_when_comm_status_error6)
{
    fakeComm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param = {};
    std::string opTag = "";
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOffloadCollOp(opTag, param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, should_trace_success_when_comm_params_valid)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.rankSize = 2;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitStreamManager();
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.trace = std::make_unique<Trace>();
    MOCKER(HrtMemAsyncCopy).stubs();

    // 执行步骤
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::UpdateProfStat).stubs().will(ignoreReturnValue());
    MOCKER_CPP_VIRTUAL(comm.GetCollService(), &CollServiceBase::LoadWithOffloadMode).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).stubs().will(ignoreReturnValue());

    // allreduce sendBuf和recvBuf地址相同
    {
        CollOpParams opParams;
        u32 buffer = 10;
        opParams.sendBuf = static_cast<void*>(&buffer);
        opParams.recvBuf = static_cast<void*>(&buffer);
        opParams.count = 2;
        opParams.dataType = DataType::FP32;
        opParams.opType = OpType::ALLREDUCE;

        std::string opTag = "";
        EXPECT_EQ(comm.LoadOffloadCollOp(opTag, opParams, nullptr), HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(CommunicatorImplTest, init_and_get_one_sided_service)
{
    CommunicatorImpl comm;
    comm.InitOneSidedService();
    HcclOneSidedService* service;
    comm.GetOneSidedService(&service);
}

TEST_F(CommunicatorImplTest, ut_GetUsedChannelCount)
{
    CommunicatorImpl comm;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    MOCKER_CPP(&CcuJettyMgr::GetUsedChannelCount).stubs().will(returnValue(1));
    EXPECT_EQ(comm.GetUsedChannelCount(0), 1);
    GlobalMockObject::verify();

    CcuJettyMgr* utCcuJettyMgr = nullptr;
    MOCKER_CPP(&CcuCommunicator::GetCcuJettyMgr).stubs().will(returnValue(utCcuJettyMgr));
    EXPECT_EQ(comm.GetUsedChannelCount(0), 0);
    GlobalMockObject::verify();
}

TEST_F(CommunicatorImplTest, ut_PrintChannelInfoCallback)
{
    CommunicatorImpl comm;
    comm.printChannelInfoCallback = nullptr;
    comm.PrintChannelInfoCallback();

    comm.printChannelInfoCallback = []() {};
    comm.PrintChannelInfoCallback();
}

TEST_F(CommunicatorImplTest, ut_GetJsonProperty_1)
{
    nlohmann::json j;
    char* propName = nullptr;
    EXPECT_THROW(GetJsonProperty(j, propName), NullPtrException);
}

TEST_F(CommunicatorImplTest, should_fail_when_AllocCommResource_not_ccu)
{
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    void* commContext = nullptr;
    u32 sendBuffer = 10;
    void* tilingData = static_cast<void*>(&sendBuffer);

    EXPECT_EQ(comm.AllocCommResource(tilingData, &commContext), HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, should_success_when_AllocCommResource_ccu)
{
    LocalRmaBuffer* rmaBuf = nullptr;

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
    auto peer1 = std::make_shared<NetInstance::Peer>(1, 1, 1, 0);
    IpAddress inputAddr(0);
    std::set<string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};

    shared_ptr<NetInstance::ConnInterface> sourceIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::ConnInterface> targetIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    shared_ptr<NetInstance::Link> link
        = std::make_shared<NetInstance::Link>(peer0, peer1, sourceIface, targetIface, LinkType::PEER2PEER, protocols);
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

#define FUSION_SUB_TASK_MAX_CPU_NUM (1U)
    typedef struct rtHostInputInfo {
        uint32_t addrOffset;
        uint32_t dataOffset;
    } rtHostInputInfo_t;

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
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());
    EXPECT_NO_THROW(comm.AllocCommResource(tilingData, &commContext));

    tilingData->commConfig.communicationEngine = 0;
    EXPECT_NO_THROW(comm.AllocCommResource(tilingData, &commContext));

    tilingData->commConfig.communicationEngine = 3;
    EXPECT_NO_THROW(comm.AllocCommResource(tilingData, &commContext));

    tilingData->commConfig.communicationEngine = 5;
    EXPECT_NO_THROW(comm.AllocCommResource(tilingData, &commContext));

    tilingData->commConfig.communicationEngine = 6;
    EXPECT_NO_THROW(comm.AllocCommResource(tilingData, &commContext));

    tilingData->commConfig.communicationEngine = 7;
    EXPECT_EQ(comm.AllocCommResource(tilingData, &commContext), HCCL_E_NOT_SUPPORT);

    free(fusionArgs.args);
    delete[] buf;
}

TEST_F(CommunicatorImplTest, should_fail_when_GetCcuTaskInfo_not_ccu)
{
    CommunicatorImpl comm;
    comm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    void* fusionArgs;
    rtCcuTaskGroup_t ccuTaskGroup;
    EXPECT_EQ(comm.GetCcuTaskInfo(fusionArgs, &ccuTaskGroup), HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, should_throw_exception_when_not_support_service)
{
    CommunicatorImpl comm;
    comm.collServices[AcceleratorState::CCU_MS] = std::make_unique<CollServiceDeviceMode>(&comm); // host 展开，ccu使用
    comm.collServices[AcceleratorState::CCU_SCHED]
        = std::make_unique<CollServiceDeviceMode>(&comm); // host 展开，ccu使用
    comm.collServices[AcceleratorState::AICPU_TS] = std::make_unique<CollServiceAiCpuImpl>(&comm); // aicpu 展开
    comm.collServices[AcceleratorState::HOSTCPU_TS]
        = std::make_unique<CollServiceDefaultImpl>(&comm); // host 展开，图模式使用

    comm.opExecuteConfig.accState = AcceleratorState::AICPU;
    EXPECT_THROW(comm.SelectCollService(), NotSupportException);
}

TEST_F(CommunicatorImplTest, should_throw_exception_SetAccelerator_when_isLoadOp_is_true)
{
    CommunicatorImpl comm;
    comm.isLoadOp = true;
    HcclAccelerator accelerator{HcclAccelerator::DEFAULT};
    bool isCcuMsAvailable = false;
    EXPECT_EQ(comm.SetAccelerator(accelerator, isCcuMsAvailable), HCCL_E_NOT_SUPPORT);
}

HcclResult HrtGetMainboardIdStub(uint32_t deviceLogicId, HcclMainboardId& hcclMainboardId)
{
    hcclMainboardId = HcclMainboardId::MAINBOARD_PCIE_STD;
    return HCCL_SUCCESS;
}
TEST_F(CommunicatorImplTest, Ut_SetAccelerator_When_CcumsAndPciestd_Return_HCCL_E_NOT_SUPPORT)
{
    // 前置条件
    CommunicatorImpl comm;
    MOCKER(HrtGetMainboardId).stubs().will(invoke(HrtGetMainboardIdStub));

    HcclAccelerator accelerator{HcclAccelerator::CCU_MS};
    bool isCcuMsAvailable = true;

    // 后置验证
    EXPECT_EQ(comm.SetAccelerator(accelerator, isCcuMsAvailable), HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, ut_RefreshSubmittedOpcnt_1)
{
    CommunicatorImpl comm;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opType = OpType::ALLTOALLV;
    EXPECT_NO_THROW(comm.RefreshSubmittedOpcnt());
}

TEST_F(CommunicatorImplTest, Ut_GetCcuMc2ServerNum_When_not_find_CCU_MS_Expect_InternalException)
{
    // 前置条件
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    // 后置验证
    EXPECT_THROW(comm.GetCcuMc2ServerNum(), InternalException);
}

TEST_F(CommunicatorImplTest, Ut_GetCcuMc2ServerNum_When_find_CCU_MS_Expect_no_throw)
{
    // 前置条件
    CommunicatorImpl* comm = new CommunicatorImpl();
    comm->collServices.emplace(AcceleratorState::CCU_MS, std::make_unique<CollServiceDeviceMode>(comm));
    comm->collServices.emplace(AcceleratorState::CCU_SCHED, std::make_unique<CollServiceDeviceMode>(comm));
    comm->opExecuteConfig.accState = AcceleratorState::CCU_MS;

    // 后置验证
    EXPECT_NO_THROW(comm->GetCcuMc2ServerNum());

    delete comm;
}

TEST_F(CommunicatorImplTest, Ut_GetCcuMc2ServerNum_When_find_CCU_SCHED_Expect_no_throw)
{
    // 前置条件
    CommunicatorImpl* comm = new CommunicatorImpl();
    comm->collServices.emplace(AcceleratorState::CCU_MS, std::make_unique<CollServiceDeviceMode>(comm));
    comm->collServices.emplace(AcceleratorState::CCU_SCHED, std::make_unique<CollServiceDeviceMode>(comm));
    comm->opExecuteConfig.accState = AcceleratorState::CCU_SCHED;

    // 后置验证
    EXPECT_NO_THROW(comm->GetCcuMc2ServerNum());

    delete comm;
}

TEST_F(CommunicatorImplTest, Ut_GetCcuMc2ServerNum_Expect_equality)
{
    // 前置条件
    CommunicatorImpl* comm = new CommunicatorImpl();
    comm->collServices.emplace(AcceleratorState::CCU_MS, std::make_unique<CollServiceDeviceMode>(comm));
    comm->collServices.emplace(AcceleratorState::CCU_SCHED, std::make_unique<CollServiceDeviceMode>(comm));
    comm->opExecuteConfig.accState = AcceleratorState::CCU_SCHED;
    u32 ccuMc2ServerNum = 0;

    // 执行步骤
    auto ret = comm->GetCcuMc2ServerNum();

    // 后置验证
    EXPECT_EQ(ret, ccuMc2ServerNum);

    delete comm;
}
TEST_F(CommunicatorImplTest, ut_should_success_when_GetSnapShotDynamicBuf)
{
    // 前置条件
    CommunicatorImpl fakeComm;
    fakeComm.CollAlgComponentInit();
    fakeComm.currentCollOperator = std::make_unique<CollOperator>();
    fakeComm.currentCollOperator->opMode = OpMode::OPBASE;
    fakeComm.submittedOpCnt = 1;
    fakeComm.commExecuteConfig.accState = AcceleratorState::CCU_MS;
    fakeComm.opExecuteConfig.accState = AcceleratorState::CCU_SCHED;
    fakeComm.isLoadOp = true;

    CollServiceDeviceMode collService{&fakeComm};
    fakeComm.collService = &collService;

    LinkInfo linkInfo{1, 0, IpAddress{"10.0.0.1"}, IpAddress{"10.0.0.2"}};
    LinkGroup utLinkGroup{vector<LinkInfo>{linkInfo}};
    vector<LinkGroup> utLinkGroups{utLinkGroup};
    MOCKER_CPP(&CcuTransportGroupMgr::GetAllTransportGroups).stubs().with().will(returnValue(utLinkGroups));

    // 执行步骤
    BinaryStream buf;
    auto ret = fakeComm.GetSnapShotDynamicBuf(buf);

    // 后置验证
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // Comm 的 GetSnapShotDynamicBuf
    u32 rOpAccState{0};
    u32 rCommAccState{0};
    bool rIsLoadOp{false};
    buf >> rOpAccState >> rCommAccState >> rIsLoadOp;
    EXPECT_EQ(static_cast<AcceleratorState::Value>(rOpAccState), fakeComm.opExecuteConfig.accState);
    EXPECT_EQ(static_cast<AcceleratorState::Value>(rCommAccState), fakeComm.commExecuteConfig.accState);
    EXPECT_EQ(rIsLoadOp, fakeComm.isLoadOp);

    // Comm 的 GetSnapShotDynamicBuf
    u32 rSubmittedOpCnt{0};
    buf >> rSubmittedOpCnt;
    EXPECT_EQ(rSubmittedOpCnt, fakeComm.submittedOpCnt);

    u32 rOpMode{0};
    buf >> rOpMode;
    EXPECT_EQ(static_cast<OpMode::Value>(rOpMode), fakeComm.currentCollOperator->opMode);

    // CollServiceDeviceMode 的 GetSnapShotDynamicBuf
    size_t rLevelRankPairs{0};
    size_t rLinkGroupPairsSize{0};
    size_t rLinkSize{0};
    buf >> rLevelRankPairs >> rLinkGroupPairsSize >> rLinkSize;
    EXPECT_EQ(rLevelRankPairs, 0);
    EXPECT_EQ(rLinkGroupPairsSize, 1);
    EXPECT_EQ(rLinkSize, 1);

    RankId rRank{0};
    u32 rDieId{0};
    buf >> rRank >> rDieId;

    IpAddress rlocalAddr{buf};
    IpAddress rRemoteAddr{buf};

    u32 rCntCkeNum{0};
    buf >> rCntCkeNum;
    EXPECT_EQ(rRank, linkInfo.rankId);
    EXPECT_EQ(rDieId, linkInfo.dieId);
    EXPECT_EQ(rlocalAddr, linkInfo.localAddr);
    EXPECT_EQ(rRemoteAddr, linkInfo.remoteAddr);
    EXPECT_EQ(rCntCkeNum, TEMP_UES_CNTCKE_NUM); // TEMP_UES_CNTCKE_NUM = 16
}

TEST_F(CommunicatorImplTest, ut_CalcTaskNum_When_Abnormal_Expect_Return_HCCL_E_INTERNAL)
{
    MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(commGetenv_stub));
    CommunicatorImpl comm;
    comm.CollAlgComponentInit();
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.collAlgComponent->rankSize_ = 0;
    MirrorTaskManager& mirrorTaskManager = comm.GetMirrorTaskManager();

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;

    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    u32 taskNum = 0;

    EXPECT_EQ(comm.CalcTaskNum(opParams.opType, opParams.dataType, 1, taskNum), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, ut_CreateCommCclBuf_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(commGetenv_stub));
    CommunicatorImpl comm;
    comm.CollAlgComponentInit();
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.devLogicId = 0;
    comm.config.hcclBufferSize = 0;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitDataBufferManager();
    MirrorTaskManager& mirrorTaskManager = comm.GetMirrorTaskManager();

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;

    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    comm.CreateCommCclBuf();
    EXPECT_EQ(comm.CreateCommCclBuf(), HcclResult::HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_CovertToCurrentCollOperator_When_AllGatherV)
{
    CommunicatorImpl comm;
    comm.rankSize = 2;

    HcclSendRecvItem sendRecvInfo;
    sendRecvInfo.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    CollOpParams collOpParams;
    collOpParams.opType = OpType::ALLGATHERV;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.dstRank = 1;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    u64 recvCounts[2] = {1, 1};
    u64 recvDispls[2] = {1, 1};
    collOpParams.vDataDes.counts = (&recvCounts);
    collOpParams.vDataDes.displs = (&recvCounts);
    collOpParams.vDataDes.dataType = DataType::INT8;

    uint64_t a = 10;
    uintptr_t devAddr = reinterpret_cast<uintptr_t>(&a);
    std::size_t devSize = 2;
    comm.cclBuffer = make_shared<DevBuffer>(10);
    string tag = "optag";
    comm.CovertToCurrentCollOperator(tag, collOpParams, OpMode::OPBASE);
}

TEST_F(CommunicatorImplTest, Ut_CovertToCurrentCollOperator_When_ReduceScatterV)
{
    CommunicatorImpl comm;
    comm.rankSize = 2;

    HcclSendRecvItem sendRecvInfo;
    sendRecvInfo.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    CollOpParams collOpParams;
    collOpParams.opType = OpType::REDUCESCATTERV;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.dstRank = 1;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    u64 sendCounts[2] = {1, 1};
    u64 sendDispls[2] = {1, 1};
    collOpParams.vDataDes.counts = (&sendCounts);
    collOpParams.vDataDes.displs = (&sendDispls);
    collOpParams.vDataDes.dataType = DataType::INT8;

    uint64_t a = 10;
    uintptr_t devAddr = reinterpret_cast<uintptr_t>(&a);
    std::size_t devSize = 2;
    comm.cclBuffer = make_shared<DevBuffer>(10);
    string tag = "optag";
    comm.CovertToCurrentCollOperator(tag, collOpParams, OpMode::OPBASE);
}

namespace {
void getInsQueue(InsQuePtr& insQueue)
{
    // ====== 配置用例基本信息 ======
    using CurrentExecutorType = InsV2AllReduceSoleExecutor<TopoMatchMesh, CcuTempAllReduceMeshMem2Mem1D>;
    using CurrentCcuInstructionType = CcuInstructionAllReduceMeshMem2Mem1D;
    using CuurentCcuContextType = CcuContextAllReduceMeshMem2Mem1D;
    auto myRank_ = 0;
    auto rankSize_ = 4;
    auto dimSize_ = {4};
    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);

    auto dataType_ = DataType::INT16;
    auto opType_ = OpType::ALLREDUCE;
    auto dataCount_ = 536870912;
    auto reduceOp_ = ReduceOp::MIN;
    auto deviceType_ = DevType::DEV_TYPE_950;

    // ====== 构造算子 ======
    std::unique_ptr<CurrentExecutorType> algoExecutor(new CurrentExecutorType());
    algoExecutor->SetMyRank(myRank_);
    algoExecutor->SetRankSize(rankSize_);
    algoExecutor->EnableDataAllign(false);
    algoExecutor->EnableDetour(false);
    algoExecutor->SetDevType(deviceType_);

    CollAlgOperator collAlgOp;
    collAlgOp.opType = opType_;
    collAlgOp.dataType = dataType_;
    collAlgOp.dataCount = dataCount_;
    collAlgOp.reduceOp = reduceOp_;
    u64 dataSize = dataCount_ * DataTypeSizeGet(dataType_);
    collAlgOp.inputMem = DevBuffer::Create(0x1000000, dataSize);
    collAlgOp.outputMem = DevBuffer::Create(0x2000000, dataSize);
    collAlgOp.scratchMem = DevBuffer::Create(0x3000000, dataSize);

    // ====== 单算子模式资源计算 ====== //
    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OPBASE;
    collAlgParams.maxTmpMemSize = 1024 * 1024 * 1024; // 1G

    CollAlgResReq algResReq;
    auto ret = algoExecutor->CalcRes(&virtTopo, algResReq);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS); // check return

    algoExecutor->vTopo_.clear();
    algoExecutor->virtRankMap_.clear();
    algoExecutor->virtRanks_.clear();

    // ====== HOST Orchestrate ====== //
    EXPECT_EQ(
        algoExecutor->Orchestrate(&virtTopo, collAlgOp, collAlgParams, insQueue),
        HcclResult::HCCL_SUCCESS); // check return
}

HcclResult Orchestrate(
    CollAlgComponent* This, const CollAlgOperator& op, const CollAlgParams& params, const string& algName,
    InsQuePtr queue)
{
    getInsQueue(queue);
    return HCCL_SUCCESS;
}

HcclResult GetProfilingInfoStub(
    s32 deviceLogicId, CcuTaskArg& ccuTaskArg, const uint64_t executorId,
    std::vector<std::vector<CcuProfilingInfo>>& ccuProfilingInfo)
{
    static int step = 0;
    std::vector<CcuProfilingInfo> profilingInfo;
    CcuProfilingInfo sqeProfInfo;
    sqeProfInfo.type = 0;
    sqeProfInfo.name = "AA::BB";
    profilingInfo.push_back(sqeProfInfo);

    CcuProfilingInfo localWaitProfInfo;
    localWaitProfInfo.type = 1;
    localWaitProfInfo.name = "LocalWait";
    profilingInfo.push_back(localWaitProfInfo);

    CcuProfilingInfo remoteWaitProfInfo;
    remoteWaitProfInfo.type = 1;
    remoteWaitProfInfo.name = "RemoteWait";
    profilingInfo.push_back(remoteWaitProfInfo);

    CcuProfilingInfo groupWaitProfInfo;
    groupWaitProfInfo.type = 1;
    groupWaitProfInfo.name = "GroupWait";
    profilingInfo.push_back(groupWaitProfInfo);

    CcuProfilingInfo groupReduceProfInfo;
    groupReduceProfInfo.type = 2;
    groupReduceProfInfo.name = "GroupReduce";
    profilingInfo.push_back(groupReduceProfInfo);

    CcuProfilingInfo gbProfInfo;
    gbProfInfo.type = 2;
    gbProfInfo.name = "GroupBroadcast";
    (void)memset_s(gbProfInfo.channelId, sizeof(gbProfInfo.channelId), 0x12, sizeof(gbProfInfo.channelId));
    profilingInfo.push_back(gbProfInfo);
    if (step == 0) {
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ++step;
    } else if (step == 1) {
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ++step;
    } else if (step == 2) {
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ++step;
    } else if (step == 3) {
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ++step;
    } else if (step == 4) {
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ++step;
    } else if (step == 5) {
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        ccuProfilingInfo.push_back(profilingInfo);
        step = 0;
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace

TEST_F(CommunicatorImplTest, Ut_CommunicatorImpl_When_EnableSuperFastLoad_Expect_LoadOpbasedCollOp_ReturnIsHCCL_SUCCESS)
{
    CallSingletons();
    MOCKER(RaTlvRequest).stubs().will(returnValue(0));
    void* mockTlvHandle = reinterpret_cast<void*>(0x1234);
    MOCKER_CPP(&HccpTlvHdcManager::GetTlvHandle).stubs().will(returnValue(mockTlvHandle));
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(rtCCULaunch).stubs().will(returnValue(RT_ERROR_NONE));
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).stubs();
    MOCKER(CcuCtxMgr::GetProfilingInfo).stubs().will(invoke(GetProfilingInfoStub));
    MOCKER_CPP(&Hccl::CcuJettyMgr::GetRemoteRankIdByChannelId).stubs().with(mockcpp::any()).will(returnValue(0x23));
    void* fakePtr = (void*)1;
    u32 fakeId = 1;
    s32 fakeDevLogId = 1;
    u32 fakeDevPhyId = 1;
    u32 fakeSqId = 2;
    u64 fakeStmMode = 3;
    SETUP_STREAM_MOCKS();
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());

    MOCKER_CPP(static_cast<void (MirrorTaskManager::*)(std::unique_ptr<TaskInfo>&&)>(&MirrorTaskManager::AddTaskInfo))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .will(invoke(Orchestrate));
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&HcclCommunicator::RegistTaskAbortHandler).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&HcclCommunicator::UnRegistTaskAbortHandler).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    CommParams commInnerParams;
    Hccl::HcclCommunicator commInner(commInnerParams);

    CommunicatorImpl& comm = *commInner.pimpl.get();
    comm.devLogicId = 0;
    comm.rankSize = 4;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    comm.InitStreamManager();
    comm.InitNotifyManager();
    comm.CollAlgComponentInit();
    comm.InitSocketManager();
    comm.notifyTimeoutCfg.notifyTimeout = 2333;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opType = OpType::ALLREDUCE;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    comm.SetCommStatus(CommStatus::COMM_READY);
    CollOpParams opParams{};
    u32 sendBuffer = 10;
    opParams.sendBuf = static_cast<void*>(&sendBuffer);
    u32 recvBuffer = 20;
    opParams.recvBuf = static_cast<void*>(&recvBuffer);
    opParams.count = 536870912;
    opParams.dataType = Hccl::DataType::INT16;
    opParams.opType = OpType::ALLREDUCE;
    opParams.reduceOp = ReduceOp::MIN;
    uint64_t tokenValue = 111;
    CcuTaskParam ccuTaskParam{};
    ccuTaskParam.dieId = 1;
    ccuTaskParam.missionId = 1;
    ccuTaskParam.instStartId = 14;
    ccuTaskParam.instCnt = 126;
    ccuTaskParam.key = 527697854;
    ccuTaskParam.argSize;
    ccuTaskParam.argSize = 13;
    ccuTaskParam.args[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(const_cast<void*>(opParams.sendBuf)));
    ccuTaskParam.args[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(const_cast<void*>(opParams.recvBuf)));
    ccuTaskParam.args[2] = tokenValue;
    ccuTaskParam.args[3] = 4096;
    ccuTaskParam.args[4] = 0;
    ccuTaskParam.args[5] = 0;
    ccuTaskParam.args[6] = 2199023255552;
    ccuTaskParam.args[7] = 4096;
    ccuTaskParam.args[8] = 0;
    ccuTaskParam.args[9] = 0;
    ccuTaskParam.args[10] = 0;
    ccuTaskParam.args[11] = 0;
    ccuTaskParam.args[12] = 0;

    comm.ccuParamsMappingKey
        = {static_cast<std::uint32_t>(opParams.reduceOp), static_cast<std::uint32_t>(opParams.dataType),
           static_cast<std::uint32_t>(opParams.count + 1)};
    std::vector<std::vector<CcuTaskParam>> ccuParams1{};
    std::vector<std::vector<CcuProfilingInfo>> ccuProfilingInfo1{};
    ccuParams1.push_back({ccuTaskParam, ccuTaskParam});
    ccuParams1.push_back({ccuTaskParam});
    ccuProfilingInfo1.resize(2);
    comm.saveCCUParams(std::move(ccuParams1), std::move(ccuProfilingInfo1), 0, CcuInstType::CCU_INS_GROUP, true);

    comm.ccuParamsMappingKey
        = {static_cast<std::uint32_t>(opParams.reduceOp), static_cast<std::uint32_t>(opParams.dataType),
           static_cast<std::uint32_t>(opParams.count + 1)};
    std::vector<std::vector<CcuTaskParam>> ccuParams2{};
    ccuParams2.push_back({ccuTaskParam});
    ccuParams2.push_back({ccuTaskParam, ccuTaskParam});
    ccuParams2.push_back({ccuTaskParam, ccuTaskParam, ccuTaskParam});
    std::vector<std::vector<CcuProfilingInfo>> ccuProfilingInfo2{};
    ccuProfilingInfo2.resize(3);
    comm.saveCCUParams(std::move(ccuParams2), std::move(ccuProfilingInfo2), 0, CcuInstType::CCU_INS_GROUP, true);
    comm.saveCCUParams(std::move(ccuParams2), std::move(ccuProfilingInfo2), 0, CcuInstType::CCU_INS_GROUP, true);

    comm.ccuParamsMappingKey
        = {static_cast<std::uint32_t>(opParams.reduceOp), static_cast<std::uint32_t>(opParams.dataType),
           static_cast<std::uint32_t>(opParams.count)};

    std::vector<std::vector<CcuTaskParam>> ccuParams{};
    ccuParams.push_back({ccuTaskParam});
    ccuParams.push_back({ccuTaskParam, ccuTaskParam});
    ccuParams.push_back({ccuTaskParam, ccuTaskParam, ccuTaskParam});
    std::vector<std::vector<CcuProfilingInfo>> ccuProfilingInfo3{};
    ccuProfilingInfo3.resize(3);
    ccuProfilingInfo3[0].resize(1);
    ccuProfilingInfo3[1].resize(2);
    ccuProfilingInfo3[2].resize(3);
    comm.saveCCUParams(std::move(ccuParams), std::move(ccuProfilingInfo3), 0, CcuInstType::CCU_INS_GROUP, true);

    Stream stream(fakePtr);
    stream.SetStmMode(fakeStmMode);
    auto streamUnique = std::make_unique<Stream>(stream.GetPtr());
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    comm.streamManager->opbase->RegisterMaster(std::move(streamUnique));
    for (int i = 0; i < 10; ++i) {
        comm.streamManager->opbase->GetOrCreateSlave();
    }
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);

    comm.superFasterLoad = true;
    comm.taskExceptionEnv = true;
    comm.enableProfilingEnv = true;
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.cclBufferSize = 0x100;
    comm.collServices.emplace(AcceleratorState::CCU_MS, std::make_unique<CollServiceDeviceMode>(&comm));
    comm.collServices.emplace(AcceleratorState::CCU_SCHED, std::make_unique<CollServiceDeviceMode>(&comm));
    comm.opExecuteConfig.accState = AcceleratorState::CCU_SCHED;
    comm.collService = comm.collServices[AcceleratorState::CCU_SCHED].get();

    for (int i = 0; i < 3; ++i) {
        u32 sendBuffer = i;
        opParams.sendBuf = static_cast<void*>(&sendBuffer);
        u32 recvBuffer = i;
        opParams.recvBuf = static_cast<void*>(&recvBuffer);
        CachedCCUParams& sendCcuParams = comm.colCcuParamMapping[opParams.opType][{
            static_cast<std::uint32_t>(opParams.reduceOp), static_cast<std::uint32_t>(opParams.dataType),
            static_cast<std::uint32_t>(opParams.count)}];
        EXPECT_EQ(
            HcclAllReduceV2(
                opParams.sendBuf, opParams.recvBuf, opParams.count, HcclDataType::HCCL_DATA_TYPE_INT16,
                HcclReduceOp::HCCL_REDUCE_MIN, static_cast<void*>(&commInner), fakePtr),
            HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(sendCcuParams.ccuParams[0].dieId, 1);
        EXPECT_EQ(sendCcuParams.ccuParams[0].missionId, 1);
        EXPECT_EQ(sendCcuParams.ccuParams[0].timeout, comm.notifyTimeoutCfg.notifyTimeout);
        EXPECT_EQ(sendCcuParams.ccuParams[0].instStartId, 14);
        EXPECT_EQ(sendCcuParams.ccuParams[0].instCnt, 126);
        EXPECT_EQ(sendCcuParams.ccuParams[0].key, 527697854);
        EXPECT_EQ(sendCcuParams.ccuParams[0].argSize, 13);
        EXPECT_EQ(
            sendCcuParams.ccuParams[0].args[0],
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(const_cast<void*>(opParams.sendBuf))));
        EXPECT_EQ(
            sendCcuParams.ccuParams[0].args[1],
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(const_cast<void*>(opParams.recvBuf))));
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[2], tokenValue);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[3], 4096);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[4], 0);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[5], 0);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[6], 2199023255552);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[7], 4096);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[8], 0);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[9], 0);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[10], 0);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[11], 0);
        EXPECT_EQ(sendCcuParams.ccuParams[0].args[12], 0);
    }

    comm.superFasterLoad = false;
    comm.taskExceptionEnv = true;
    comm.enableProfilingEnv = true;
}

TEST_F(CommunicatorImplTest, Ut_LoadOpbasedCollOp_When_Alg_Is_Not_Support_Then_Throw_Exception)
{
    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHERV;
    opParams.reduceOp = ReduceOp::SUM;

    fakeComm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;

    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect).stubs().will(returnValue(HCCL_E_NOT_SUPPORT));
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, nullptr), HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, ut_OpAcceleratorStateFallback_When_OtherAccelerator_Expect_ThrowNotSupportException)
{
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    EXPECT_THROW(comm.OpAcceleratorStateFallback(), NotSupportException);
}

TEST_F(CommunicatorImplTest, ut_OpAcceleratorStateFallback_When_CCU_SCHED_Expect_NoThrow)
{
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_SCHED;
    EXPECT_NO_THROW(comm.OpAcceleratorStateFallback());
}

TEST_F(CommunicatorImplTest, ut_ReLoadOpbasedOp_When_HOSTCPU_TS_Expect_returnHCCL_E_NOT_SUPPORT)
{
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    fakeComm.opExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
    EXPECT_EQ(fakeComm.ReLoadOpbasedOp(), HcclResult::HCCL_E_NOT_SUPPORT); // CollServiceDefaultImpl NOT_SUPPORT

    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    EXPECT_EQ(fakeComm.ReLoadOpbasedOp(), HcclResult::HCCL_E_PARA); // CheckOpDataTypeOpbase HCCL_E_PARA

    fakeComm.curOpParams = opParams;
    fakeComm.currentCollOperator = nullptr;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
    EXPECT_EQ(fakeComm.ReLoadOpbasedOp(), HcclResult::HCCL_E_PTR); // currentCollOperator == nullptr
}

TEST_F(CommunicatorImplTest, ut_ReLoadOpbasedOp_When_AICPU_TS_Expect_returnHCCL_SUCCESS)
{
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;
    fakeComm.curOpParams = opParams;

    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    MOCKER_CPP(&CollServiceAiCpuImpl::LoadWithOpBasedModeNoRegister)
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    EXPECT_EQ(fakeComm.ReLoadOpbasedOp(), HcclResult::HCCL_SUCCESS);

    fakeComm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    EXPECT_THROW(fakeComm.ReLoadOpbasedOp(), NotSupportException); // CCU::ReLoadWithOffloadMode NotSupport
}

TEST_F(LoadOffloadCollOpTest, Ut_LoadOffloadCollOp_When_HOSTCPU_TS_Expect_returnHCCL_E_NOT_SUPPORT)
{
    // 设置条件
    std::string opTag = "test";
    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    void* stream = nullptr;

    // 执行测试
    HcclResult result = fakeComm.LoadOffloadCollOp(opTag, opParams, stream);

    // 验证结果
    EXPECT_EQ(result, HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(LoadOffloadCollOpTest, Ut_ReLoadOffloadOp_When_HOSTCPU_TS_Expect_returnHCCL_E_NOT_SUPPORT)
{
    // 执行测试
    HcclResult result = fakeComm.ReLoadOffloadOp();

    // 验证结果
    EXPECT_EQ(result, HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, ut_ReLoadOffloadOp_When_HOSTCPU_TS_Expect_returnHCCL_E_NOT_SUPPORT)
{
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    EXPECT_EQ(fakeComm.ReLoadOffloadOp(), HcclResult::HCCL_E_PARA); // CheckOpDataTypeOpbase HCCL_E_PARA

    fakeComm.curOpParams = opParams;
    fakeComm.currentCollOperator = nullptr;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
    EXPECT_EQ(fakeComm.ReLoadOffloadOp(), HcclResult::HCCL_E_PTR); // currentCollOperator == nullptr
}

TEST_F(CommunicatorImplTest, ut_ReLoadOffloadOp_When_AICPU_TS_Expect_returnHCCL_SUCCESS)
{
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;
    fakeComm.curOpParams = opParams;

    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    MOCKER_CPP(&CollServiceAiCpuImpl::LoadWithOffloadModeNoRegister)
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    EXPECT_EQ(fakeComm.ReLoadOffloadOp(), HcclResult::HCCL_SUCCESS);

    fakeComm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    EXPECT_THROW(fakeComm.ReLoadOffloadOp(), NotSupportException); // CCU::ReLoadWithOffloadMode NotSupport
}

TEST_F(CommunicatorImplTest, Ut_CommunicatorImpl_When_SingleRankProc_Expect_OK_ReturnIsHCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.rankSize = 1;
    CollOpParams opParams{};
    u32 sendBuffer = 1;
    opParams.sendBuf = static_cast<void*>(&sendBuffer);
    opParams.recvBuf = static_cast<void*>(&sendBuffer);
    opParams.count = 536870912;
    opParams.dataType = Hccl::DataType::INT16;
    opParams.opType = OpType::BATCHSENDRECV;
    opParams.reduceOp = ReduceOp::MIN;
    void* stream = static_cast<void*>(&sendBuffer);
    EXPECT_NO_THROW(comm.SingleRankProc(opParams, stream));
}

TEST_F(CommunicatorImplTest, Ut_LoadOffloadCollOp_When_dataTpye_fail_Expect_HCCL_E_PARA)
{
    // 前置条件
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.rankSize = 2;
    comm.isAicpuKernelLaunched = false;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    comm.SetCommStatus(CommStatus::COMM_READY);
    MOCKER(HrtMemAsyncCopy).stubs();
    CollOpParams opParams;
    u32 buffer = 10;
    opParams.sendBuf = static_cast<void*>(&buffer);
    opParams.recvBuf = static_cast<void*>(&buffer);
    opParams.count = 1;
    opParams.dataType = DataType::INT64;
    opParams.opType = OpType::ALLREDUCE;
    std::string opTag = "";

    // 执行步骤
    auto ret = comm.LoadOffloadCollOp(opTag, opParams, nullptr);

    // 后置验证
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_AppendLocalDieId_When_OneP_return)
{
    CommunicatorImpl comm;
    comm.rankSize = 1;

    EXPECT_NO_THROW(comm.AppendLocalDieIdForLinks());
}

TEST_F(CommunicatorImplTest, St_CheckAcceleratorConsistency)
{
    CommunicatorImpl comm;
    EXPECT_NO_THROW(comm.CheckAcceleratorConsistency(AcceleratorState::AIV, AcceleratorState::AIV));
}

TEST_F(CommunicatorImplTest, Ut_GetNetLayers_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);

    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum;
    auto ret = comm.GetNetLayers(&netLayers, &netLayerNum);
    EXPECT_EQ(netLayers[0], 0);
    EXPECT_EQ(netLayerNum, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetInstSizeByNetLayer_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);

    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t rankNum = 0;
    auto ret = comm.GetInstSizeByNetLayer(netLayer, &rankNum);
    EXPECT_EQ(rankNum, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetInstRanksByNetLayer_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);

    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* ranks = nullptr;
    uint32_t rankNum;
    auto ret = comm.GetInstRanksByNetLayer(netLayer, &ranks, &rankNum);
    EXPECT_EQ(ranks[0], 0);
    EXPECT_EQ(rankNum, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetInstRanksByNetLayer_When_InvalidLayer_Expect_Return_HCCL_E_PTR)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);

    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    netLayer = 3;
    uint32_t* ranks = nullptr;
    uint32_t rankNum;
    auto ret = comm.GetInstRanksByNetLayer(netLayer, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CommunicatorImplTest, Ut_GetInstTopoTypeByNetLayer_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);

    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t netType;
    auto ret = comm.GetInstTopoTypeByNetLayer(netLayer, &netType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(netType, COMM_TOPO_CUSTOM);
}

TEST_F(CommunicatorImplTest, Ut_GetInstTopoTypeByNetLayer_When_InvalidLayer_Expect_Return_HCCL_E_PTR)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    netLayer = 3;
    uint32_t netType;
    auto ret = comm.GetInstTopoTypeByNetLayer(netLayer, &netType);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CommunicatorImplTest, Ut_GetInstSizeListByNetLayer_When_InvalidLayer_Expect_ReturnHCCL_E_PARA)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    netLayer = 3;
    uint32_t* instSizeList = nullptr;
    uint32_t listSize;
    auto ret = comm.GetInstSizeListByNetLayer(netLayer, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_GetInstSizeListByNetLayer_When_InputValue_Expect_ReturnHCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);

    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* instSizeList = nullptr;
    uint32_t listSize;
    auto ret = comm.GetInstSizeListByNetLayer(netLayer, &instSizeList, &listSize);
    EXPECT_EQ(instSizeList[0], 1);
    EXPECT_EQ(listSize, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetLinks_When_netLayer02p_InputValue_Expect_Return_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2pclos.json"};
    comm.rankGraph = rankGraphBuilder.Build(RankTable2pClos, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    uint32_t srcRank = 0;
    uint32_t dstRank = 1;
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    auto ret = comm.GetLinks(0, srcRank, dstRank, &linkList, &listSize);
    EXPECT_EQ(listSize, 5);
    EXPECT_EQ(linkList[3].linkAttr.hop, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetLinks_When_netLayer064Plus1_InputValue_Expect_Return_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo64plus1.json"};
    comm.rankGraph = rankGraphBuilder.Build(RANK_TABLE_4P_REPLACE_RANK1, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    CommLink* linkList1 = nullptr;
    uint32_t listSize1 = 0;
    auto ret1 = comm.GetLinks(0, 2, 1, &linkList1, &listSize1); // 斜向 rank1为replace
    EXPECT_EQ(listSize1, 0);                                    // 无连接
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    auto ret2 = comm.GetLinks(0, 1, 3, &linkList, &listSize); // db到直连d
    EXPECT_EQ(listSize, 1);                                   // 只有一条peer2peer
    EXPECT_EQ(ret2, HCCL_SUCCESS);

    CommLink* linkListD2D1 = nullptr;
    uint32_t listSizeD2D1 = 0;
    auto ret3 = comm.GetLinks(0, 2, 3, &linkListD2D1, &listSizeD2D1); // db到直连d X/Y轴
    EXPECT_EQ(listSizeD2D1, 5);
    EXPECT_EQ(ret3, HCCL_SUCCESS);

    CommLink* linkListD2D2 = nullptr;
    uint32_t listSizeD2D2 = 0;
    auto ret4 = comm.GetLinks(0, 0, 3, &linkListD2D2, &listSizeD2D2); // db到直连d 斜向
    EXPECT_EQ(listSizeD2D2, 4);
    EXPECT_EQ(ret4, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetTopoInstsByLayer_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInsts_[topoInstId] = topoInstance;

    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());

    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* topoInsts = nullptr;
    uint32_t topoInsNum = 0;
    auto ret = comm.GetTopoInstsByLayer(netLayer, &topoInsts, &topoInsNum);

    EXPECT_EQ(topoInsts[0], 0);
    EXPECT_EQ(topoInsNum, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetTopoInstsByLayer_When_InVaildLayer_Expect_Return_HCCL_E_PTR)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInsts_[topoInstId] = topoInstance;

    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());

    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    netLayer = 3;
    uint32_t* topoInsts = nullptr;
    uint32_t topoInsNum = 0;
    auto ret = comm.GetTopoInstsByLayer(netLayer, &topoInsts, &topoInsNum);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CommunicatorImplTest, Ut_GetTopoInstsByLayer_When_ErrorNetType_Expect_Return_HCCL_E_PARA)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 1;
    string netInstId = "layer1";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    topoInstance->topoType = Hccl::TopoType::CLOS;
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInsts_[topoInstId] = topoInstance;
    auto netInstance = std::make_shared<ClosNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* topoInsts = nullptr;
    uint32_t topoInsNum = 0;
    auto ret = comm.GetTopoInstsByLayer(netLayer, &topoInsts, &topoInsNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_GetTopoType_When_ErrorNetType_Expect_Return_HCCL_E_PARA)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 1;
    string netInstId = "layer1";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    topoInstance->topoType = Hccl::TopoType::CLOS;
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInsts_[topoInstId] = topoInstance;
    auto netInstance = std::make_shared<ClosNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    CommTopo topoType;
    auto ret = comm.GetTopoType(netLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_GetTopoType_When_InvalidLayer_Expect_Return_HCCL_E_PTR)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInsts_[topoInstId] = topoInstance;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    netLayer = 3;
    CommTopo topoType;
    auto ret = comm.GetTopoType(netLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CommunicatorImplTest, Ut_GetTopoType_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInstance->topoType = Hccl::TopoType::CLOS;
    topoInsts_[topoInstId] = topoInstance;

    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());

    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    CommTopo topoType;
    auto ret = comm.GetTopoType(netLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetRanksByTopoInst_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    std::set<RankId> rankSet = {0};
    topoInstance->ranks = std::move(rankSet);
    topoInsts_[topoInstId] = topoInstance;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* ranks = nullptr;
    uint32_t rankNum;
    auto ret = comm.GetRanksByTopoInst(netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ranks[0], 0);
    EXPECT_EQ(rankNum, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetRanksByTopoInst_When_InvalidLayer_Expect_Return_HCCL_E_PTR)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 0;
    string netInstId = "test";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    std::set<RankId> rankSet = {0};
    topoInstance->ranks = std::move(rankSet);
    topoInsts_[topoInstId] = topoInstance;
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    netLayer = 3;
    uint32_t* ranks = nullptr;
    uint32_t rankNum;
    auto ret = comm.GetRanksByTopoInst(netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CommunicatorImplTest, Ut_GetRanksByTopoInst_When_ErrorNetType_Expect_Return_HCCL_E_PARA)
{
    MockCommunicatorImpl();
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    comm.rankGraph = make_unique<RankGraph>(0);
    u32 netLayer = 1;
    string netInstId = "layer1";
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;

    u32 topoInstId = 0;
    auto topoInstance = std::make_shared<NetInstance::TopoInstance>(topoInstId);
    topoInstance->topoType = Hccl::TopoType::CLOS;
    std::unordered_map<u32, std::shared_ptr<NetInstance::TopoInstance>> topoInsts_;
    topoInsts_[topoInstId] = topoInstance;
    auto netInstance = std::make_shared<ClosNetInstance>(netLayer, netInstId);
    netInstance->topoInsts_ = std::move(topoInsts_);

    shared_ptr<NetInstance::Peer> peer0 = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    peer0->AddNetInstance(netInstance);
    netInstance->AddRankId(peer0->GetRankId());
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddNetInstance(netInstance);
    comm.rankGraph->netInsts_[netLayer].emplace(netInstId, netInstance);

    uint32_t* ranks = nullptr;
    uint32_t rankNum;
    auto ret = comm.GetRanksByTopoInst(netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_TryInitCcuFeature_When_AccStateIsNotCcu_Expect_OK)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&TpManager::Init).stubs();
    CommunicatorImpl comm;
    comm.devLogicId = 40; // 40: 避免影响其他用例
    comm.rankSize = 2;
    comm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    comm.TryInitCcuFeature();
    comm.isAicpuKernelLaunched = false;
}

class FakeAivCollAlgComponent : public CollAlgComponent {
public:
    FakeAivCollAlgComponent() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {};
    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue) override
    {
        std::vector<LinkData> links;
        LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
        links.push_back(link);
        AivOpArgs aivOpArgs;
        std::unique_ptr<Instruction> aivIns = std::make_unique<AivInstruction>(links, aivOpArgs);
        queue->Append(std::move(aivIns));
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult Orchestrate(
        const CollAlgOperator& op, const CollAlgParams& params, const string& algName, PrimQuePtr queue) override
    {
        return HcclResult::HCCL_SUCCESS;
    }
};

TEST_F(CommunicatorImplTest, ut_GetAlgExecParam_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    std::shared_ptr<FakeAivCollAlgComponent> collAlgComponent = std::make_shared<FakeAivCollAlgComponent>();
    fakeComm.collAlgComponent = collAlgComponent;
    fakeComm.commExecuteConfig.accState = AcceleratorState::AIV;

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->comm_ = &fakeComm;
    dfxOpInfo->op_ = op;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::INT8;
    opParams.opType = OpType::ALLREDUCE;
    opParams.count = 1024;
    opParams.reduceOp = Hccl::ReduceOp::SUM;

    u32 sendBuffer = 10;
    opParams.sendBuf = static_cast<void*>(&sendBuffer);
    u32 recvBuffer = 20;
    opParams.recvBuf = static_cast<void*>(&recvBuffer);

    bool clearEnable = true;
    void* commContext = nullptr;
    u64 len = 0;
    int32_t aivCoreLimit = 2;

    void* addr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    MOCKER(HrtFree).stubs();
    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&))
        .stubs()
        .will(ignoreReturnValue());
    MOCKER_CPP(&UbMemoryTransportMgr::BatchCreateTransport)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&UbMemoryTransportMgr::TransportsConnect).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(
        fakeComm.GetAlgExecParam(opParams, clearEnable, commContext, len, aivCoreLimit), HcclResult::HCCL_SUCCESS);

    MOCKER_CPP(&CommunicatorImpl::HcomSelectAlg)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    fakeComm.commExecuteConfig.accState = AcceleratorState::CCU_MS;
    fakeComm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    EXPECT_EQ(
        fakeComm.GetAlgExecParam(opParams, clearEnable, commContext, len, aivCoreLimit),
        HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(CommunicatorImplTest, ut_Single_Rank_With_SendRecv_Expect_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.SetCommStatus(CommStatus::COMM_READY);
    comm.rankSize = 1;

    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.sendBuf = malloc(100);
    opParams.recvBuf = nullptr;
    opParams.opType = OpType::SEND;
    opParams.reduceOp = ReduceOp::SUM;

    EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HCCL_SUCCESS);

    free(opParams.sendBuf);
}

TEST_F(CommunicatorImplTest, Ut_AllocCollOpResource_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocCollOpResourceNoRegister)
        .stubs()
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).stubs();

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.commExecuteConfig = opConfig;

    CollOpParams param{};
    param.staticAddr = true;
    param.staticShape = true;
    param.dataType = DataType::INT8;
    param.opType = OpType::ALLREDUCE;
    param.commEngine = HcclAccelerator::AICPU_TS;
    void* addr = nullptr;
    auto ret = fakeComm.AllocCollOpResource(param, &addr);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);
}

TEST_F(CommunicatorImplTest, Ut_AllocCollOpResource_When_Not_AiCpu_Expect_ReturnIsHCCL_E_PARA)
{
    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocCollOpResourceNoRegister)
        .stubs()
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.commExecuteConfig = opConfig;

    CollOpParams param{};
    param.staticAddr = true;
    param.staticShape = true;
    param.dataType = DataType::INT8;
    param.opType = OpType::ALLREDUCE;
    void* addr = nullptr;
    auto ret = fakeComm.AllocCollOpResource(param, &addr);
    EXPECT_EQ(HcclResult::HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(CommunicatorImplTest, Ut_AllocCollOpResource_When_Status_Error_Expect_ReturnIsHCCL_E_INTERNAL)
{
    fakeComm.SetCommStatus(CommStatus::COMM_ERROR);
    CollOpParams param{};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    param.commEngine = HcclAccelerator::AICPU_TS;
    void* addr = nullptr;
    auto ret = fakeComm.AllocCollOpResource(param, &addr);
    EXPECT_EQ(HcclResult::HCCL_E_INTERNAL, ret);
}

TEST_F(CommunicatorImplTest, Ut_AllocCollOpResource_When_Status_Suspended_Expect_ReturnIsHCCL_E_SUSPENDING)
{
    fakeComm.isSuspended = true;
    CollOpParams param{};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    param.commEngine = HcclAccelerator::AICPU_TS;
    void* addr = nullptr;
    auto ret = fakeComm.AllocCollOpResource(param, &addr);
    EXPECT_EQ(HcclResult::HCCL_E_SUSPENDING, ret);
}

TEST_F(CommunicatorImplTest, Ut_AllocCollOpResource_When_Service_Default_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::HOSTCPU_TS;
    fakeComm.commExecuteConfig = opConfig;

    CollOpParams param{};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    param.commEngine = HcclAccelerator::AICPU_TS;
    void* addr = nullptr;
    auto ret = fakeComm.AllocCollOpResource(param, &addr);
    EXPECT_EQ(HcclResult::HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(CommunicatorImplTest, Ut_AllocCollOpResource_When_DataType_Check_Fail_Expect_ReturnIsHCCL_E_PARA)
{
    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.commExecuteConfig = opConfig;

    CollOpParams param{};
    param.staticAddr = true;
    param.staticShape = true;
    param.dataType = DataType::HIF8;
    param.opType = OpType::ALLREDUCE;
    param.commEngine = HcclAccelerator::AICPU_TS;
    void* addr = nullptr;
    auto ret = fakeComm.AllocCollOpResource(param, &addr);
    EXPECT_EQ(HcclResult::HCCL_E_PARA, ret);
}

TEST_F(CommunicatorImplTest, Ut_GetEndpointNum_When_Valid_Return_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    HcclCommConfig config;
    CommParams params;

    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2p.json"};
    comm.rankGraph = rankGraphBuilder.Build(RankTable2pEnd, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    uint32_t num = 0;
    uint32_t topoInstId = 0;
    auto ret = comm.GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_GetEndpointDesc_When_Valid_Return_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;

    HcclCommConfig config;
    CommParams params;

    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2p.json"};
    comm.rankGraph = rankGraphBuilder.Build(RankTable2pEnd, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    uint32_t layer = 0;
    uint32_t num = 2;
    uint32_t topoInstId = 0;
    HcclResult ret = comm.GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);

    uint32_t descNum = num;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    ret = comm.GetEndpointDesc(layer, topoInstId, &descNum, endPointDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (uint32_t i = 0; i < num; ++i) {
        EXPECT_NE(endPointDesc[i].commAddr.type, COMM_ADDR_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].loc.locType, ENDPOINT_LOC_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].protocol, COMM_PROTOCOL_RESERVED);
    }
    delete[] endPointDesc;
}

TEST_F(CommunicatorImplTest, Ut_GetEndpointDesc_When_NoPeer_Return_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;

    HcclCommConfig config;
    CommParams params;

    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2pclos.json"};
    comm.rankGraph = rankGraphBuilder.Build(RankTable2pClos, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    uint32_t layer = 0;
    uint32_t num = 0;
    uint32_t topoInstId = 0;
    auto ret = comm.GetEndpointNum(0, topoInstId, &num);
    uint32_t descNum = num;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    ret = comm.GetEndpointDesc(layer, topoInstId, &descNum, endPointDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (uint32_t i = 0; i < num; ++i) {
        EXPECT_NE(endPointDesc[i].commAddr.type, COMM_ADDR_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].loc.locType, ENDPOINT_LOC_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].protocol, COMM_PROTOCOL_RESERVED);
    }
    delete[] endPointDesc;
}

TEST_F(CommunicatorImplTest, Ut_GetInfo_When_BW_COEFF_Return_Success)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;

    HcclCommConfig config;
    CommParams params;

    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2pclos.json"};
    comm.rankGraph = rankGraphBuilder.Build(RankTable2pClos, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    uint32_t num = 0;
    uint32_t topoInstId = 0;
    auto ret = comm.GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);
    uint32_t descNum = num;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    uint32_t layer = 0;
    ret = comm.GetEndpointDesc(layer, topoInstId, &descNum, endPointDesc);
    for (uint32_t i = 0; i < num; ++i) {
        uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
        EndpointAttrBwCoeff bwCoeff{};
        ret = comm.GetEndpointInfo(0, &endPointDesc[i], ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff);
        EXPECT_EQ(bwCoeff, 1);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    delete[] endPointDesc;
}

TEST_F(CommunicatorImplTest, Ut_GetInfo_When_endpointDecsNull_Return_Hccl_E_PTR)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;

    HcclCommConfig config;
    CommParams params;

    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2pclos.json"};
    comm.rankGraph = rankGraphBuilder.Build(RankTable2pClos, topoFilePath, 0);
    EXPECT_NE(comm.rankGraph, nullptr);

    uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
    EndpointAttrBwCoeff bwCoeff{};
    HcclResult ret = comm.GetEndpointInfo(0, nullptr, ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

void MockForInitAndLaunchDpuKernel()
{
    MOCKER(aclrtGetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));
    MOCKER_CPP(&CommunicatorImpl::WaitDpuKernelThreadTerminate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::GetKFCWorkSpaceVA).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::PrepareDpuKernelResource).stubs().will(returnValue(HCCL_SUCCESS));
}

TEST_F(CommunicatorImplTest, Ut_InitAndLaunchDpuKernel_When_LaunchFailed_Expect_Return_HCCL_E_INTERNAL)
{
    CommunicatorImpl comm;
    comm.devPhyId = 0;
    comm.isDpuKernelLaunched = true;
    MockForInitAndLaunchDpuKernel();
    MOCKER_CPP(&CommunicatorImpl::LaunchDpuKernel).stubs().will(returnValue(HCCL_E_INTERNAL));
    HcclResult ret = comm.InitAndLaunchDpuKernel();
}

TEST_F(CommunicatorImplTest, Ut_When_DestroyDpuKernelResource_Expect_DpuStream_Uninit_Failed)
{
    CommunicatorImpl comm;
    comm.devPhyId = 0;
    comm.isDpuKernelLaunched = true;
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER_CPP(&CommunicatorImpl::WaitDpuKernelThreadTerminate).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = comm.DestroyDpuKernelResource();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(CommunicatorImplTest, Ut_When_DestroyDpuKernelResource_Expect_ResetDpuDevice_Failed)
{
    CommunicatorImpl comm;
    comm.devPhyId = 0;
    comm.isDpuKernelLaunched = true;
    MockForInitAndLaunchDpuKernel();
    MOCKER_CPP(&CommunicatorImpl::LaunchDpuKernel).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = comm.InitAndLaunchDpuKernel();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = comm.DestroyDpuKernelResource();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

namespace Hccl {
HcclResult HrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t& value);
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_CacheHitSizeMatch_Expect_Success)
{
    uint64_t size = 1024;
    void* addr = nullptr;
    bool newCreated = true;
    std::string tag = "DPUTAG";

    auto buf = DevBuffer::Create(static_cast<uintptr_t>(0x1000), size);
    ASSERT_NE(buf, nullptr);
    fakeComm.tagWorkspaceVAMap_.insert({tag, buf});

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, &newCreated);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(addr, reinterpret_cast<void*>(0x1000));
    EXPECT_FALSE(newCreated);
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_CacheHitSizeMismatch_Expect_ParaError)
{
    uint64_t size = 2048;
    void* addr = nullptr;
    bool newCreated = true;
    std::string tag = "DPUTAG";

    auto buf = DevBuffer::Create(static_cast<uintptr_t>(0x1000), 1024);
    ASSERT_NE(buf, nullptr);
    fakeComm.tagWorkspaceVAMap_.insert({tag, buf});

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, &newCreated);

    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_CacheHitNewCreatedNull_Expect_Success)
{
    uint64_t size = 1024;
    void* addr = nullptr;
    std::string tag = "DPUTAG";

    auto buf = DevBuffer::Create(static_cast<uintptr_t>(0x1000), size);
    ASSERT_NE(buf, nullptr);
    fakeComm.tagWorkspaceVAMap_.insert({tag, buf});

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, nullptr);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(addr, reinterpret_cast<void*>(0x1000));
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_PciePath_Expect_Success)
{
    GlobalMockObject::verify();
    uint64_t size = 1024;
    void* addr = nullptr;
    bool newCreated = false;
    std::string tag = "DPUTAG";

    int64_t connType = 0; // HOST_DEVICE_CONNECT_TYPE_PCIE
    static uint8_t pcieAccessMem = 0;
    void* stubRegAddr = &pcieAccessMem;
    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(reinterpret_cast<void*>(0x2000)));
    MOCKER(halHostRegister)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&stubRegAddr))
        .will(returnValue(DRV_ERROR_NONE));
    MOCKER(HrtFree).stubs();

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, &newCreated);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(addr, nullptr);
    EXPECT_TRUE(newCreated);
    EXPECT_TRUE(fakeComm.tagWorkspaceVAMap_.find(tag) != fakeComm.tagWorkspaceVAMap_.end());
    EXPECT_EQ(fakeComm.tagDpuShmemArgsMap_[tag].va_, reinterpret_cast<void*>(0x2000));
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_PcieHalHostRegisterFail_Expect_DrvError)
{
    GlobalMockObject::verify();
    uint64_t size = 1024;
    void* addr = nullptr;
    bool newCreated = true;
    std::string tag = "DPUTAG";

    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(reinterpret_cast<void*>(0x2000)));
    int64_t connType = 0; // HOST_DEVICE_CONNECT_TYPE_PCIE
    void* stubRegAddr = reinterpret_cast<void*>(0xBEEF0000);
    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(halHostRegister)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(DRV_ERROR_INNER_ERR));
    MOCKER(HrtFree).stubs();

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, &newCreated);

    EXPECT_EQ(ret, HCCL_E_DRV);
    auto iter = fakeComm.tagWorkspaceVAMap_.find(tag);
    EXPECT_EQ(iter, fakeComm.tagWorkspaceVAMap_.end());
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_UbPath_Expect_Success)
{
    GlobalMockObject::verify();
    constexpr u64 ALIGN_4K = 4096U;
    uint64_t size = 1024;
    void* addr = nullptr;
    bool newCreated = false;
    std::string tag = "DPUTAG";
    int64_t connType = 2; // HOST_DEVICE_CONNECT_TYPE_UB

    void* stubRegAddr = reinterpret_cast<void*>(0xBEEF0000);

    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(halHostRegister)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&stubRegAddr))
        .will(returnValue(DRV_ERROR_NONE));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(reinterpret_cast<void*>(0x3000)));

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, &newCreated);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(addr, nullptr);
    EXPECT_TRUE(newCreated);
    EXPECT_TRUE(fakeComm.tagWorkspaceVAMap_.find(tag) != fakeComm.tagWorkspaceVAMap_.end());

    EXPECT_NE(fakeComm.tagDpuShmemArgsMap_[tag].va_, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(fakeComm.tagDpuShmemArgsMap_[tag].va_) % ALIGN_4K, 0);
    // free the malloc'd memory allocated by GetKFCWorkSpaceVA in UB path
    free(fakeComm.tagDpuShmemArgsMap_[tag].originVa_);
    fakeComm.tagDpuShmemArgsMap_[tag].originVa_ = nullptr;
    fakeComm.tagDpuShmemArgsMap_[tag].va_ = nullptr;
}

TEST_F(CommunicatorImplTest, Ut_GetKFCWorkSpaceVA_When_UnsupportedConnectType_Expect_NotSupport)
{
    GlobalMockObject::verify();
    uint64_t size = 1024;
    void* addr = nullptr;
    bool newCreated = true;
    std::string tag = "DPUTAG";
    int64_t connType = 1; // HOST_DEVICE_CONNECT_TYPE_HCCS;// unsupported connect type

    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connType))
        .will(returnValue(HCCL_SUCCESS));

    auto ret = fakeComm.GetKFCWorkSpaceVA(tag, &size, &addr, &newCreated);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

class TryFastCcuLaunchTest : public CommunicatorImplTest {
protected:
    void SetUp() override
    {
        CommunicatorImplTest::SetUp();
        // 初始化测试环境
        fakeStreamPtr = (void*)1;
        sendBuffer = 10;
        fakeOpParams.sendBuf = static_cast<void*>(&sendBuffer);
        recvBuffer = 20;
        fakeOpParams.recvBuf = static_cast<void*>(&recvBuffer);
        fakeOpParams.opType = OpType::ALLREDUCE;
        fakeOpParams.dataType = DataType::FP32;
        fakeOpParams.reduceOp = ReduceOp::SUM;
        fakeOpParams.count = 1024;
        fakeComm.commExecuteConfig.accState = AcceleratorState::CCU_SCHED;
        MOCKER_CPP(&CommunicatorImpl::InitCcuSuperFastLoad).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER_CPP(&CommunicatorImpl::ExecuteFastCcuLaunch).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        CcuTaskParam ccuTaskParam{};
        std::vector<std::vector<CcuTaskParam>> ccuIns;
        std::vector<std::vector<CcuProfilingInfo>> ccuProInfo;
        ccuIns.push_back({ccuTaskParam});
        ccuProInfo.resize(1);
        fakeComm.currentCollOperator->opType = fakeOpParams.opType;
        fakeComm.ccuParamsMappingKey
            = {static_cast<std::uint32_t>(fakeOpParams.reduceOp), static_cast<std::uint32_t>(fakeOpParams.dataType),
               static_cast<std::uint32_t>(fakeOpParams.count)};
        fakeComm.saveCCUParams(std::move(ccuIns), std::move(ccuProInfo), 0, CcuInstType::CCU_INS_GROUP, true);
    }

private:
    CollOpParams fakeOpParams{};
    u32 sendBuffer;
    u32 recvBuffer;
    void* fakeStreamPtr;
};

TEST_F(TryFastCcuLaunchTest, Ut_TryFastCcuLaunch_When_NoNeedProf_Expect_ReturnTrue)
{
    // when
    fakeComm.enableProfilingEnv = false;
    // then
    EXPECT_EQ(fakeComm.TryFastCcuLaunch(fakeOpParams, fakeStreamPtr), true);
}

TEST_F(TryFastCcuLaunchTest, Ut_TryFastCcuLaunch_When_NeedProf_Expect_ReturnTrue)
{
    // when
    fakeComm.enableProfilingEnv = true;
    // then
    EXPECT_EQ(fakeComm.TryFastCcuLaunch(fakeOpParams, fakeStreamPtr), true);
}

TEST_F(TryFastCcuLaunchTest, Ut_TryFastCcuLaunch_When_NoCcu_Expect_ReturnFalse)
{
    // when
    fakeComm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    // then
    EXPECT_EQ(fakeComm.TryFastCcuLaunch(fakeOpParams, fakeStreamPtr), false);
}

TEST_F(TryFastCcuLaunchTest, Ut_TryFastCcuLaunch_When_OpNoSupportFastLaunch_Expect_ReturnFalse)
{
    // when
    fakeOpParams.opType = OpType::ALLGATHERV;
    // then
    EXPECT_EQ(fakeComm.TryFastCcuLaunch(fakeOpParams, fakeStreamPtr), false);
}

// 测试 SaveDpuStreamId - 正常情况
TEST_F(CommunicatorImplTest, Ut_SaveDpuStreamId_When_Normal_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    // 构造一个假的 aclrtStream
    aclrtStream fakeStream = (aclrtStream)0x12345678;
    comm.dpuStream = fakeStream;
    // mock HrtGetStreamId 返回固定值
    s32 fakeStreamId = 0;
    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclResult ret = comm.SaveDpuStreamId();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(comm.GetDpuStreamId(), static_cast<u32>(fakeStreamId));
}

// 测试 LaunchDpuKernel - aclrtLaunchKernelWithHostArgs 返回成功
TEST_F(CommunicatorImplTest, Ut_LaunchDpuKernel_When_aclrtLaunchKernelWithHostArgsSuccess_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    comm.id = "test_comm";
    comm.devLogicId = 0;
    aclrtStream fakeStream = reinterpret_cast<aclrtStream>(0x11111111);
    comm.dpuStream = fakeStream;
    aclrtFuncHandle fakeFuncHandle = reinterpret_cast<aclrtFuncHandle>(0x22222222);

    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = comm.LaunchDpuKernel(fakeFuncHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorperty_When_MissingProperty_Expect_Throw)
{
    nlohmann::json obj = nlohmann::json::object();
    EXPECT_THROW(GetJsonProperty(obj, "missing", true), InvalidParamsException);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorpertyUInt_When_MissingProperty_Expect_Throw)
{
    nlohmann::json obj = nlohmann::json::object();
    EXPECT_THROW(GetJsonPropertyUInt(obj, "missing", true, 0), InvalidParamsException);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorpertyUInt_When_ValueExceedsUint32Max_Expect_Throw)
{
    nlohmann::json obj;
    obj["test"] = INT64_MAX;
    EXPECT_THROW(GetJsonPropertyUInt(obj, "test", true, 0), InvalidParamsException);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorpertySInt_When_MissingProperty_Expect_Throw)
{
    nlohmann::json obj = nlohmann::json::object();
    EXPECT_THROW(GetJsonPropertySInt(obj, "missing", true, 0), InvalidParamsException);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorpertySInt_When_ValueExceedsSint32Max_Expect_Throw)
{
    nlohmann::json obj;
    obj["test"] = INT64_MAX;
    EXPECT_THROW(GetJsonPropertySInt(obj, "test", true, 0), InvalidParamsException);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorpertyList_When_MissingProperty_Expect_Throw)
{
    nlohmann::json obj = nlohmann::json::object();
    nlohmann::json listObj;
    EXPECT_THROW(GetJsonPropertyList(obj, "missing", listObj), InvalidParamsException);
}

TEST_F(TryFastCcuLaunchTest, GetJsonPorpertyList_When_NotArray_Expect_Throw)
{
    nlohmann::json obj;
    obj["test"] = "not_array";
    nlohmann::json listObj;
    EXPECT_THROW(GetJsonPropertyList(obj, "test", listObj), InvalidParamsException);
}

// dpu相关新增
TEST_F(CommunicatorImplTest, Ut_WaitDpuKernelThreadTerminate_When_NotLaunched_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    comm.isDpuKernelLaunched = false;

    HcclResult ret = comm.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_WaitDpuKernelThreadTerminate_When_VaNull_Expect_ReturnMemoryError)
{
    CommunicatorImpl comm;
    comm.isDpuKernelLaunched = true;
    comm.tagDpuShmemArgsMap_["DPUTAG"].va_ = nullptr;
    comm.tagDpuShmemArgsMap_["DPUTAG"].accessVA_ = nullptr;

    HcclResult ret = comm.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

TEST_F(CommunicatorImplTest, Ut_WaitDpuKernelThreadTerminate_When_AlreadyExited_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    comm.isDpuKernelLaunched = true;

    uint8_t shmemData[64] = {0};
    shmemData[0] = 3;
    comm.tagDpuShmemArgsMap_["DPUTAG"].va_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTAG"].accessVA_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTAG"].connectType_ = 0;

    HcclResult ret = comm.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_WaitDpuKernelThreadTerminate_When_NeedTerminate_Expect_WriteSecondAndWaitThird)
{
    CommunicatorImpl comm;
    comm.isDpuKernelLaunched = true;

    uint8_t shmemData[64] = {0};
    shmemData[0] = 0;
    comm.tagDpuShmemArgsMap_["DPUTAG"].va_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTAG"].accessVA_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTAG"].connectType_ = 0;

    std::thread signalThread([&shmemData]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        shmemData[0] = 3;
    });

    HcclResult ret = comm.WaitDpuKernelThreadTerminate();
    signalThread.join();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(shmemData[0], 3);
}

TEST_F(CommunicatorImplTest, Ut_WaitDpuKernelThreadTerminate_When_UbPath_Expect_UseVaAsHostPtr)
{
    CommunicatorImpl comm;
    comm.isDpuKernelLaunched = true;

    uint8_t shmemData[64] = {0};
    shmemData[0] = 3;
    comm.tagDpuShmemArgsMap_["DPUTAG"].va_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTAG"].accessVA_ = reinterpret_cast<void*>(0xDEAD);
    comm.tagDpuShmemArgsMap_["DPUTAG"].connectType_ = 2;

    HcclResult ret = comm.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_DestroyDpuTaskexpShmemInDevice_When_VaNull_Expect_ReturnMemoryError)
{
    CommunicatorImpl comm;
    comm.isDpuKernelLaunched = true;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].va_ = nullptr;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].accessVA_ = nullptr;

    HcclResult ret = comm.DestroyDpuTaskexpShmemInDevice();
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

TEST_F(CommunicatorImplTest, Ut_DestroyDpuTaskexpShmemInDevice_When_DeviceResponseSuccess_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;

    uint8_t shmemData[64] = {0};
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].va_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].accessVA_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].connectType_ = 0;

    std::thread responseThread([&shmemData]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        shmemData[0] = 0;
    });

    HcclResult ret = comm.DestroyDpuTaskexpShmemInDevice();
    responseThread.join();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(shmemData[0], 0);
}

TEST_F(CommunicatorImplTest, Ut_DestroyDpuTaskexpShmemInDevice_When_UbPath_Expect_UseVaAsHostPtr)
{
    CommunicatorImpl comm;

    uint8_t shmemData[64] = {0};
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].va_ = shmemData;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].accessVA_ = reinterpret_cast<void*>(0xDEAD);
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].connectType_ = 2;

    std::thread responseThread([&shmemData]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        shmemData[0] = 0;
    });

    HcclResult ret = comm.DestroyDpuTaskexpShmemInDevice();
    responseThread.join();

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

void SetupAicpuKernelTestComm(CommunicatorImpl& comm, aclError streamCreateRet)
{
    comm.id = "aicpuTest";
    comm.devLogicId = 0;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].accessVA_ = reinterpret_cast<void*>(0x1000);

    MOCKER_CPP(&CommunicatorImpl::GetAicpuKernelFuncHandle)
        .stubs()
        .will(returnValue(reinterpret_cast<aclrtFuncHandle>(0x1)));
    MOCKER(aclrtCreateStreamWithConfig)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(streamCreateRet));
}

TEST_F(CommunicatorImplTest, Ut_InitAndLaunchAicpuKernel_When_CreateStreamFail_Expect_ReturnInternalError)
{
    CommunicatorImpl comm;
    SetupAicpuKernelTestComm(comm, ACL_ERROR_INVALID_PARAM);

    HcclResult ret = comm.InitAndLaunchAicpuKernel();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, Ut_InitAndLaunchAicpuKernel_When_LaunchFail_Expect_ReturnInternalError)
{
    CommunicatorImpl comm;
    SetupAicpuKernelTestComm(comm, ACL_SUCCESS);
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(static_cast<rtError_t>(1)));

    HcclResult ret = comm.InitAndLaunchAicpuKernel();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(CommunicatorImplTest, Ut_InitAndLaunchAicpuKernel_When_DestroyStreamFail_Expect_ReturnRuntimeError)
{
    CommunicatorImpl comm;
    SetupAicpuKernelTestComm(comm, ACL_SUCCESS);
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(RT_ERROR_NONE));
    MOCKER(HcclStreamSynchronize).stubs();
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));

    HcclResult ret = comm.InitAndLaunchAicpuKernel();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(CommunicatorImplTest, Ut_InitAndLaunchAicpuKernel_When_AllSuccess_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    SetupAicpuKernelTestComm(comm, ACL_SUCCESS);
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(RT_ERROR_NONE));
    MOCKER(HcclStreamSynchronize).stubs();
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = comm.InitAndLaunchAicpuKernel();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_Destructor_When_TaskExpMemMapHasEntry_Expect_EraseEntry)
{
    CommunicatorImpl* comm = new CommunicatorImpl();
    std::string savedId = "destrTest_" + std::to_string(reinterpret_cast<uintptr_t>(comm));
    comm->id = savedId;
    comm->devLogicId = 0;
    comm->isDpuKernelLaunched = true;
    comm->isAicpuKernelLaunched = false;

    g_taskExpMemMap[savedId][0] = reinterpret_cast<void*>(0x1234);
    ASSERT_TRUE(g_taskExpMemMap.find(savedId) != g_taskExpMemMap.end());

    MOCKER_CPP(&CommunicatorImpl::DestroyDpuKernelResource).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::DestroyDpuTaskexpShmemInDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::DestroyKFCWorkSpaceVA).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::NotifyAicpuDestroyComm).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::DeInitPreResource).stubs();

    delete comm;

    EXPECT_TRUE(g_taskExpMemMap.find(savedId) == g_taskExpMemMap.end());
}

TEST_F(CommunicatorImplTest, Ut_NotifyAicpuDestroyComm_When_NotLaunched_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    comm.isAicpuKernelLaunched = false;

    HcclResult ret = comm.NotifyAicpuDestroyComm();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_NotifyAicpuDestroyComm_When_KfcControlNull_Expect_ReturnSuccess)
{
    CommunicatorImpl comm;
    comm.isAicpuKernelLaunched = true;
    comm.kfcControlTransferH2D = nullptr;

    HcclResult ret = comm.NotifyAicpuDestroyComm();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommunicatorImplTest, Ut_LaunchDpuKernel_When_DpuTaskexpShmemValid_Expect_TaskexceptionVaSet)
{
    CommunicatorImpl comm;
    comm.id = "launchTest";
    comm.devLogicId = 0;
    comm.hostShareBuf = malloc(1024);

    void* taskexpVa = reinterpret_cast<void*>(0xABCD);
    comm.tagDpuShmemArgsMap_["DPUTAG"].va_ = reinterpret_cast<void*>(0x2000);
    comm.tagDpuShmemArgsMap_["DPUTAG"].accessVA_ = reinterpret_cast<void*>(0x3000);
    comm.tagDpuShmemArgsMap_["DPUTAG"].connectType_ = 0;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].va_ = reinterpret_cast<void*>(0x4000);
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].accessVA_ = taskexpVa;
    comm.tagDpuShmemArgsMap_["DPUTASKEXCEPTION"].connectType_ = 0;

    aclrtFuncHandle fakeHandle = reinterpret_cast<aclrtFuncHandle>(0x1);
    MOCKER_CPP(&CommunicatorImpl::SaveDpuStreamId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = comm.LaunchDpuKernel(fakeHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(comm.hostArgs.taskexceptionVa, taskexpVa);

    MOCKER_CPP(&CommunicatorImpl::DestroyDpuKernelResource).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::DestroyDpuTaskexpShmemInDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::DestroyKFCWorkSpaceVA).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::NotifyAicpuDestroyComm).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::DeInitPreResource).stubs();

    free(comm.hostShareBuf);
    comm.hostShareBuf = nullptr;
}

TEST_F(CommunicatorImplTest, Ut_InitAndLaunchDpuKernel_When_DpuTaskexpShmemAllocated_Expect_BothTagsInShmemMap)
{
    CommunicatorImpl comm;
    comm.devPhyId = 0;
    comm.id = "dualTagTest";

    MockForInitAndLaunchDpuKernel();
    MOCKER_CPP(&CommunicatorImpl::LaunchDpuKernel).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = comm.InitAndLaunchDpuKernel();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_TRUE(comm.tagDpuShmemArgsMap_.find("DPUTAG") != comm.tagDpuShmemArgsMap_.end());
    EXPECT_TRUE(comm.tagDpuShmemArgsMap_.find("DPUTASKEXCEPTION") != comm.tagDpuShmemArgsMap_.end());
}
