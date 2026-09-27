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
#include "hccp_ctx.h"
#include "hccp_common.h"
#include "base_config_legacy.h"
#include "cfg_field.h"
#include "env_config_v2.h"
#include "env_func.h"
#include "virtual_topo.h"
#include "dev_ub_connection.h"
#include "virtual_topo_stub.h"
#include "coll_alg_component_builder.h"
#include <stdexcept>
#include <string>
#include <thread>
#include <hccl/hccl_types.h>
#include "coll_service_default_impl.h"
#include "op_params_checker.h"
#include "mc2_type.h"
#include "json_parser.h"
#include "hccl_communicator.h"
#include "net_instance.h"
#include "rank_graph_builder.h"
#include "phy_topo_builder.h"
#include "detour_service.h"
#include "sal.h"
#include "rank_gph.h"
#include "base_config_legacy.h"
#include "env_config_v2.h"
#include "coll_service_device_mode.h"
#include "communicator_callback.h"
#include "hccp_tlv_hdc_manager.h"
#include "ccu_driver_handle.h"

#include "hdc_lite.h"
#include "kfc.h"
#include "coll_service_device_mode.h"
#include "ccu_communicator.h"

#include "stream_manager.h"
#include "virtual_topo_stub.h"
#include "ins_all_reduce_sole_executor.h"
#include "topo_match_mesh.h"
#include "ccu_temp_all_reduce_mesh_1D_one_shot.h"
#include "ccu_context_all_reduce_mesh1d_one_shot.h"
#include "ccu_instruction_all_reduce_mesh1d_one_shot.h"
#include "ccu_instruction_all_reduce_mesh1d_multimission.h"
#include "ccu_context_all_reduce_mesh1d_multimission.h"
#include "ccu_temp_all_reduce_mesh_1D_multimission.h"
#include "ccu_ins_group.h"
#include "coll_alg_component.h"
#include "../common/env_config_stub.h"
#include "stream_utils.h"
#include "op_base_v2.h"
#include "hccl.h"
#include "ccu_component.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "tp_manager.h"
#include "ranktable_stub_clos.h"
#include "task_abort_handler_v2.h"
#undef private
#undef protected

using namespace Hccl;
std::map<std::string, std::string> envCommstCfgMap = defaultEnvCfgMap;
constexpr u32 TEMP_UES_CNTCKE_NUM = 16;

char* commstGetenv_stub(const char* __name)
{
    char* ret = const_cast<char*>(envCommstCfgMap[std::string(__name)].c_str());
    return ret;
}

class CommunicatorImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }
};

class FakeCollAlgComponent : public CollAlgComponent {
public:
    FakeCollAlgComponent() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {};
    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue) override
    {
        return HCCL_SUCCESS;
    }

    HcclResult Orchestrate(
        const CollAlgOperator& op, const CollAlgParams& params, const string& algName, PrimQuePtr queue) override
    {
        return HCCL_SUCCESS;
    }
};

class FakeCollAlgComponentWithError : public CollAlgComponent {
public:
    FakeCollAlgComponentWithError() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {}
    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue) override
    {
        return HCCL_E_INTERNAL;
    }
};

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

const char filePath[] = "ranktable.json";

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

void CommImplSendStub1() { THROW<InternalException>("HcclException &e"); }

#define SETUP_COMM_RESOURCE_MOCKS()                                                                     \
    do {                                                                                                \
        MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());         \
        Buffer* buf = nullptr;                                                                          \
        LocalRmaBuffer* rmaBuf = nullptr;                                                               \
        MOCKER_CPP(&DataBufManager::Get)                                                                \
            .stubs()                                                                                    \
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())                                       \
            .will(returnValue(buf));                                                                    \
        MOCKER_CPP(                                                                                     \
            &LocalRmaBufManager::Reg,                                                                   \
            LocalRmaBuffer* (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>, \
                                                    const PortData&, LinkProtocol))                     \
            .stubs()                                                                                    \
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())                                       \
            .will(returnValue(rmaBuf));                                                                 \
    } while (0)

#define SETUP_SNAPSHOT_COMM_PARAMS(collSvc)                                                 \
    do {                                                                                    \
        comm.collService = &(collSvc);                                                      \
        SnapShotComm snapShotComm;                                                          \
        u32 step = 1;                                                                       \
        CommParams commParams("test_comm_id", 0, 4, 0, DevType::DEV_TYPE_950, false, true); \
        HcclCommConfig config;                                                              \
        strcpy(config.reserved, "test_reserved");                                           \
        config.hcclBufferSize = 1024;                                                       \
        config.hcclDeterministic = 1;                                                       \
        strcpy(config.hcclCommName, "test_comm_name");                                      \
        strcpy(config.hcclUdi, "test_udi");                                                 \
        RankTableInfo ranktableInfo;                                                        \
        ranktableInfo.version = "2.0";                                                      \
        ranktableInfo.rankCount = 0;                                                        \
        TopoInfo topoInfo;                                                                  \
        topoInfo.version = "2.0";                                                           \
        topoInfo.peerCount = 0;                                                             \
        topoInfo.edgeCount = 0;                                                             \
    } while (0)

TEST(CommunicatorImplTest, load_opbased_coll_op_test)
{
    CommParams params;
    CollOpParams opParams;
    HcclCommunicator hcclCommunicator(params);
    void* stream = nullptr;
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::LoadOpbasedCollOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcclCommunicator.LoadOpbasedCollOp(opParams, stream);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    opParams.opType = OpType::ALLTOALL;
    EXPECT_EQ(HCCL_SUCCESS, hcclCommunicator.LoadOpbasedCollOp(opParams, stream));
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, set_coll_offload_slave_streams_test)
{
    CommParams params;
    OpType opType = OpType::ALLREDUCE;
    u64 dataSize = 100;
    CollOffloadOpResReq resReq1;
    HcclCommunicator hcclCommunicator(params);
    MOCKER_CPP(&CommunicatorImpl::CalcCollOffloadOpRes).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcclCommunicator.CalcCollOffloadOpRes(opType, dataSize, HCCL_DATA_TYPE_INT8, resReq1);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST(CommunicatorImplTest, set_coll_offload_slav_streams_test)
{
    CommParams params;
    HcclCommunicator hcclCommunicator(params);
    std::vector<void*> slaveStreams;
    std::string opTag = "test";

    MOCKER_CPP(&CommunicatorImpl::SetCollOffloadSlaveStreams)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HCCL_SUCCESS, hcclCommunicator.SetCollOffloadSlaveStreams(opTag, slaveStreams));
}

TEST(CommunicatorImplTest, load_offload_coll_op_test)
{
    std::string opTag = "opTag";
    CollOpParams opParams;
    CommParams params;
    HcclCommunicator hcclCommunicator(params);

    MOCKER_CPP(&CommunicatorImpl::LoadOffloadCollOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcclCommunicator.LoadOffloadCollOp(opTag, opParams, nullptr);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST(CommunicatorImplTest, should_return_failed_when_calling_create_subcomm_with_comm_uninitialized)
{
    CommunicatorImpl comm;
    CommParams params;
    std::vector<u32> rankIds;
    CommunicatorImpl subCommImpl;
    HcclCommConfig subConfig;
    EXPECT_EQ(HcclResult::HCCL_E_INTERNAL, comm.CreateSubComm(params, rankIds, &subCommImpl, subConfig));
}

TEST(CommunicatorImplTest, should_return_failed_when_calling_create_subcomm_with_comm_uninitialized_1)
{
    CommunicatorImpl comm;
    CommParams params;
    std::vector<u32> rankIds;
    CommunicatorImpl subCommImpl;
    EXPECT_EQ(HCCL_E_INTERNAL, comm.CreateSubComm(params, rankIds, &subCommImpl));
}

TEST(CommunicatorImplTest, should_return_success_when_calling_suspend_without_aicpu_kernel_launched)
{
    CommunicatorImpl comm;
    EXPECT_EQ(false, comm.isSuspended);
    EXPECT_EQ(false, comm.isAicpuKernelLaunched);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, comm.Suspend());
    EXPECT_EQ(true, comm.isSuspended);

    // 已经处于isSuspended == true，重复Suspend()无效
    comm.isSuspended = true;
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, comm.Suspend());
}

TEST(CommunicatorImplTest, should_return_success_when_calling_clean_without_aicpu_kernel_launched)
{
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

TEST(CommunicatorImplTest, should_return_success_when_calling_suspend_with_aicpu_kernel_launched)
{
    MOCKER(HrtDrvMemCpy).stubs().with().will(invoke(HrtDrvMemCpyStub));

    CommunicatorImpl comm;
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
    // 这里是模拟对应的内容
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

TEST(CommunicatorImplTest, should_return_success_when_calling_clean_with_aicpu_kernel_launched)
{
    MOCKER(HrtDrvMemCpy).stubs().with().will(invoke(HrtDrvMemCpyStub));

    CommunicatorImpl comm;
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

TEST(CommunicatorImplTest, St_NsRecovery_Clean_When_Not_Load_Op_Expect_Success)
{
    CommunicatorImpl comm;
    comm.isSuspended = true;
    comm.isCleaned = false;
    comm.commExecuteConfig.accState = AcceleratorState::CCU_MS;

    EXPECT_EQ(comm.Clean(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(comm.isCleaned, true);
}

TEST(CommunicatorImplTest, St_NsRecovery_Resume_When_Not_Load_Op_Expect_Success)
{
    CommunicatorImpl comm;
    comm.status = CommStatus::COMM_READY;
    comm.isSuspended = true;
    comm.isCleaned = true;
    comm.commExecuteConfig.accState = AcceleratorState::CCU_MS;

    EXPECT_EQ(comm.Resume(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(comm.isSuspended, false);
    EXPECT_EQ(comm.isCleaned, false);
}

TEST(CommunicatorImplTest, initvittualtopo_check_fail)
{
    s32 myRank = 0;
    std::unique_ptr<RankGraph> inputVirtualTopo = std::make_unique<RankGraph>(myRank);
    CommunicatorImpl comm;
    EXPECT_THROW(comm.InitRankGraph(inputVirtualTopo), InvalidParamsException);
}

TEST(CommunicatorImplTest, should_return_success_when_normal_calling_new_init_with_two_parameters_new_return_error)
{
    CommunicatorImpl comm;
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    comm.initFlag = true;

    const string rankTablePath = "ranktable.json";
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    EXPECT_EQ(comm.Init(params, rankTablePath, config), HCCL_E_INTERNAL);
}

TEST(CommunicatorImplTest, init_common_data)
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(1));
    CommunicatorImpl comm;
    CommParams params;
    HcclCommunicator hcclCommunicator(params);
    comm.InitCommonData(params);
}

TEST(CommunicatorImplTest, should_return_rank_size_when_calling_get_rank_size)
{
    CommunicatorImpl comm;
    u32 rankSize = 10;
    comm.rankSize = rankSize;
    cout << "comm.GetRankSize() = " << comm.GetRankSize() << "rankSize = " << rankSize << endl;
    EXPECT_EQ(comm.GetRankSize(), rankSize);
}

TEST(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperator)
{
    CommunicatorImpl fakeComm;
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();
    MOCKER(HrtStreamDestroy).stubs();

    // 资源初始化
    SETUP_COMM_RESOURCE_MOCKS();
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
    fakeComm.status = CommStatus::COMM_READY;
    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
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
    peer0->AddConnInterface(connInterface);
    fakeComm.rankGraph->AddPeer(peer0);
    fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
    fakeComm.trace = std::make_unique<Trace>();

    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    OpExecuteConfig opConfig; // aicpu 展开
    opConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.opExecuteConfig = opConfig;
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    // 算法组件初始化
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    collAlgOpReq.resReq.primQueueNum = 1;
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
            const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opConfig,
            CollOffloadOpResReq& resReq))
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

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->comm_ = &fakeComm;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, ptr1), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST(CommunicatorImplTest, should_trace_success_when_comm_params_valid)
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
    comm.status = CommStatus::COMM_READY;
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
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, CovertToCurrentCollOperator)
{
    CommunicatorImpl comm;

    HcclSendRecvItem sendRecvInfo;
    sendRecvInfo.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    CollOpParams collOpParams;
    collOpParams.opType = OpType::BATCHSENDRECV;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = 1;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(&sendRecvInfo);

    uint64_t a = 10;
    uintptr_t devAddr = reinterpret_cast<uintptr_t>(&a);
    std::size_t devSize = 2;
    comm.cclBuffer = make_shared<DevBuffer>(10);
    string tag = "optag";
    comm.CovertToCurrentCollOperator(tag, collOpParams, OpMode::OPBASE);
}

TEST(CommunicatorImplTest, LoadOpbasedCollOp_success_CovertToCurrentCollOperator_allgather)
{
    CommunicatorImpl fakeComm;
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();
    MOCKER(HrtStreamDestroy).stubs();

    // 资源初始化
    SETUP_COMM_RESOURCE_MOCKS();
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
    fakeComm.status = CommStatus::COMM_READY;
    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
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
    peer0->AddConnInterface(connInterface);
    fakeComm.rankGraph->AddPeer(peer0);
    fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
    fakeComm.trace = std::make_unique<Trace>();

    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    OpExecuteConfig opConfig; // aicpu 展开
    opConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.opExecuteConfig = opConfig;
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    // 算法组件初始化
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    collAlgOpReq.resReq.primQueueNum = 1;
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
            const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opConfig,
            CollOffloadOpResReq& resReq))
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

    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLGATHER;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->comm_ = &fakeComm;
    MirrorTaskManager& mirrorTaskManager = fakeComm.GetMirrorTaskManager();
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;
    EXPECT_EQ(fakeComm.SetCollOffloadScratchBuf("test", (void*)0x100, 0x100), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeComm.LoadOpbasedCollOp(opParams, ptr1), HcclResult::HCCL_SUCCESS);

    string tag = "tag";
    EXPECT_NO_THROW(fakeComm.CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD));
}

TEST(CommunicatorImplTest, CalcA2ASendRecvMem_test)
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

TEST(CommunicatorImplTest, CalcA2ASendRecvMem_test_2)
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

TEST(CommunicatorImplTest, should_fail_when_LoadOpbasedCollOp_catch_HcclException)
{
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();
    comm.status = CommStatus::COMM_READY;
    comm.rankSize = 2;

    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(invoke(CommImplSendStub1));

    EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_E_INTERNAL);
}

TEST(CommunicatorImplTest, should_fail_when_LoadOpbasedCollOp_catch_unknown_exception)
{
    CommunicatorImpl comm;
    comm.status = CommStatus::COMM_READY;
    comm.rankSize = 2;
    CollOpParams opParams;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLREDUCE;

    std::string str("...");
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(str));

    EXPECT_EQ(comm.LoadOpbasedCollOp(opParams, nullptr), HcclResult::HCCL_E_INTERNAL);
}

TEST(CommunicatorImplTest, LoadOpbasedCollOp_rankSize_1_test)
{
    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.rankSize = 1;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    comm.status = CommStatus::COMM_READY;
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

TEST(CommunicatorImplTest, should_throw_exception_when_ccu_and_aicpu_both_enabled)
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

TEST(CommunicatorImplTest, should_no_throw_exception_when_only_ccu)
{
    CommunicatorImpl comm;
    comm.collServices[AcceleratorState::CCU_MS]
        = std::make_unique<CollServiceDeviceMode>(&comm); // host 展开，图模式使用
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    EXPECT_NO_THROW(comm.SelectCollService());
}

// ok
TEST(CommunicatorImplTest, RecoverComm_NormalCase)
{
    // 打桩所有初始化函数，使其返回成功
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtOpenTsdProcess).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtRaTlvInit).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(1)));
    MOCKER(RaInit).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(RaTlvInit).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&CommunicatorImpl::RecoverRankGraphData).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
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
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    CommunicatorImpl comm;
    comm.RegisterAcceStateCallBack(CommunicatorCallback());
    CollServiceAiCpuImpl collService{&comm};
    SETUP_SNAPSHOT_COMM_PARAMS(collService);

    snapShotComm.rankTableInfo = ranktableInfo;
    snapShotComm.topoInfo = topoInfo;
    comm.myRank = 0;
    comm.rankGraph = make_unique<RankGraph>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);

    const char* filePath = "test";
    HcclResult result = comm.RecoverComm(snapShotComm, step, filePath);

    // 检查结果
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(comm.status, CommStatus::COMM_RESUMING);
    EXPECT_TRUE(comm.initFlag);
}

// ok
TEST(CommunicatorImplTest, RecoverComm_StdException)
{
    CommunicatorImpl comm;
    comm.initFlag = false;
    comm.status = CommStatus::COMM_IDLE;
    SnapShotComm snapShotComm;
    u32 step = 0;
    const char* filePath = "test";
    HcclResult result = comm.RecoverComm(snapShotComm, step, filePath);
}

// //OK
TEST(CommunicatorImplTest, RecoverSubComm_InitFlagTrue)
{
    CommunicatorImpl comm;
    comm.initFlag = true;
    SnapShotComm snapShotComm;
    u32 step = 0;

    const char* filePath = "test";
    HcclResult result = comm.RecoverComm(snapShotComm, step, filePath);
}

TEST(CommunicatorImplTest, RecoverComm_SubCommNormalCase)
{
    // 打桩所有初始化函数，使其返回成功
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtOpenTsdProcess).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtRaTlvInit).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(1)));
    MOCKER(RaInit).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(RaTlvInit).stubs().with(mockcpp::any()).will(returnValue(0));
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
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    CommunicatorImpl comm;
    comm.RegisterAcceStateCallBack(CommunicatorCallback());
    CollServiceAiCpuImpl collService{&comm};
    SETUP_SNAPSHOT_COMM_PARAMS(collService);

    snapShotComm.rankTableInfo = ranktableInfo;
    snapShotComm.topoInfo = topoInfo;
    comm.myRank = 0;
    comm.rankGraph = make_unique<RankGraph>(0);
    comm.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);

    const char* filePath = "test";
    HcclResult result = comm.RecoverComm(snapShotComm, step, filePath);

    // 检查结果
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(comm.status, CommStatus::COMM_RESUMING);
    EXPECT_TRUE(comm.initFlag);
}

TEST(CommunicatorImplTest, RecoverComm_SubComStdException)
{
    s32 myRank = 0;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(myRank);

    CommunicatorImpl comm;
    SnapShotSubComm snapShotComm;
    u32 step = 1;

    HcclResult result = comm.RecoverComm(snapShotComm, virtualTopo, step);
}

TEST(CommunicatorImplTest, RecoverComm_SubComInitFlagTrue)
{
    CommunicatorImpl comm;
    comm.initFlag = true;
    SnapShotSubComm snapShotComm;
    s32 myRank = 0;
    std::unique_ptr<RankGraph> virtualTopo = std::make_unique<RankGraph>(myRank);
    u32 step = 0;

    HcclResult result = comm.RecoverComm(snapShotComm, virtualTopo, step);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(comm.status, CommStatus::COMM_IDLE);
}

TEST(CommunicatorImplTest, RecoverSubComm_InitFlagFalse)
{
    unique_ptr<CommunicatorImpl> subCommImpl;
    CommunicatorImpl comm;
    SnapShotSubComm snapShotSubComm;
    snapShotSubComm.rankIds = {0, 1, 2};
    u32 step = 10;

    comm.initFlag = false;
    HcclResult result = comm.RecoverSubComm(snapShotSubComm, subCommImpl.get(), step);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(comm.status, CommStatus::COMM_IDLE);
}

TEST(CommunicatorImplTest, init_and_get_one_sided_service)
{
    CommunicatorImpl comm;
    comm.InitOneSidedService();
    HcclOneSidedService* service;
    comm.GetOneSidedService(&service);
}

TEST(CommunicatorImplTest, opbased_ccu_CheckOpDataTypeOpbase_should_success_when_opdatatype_is_supported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::SCATTER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    string tag = "tag";
    bool isAiv = false;
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, offload_ccu_CheckOpDataTypeOffload_should_success_when_opdatatype_is_supported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    string tag = "tag";
    bool isAiv = false;
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, opbased_aicpu_CheckOpDataTypeOpbase_should_success_when_opdatatype_is_supported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce
        = {OpType::ALLGATHER, OpType::SCATTER, OpType::BROADCAST, OpType::SEND, OpType::RECV};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,    DataType::INT16,   DataType::INT32,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32,  DataType::FP16,    DataType::FP32,   DataType::BFP16, DataType::HIF8,
           DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    string tag = "tag";
    bool isAiv = true;
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_return_success_when_check_datatype_aicpu_opbased_batchsendrecv)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<HcclDataType> datatypeWithoutReduce = {
        HcclDataType::HCCL_DATA_TYPE_INT8,   HcclDataType::HCCL_DATA_TYPE_INT16,   HcclDataType::HCCL_DATA_TYPE_INT32,
        HcclDataType::HCCL_DATA_TYPE_INT64,  HcclDataType::HCCL_DATA_TYPE_UINT8,   HcclDataType::HCCL_DATA_TYPE_UINT16,
        HcclDataType::HCCL_DATA_TYPE_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT64,  HcclDataType::HCCL_DATA_TYPE_FP16,
        HcclDataType::HCCL_DATA_TYPE_FP32,   HcclDataType::HCCL_DATA_TYPE_FP64,    HcclDataType::HCCL_DATA_TYPE_BFP16,
        HcclDataType::HCCL_DATA_TYPE_HIF8,   HcclDataType::HCCL_DATA_TYPE_FP8E4M3, HcclDataType::HCCL_DATA_TYPE_FP8E5M2,
        HcclDataType::HCCL_DATA_TYPE_FP8E8M0};

    bool isAiv = true;
    opParams.opType = OpType::BATCHSENDRECV;
    HcclSendRecvItem* sendRecvItemdata = nullptr;
    sendRecvItemdata = new HcclSendRecvItem[1];
    opParams.batchSendRecvDataDes.itemNum = 1;
    for (auto dtype : datatypeWithoutReduce) {
        sendRecvItemdata->dataType = dtype;
        opParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(sendRecvItemdata);
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    delete[] sendRecvItemdata;
}

TEST(CommunicatorImplTest, should_return_error_when_check_unsupported_datatype_aicpu_opbased_batchsendrecv)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<HcclDataType> datatypeWithoutReduce = {HcclDataType::HCCL_DATA_TYPE_INT128};
    bool isAiv = true;
    opParams.opType = OpType::BATCHSENDRECV;
    HcclSendRecvItem* sendRecvItemdata = nullptr;
    sendRecvItemdata = new HcclSendRecvItem[1];
    opParams.batchSendRecvDataDes.itemNum = 1;
    for (auto dtype : datatypeWithoutReduce) {
        sendRecvItemdata->dataType = dtype;
        opParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(sendRecvItemdata);
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_E_PARA);
    }
    delete[] sendRecvItemdata;
}

TEST(CommunicatorImplTest, offload_aicpu_CheckOpDataTypeOffload_should_success_when_opdatatype_is_supported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    string tag = "tag";
    bool isAiv = false;
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, offload_host_CheckOpDataTypeOffload_should_success_when_opdatatype_is_supported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceDefaultImpl collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    opParams.opType = OpType::ALLGATHER;
    comm.opExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CommunicatorImpl::ConvertCollOperatorA2A).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::ALLREDUCE, OpType::REDUCESCATTER};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::BROADCAST, OpType::SEND, OpType::RECV};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    string tag = "tag";
    bool isAiv = false;
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_SUCCESS);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, opbased_ccu_CheckOpDataTypeOpbase_should_throw_error_when_opdatatype_is_unsupported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::INT128;
    opParams.opType = OpType::ALLGATHER;
    string tag = "tag";
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    bool isAiv = true;
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::SCATTER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT64, DataType::UINT64,   DataType::UINT16,  DataType::UINT32,  DataType::FP64, DataType::INT128,
           DataType::HIF8,  DataType::BF16_SAT, DataType::FP8E4M3, DataType::FP8E5M2, DataType::UINT8};
    std::vector<DataType> datatypeWithoutReduce = {DataType::BF16_SAT};
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_E_PARA);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_E_PARA);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_E_PARA);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_E_PARA);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, opbased_aicpu_CheckOpDataTypeOpbase_should_throw_error_when_opdatatype_is_unsupported)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    comm.InitMirrorTaskManager();
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
    comm.InitProfilingReporter();

    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.status = CommStatus::COMM_READY;
    CollServiceAiCpuImpl collService{&comm};
    comm.collService = &collService;
    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::INT128;
    opParams.opType = OpType::ALLGATHER;
    string tag = "tag";
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce
        = {OpType::ALLGATHER, OpType::SEND, OpType::RECV, OpType::SCATTER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::UINT8, DataType::UINT16,   DataType::UINT32,  DataType::INT128,
           DataType::HIF8,  DataType::BF16_SAT, DataType::FP8E4M3, DataType::FP8E5M2};
    std::vector<DataType> datatypeWithoutReduce = {DataType::INT128, DataType::BF16_SAT};
    bool isAiv = true;
    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_E_PARA);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(
                    opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
                HcclResult::HCCL_E_PARA);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_E_PARA);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(
                opParams, comm.GetOpCcuFeatureFlag(), comm.GetOpAiCpuTSFeatureFlag(), isAiv),
            HcclResult::HCCL_E_PARA);
    }
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_suc_when_check_datatype_mc2_highP)
{
    Mc2CommConfig config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> optypeWithoutReduce = {
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALL),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALLV),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_HALF_ALLTOALLV)};
    std::vector<uint32_t> dataTypeHighP
        = {static_cast<uint32_t>(DataType::INT16), static_cast<uint32_t>(DataType::INT32),
           static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.dataType = dtype;
            config.outputDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2(config), HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.dataType = dtype;
            config.outputDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2(config), HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST(CommunicatorImplTest, should_suc_when_check_datatype_mc2_lowP)
{
    Mc2CommConfig config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> inputDataType
        = {static_cast<uint32_t>(DataType::INT8), static_cast<uint32_t>(DataType::FP8E5M2),
           static_cast<uint32_t>(DataType::FP8E4M3), static_cast<uint32_t>(DataType::HIF8)};
    std::vector<uint32_t> outputDataType
        = {static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : inputDataType) {
            config.dataType = dtype;
            for (auto outDtype : outputDataType) {
                config.outputDataType = outDtype;
                EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2(config), HcclResult::HCCL_SUCCESS);
            }
        }
    }
}

TEST(CommunicatorImplTest, should_fail_when_check_unsupported_datatype_mc2_lowP)
{
    Mc2CommConfig config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.dataType = static_cast<uint32_t>(DataType::INT16);
    config.outputDataType = static_cast<uint32_t>(DataType::FP16);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);

    config.dataType = static_cast<uint32_t>(DataType::INT8);
    config.outputDataType = static_cast<uint32_t>(DataType::INT32);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);
}

TEST(CommunicatorImplTest, should_fail_when_check_unsupported_datatype_mc2_highP)
{
    Mc2CommConfig config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.dataType = static_cast<uint32_t>(DataType::INT64);
    config.outputDataType = static_cast<uint32_t>(DataType::INT64);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);
}

TEST(CommunicatorImplTest, should_fail_when_check_unsupported_datatype_mc2_optype_without_reduce)
{
    Mc2CommConfig config;

    std::vector<uint32_t> optypeWithoutReduce = {
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALL),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALLV),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_HALF_ALLTOALLV)};

    for (auto optype : optypeWithoutReduce) {
        config.opType = optype;
        config.dataType = static_cast<uint32_t>(DataType::INT8);
        config.outputDataType = static_cast<uint32_t>(DataType::INT16);
        EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);
    }
}

TEST(CommunicatorImplTest, st_GetUsedChannelCount)
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

TEST(CommunicatorImplTest, st_PrintChannelInfoCallback)
{
    CommunicatorImpl comm;
    comm.printChannelInfoCallback = nullptr;
    comm.PrintChannelInfoCallback();

    comm.printChannelInfoCallback = []() {};
    comm.PrintChannelInfoCallback();
}

TEST(CommunicatorImplTest, ut_RefreshSubmittedOpcnt_1)
{
    CommunicatorImpl comm;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opType = OpType::ALLTOALLV;
    EXPECT_NO_THROW(comm.RefreshSubmittedOpcnt());
}

TEST(CommunicatorImplTest, st_should_success_when_GetSnapShotDynamicBuf)
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

TEST(CommunicatorImplTest, should_no_throw_exception_SetAccelerator_GetAccelerator)
{
    CommunicatorImpl comm;
    comm.rankSize = 1;
    comm.rankGraph = make_unique<RankGraph>(0);
    u32 level = 0;
    string groupId = "groupId";
    shared_ptr<NetInstance> fabGroup = make_shared<InnerNetInstance>(level, groupId);
    comm.rankGraph->netInsts_[level].emplace(groupId, fabGroup);

    comm.RegisterAcceStateCallBack(CommunicatorCallback());
    HcclAccelerator accelerator{HcclAccelerator::DEFAULT}; // DEFAULT
    bool isCcuMsAvailable = true;
    comm.SetAccelerator(accelerator, isCcuMsAvailable);

    accelerator = HcclAccelerator::HOSTCPU_TS; // HOSTCPU_TS
    isCcuMsAvailable = false;
    comm.SetAccelerator(accelerator, isCcuMsAvailable);

    accelerator = HcclAccelerator::AICPU_TS; // AICPU_TS
    isCcuMsAvailable = false;
    comm.SetAccelerator(accelerator, isCcuMsAvailable);

    accelerator = HcclAccelerator::AIV; // AIV
    isCcuMsAvailable = false;
    comm.SetAccelerator(accelerator, isCcuMsAvailable);

    accelerator = HcclAccelerator::AIV_ONLY; // AIV_ONLY
    isCcuMsAvailable = false;
    EXPECT_EQ(comm.SetAccelerator(accelerator, isCcuMsAvailable), HCCL_E_NOT_SUPPORT);

    accelerator = HcclAccelerator::CCU_MS; // CCU_MS
    isCcuMsAvailable = false;
    comm.SetAccelerator(accelerator, isCcuMsAvailable);

    accelerator = HcclAccelerator::CCU_SCHED; // CCU_SCHED
    isCcuMsAvailable = false;
    comm.SetAccelerator(accelerator, isCcuMsAvailable);

    accelerator = HcclAccelerator::AICPU; // AICPU
    isCcuMsAvailable = false;
    EXPECT_EQ(comm.SetAccelerator(accelerator, isCcuMsAvailable), HCCL_E_NOT_SUPPORT);

    comm.rankGraph = nullptr;
    accelerator = static_cast<HcclAccelerator::Value>(8); // other
    isCcuMsAvailable = true;
    EXPECT_EQ(comm.SetAccelerator(accelerator, isCcuMsAvailable), HCCL_E_NOT_SUPPORT);

    comm.rankGraph = make_unique<RankGraph>(0);
    accelerator = static_cast<HcclAccelerator::Value>(8); // other
    isCcuMsAvailable = false;
    EXPECT_EQ(comm.SetAccelerator(accelerator, isCcuMsAvailable), HCCL_E_NOT_SUPPORT);

    comm.commExecuteConfig.accState = AcceleratorState::HOSTCPU_TS;
    HcclResult ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(accelerator, 1);

    comm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(accelerator, 2);

    comm.commExecuteConfig.accState = AcceleratorState::AIV;
    ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(accelerator, 3);

    comm.commExecuteConfig.accState = AcceleratorState::CCU_MS;
    ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(accelerator, 5);

    comm.commExecuteConfig.accState = AcceleratorState::CCU_SCHED;
    ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(accelerator, 6);

    comm.commExecuteConfig.accState = AcceleratorState::AICPU;
    ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(accelerator, 7);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    comm.commExecuteConfig.accState = AcceleratorState::CCU_FALLBACK;
    ret = comm.GetAccelerator(&accelerator);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST(CommunicatorImplTest, St_GetCcuMc2ServerNum_When_CCU_SCHED_Expect_equality)
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

TEST(CommunicatorImplTest, St_CovertToCurrentCollOperator_When_AllGatherV)
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

TEST(CommunicatorImplTest, St_CovertToCurrentCollOperator_When_ReduceScatterV)
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
    using CurrentExecutorType = InsAllReduceSoleExecutor<TopoMatchMesh, CcuTempAllReduceMesh1DMultiMission>;
    using CurrentCcuInstructionType = CcuInstructionAllReduceMesh1DMultiMission;
    using CuurentCcuContextType = CcuContextAllReduceMesh1DMultiMission;

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

TEST(CommunicatorImplTest, St_CommunicatorImpl_When_EnableSuperFastLoad_Expect_LoadOpbasedCollOp_ReturnIsHCCL_SUCCESS)
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
    MOCKER(HrtGetStreamId).stubs().will(returnValue(fakeId));
    MOCKER(HrtGetDevice).stubs().will(returnValue(fakeDevLogId));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId)));
    MOCKER(HrtStreamGetSqId).stubs().will(returnValue(fakeSqId));
    MOCKER(HrtStreamDestroy).stubs();
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());

    MOCKER_CPP(&Hccl::MirrorTaskManager::AddTaskInfo).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .will(invoke(Orchestrate));
    SETUP_COMM_RESOURCE_MOCKS();
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
    fakeComm.status = CommStatus::COMM_READY;
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
    fakeComm.aivOffloadTagBuffer = DevBuffer::Create(0x100, 10);
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
    peer0->AddConnInterface(connInterface);
    fakeComm.rankGraph->AddPeer(peer0);
    fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
    fakeComm.trace = std::make_unique<Trace>();

    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    CollServiceDeviceMode collService{&fakeComm};
    fakeComm.collService = &collService;
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocOpMem).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AddPostToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::AddWaitToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x100000));
    std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl;

    std::shared_ptr<FakeAivCollAlgComponent> collAlgComponent = std::make_shared<FakeAivCollAlgComponent>();
    fakeComm.collAlgComponent = collAlgComponent;

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

    MOCKER_CPP(&SocketManager::BatchCreateSockets).stubs().with(mockcpp::any()).will(ignoreReturnValue());
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
    fakeComm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    EXPECT_EQ(
        fakeComm.GetAlgExecParam(opParams, clearEnable, commContext, len, aivCoreLimit),
        HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST(CommunicatorImplTest, st_Single_Rank_With_SendRecv_Expect_HCCL_SUCCESS)
{
    CommunicatorImpl comm;
    comm.status = CommStatus::COMM_READY;
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
