/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include "ccu_kernel_arg.h"
#include "ccu_kernel_signature.h"
#include "base.h"
#define private public
#include "ccu_task_param_v1.h"
#include "ccu_kernel.h"
#include "ccu_kernel_resource.h"
#include "ccu_rep_context_v1.h"
#include "ccu_task_arg_v1.h"
#include "string_util.h"
#include "task_param.h"
#include "channel_process.h"
#include "hcomm_c_adpt.h"
#include "local_notify_impl.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"
#include "hccl_api_base_test.h"
#include "ccu_kernel_mgr.h"
#include "hcomm_c_adpt.h"
#include "ccu_urma_channel.h"
#include "ccu_comp.h"
#include "ccuTaskException.h"
#include "hcclCommTaskException.h"
#include "ccu_rep_assign_v1.h"
#include "../../../../../src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "comms/ccu/ccu_representation/reps/data/ccu_rep_read_v1.h"
#include "comms/ccu/ccu_representation/reps/data/ccu_rep_write_v1.h"
#include "comms/ccu/ccu_representation/reps/data/ccu_rep_bufread_v1.h"
#include "comms/ccu/ccu_representation/reps/data/ccu_rep_bufwrite_v1.h"
#include "comms/ccu/ccu_representation/reps/sync/ccu_rep_loc_wait_event.h"
#include "comms/ccu/ccu_representation/reps/sync/ccu_rep_loc_wait_notify.h"

namespace hcomm {

constexpr size_t CCU_CTX_RAW_CAPACITY = 64;

// 模拟CcuUrmaChannel

class MockCcuUrmaChannel {
public:
    MockCcuUrmaChannel(uint16_t channelId = 101, uint32_t dieId = 0) : channelId_(channelId), dieId_(dieId) {}
    uint16_t GetChannelId() { return channelId_; }
    uint32_t GetDieId() { return dieId_; }

private:
    uint16_t channelId_;
    uint32_t dieId_;
};

// 桩函数。 模拟HcommChannelGet
static MockCcuUrmaChannel* g_mockChannel = nullptr;

HcclResult HcommChannelGet(ChannelHandle channel, void** channelPtr)
{
    if (channelPtr == nullptr)
        return HCCL_E_PTR;
    *channelPtr = g_mockChannel;
    return HCCL_SUCCESS;
}

// 桩函数，模拟SaveDfxTaskInfo
static HcclResult g_saveDfxRet = HCCL_SUCCESS;
HcclResult SaveDfxTaskInfo(const HcclComm comm, const Hccl::TaskParam& taskParam) { return g_saveDfxRet; }

void SetMockCcuUrmaChannel(MockCcuUrmaChannel* channel) { g_mockChannel = channel; }
void SetSaveDfxTaskInfoRet(HcclResult ret) { g_saveDfxRet = ret; }
void ResetStubs()
{
    g_mockChannel = nullptr;
    g_saveDfxRet = HCCL_SUCCESS;
}

// 测试子类
class TestCcuKernel : public CcuKernel {
public:
    using CcuKernel::CcuKernel;
    HcclResult Algorithm() override { return HCCL_SUCCESS; }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg&) override { return {1, 2, 3, 4}; }
    // 暴露私有成员用于验证（利用-fno-access-control编译选项，无需修改源码）
    HcclResult GeneTaskParam(const CcuTaskArg& arg, std::vector<CcuTaskParam>& taskParams)
    {
        return HcclResult::HCCL_SUCCESS;
    }

    void SetLoadArgIndex(uint32_t idx) { loadArgIndex_ = idx; }
    void SetMissionId(uint32_t id) { missionId = id; }
    void SetDieId(uint32_t id) { dieId = id; }
    uint32_t GetDieId() const { return dieId; }
    void Work()
    {
        std::shared_ptr<CcuRep::CcuRepBase> rep;
        allLgProfilingReps.push_back(rep);
    }
};

// 测试数据
class MockCcuKernelArg : public hcomm::CcuKernelArg {
public:
    CcuKernelSignature GetKernelSignature() const override { return CcuKernelSignature{}; }
};

class CcuKernelTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 1. 初始化桩函数
        hcomm::ResetStubs();
        // 2. 创建桩函数通道
        mockChannel_ = new MockCcuUrmaChannel(101, 0);
        hcomm::SetMockCcuUrmaChannel(mockChannel_);

        // 3. 构造CcuKernelArg
        kernelArg_.channels = {0x1001, 0x1002};

        // 4. 创建测试实例
        kernel_ = std::make_unique<TestCcuKernel>(kernelArg_);
        kernel_->SetDieId(0);
        kernel_->SetMissionId(100);
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
        // 清理桩函数
        delete mockChannel_;
        hcomm::ResetStubs();
    };
    // 修改 CcuKernelTest 类成员变量
    MockCcuKernelArg kernelArg_;
    std::unique_ptr<TestCcuKernel> kernel_;
    MockCcuUrmaChannel* mockChannel_ = nullptr;

    // 通用测试参数
    const hcomm::GroupInfo testGroupInfo{1, 2, 3};
    const std::vector<ChannelHandle> testChannelsVec{0x1001, 0x1002};
    const ChannelHandle testChannelsArr = 0x1001;
    const HcclDataType testDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    const HcclDataType testOutputDataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    const HcclReduceOp testReduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
};

class Ccukernel_ReportProfilingTest : public hcomm::CcuKernelTest {
protected:
    void SetUp() override
    {
        CcuKernelTest::SetUp();
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
        // 初始化测试数据
        testThreadHandle = 0x1000;
        testExecId = 0x2000;

        // 有效ProfilingInfo
        validProfInfo.dieId = 1;
        validProfInfo.missionId = 100;
        validProfInfo.instrId = 200;
        validProfInfo.channelId[0] = 101;
        validProfInfo.channelId[1] = 102;

        // 无效channelId的ProfilingInfo
        invalidChannelProfInfo.dieId = 2;
        invalidChannelProfInfo.missionId = 200;
        invalidChannelProfInfo.instrId = 300;
        invalidChannelProfInfo.channelId[0] = INVALID_VALUE_CHANNELID;
    }

    // 测试数据
    ThreadHandle testThreadHandle;
    uint64_t testExecId;
    CcuProfilingInfo validProfInfo;
    CcuProfilingInfo invalidChannelProfInfo;
};

TEST_F(CcuKernelTest, AddCcuProfilingInfo_Normal)
{
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel* channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    void* channelHandle = static_cast<void*>(channel);
    void** handle{nullptr};
    handle = &channelHandle;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    kernel_->Work();
    HcclResult ret = kernel_->AddCcuProfiling(
        testGroupInfo, testChannelsVec, testDataType, testOutputDataType, testReduceOp, "GroupReduce");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuKernelTest, AddProfilingInfo_Normal)
{
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel* channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    void* channelHandle = static_cast<void*>(channel);
    void** handle{nullptr};
    handle = &channelHandle;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    kernel_->Work();
    HcclResult ret
        = kernel_->AddCcuProfiling(&testChannelsArr, 1, testDataType, testOutputDataType, testReduceOp, "GroupReduce");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(Ccukernel_ReportProfilingTest, Ut_HcclReportAivKernel_Normal)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[128] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 6;
    config.hcclRdmaServiceLevel = 0;
    config.hcclRdmaTrafficClass = 0;
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ret = HcclReportAivKernel(comm, 12345);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(Ccukernel_ReportProfilingTest, Ut_HcclReportAicpuKernel_Normal)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommDfx::ReportKernel).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommDfx::GetOpModeFlags)
        .stubs()
        .with(outBound(false), outBound(false))
        .will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[128] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 6;
    config.hcclRdmaServiceLevel = 0;
    config.hcclRdmaTrafficClass = 0;
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    char kernelName[128] = "Hello";
    ret = HcclReportAicpuKernel(comm, 12345, kernelName);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuKernelTest, GetCcuProfilingInfo_EmptyCache)
{
    // Ensure profiling cache is empty
    kernel_->GetProfilingInfo().clear();

    std::vector<CcuProfilingInfo> out;
    CcuTaskArg arg{}; // default task arg
    HcclResult ret = kernel_->GetCcuProfilingInfo(arg, out);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(out.empty());
}

TEST_F(CcuKernelTest, GetCcuProfilingInfo_TaskProfilingInfo)
{
    // Prepare a task profiling entry in the kernel's profiling cache
    kernel_->GetProfilingInfo().clear();
    CcuProfilingInfo prof;
    prof.type = static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING);
    prof.name = "UnitTestTaskProfiling";
    prof.dieId = 1;
    prof.missionId = 0; // will be overwritten by GetCcuProfilingInfo
    prof.instrId = 0;   // will be overwritten by GetCcuProfilingInfo

    kernel_->GetProfilingInfo().push_back(prof);

    // Set expected mission and instr ids
    kernel_->SetMissionId(1234);
    kernel_->SetInstrId(4321);

    std::vector<CcuProfilingInfo> out;
    CcuTaskArg arg{};
    HcclResult ret = kernel_->GetCcuProfilingInfo(arg, out);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].missionId, 210);
    EXPECT_EQ(out[0].instrId, 4321u);
    EXPECT_EQ(out[0].name, "UnitTestTaskProfiling");
    EXPECT_EQ(out[0].dieId, 1u);
}

// +++++++++++++ CcuTaskException测试 +++++++++++++
class HcommDfxTaskExceptionTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 可以在这里设置一些全局状态或环境变量
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
        // 清理全局状态或环境变量
    }
};

TEST_F(HcommDfxTaskExceptionTest, ProccessCcuException_Normal)
{
    // 构造异常信息和任务信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.expandInfo.u.ccuInfo.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].status = 0x01;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].subStatus = 0x02;
    uint8_t panicLog[128] = {}; // 模拟panic日志内容
    std::memcpy(exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].panicLog, panicLog, sizeof(panicLog));

    hccl::TaskPara a;
    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo{1, 2, 3, taskParam, nullptr, true};
    MOCKER_CPP(&CcuComponent::CleanTaskKillState).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::CleanDieCkes).stubs().will(returnValue(HCCL_SUCCESS));

    // 调用处理函数
    CcuTaskException::ProcessCcuException(&exceptionInfo, taskInfo);

    // 验证输出
    EXPECT_TRUE(true);
}

TEST_F(HcommDfxTaskExceptionTest, GetGroupRankInfo_Normal)
{
    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    hccl::ManagerCallbacks callbacks;
    callbacks.getAicpuCommState = []() {
        return true;
    };
    callbacks.setAicpuCommState = [](bool) {};
    callbacks.kernelLaunchAicpuCommInit = []() {
        return HCCL_SUCCESS;
    };
    std::string commName = "TestComm";
    hccl::CollComm collComm(nullptr, 1, commName, callbacks);
    taskInfo.dfxOpInfo_->comm_ = &collComm;
    std::string testCommStr = "TestComm";
    MOCKER_CPP(&hccl::CollComm::GetCommId).stubs().will(returnValue(testCommStr));
    MOCKER_CPP(&hccl::CollComm::GetRankSize).stubs().will(returnValue(8u));
    MOCKER_CPP(&hccl::CollComm::GetMyRankId).stubs().will(returnValue(3u));
    std::string result = CcuTaskException::GetGroupRankInfo(taskInfo);
    EXPECT_NE(result, "");
    EXPECT_TRUE(result.find("group:[TestComm]") != std::string::npos);
    EXPECT_TRUE(result.find("rankSize[8]") != std::string::npos);
    EXPECT_TRUE(result.find("rankId[3]") != std::string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GetGroupRankInfo_Nullptr)
{
    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);
    taskInfo.dfxOpInfo_ = nullptr;
    std::string result = CcuTaskException::GetGroupRankInfo(taskInfo);
    EXPECT_EQ(result, "");
}

struct CcumDfxInfoForTest {
    unsigned int queryResult; // 0:success, 1:fail
    unsigned int ccumSqeRecvCnt;
    unsigned int ccumSqeSendCnt;
    unsigned int ccumMissionDfx;
    unsigned int ccumSqeDropCnt;
    unsigned int ccumSqeAddrLenErrDropCnt;
    unsigned int lqcCcuSecReg0;
    unsigned int ccumTifSqeCnt;
    unsigned int ccumTifCqeCnt;
    unsigned int ccumCifSqeCnt;
    unsigned int ccumCifCqeCnt;
};

TEST_F(HcommDfxTaskExceptionTest, PrintPanicLogInfo_Normal)
{
    uint8_t panicLog[128] = {};
    struct CcumDfxInfoForTest* info = reinterpret_cast<struct CcumDfxInfoForTest*>(panicLog);
    info->queryResult = 0; // success
    info->ccumSqeRecvCnt = 100;
    info->ccumSqeSendCnt = 200;
    info->ccumMissionDfx = 300;
    info->ccumTifSqeCnt = 400;
    info->ccumTifCqeCnt = 500;
    info->ccumCifSqeCnt = 600;
    info->ccumCifCqeCnt = 700;
    info->ccumSqeDropCnt = 800;
    info->ccumSqeAddrLenErrDropCnt = 900;
    info->lqcCcuSecReg0 = 1; // enable

    // 调用函数
    EXPECT_NO_THROW(CcuTaskException::PrintPanicLogInfo(panicLog));
}

TEST_F(HcommDfxTaskExceptionTest, PrintPaniclogInfo) { EXPECT_NO_THROW(CcuTaskException::PrintPanicLogInfo(nullptr)); }

TEST_F(HcommDfxTaskExceptionTest, GetCcuCKEValue_Normal)
{
    // 模拟hrtGetDevicePhyIdByIndex
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));

    // 模拟HccpRaCustomChannel
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_SUCCESS));

    uint16_t result = CcuTaskException::GetCcuCKEValue(0, 0, 0);
    EXPECT_EQ(result, 0); // INVALID_U16
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuCKEValue_GetDevicePhyIdFail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_E_PARA));

    uint16_t result = CcuTaskException::GetCcuCKEValue(0, 0, 0);
    EXPECT_EQ(result, 65535);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuCKEValue_CustomChannelFail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_E_PARA));

    uint16_t result = CcuTaskException::GetCcuCKEValue(0, 0, 0);
    EXPECT_EQ(result, 65535);
}

TEST_F(HcommDfxTaskExceptionTest, GetMissContectF)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_SUCCESS));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, CCU_CTX_RAW_CAPACITY);
    EXPECT_EQ(raw[0], 0u);
}

TEST_F(HcommDfxTaskExceptionTest, GetDevidFail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_E_PARA));

    uint8_t raw[CCU_CTX_RAW_CAPACITY];
    (void)memset_s(raw, sizeof(raw), 0xAA, sizeof(raw));
    size_t copiedLen = 123;
    EXPECT_EQ(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_E_PARA);
    EXPECT_EQ(copiedLen, 0U);
    EXPECT_EQ(raw[0], 0u);
}

TEST_F(HcommDfxTaskExceptionTest, GetMissContectFail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_E_PARA));

    uint8_t raw[CCU_CTX_RAW_CAPACITY];
    (void)memset_s(raw, sizeof(raw), 0xBB, sizeof(raw));
    size_t copiedLen = 123;
    EXPECT_EQ(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_E_PARA);
    EXPECT_EQ(copiedLen, 0U);
    EXPECT_EQ(raw[0], 0u);
}

constexpr uint64_t INVALID_U64_VAL = 18446744073709551615ULL;
TEST_F(HcommDfxTaskExceptionTest, GetCcuGSAValue)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_SUCCESS));

    uint64_t result = CcuTaskException::GetCcuGSAValue(0, 0, 0);
    EXPECT_EQ(result, 0u);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuGSAValuegetdevidfailed)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuGSAValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuGSAValuechannelfail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuGSAValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuXnValue)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_SUCCESS));

    uint64_t result = CcuTaskException::GetCcuXnValue(0, 0, 0);
    EXPECT_EQ(result, 0u);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuXnValuedevidfailed)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuXnValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuXnValuechannelfail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuXnValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsg)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(1024);
    EXPECT_EQ(result, "");
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgzero)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgDefault)
{
    CcuErrorInfo ccuErrorInfo = {};
    string result = CcuTaskException::GetCcuErrorMsgDefault(ccuErrorInfo);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgMission)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::MISSION;
    string result = CcuTaskException::GetCcuErrorMsgMission(ccuErrorInfo);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuLoopContext_Normal)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaCustomChannel).stubs().with(any(), any(), any(), any()).will(returnValue(HCCL_SUCCESS));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, CCU_CTX_RAW_CAPACITY);
    EXPECT_EQ(raw[0], 0u);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuLoopContext_GetDevicePhyIdFail)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any()).will(returnValue(HCCL_E_PARA));
    uint8_t raw[CCU_CTX_RAW_CAPACITY];
    (void)memset_s(raw, sizeof(raw), 0xCC, sizeof(raw));
    size_t copiedLen = 123;
    EXPECT_EQ(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_E_PARA);
    EXPECT_EQ(copiedLen, 0U);
    EXPECT_EQ(raw[0], 0u);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_Normal)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 1;
    baseInfo.currentInsId = 0;
    baseInfo.status = 0x0102;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
}

TEST_F(HcommDfxTaskExceptionTest, GetRankIdByChannelId_Normal)
{
    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    RankId rankId = CcuTaskException::GetRankIdByChannelId(101, taskInfo, 0);

    // 验证输出
    EXPECT_TRUE(true);
}

TEST_F(HcommDfxTaskExceptionTest, GetAddrPairByChannelId_Normal)
{
    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    auto addrPair = CcuTaskException::GetAddrPairByChannelId(101, taskInfo, 0);

    EXPECT_TRUE(true);
}

TEST_F(HcommDfxTaskExceptionTest, GetMSIdPerDie_Normal)
{
    uint16_t result1 = CcuTaskException::GetMSIdPerDie(0x8001);
    EXPECT_EQ(result1, 0x0001);

    uint16_t result2 = CcuTaskException::GetMSIdPerDie(0x7FFF);
    EXPECT_EQ(result2, 0x7FFF);

    uint16_t result3 = CcuTaskException::GetMSIdPerDie(0);
    EXPECT_EQ(result3, 0);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_UnsupportedOpcode)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 1;
    baseInfo.currentInsId = 0;
    baseInfo.status = 0x0100;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_TRUE(string(errorInfo[0].msg.mission.missionError).find("Unsupported Opcode") != string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_LocalOpError)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 1;
    baseInfo.missionId = 2;
    baseInfo.currentInsId = 5;
    baseInfo.status = 0x0201;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_TRUE(string(errorInfo[0].msg.mission.missionError).find("Local Length Error") != string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_RemoteOpError)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 3;
    baseInfo.currentInsId = 10;
    baseInfo.status = 0x0301;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_TRUE(string(errorInfo[0].msg.mission.missionError).find("Remote Unsupported Request") != string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_TransactionRetryExceeded)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 4;
    baseInfo.currentInsId = 15;
    baseInfo.status = 0x0400;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_TRUE(
        string(errorInfo[0].msg.mission.missionError).find("Transaction Retry Counter Exceeded") != string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_TransactionAckTimeout)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 5;
    baseInfo.currentInsId = 20;
    baseInfo.status = 0x0500;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_JettyWorkRequestFlushed)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 6;
    baseInfo.currentInsId = 25;
    baseInfo.status = 0x0600;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_CCUAAlgTaskError)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 7;
    baseInfo.currentInsId = 30;
    baseInfo.status = 0x0700;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_MemoryECCError)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 8;
    baseInfo.currentInsId = 35;
    baseInfo.status = 0x0800;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_CCUMExecuteError)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 9;
    baseInfo.currentInsId = 40;
    baseInfo.status = 0x0901;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_TRUE(string(errorInfo[0].msg.mission.missionError).find("SQE instr and key not match") != string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_CCUAExecuteError)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 10;
    baseInfo.currentInsId = 45;
    baseInfo.status = 0x0A02;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_TRUE(string(errorInfo[0].msg.mission.missionError).find("SLVERR") != string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_HighPartNotInMap)
{
    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.missionId = 11;
    baseInfo.currentInsId = 50;
    baseInfo.status = 0x1100;

    vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenStatusInfo(baseInfo, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLoop_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::LOOP;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOOP;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 10;
    ccuErrorInfo.msg.loop.startInstrId = 0;
    ccuErrorInfo.msg.loop.endInstrId = 100;
    ccuErrorInfo.msg.loop.loopEngineId = 5;
    ccuErrorInfo.msg.loop.loopCnt = 10;
    ccuErrorInfo.msg.loop.loopCurrentCnt = 5;
    ccuErrorInfo.msg.loop.addrStride = 256;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLoop(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLoopGroup_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::LOOP_GROUP;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOOPGROUP;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 20;
    ccuErrorInfo.msg.loopGroup.startLoopInsId = 0;
    ccuErrorInfo.msg.loopGroup.loopInsCnt = 8;
    ccuErrorInfo.msg.loopGroup.expandOffset = 1;
    ccuErrorInfo.msg.loopGroup.expandCnt = 4;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLoopGroup(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLocPostSem_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOC_RECORD_EVENT;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 5;
    ccuErrorInfo.msg.waitSignal.signalId = 10;
    ccuErrorInfo.msg.waitSignal.signalMask = 0xFF;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xF0;
    ccuErrorInfo.msg.waitSignal.paramId = 1;
    ccuErrorInfo.msg.waitSignal.paramValue = 12345;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLocPostSem(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLocWaitEvent_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOC_WAIT_EVENT;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 6;
    ccuErrorInfo.msg.waitSignal.signalId = 20;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0F;
    ccuErrorInfo.msg.waitSignal.signalValue = 0x05;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLocWaitEvent(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLocWaitNotify_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOC_WAIT_NOTIFY;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 7;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLocWaitNotify(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgRemPostSem_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::REM_POST_SEM;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 8;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgRemPostSem(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgRemWaitSem_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::REM_WAIT_SEM;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 9;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgRemWaitSem(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgRemPostVar_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::REM_POST_VAR;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 11;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgRemPostVar(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgPostSharedSem_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRep::CcuRepType::RECORD_SHARED_NOTIFY;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 12;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgPostSharedSem(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgRead_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::READ;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 13;
    ccuErrorInfo.msg.transMem.locAddr = 0x1000;
    ccuErrorInfo.msg.transMem.locToken = 0x1;
    ccuErrorInfo.msg.transMem.rmtAddr = 0x2000;
    ccuErrorInfo.msg.transMem.rmtToken = 0x2;
    ccuErrorInfo.msg.transMem.len = 1024;
    ccuErrorInfo.msg.transMem.channelId = 101;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgRead(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgWrite_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::WRITE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 14;
    ccuErrorInfo.msg.transMem.locAddr = 0x1000;
    ccuErrorInfo.msg.transMem.locToken = 0x1;
    ccuErrorInfo.msg.transMem.rmtAddr = 0x2000;
    ccuErrorInfo.msg.transMem.rmtToken = 0x2;
    ccuErrorInfo.msg.transMem.len = 2048;
    ccuErrorInfo.msg.transMem.channelId = 102;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgWrite(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLocalCpy_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOCAL_CPY;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 15;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLocalCpy(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgLocalReduce_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::LOCAL_REDUCE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 16;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgLocalReduce(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgBufRead_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BUF_READ;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 17;
    ccuErrorInfo.msg.bufTransMem.bufId = 5;
    ccuErrorInfo.msg.bufTransMem.addr = 0x3000;
    ccuErrorInfo.msg.bufTransMem.len = 512;
    ccuErrorInfo.msg.bufTransMem.channelId = 103;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgBufRead(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgBufWrite_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BUF_WRITE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 18;
    ccuErrorInfo.msg.bufTransMem.bufId = 6;
    ccuErrorInfo.msg.bufTransMem.addr = 0x4000;
    ccuErrorInfo.msg.bufTransMem.len = 256;
    ccuErrorInfo.msg.bufTransMem.channelId = 104;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgBufWrite(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgBufLocRead_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BUF_LOC_READ;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 19;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgBufLocRead(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgBufLocWrite_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::BUF_TRANS_MEM;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BUF_LOC_WRITE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 21;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgBufLocWrite(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgBufReduce_DefaultValues)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::BUF_REDUCE;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BUF_REDUCE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 22;
    ccuErrorInfo.msg.bufReduce.count = 4;
    ccuErrorInfo.msg.bufReduce.dataType = 1;
    ccuErrorInfo.msg.bufReduce.outputDataType = 0;
    ccuErrorInfo.msg.bufReduce.opType = 0;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgBufReduce(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgByType_MISSION)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::MISSION;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BASE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 0;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgByType(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgByType_DefaultType)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::DEFAULT;
    ccuErrorInfo.repType = CcuRep::CcuRepType::BASE;
    ccuErrorInfo.dieId = 0;
    ccuErrorInfo.missionId = 1;
    ccuErrorInfo.instrId = 0;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

    string result = CcuTaskException::GetCcuErrorMsgByType(ccuErrorInfo, taskInfo, 0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuLenErrorMsg_LargeValue)
{
    constexpr uint64_t CCU_MSG_256MB_LEN = 256 * 1024 * 1024;
    string result = CcuTaskException::GetCcuLenErrorMsg(CCU_MSG_256MB_LEN);
    EXPECT_TRUE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuLenErrorMsg_NormalValue)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(1024 * 1024);
    EXPECT_TRUE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuLenErrorMsg_Exceed256MB)
{
    constexpr uint64_t CCU_MSG_256MB_LEN = 256 * 1024 * 1024;
    uint64_t exceedLen = static_cast<uint64_t>(CCU_MSG_256MB_LEN) + 1;
    string result = CcuTaskException::GetCcuLenErrorMsg(exceedLen);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, GetCcuLenErrorMsg_Zero)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(0);
    EXPECT_FALSE(result.empty());
}

TEST_F(HcommDfxTaskExceptionTest, ProcessCcuException_MultipleMission)
{
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.expandInfo.u.ccuInfo.ccuMissionNum = 2;
    for (uint32_t i = 0; i < 2; ++i) {
        exceptionInfo.expandInfo.u.ccuInfo.missionInfo[i].status = 0x01;
        exceptionInfo.expandInfo.u.ccuInfo.missionInfo[i].subStatus = 0x02;
    }

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, true);
    MOCKER_CPP(&CcuComponent::CleanTaskKillState).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::CleanDieCkes).stubs().will(returnValue(HCCL_SUCCESS));

    CcuTaskException::ProcessCcuException(&exceptionInfo, taskInfo);
    EXPECT_TRUE(true);
}

TEST_F(HcommDfxTaskExceptionTest, ProcessCcuException_CleanTaskKillStateFail)
{
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.expandInfo.u.ccuInfo.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].status = 0x01;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].subStatus = 0x00;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, true);
    MOCKER_CPP(&CcuComponent::CleanTaskKillState).stubs().will(returnValue(HCCL_E_PARA));
    MOCKER_CPP(&CcuComponent::CleanDieCkes).stubs().will(returnValue(HCCL_SUCCESS));

    CcuTaskException::ProcessCcuException(&exceptionInfo, taskInfo);
    EXPECT_TRUE(true);
}

TEST_F(HcommDfxTaskExceptionTest, ProcessCcuException_CleanDieCkesFail)
{
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.expandInfo.u.ccuInfo.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].status = 0x01;
    exceptionInfo.expandInfo.u.ccuInfo.missionInfo[0].subStatus = 0x00;

    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, true);
    MOCKER_CPP(&CcuComponent::CleanTaskKillState).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::CleanDieCkes).stubs().will(returnValue(HCCL_E_PARA));

    CcuTaskException::ProcessCcuException(&exceptionInfo, taskInfo);
    EXPECT_TRUE(true);
}

TEST_F(HcommDfxTaskExceptionTest, GetGroupRankInfo_WithValidComm)
{
    Hccl::ParaCcu paraCcu = {};
    Hccl::TaskParam taskParam
        = {.taskType = Hccl::TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .isMaster = false,
           .taskPara = {.Ccu = paraCcu},
           .ccuDetailInfo = nullptr};
    Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);
    taskInfo.dfxOpInfo_ = std::make_shared<Hccl::DfxOpInfo>();
    hccl::ManagerCallbacks callbacks;
    callbacks.getAicpuCommState = []() {
        return true;
    };
    callbacks.setAicpuCommState = [](bool) {};
    callbacks.kernelLaunchAicpuCommInit = []() {
        return HCCL_SUCCESS;
    };
    std::string commName = "TestCommGroup";
    hccl::CollComm collComm(nullptr, 1, commName, callbacks);
    taskInfo.dfxOpInfo_->comm_ = &collComm;
    MOCKER_CPP(&hccl::CollComm::GetCommId).stubs().will(returnValue(commName));
    MOCKER_CPP(&hccl::CollComm::GetRankSize).stubs().will(returnValue(16u));
    MOCKER_CPP(&hccl::CollComm::GetMyRankId).stubs().will(returnValue(7u));

    std::string result = CcuTaskException::GetGroupRankInfo(taskInfo);
    EXPECT_NE(result, "");
    EXPECT_TRUE(result.find("group:[TestCommGroup]") != std::string::npos);
    EXPECT_TRUE(result.find("rankSize[16]") != std::string::npos);
    EXPECT_TRUE(result.find("rankId[7]") != std::string::npos);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_ExceptionInfo_Nullptr)
{
    rtExceptionInfo_t* exceptionInfo = nullptr;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(exceptionInfo));
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_NotFound_DeviceId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.DestroyQueue(0, 0);
    globalMirrorTasks.DestroyQueue(0, 1);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_NotFound_StreamId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);
    globalMirrorTasks.DestroyQueue(0, 1);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 1;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_NotFound_TaskId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, 1, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_DfxOpInfo_Nullptr)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, 0, 0, taskParam, nullptr);
    globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_IsIndop_False)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = false;
    shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_TaskType_TaskCCU)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_TaskType_NotCCU)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_Queue_Nullptr)
{
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::PrintTaskContextInfo(0, 0, 0));
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_Task_Not_Found)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);

    EXPECT_NO_THROW(hcomm::TaskExceptionHost::PrintTaskContextInfo(0, 0, 0));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_TaskId_Greater_Then_ErrTaskId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    for (u32 i = 0; i < 5; i++) {
        Hccl::TaskParam taskParam{};
        taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
        shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
        dfxOpInfo->isIndop_ = true;
        shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, i + 10, 0, taskParam, dfxOpInfo);
        globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);
    }

    EXPECT_NO_THROW(hcomm::TaskExceptionHost::PrintTaskContextInfo(0, 0, 5));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(HcommDfxTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_Success)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    for (u32 i = 0; i < 5; i++) {
        Hccl::TaskParam taskParam{};
        taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
        shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
        dfxOpInfo->isIndop_ = true;
        shared_ptr<Hccl::TaskInfo> taskInfo = make_shared<Hccl::TaskInfo>(0, i, 0, taskParam, dfxOpInfo);
        globalMirrorTasks.GetQueue(0, 0)->Append(taskInfo);
    }

    EXPECT_NO_THROW(hcomm::TaskExceptionHost::PrintTaskContextInfo(0, 0, 4));

    globalMirrorTasks.DestroyQueue(0, 0);
}

namespace CcuRep {

    class MockCcuRepBase : public CcuRepBase {
    public:
        MockCcuRepBase(CcuRepType type = CcuRepType::BASE, uint16_t startId = 0, uint16_t count = 1)
            : type_(type),
              startId_(startId),
              count_(count)
        {}

        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override { return true; }

        std::string Describe() override { return std::string("MockCcuRepBase"); }

        uint16_t StartInstrId() const { return startId_; }

        uint16_t InstrCount() override { return count_; }

        CcuRepType Type() const { return type_; }

    private:
        CcuRepType type_;
        uint16_t startId_;
        uint16_t count_;
    };

} // namespace CcuRep

class CcuRepContextTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mockChannel_ = new MockCcuUrmaChannel(101, 0);
        g_mockChannel = mockChannel_;
    }

    void TearDown() override
    {
        delete mockChannel_;
        g_mockChannel = nullptr;
        GlobalMockObject::verify();
    }

    MockCcuUrmaChannel* mockChannel_;
};

TEST_F(CcuRepContextTest, Constructor_Normal)
{
    CcuRep::CcuRepContext context;
    EXPECT_NE(context.CurrentBlock(), nullptr);
}

TEST_F(CcuRepContextTest, CurrentBlock_Normal)
{
    CcuRep::CcuRepContext context;
    auto block = context.CurrentBlock();
    EXPECT_NE(block, nullptr);
}

TEST_F(CcuRepContextTest, SetCurrentBlock_Normal)
{
    CcuRep::CcuRepContext context;
    auto newBlock = std::make_shared<CcuRep::CcuRepBlock>("test_block");
    context.SetCurrentBlock(newBlock);
    EXPECT_EQ(context.CurrentBlock(), newBlock);
}

TEST_F(CcuRepContextTest, Append_Normal)
{
    CcuRep::CcuRepContext context;
    auto rep = std::make_shared<CcuRep::MockCcuRepBase>(CcuRep::CcuRepType::BASE);
    context.Append(rep);
    auto& reps = context.GetRepSequence();
    EXPECT_EQ(reps.size(), 1);
    EXPECT_EQ(reps[0], rep);
}

TEST_F(CcuRepContextTest, GetRepSequence_Empty)
{
    CcuRep::CcuRepContext context;
    auto& reps = context.GetRepSequence();
    EXPECT_EQ(reps.size(), 0);
}

TEST_F(CcuRepContextTest, GetRepByInstrId_NotFound)
{
    CcuRep::CcuRepContext context;
    auto rep = std::make_shared<CcuRep::MockCcuRepBase>(CcuRep::CcuRepType::BASE, 10, 5);
    context.Append(rep);

    auto foundRep = context.GetRepByInstrId(20);
    EXPECT_EQ(foundRep, nullptr);
}

TEST_F(CcuRepContextTest, SetDieId_GetDieId)
{
    CcuRep::CcuRepContext context;
    uint32_t dieId = 5;
    context.SetDieId(dieId);
    EXPECT_EQ(context.GetDieId(), dieId);
}

TEST_F(CcuRepContextTest, SetMissionId_GetMissionId)
{
    CcuRep::CcuRepContext context;
    uint32_t missionId = 100;
    context.SetMissionId(missionId);
    EXPECT_EQ(context.GetMissionId(), missionId);
}

TEST_F(CcuRepContextTest, SetMissionId_OnlyOnce)
{
    CcuRep::CcuRepContext context;
    context.SetMissionId(100);
    context.SetMissionId(200);
    EXPECT_EQ(context.GetMissionId(), 100);
}

TEST_F(CcuRepContextTest, SetMissionKey_GetMissionKey)
{
    CcuRep::CcuRepContext context;
    uint32_t missionKey = 300;
    context.SetMissionKey(missionKey);
    EXPECT_EQ(context.GetMissionKey(), missionKey);
}

TEST_F(CcuRepContextTest, GetProfilingInfo_Empty)
{
    CcuRep::CcuRepContext context;
    auto& profilingInfo = context.GetProfilingInfo();
    EXPECT_EQ(profilingInfo.size(), 0);
}

TEST_F(CcuRepContextTest, GetLGProfilingInfo_Empty)
{
    CcuRep::CcuRepContext context;
    auto& lgProfilingInfo = context.GetLGProfilingInfo();
    EXPECT_EQ(lgProfilingInfo.ccuProfilingInfos.size(), 0);
}

TEST_F(CcuRepContextTest, GetWaiteCkeProfilingReps_Empty)
{
    CcuRep::CcuRepContext context;
    auto& waitReps = context.GetWaiteCkeProfilingReps();
    EXPECT_EQ(waitReps.size(), 0);
}

TEST_F(CcuRepContextTest, CollectProfilingReps_AssignVarToVar)
{
    CcuRep::CcuRepContext context;
    auto rep = std::make_shared<CcuRep::CcuRepAssign>(CcuRep::Variable(&context), CcuRep::Variable(&context));
    rep->subType = CcuRep::AssignSubType::VAR_TO_VAR;
    context.CollectProfilingReps(rep);

    auto& profilingInfo = context.GetLGProfilingInfo();
    EXPECT_EQ(profilingInfo.assignProfilingReps.size(), 1);
}

TEST_F(CcuRepContextTest, AddSqeProfiling_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    context.AddSqeProfiling();

    auto& profilingInfo = context.GetProfilingInfo();
    EXPECT_EQ(profilingInfo.size(), 1);
    EXPECT_EQ(profilingInfo[0].type, (uint8_t)CcuProfilinType::CCU_TASK_PROFILING);
    EXPECT_EQ(profilingInfo[0].name, "CCU_KERNEL");
    EXPECT_EQ(profilingInfo[0].dieId, 1);
}

TEST_F(CcuRepContextTest, AddProfiling_WithNameAndMask)
{
    CcuRep::CcuRepContext context;
    std::string name = "TestProfiling";
    uint32_t mask = 0xFF;
    int32_t ret = context.AddProfiling(name, mask);

    EXPECT_EQ(ret, 0);
    auto& profilingInfo = context.GetProfilingInfo();
    EXPECT_EQ(profilingInfo.size(), 1);
    EXPECT_EQ(profilingInfo[0].name, name);
    EXPECT_EQ(profilingInfo[0].mask, mask);
}

TEST_F(CcuRepContextTest, AddProfiling_WithChannelsNullPtr)
{
    CcuRep::CcuRepContext context;
    int32_t ret = context.AddProfiling(nullptr, 2);
    EXPECT_NE(ret, 0);
}
// ========== GenErrorInfo系列函数测试 ==========

class GenErrorInfoTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(HccpRaTlvRequestForCustomChannel)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        baseInfo_.dieId = 0;
        baseInfo_.deviceId = 0;
        baseInfo_.missionId = 1;
        baseInfo_.currentInsId = 10;
        baseInfo_.status = 0x0101;
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
    ErrorInfoBase baseInfo_;
};

class MockTaskCcuRepBase : public CcuRep::CcuRepBase {
public:
    MockTaskCcuRepBase(CcuRep::CcuRepType type, uint16_t instrId = 10)
    {
        this->type = type;
        this->instrId = instrId;
    }
    ~MockTaskCcuRepBase() override = default;
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "MockRep"; }
    CcuRep::CcuRepType GetType() const { return type; }
};

class MockCcuRepLocRecordEvent : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocRecordEvent(uint16_t eventId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::LOC_RECORD_EVENT;
        this->instrId = 10;
        eventId_ = eventId;
        mask_ = mask;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "LocRecordEvent"; }
    uint16_t GetEventId() { return eventId_; }
    uint32_t GetMask() { return mask_; }

private:
    uint16_t eventId_;
    uint32_t mask_;
};

class MockCcuRepLocWaitEvent : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocWaitEvent(uint16_t eventId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::LOC_WAIT_EVENT;
        this->instrId = 11;
        eventId_ = eventId;
        mask_ = mask;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "LocWaitEvent"; }
    uint16_t GetEventId() { return eventId_; }
    uint32_t GetMask() { return mask_; }

private:
    uint16_t eventId_;
    uint32_t mask_;
};

class MockCcuRepRemPostSem : public CcuRep::CcuRepBase {
public:
    MockCcuRepRemPostSem(uint32_t id, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::REM_POST_SEM;
        this->instrId = 13;
        id_ = id;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "RemPostSem"; }
    uint32_t GetId() { return id_; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint32_t id_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepRemWaitSem : public CcuRep::CcuRepBase {
public:
    MockCcuRepRemWaitSem(uint32_t id, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::REM_WAIT_SEM;
        this->instrId = 14;
        id_ = id;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "RemWaitSem"; }
    uint32_t GetId() { return id_; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint32_t id_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepRemPostVar : public CcuRep::CcuRepBase {
public:
    MockCcuRepRemPostVar(
        uint32_t rmtCkeId, uint32_t rmtXnId, uint32_t paramIndex, uint32_t semIndex, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::REM_POST_VAR;
        this->instrId = 15;
        rmtCkeId_ = rmtCkeId;
        rmtXnId_ = rmtXnId;
        paramIndex_ = paramIndex;
        semIndex_ = semIndex;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "RemPostVar"; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetRmtCkeId() { return rmtCkeId_; }
    uint32_t GetRmtXnId() { return rmtXnId_; }
    uint32_t GetParamIndex() { return paramIndex_; }
    CcuRep::Variable GetParam()
    {
        CcuRep::Variable v;
        v.Reset(100);
        return v;
    }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint32_t rmtCkeId_;
    uint32_t rmtXnId_;
    uint32_t paramIndex_;
    uint32_t semIndex_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepRecordSharedNotify : public CcuRep::CcuRepBase {
public:
    MockCcuRepRecordSharedNotify(uint16_t notifyId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::RECORD_SHARED_NOTIFY;
        this->instrId = 16;
        notifyId_ = notifyId;
        mask_ = mask;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "RecordSharedNotify"; }
    uint32_t GetMask() { return mask_; }
    uint16_t GetNotifyId() { return notifyId_; }

private:
    uint16_t notifyId_;
    uint32_t mask_;
};

class MockCcuRepRead : public CcuRep::CcuRepBase {
public:
    MockCcuRepRead(
        uint16_t locAddrId, uint16_t locTokenId, uint16_t remAddrId, uint16_t remTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::READ;
        this->instrId = 17;
        locAddrId_ = locAddrId;
        locTokenId_ = locTokenId;
        remAddrId_ = remAddrId;
        remTokenId_ = remTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "Read"; }
    uint16_t GetLocAddrId() { return locAddrId_; }
    uint16_t GetLocTokenId() { return locTokenId_; }
    uint16_t GetRemAddrId() { return remAddrId_; }
    uint16_t GetRemTokenId() { return remTokenId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint16_t locAddrId_;
    uint16_t locTokenId_;
    uint16_t remAddrId_;
    uint16_t remTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepWrite : public CcuRep::CcuRepBase {
public:
    MockCcuRepWrite(
        uint16_t locAddrId, uint16_t locTokenId, uint16_t remAddrId, uint16_t remTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask, uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::WRITE;
        this->instrId = 18;
        locAddrId_ = locAddrId;
        locTokenId_ = locTokenId;
        remAddrId_ = remAddrId;
        remTokenId_ = remTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "Write"; }
    uint16_t GetLocAddrId() { return locAddrId_; }
    uint16_t GetLocTokenId() { return locTokenId_; }
    uint16_t GetRemAddrId() { return remAddrId_; }
    uint16_t GetRemTokenId() { return remTokenId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint16_t locAddrId_;
    uint16_t locTokenId_;
    uint16_t remAddrId_;
    uint16_t remTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepLocalCpy : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocalCpy(
        uint16_t srcAddrId, uint16_t srcTokenId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::LOCAL_CPY;
        this->instrId = 19;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "LocalCpy"; }
    uint16_t GetSrcAddrId() { return srcAddrId_; }
    uint16_t GetSrcTokenId() { return srcTokenId_; }
    uint16_t GetDstAddrId() { return dstAddrId_; }
    uint16_t GetDstTokenId() { return dstTokenId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }

private:
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
};

class MockCcuRepLocalReduce : public CcuRep::CcuRepBase {
public:
    MockCcuRepLocalReduce(
        uint16_t srcAddrId, uint16_t srcTokenId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId,
        uint16_t semId, uint32_t mask, uint16_t dataType, uint16_t opType)
    {
        this->type = CcuRep::CcuRepType::LOCAL_REDUCE;
        this->instrId = 20;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        dataType_ = dataType;
        opType_ = opType;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "LocalReduce"; }
    uint16_t GetSrcAddrId() { return srcAddrId_; }
    uint16_t GetSrcTokenId() { return srcTokenId_; }
    uint16_t GetDstAddrId() { return dstAddrId_; }
    uint16_t GetDstTokenId() { return dstTokenId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }
    uint16_t GetDataType() { return dataType_; }
    uint16_t GetOpType() { return opType_; }

private:
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint16_t dataType_;
    uint16_t opType_;
};

class MockCcuRepBufRead : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufRead(
        uint16_t dstAddrId, uint16_t srcAddrId, uint16_t srcTokenId, uint16_t lenId, uint16_t semId, uint32_t mask,
        uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::BUF_READ;
        this->instrId = 21;
        dstAddrId_ = dstAddrId;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "BufRead"; }
    uint16_t GetDstAddrId() { return dstAddrId_; }
    uint16_t GetSrcAddrId() { return srcAddrId_; }
    uint16_t GetSrcTokenId() { return srcTokenId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint16_t dstAddrId_;
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepBufWrite : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufWrite(
        uint16_t srcId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId, uint16_t semId, uint32_t mask,
        uint32_t channelId)
    {
        this->type = CcuRep::CcuRepType::BUF_WRITE;
        this->instrId = 22;
        srcId_ = srcId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
        channelId_ = channelId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "BufWrite"; }
    uint16_t GetSrcId() { return srcId_; }
    uint16_t GetDstAddrId() { return dstAddrId_; }
    uint16_t GetDstTokenId() { return dstTokenId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }
    uint32_t GetChannelId() { return channelId_; }

private:
    uint16_t srcId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
    uint32_t channelId_;
};

class MockCcuRepBufLocRead : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufLocRead(
        uint16_t srcAddrId, uint16_t srcTokenId, uint16_t dstId, uint16_t lenId, uint16_t semId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::BUF_LOC_READ;
        this->instrId = 23;
        srcAddrId_ = srcAddrId;
        srcTokenId_ = srcTokenId;
        dstId_ = dstId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "BufLocRead"; }
    uint16_t GetSrcAddrId() { return srcAddrId_; }
    uint16_t GetSrcTokenId() { return srcTokenId_; }
    uint16_t GetDstId() { return dstId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }

private:
    uint16_t srcAddrId_;
    uint16_t srcTokenId_;
    uint16_t dstId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
};

class MockCcuRepBufLocWrite : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufLocWrite(
        uint16_t srcId, uint16_t dstAddrId, uint16_t dstTokenId, uint16_t lenId, uint16_t semId, uint32_t mask)
    {
        this->type = CcuRep::CcuRepType::BUF_LOC_WRITE;
        this->instrId = 24;
        srcId_ = srcId;
        dstAddrId_ = dstAddrId;
        dstTokenId_ = dstTokenId;
        lenId_ = lenId;
        semId_ = semId;
        mask_ = mask;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "BufLocWrite"; }
    uint16_t GetSrcAddrId() { return srcId_; }
    uint16_t GetDstTokenId() { return dstTokenId_; }
    uint16_t GetDstAddrId() { return dstAddrId_; }
    uint16_t GetLenId() { return lenId_; }
    uint16_t GetSemId() { return semId_; }
    uint32_t GetMask() { return mask_; }

private:
    uint16_t srcId_;
    uint16_t dstAddrId_;
    uint16_t dstTokenId_;
    uint16_t lenId_;
    uint16_t semId_;
    uint32_t mask_;
};

class MockCcuRepBufReduce : public CcuRep::CcuRepBase {
public:
    MockCcuRepBufReduce(
        uint16_t count, uint16_t dataType, uint16_t outputDataType, uint16_t opType, uint16_t semId, uint32_t mask,
        const std::vector<CcuRep::CcuBuf>& mem, uint16_t xnLengthId)
    {
        this->type = CcuRep::CcuRepType::BUF_REDUCE;
        this->instrId = 25;
        count_ = count;
        dataType_ = dataType;
        outputDataType_ = outputDataType;
        opType_ = opType;
        semId_ = semId;
        mask_ = mask;
        mem_ = mem;
        xnLengthId_ = xnLengthId;
    }
    bool Translate(CcuInstr*& instr, uint16_t& instrId, const CcuRep::TransDep& dep) override { return true; }
    std::string Describe() override { return "BufReduce"; }
    const std::vector<CcuRep::CcuBuf>& GetMem() { return mem_; }
    uint16_t GetCount() { return count_; }
    uint16_t GetDataType() { return dataType_; }
    uint16_t GetOutputDataType() { return outputDataType_; }
    uint16_t GetOpType() { return opType_; }
    uint16_t GetXnLengthId() { return xnLengthId_; }
    uint32_t GetMask() { return mask_; }
    uint16_t GetSemId() { return semId_; }

private:
    uint16_t count_;
    uint16_t dataType_;
    uint16_t outputDataType_;
    uint16_t opType_;
    uint16_t semId_;
    uint32_t mask_;
    std::vector<CcuRep::CcuBuf> mem_;
    uint16_t xnLengthId_;
};

TEST_F(CcuRepContextTest, AddProfilingnameandmask_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t mask = 0;
    int ret = context.AddProfiling(name, mask);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(CcuRepContextTest, AddProfilingchannel_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t mask = 0;
    uint32_t signalIndex = 0;
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel* channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(channel);
    void** handle{nullptr};
    void* voidHandle = reinterpret_cast<void*>(channelHandle);
    handle = &voidHandle;
    uint32_t locCkeId = 1;
    uint32_t channelid = 2;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetLocCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(locCkeId))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(channelid));

    int ret = context.AddProfiling(channelHandle, name, signalIndex, mask);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(CcuRepContextTest, AddProfilingchannelNum_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t num = 1;
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel* channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(channel);
    void** handle{nullptr};
    void* voidHandle = reinterpret_cast<void*>(channelHandle);
    handle = &voidHandle;
    uint32_t channelid = 2;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(channelid));

    int ret = context.AddProfiling(&channelHandle, num);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(CcuRepContextTest, AddProfilingchannelNumMuch_Normal)
{
    CcuRep::CcuRepContext context;
    context.SetDieId(1);
    std::string name = "CCU_KERNEL";
    uint32_t num = 1;
    EndpointHandle locEndpointHandle;
    HcommChannelDesc channelDesc;
    Channel* channel = new (std::nothrow) CcuUrmaChannel(locEndpointHandle, channelDesc);
    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(channel);
    void** handle{nullptr};
    void* voidHandle = reinterpret_cast<void*>(channelHandle);
    handle = &voidHandle;
    uint32_t channelid = 2;
    uint32_t missonid = 2;
    HcommDataType hcommDataType = HcommDataType::HCOMM_DATA_TYPE_INT8;
    HcommDataType hcommOutputDataType = HcommDataType::HCOMM_DATA_TYPE_INT8;
    HcommReduceOp hcommOpType = HcommReduceOp::HCOMM_REDUCE_SUM;
    MOCKER_CPP(&ChannelProcess::ChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(handle, sizeof(handle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(channelid));
    MOCKER_CPP(&CcuRepContext::GetMissionId).stubs().will(returnValue(missonid));
    std::shared_ptr<CcuRepBase> baserep;
    context.allLgProfilingReps.push_back(baserep);
    int ret = context.AddProfiling(&channelHandle, num, hcommDataType, hcommOutputDataType, hcommOpType);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

namespace {

    class MockHccpRaCustomChannel {
    public:
        static uint32_t dieId_;
        static uint32_t callCount_;
        static uint32_t lastOpCode_;

        static void Setup()
        {
            callCount_ = 0;
            lastOpCode_ = 0;
        }
    };

    uint32_t MockHccpRaCustomChannel::dieId_ = 0;
    uint32_t MockHccpRaCustomChannel::callCount_ = 0;
    uint32_t MockHccpRaCustomChannel::lastOpCode_ = 0;

} // namespace

class CcuComponentTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(CcuComponentTest, CcuCleanTaskKillState_InvalidParam)
{
    auto ret = CcuComponent::GetInstance(0).CcuCleanTaskKillState(-1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);

    ret = CcuComponent::GetInstance(0).CcuCleanTaskKillState(256);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, CleanDieCkes_InvalidDieId)
{
    constexpr uint8_t MAX_IODIE_NUM = 2;
    auto ret = CcuComponent::GetInstance(0).CleanDieCkes(MAX_IODIE_NUM);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, CcuComponentTestSetProcess)
{
    CcuOpcodeType opCode = CcuOpcodeType::CCU_U_OP_CLEAN_TASKKILL_STATE;
    // 模拟HccpRaTlvRequestForCustomChannel
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    auto ret = CcuComponent::GetInstance(0).SetProcess(opCode);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, CcuComponentSetTaskKillDone)
{
    MOCKER_CPP(&CcuComponent::SetProcess).stubs().will(returnValue(HCCL_SUCCESS));
    CcuComponent::GetInstance(0).status = CcuComponent::GetInstance(0).CcuTaskKillStatus::TASK_KILL;
    auto ret = CcuComponent::GetInstance(0).SetTaskKillDone();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, CcuSetTaskKillDonefailed)
{
    auto ret = CcuComponent::GetInstance(0).CcuSetTaskKillDone(-1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuComponentTest, CleanDiId)
{
    constexpr uint8_t IODIE = 0;
    uint32_t ckeNum = 1;
    MOCKER_CPP(&CcuResSpecifications::GetCkeNum)
        .stubs()
        .with(mockcpp::any(), outBound(ckeNum))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CcuComponent::GetInstance(0).dieEnableFlags_[0] = true;
    auto ret = CcuComponent::GetInstance(0).CleanDieCkes(IODIE);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// // ==================== 补充覆盖率测试 ====================

// // --- GetCcuChannelHandleById: 空 map 返回 HCCL_E_NOT_FOUND ---
// extern std::unordered_map<uint16_t, uint64_t> g_channelIdToHandle;
// extern std::mutex g_channelMapMutex;

// TEST_F(HcommDfxTaskExceptionTest, GetCcuChannelHandleById_Found)
// {
//     {
//         std::lock_guard<std::mutex> lock(g_channelMapMutex);
//         g_channelIdToHandle[99] = 0xABCD1234;
//     }

//     u64 handle = 0;
//     HcclResult ret = CcuTaskException::GetCcuChannelHandleById(99, handle);
//     EXPECT_EQ(ret, HCCL_SUCCESS);
//     EXPECT_EQ(handle, static_cast<u64>(0xABCD1234));

//     {
//         std::lock_guard<std::mutex> lock(g_channelMapMutex);
//         g_channelIdToHandle.erase(99);
//     }
// }

// // --- GenStatusInfo: CCU_MISSION_TASK_KILLED 分支 (status = 0x0902) ---
// TEST_F(HcommDfxTaskExceptionTest, GenStatusInfo_CcuMissionTaskKilled)
// {
//     ErrorInfoBase baseInfo = {};
//     baseInfo.deviceId = 0;
//     baseInfo.dieId = 0;
//     baseInfo.missionId = 1;
//     baseInfo.currentInsId = 5;
//     // highPart=0x09 (CCUM_EXECUTE_ERROR), lowPart=0x02 (CCU_MISSION_TASK_KILLED)
//     baseInfo.status = static_cast<uint16_t>((0x09 << 8) | 0x02);

//     vector<CcuErrorInfo> errorInfo;
//     EXPECT_NO_THROW(CcuTaskException::GenStatusInfo(baseInfo, errorInfo));
//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
//     // 确认包含 TASK_KILLED 对应子状态描述
//     EXPECT_TRUE(string(errorInfo[0].msg.mission.missionError).find("CCU Mission Task Killed") != string::npos);
// }

// // --- GenErrorInfoDefault: 直接调用验证填充 DEFAULT 类型 ---
// TEST_F(HcommDfxTaskExceptionTest, GenErrorInfoDefault_FillsDefaultType)
// {
//     // 构造一个仅有基本 CcuRepBase 功能的 stub
//     // CcuRepBase 有纯虚函数，用 NopRep 继承覆盖
//     class NopRep : public CcuRep::CcuRepBase {
//     public:
//         NopRep() : CcuRepBase()
//         {
//             type = CcuRep::CcuRepType::BASE;
//         }
//         bool Translate(CcuRep::CcuInstr *&, uint16_t &, const CcuRep::TransDep &) override
//         {
//             return false;
//         }
//         std::string Describe() override
//         {
//             return "NopRep";
//         }
//     };

//     auto repBase = std::make_shared<NopRep>();
//     ErrorInfoBase baseInfo{0, 1, 2, 10, 0x0102};
//     vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoDefault(baseInfo, repBase, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::DEFAULT);
//     EXPECT_EQ(errorInfo[0].dieId, 1u);
//     EXPECT_EQ(errorInfo[0].missionId, 2u);
// }

// // --- GenErrorInfoByRepType: type 不在 HANDLER_MAP，走 default 路径 ---
// TEST_F(HcommDfxTaskExceptionTest, GenErrorInfoByRepType_UnknownType_GoesToDefault)
// {
//     class NopRep : public CcuRep::CcuRepBase {
//     public:
//         NopRep() : CcuRepBase()
//         {
//             type = CcuRep::CcuRepType::NOP; // NOP 不在 HANDLER_MAP
//         }
//         bool Translate(CcuRep::CcuInstr *&, uint16_t &, const CcuRep::TransDep &) override
//         {
//             return false;
//         }
//         std::string Describe() override
//         {
//             return "NopRep";
//         }
//     };

//     auto repBase = std::make_shared<NopRep>();
//     ErrorInfoBase baseInfo{0, 0, 0, 0, 0x0101};
//     vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoByRepType(baseInfo, repBase, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::DEFAULT);
// }

// // --- PrintCcuErrorLog: empty input -> no crash ---
// TEST_F(HcommDfxTaskExceptionTest, PrintCcuErrorLog_EmptyList)
// {
//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     vector<CcuErrorInfo> errorInfos;
//     EXPECT_NO_THROW(CcuTaskException::PrintCcuErrorLog(errorInfos, taskInfo, 0));
// }

// // --- PrintCcuErrorLog: MISSION type -> GetCcuErrorMsgByType called ---
// TEST_F(HcommDfxTaskExceptionTest, PrintCcuErrorLog_WithMissionInfo)
// {
//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     CcuErrorInfo info{};
//     info.type = CcuErrorType::MISSION;
//     info.repType = CcuRep::CcuRepType::BASE;
//     info.dieId = 0;
//     info.missionId = 1;
//     info.instrId = 0;
//     const char errMsg[] = "Test error";
//     (void)strncpy_s(info.msg.mission.missionError, sizeof(info.msg.mission.missionError), errMsg, sizeof(errMsg));

//     vector<CcuErrorInfo> errorInfos{info};
//     EXPECT_NO_THROW(CcuTaskException::PrintCcuErrorLog(errorInfos, taskInfo, 0));
// }

// // --- GetCcuErrorMsgByType: LOOP_GROUP type ---
// TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgByType_LoopGroup)
// {
//     CcuErrorInfo ccuErrorInfo = {};
//     ccuErrorInfo.type = CcuErrorType::LOOP_GROUP;
//     ccuErrorInfo.repType = CcuRep::CcuRepType::LOOPGROUP;
//     ccuErrorInfo.instrId = 3;
//     ccuErrorInfo.msg.loopGroup.startLoopInsId = 0;
//     ccuErrorInfo.msg.loopGroup.loopInsCnt = 4;
//     ccuErrorInfo.msg.loopGroup.expandOffset = 1;
//     ccuErrorInfo.msg.loopGroup.expandCnt = 2;

//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     string result = CcuTaskException::GetCcuErrorMsgByType(ccuErrorInfo, taskInfo, 0);
//     EXPECT_FALSE(result.empty());
//     EXPECT_TRUE(result.find("LoopGroup") != string::npos);
// }

// // --- GetCcuErrorMsgByType: LOOP type ---
// TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgByType_Loop)
// {
//     CcuErrorInfo ccuErrorInfo = {};
//     ccuErrorInfo.type = CcuErrorType::LOOP;
//     ccuErrorInfo.repType = CcuRep::CcuRepType::LOOP;
//     ccuErrorInfo.instrId = 7;
//     ccuErrorInfo.msg.loop.startInstrId = 0;
//     ccuErrorInfo.msg.loop.endInstrId = 10;
//     ccuErrorInfo.msg.loop.loopEngineId = 2;
//     ccuErrorInfo.msg.loop.loopCnt = 5;
//     ccuErrorInfo.msg.loop.loopCurrentCnt = 3;
//     ccuErrorInfo.msg.loop.addrStride = 128;

//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     string result = CcuTaskException::GetCcuErrorMsgByType(ccuErrorInfo, taskInfo, 0);
//     EXPECT_FALSE(result.empty());
//     EXPECT_TRUE(result.find("Loop") != string::npos);
// }

// // --- GetMSIdPerDie: bit-masking logic ---
// TEST_F(HcommDfxTaskExceptionTest, GetMSIdPerDie_MasksHighBit)
// {
//     // GetMSIdPerDie returns msId & 0x7fff
//     uint16_t msIdWithHighBit = 0x8005;
//     uint16_t expected = msIdWithHighBit & 0x7FFF; // 5
//     EXPECT_EQ(CcuTaskException::GetMSIdPerDie(msIdWithHighBit), expected);
// }

// TEST_F(HcommDfxTaskExceptionTest, GetMSIdPerDie_NoHighBit)
// {
//     uint16_t msId = 42;
//     EXPECT_EQ(CcuTaskException::GetMSIdPerDie(msId), msId);
// }

// // --- GetCcuErrorMsgLocPostSem: via WAIT_SIGNAL type ---
// TEST_F(HcommDfxTaskExceptionTest, GetCcuErrorMsgByType_WaitSignal_LocPostSem)
// {
//     CcuErrorInfo ccuErrorInfo = {};
//     ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
//     ccuErrorInfo.repType = CcuRep::CcuRepType::LOC_RECORD_EVENT;
//     ccuErrorInfo.instrId = 1;
//     ccuErrorInfo.msg.waitSignal.signalId = 3;
//     ccuErrorInfo.msg.waitSignal.signalValue = 0x00FF;
//     ccuErrorInfo.msg.waitSignal.signalMask = 0xFF00;

//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     string result = CcuTaskException::GetCcuErrorMsgByType(ccuErrorInfo, taskInfo, 0);
//     EXPECT_FALSE(result.empty());
// }

// // --- RankIdByChannelId: taskType not CCU ---
// TEST_F(HcommDfxTaskExceptionTest, GetRankIdByChannelId_TaskTypeNotCcu)
// {
//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_RDMA,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     RankId rankId = CcuTaskException::GetRankIdByChannelId(0, taskInfo, 0);
//     EXPECT_EQ(rankId, static_cast<RankId>(INVALID_UINT));
// }

// // --- GetRankIdByChannelId: dfxOpInfo is nullptr ---
// TEST_F(HcommDfxTaskExceptionTest, GetRankIdByChannelId_DfxOpInfoNullptr)
// {
//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);
//     taskInfo.dfxOpInfo_ = nullptr;

//     RankId rankId = CcuTaskException::GetRankIdByChannelId(0, taskInfo, 0);
//     EXPECT_EQ(rankId, static_cast<RankId>(INVALID_UINT));
// }

// // --- GetAddrPairByChannelId: taskType not CCU ---
// TEST_F(HcommDfxTaskExceptionTest, GetAddrPairByChannelId_TaskTypeNotCcu)
// {
//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_RDMA,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);

//     auto result = CcuTaskException::GetAddrPairByChannelId(0, taskInfo, 0);
//     // 返回 dummy 空地址对
//     EXPECT_TRUE(result.first.Describe().empty() || !result.first.Describe().empty()); // 不崩溃即可
// }

// // --- GetAddrPairByChannelId: dfxOpInfo nullptr ---
// TEST_F(HcommDfxTaskExceptionTest, GetAddrPairByChannelId_DfxOpInfoNullptr)
// {
//     Hccl::ParaCcu paraCcu = {};
//     Hccl::TaskParam taskParam = {.taskType = Hccl::TaskParamType::TASK_CCU,
//         .beginTime = 0,
//         .endTime = 0,
//         .isMaster = false,
//         .taskPara = {.Ccu = paraCcu},
//         .ccuDetailInfo = nullptr};
//     Hccl::TaskInfo taskInfo(1, 2, 3, taskParam, nullptr, false);
//     taskInfo.dfxOpInfo_ = nullptr;

//     auto result = CcuTaskException::GetAddrPairByChannelId(0, taskInfo, 0);
//     EXPECT_TRUE(true); // 不崩溃即可
// }

// TEST_F(GenErrorInfoTest, GenErrorInfoRead_UsesRepFields)
// {
//     CcuRep::Address locAddr;
//     CcuRep::Variable locToken;
//     CcuRep::Address remAddr;
//     CcuRep::Variable remToken;
//     CcuRep::Variable len;
//     CcuRep::CompletedEvent sem;

//     locAddr.Reset(11);
//     locToken.Reset(12);
//     remAddr.Reset(13);
//     remToken.Reset(14);
//     len.Reset(15);
//     sem.Reset(16);

//     auto rep = std::make_shared<CcuRep::CcuRepRead>(static_cast<ChannelHandle>(101),
//         CcuRep::LocalAddr(locAddr, locToken), CcuRep::RemoteAddr(remAddr, remToken), len, sem, 0x5A);
//     std::vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoRead(baseInfo_, rep, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
//     EXPECT_EQ(errorInfo[0].msg.transMem.signalId, 16u);
//     EXPECT_EQ(errorInfo[0].msg.transMem.signalMask, 0x5Au);
//     EXPECT_EQ(errorInfo[0].msg.transMem.channelId, 0u);
// }

// TEST_F(GenErrorInfoTest, GenErrorInfoWrite_UsesRepFields)
// {
//     CcuRep::Address locAddr;
//     CcuRep::Variable locToken;
//     CcuRep::Address remAddr;
//     CcuRep::Variable remToken;
//     CcuRep::Variable len;
//     CcuRep::CompletedEvent sem;

//     locAddr.Reset(21);
//     locToken.Reset(22);
//     remAddr.Reset(23);
//     remToken.Reset(24);
//     len.Reset(25);
//     sem.Reset(26);

//     auto rep = std::make_shared<CcuRep::CcuRepWrite>(static_cast<ChannelHandle>(102),
//         CcuRep::RemoteAddr(remAddr, remToken), CcuRep::LocalAddr(locAddr, locToken), len, sem, 0xA5);
//     std::vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoWrite(baseInfo_, rep, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
//     EXPECT_EQ(errorInfo[0].msg.transMem.signalId, 26u);
//     EXPECT_EQ(errorInfo[0].msg.transMem.signalMask, 0xA5u);
//     EXPECT_EQ(errorInfo[0].msg.transMem.channelId, 0u);
// }

// TEST_F(GenErrorInfoTest, GenErrorInfoBufRead_UsesRepFields)
// {
//     CcuRep::Address srcAddr;
//     CcuRep::Variable srcToken;
//     CcuRep::CcuBuf dst;
//     CcuRep::Variable len;
//     CcuRep::CompletedEvent sem;

//     srcAddr.Reset(31);
//     srcToken.Reset(32);
//     dst.Reset(0x8009);
//     len.Reset(33);
//     sem.Reset(34);

//     auto rep = std::make_shared<CcuRep::CcuRepBufRead>(
//         static_cast<ChannelHandle>(103), CcuRep::RemoteAddr(srcAddr, srcToken), dst, len, sem, 0x12);
//     std::vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoBufRead(baseInfo_, rep, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.bufId, 9u);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 34u);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, 0x12u);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.channelId, 0u);
// }

// TEST_F(GenErrorInfoTest, GenErrorInfoBufWrite_UsesRepFields)
// {
//     CcuRep::CcuBuf src;
//     CcuRep::Address dstAddr;
//     CcuRep::Variable dstToken;
//     CcuRep::Variable len;
//     CcuRep::CompletedEvent sem;

//     src.Reset(0x800A);
//     dstAddr.Reset(41);
//     dstToken.Reset(42);
//     len.Reset(43);
//     sem.Reset(44);

//     auto rep = std::make_shared<CcuRep::CcuRepBufWrite>(
//         static_cast<ChannelHandle>(104), src, CcuRep::RemoteAddr(dstAddr, dstToken), len, sem, 0x34);
//     std::vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoBufWrite(baseInfo_, rep, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.bufId, 10u);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 44u);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, 0x34u);
//     EXPECT_EQ(errorInfo[0].msg.bufTransMem.channelId, 0u);
// }

// TEST_F(GenErrorInfoTest, GenErrorInfoLocWaitEvent_UsesRepFields)
// {
//     CcuRep::CompletedEvent event;
//     event.Reset(51);
//     event.SetMask(0xFF);

//     auto rep = std::make_shared<CcuRep::CcuRepLocWaitEvent>(event, true);
//     std::vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoLocWaitEvent(baseInfo_, rep, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
//     EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 51u);
//     EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, 0xFFu);
// }

// TEST_F(GenErrorInfoTest, GenErrorInfoLocWaitNotify_UsesRepFields)
// {
//     CcuRep::LocalNotify notify;
//     notify.Reset(52);

//     auto rep = std::make_shared<CcuRep::CcuRepLocWaitNotify>(notify, 0xAA, true);
//     std::vector<CcuErrorInfo> errorInfo;

//     CcuTaskException::GenErrorInfoLocWaitNotify(baseInfo_, rep, errorInfo);

//     EXPECT_EQ(errorInfo.size(), 1u);
//     EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
//     EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 52u);
//     EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, 0xAAu);
// }

} // namespace hcomm

#undef private
