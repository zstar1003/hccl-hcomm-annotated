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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>

#include "const_val.h"
#include "orion_adapter_rts.h"
#include "ccu_error_info.h"
#include "global_mirror_tasks.h"
#include "mirror_task_manager.h"
#include "ccu_dfx.h"
#include "mc2_global_mirror_tasks.h"
#include "ccu_device_manager.h"
#include "ccu_component.h"
#include <adapter_error_manager_pub.h>

void RptInputErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    (void)error_code;
    (void)key;
    (void)value;
}

#define private public
#define protected public
#include "task_exception_handler.h"
#include "communicator_impl.h"
#include "coll_service_device_mode.h"
#include "ccu_transport_manager.h"
#include "mc2_compont.h"
#include "task_exception_test_common.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;
using namespace CcuRep;

class TaskExceptionHandlerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskExceptionHandlerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskExceptionHandlerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TaskExceptionHandlerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TaskExceptionHandlerTest TearDown" << std::endl;
    }

    unique_ptr<TaskInfo> InitTaskInfo(u32 streamId = 0, u32 taskId = 0, u32 remoteRank = 0)
    {
        TaskParam taskParam{};
        shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
        return make_unique<TaskInfo>(streamId, taskId, remoteRank, taskParam, dfxOpInfo);
    }

    void AddNullDfxCcuTask(MirrorTaskManager& mgr)
    {
        auto curTaskInfo = InitTaskInfo(0, 0);
        curTaskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
        curTaskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
        mgr.AddTaskInfo(std::move(curTaskInfo));
    }

    rtExceptionInfo_t MakeCcuExceptionInfo() const
    {
        SetupAndProcessCcuException();
        rtExceptionInfo_t exceptionInfo{};
        return exceptionInfo;
    }

    void StubCcuExceptionProcess() const;
};

TEST_F(TaskExceptionHandlerTest, test_process_when_task_less_than_50)
{
    // 打桩 GlobalMirrorTasks
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1); // diveceId 0
    shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
    dfxOpInfo->commIndex_ = 3;
    dfxOpInfo->op_.dataCount = 0xff;
    dfxOpInfo->op_.reduceOp = ReduceOp::PROD;
    dfxOpInfo->op_.dataType = DataType::FP64;
    dfxOpInfo->algType_ = AlgType{AlgType::RING}.Describe();
    dfxOpInfo->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    dfxOpInfo->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    // 加入一些 Task 数据
    // 在异常 Task 前加入一些 Task
    auto taskInfo0 = InitTaskInfo(0, 0);
    taskInfo0->taskParam_.taskType = TaskParamType::TASK_SDMA;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo0));
    // TASK_SDMA
    auto taskInfo1 = InitTaskInfo(0, 1);
    taskInfo1->taskParam_.taskType = TaskParamType::TASK_SDMA;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo1));
    // TASK_RDMA
    auto taskInfo2 = InitTaskInfo(0, 2);
    taskInfo2->taskParam_.taskType = TaskParamType::TASK_RDMA;
    taskInfo2->taskParam_.taskPara.DMA.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo2));
    // TASK_SEND_PAYLOAD
    auto taskInfo3 = InitTaskInfo(0, 3);
    taskInfo3->taskParam_.taskType = TaskParamType::TASK_SEND_PAYLOAD;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo3));
    // TASK_REDUCE_INLINE
    auto taskInfo4 = InitTaskInfo(0, 4);
    taskInfo4->taskParam_.taskType = TaskParamType::TASK_REDUCE_INLINE;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo4));
    // TASK_REDUCE_TBE
    auto taskInfo5 = InitTaskInfo(0, 5);
    taskInfo5->taskParam_.taskType = TaskParamType::TASK_REDUCE_TBE;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo5));
    // TASK_NOTIFY_RECORD
    auto taskInfo6 = InitTaskInfo(0, 6);
    taskInfo6->taskParam_.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskInfo6->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo6));
    // TASK_NOTIFY_WAIT
    auto taskInfo7 = InitTaskInfo(0, 7);
    taskInfo7->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo7->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo7));
    // TASK_SEND_NOTIFY
    auto taskInfo8 = InitTaskInfo(0, 8);
    taskInfo8->taskParam_.taskType = TaskParamType::TASK_SEND_NOTIFY;
    taskInfo8->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo8));
    // TASK_WRITE_WITH_NOTIFY
    auto taskInfo9 = InitTaskInfo(0, 9);
    taskInfo9->taskParam_.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
    taskInfo9->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo9));
    // TASK_WRITE_REDUCE_WITH_NOTIFY
    auto taskInfo10 = InitTaskInfo(0, 10);
    taskInfo10->taskParam_.taskType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
    taskInfo10->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo10));
    // 加入当前异常 Task
    auto curTaskInfo = InitTaskInfo(0, 11); // streamId 0, taskId 11
    curTaskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    curTaskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    curTaskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(curTaskInfo));

    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();
    // 调用 TaskExceptionHandler::Process() 打印异常DFX信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 11; // 当前异常TaskId
    TaskExceptionHandler::Process(&exceptionInfo);

    globalMirrorTasks.DestroyQueue(0, 0); // diveceId 0, streamId 0
}

HcclResult MockGetCcuErrorMsg(
    s32 deviceId, uint16_t missionStatus, uint16_t currIns, const ParaCcu& ccuTaskParam,
    const std::string& groupRankContent, std::vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo loopGroupErrorInfo{};
    loopGroupErrorInfo.type = CcuErrorType::LOOP_GROUP;
    loopGroupErrorInfo.repType = CcuRepType::LOOPGROUP;
    loopGroupErrorInfo.instrId = 1;
    loopGroupErrorInfo.msg.loopGroup.startLoopInsId = 17;
    loopGroupErrorInfo.msg.loopGroup.loopInsCnt = 5;
    loopGroupErrorInfo.msg.loopGroup.expandOffset = 3;
    loopGroupErrorInfo.msg.loopGroup.expandCnt = 2;
    errorInfo.push_back(loopGroupErrorInfo);

    CcuErrorInfo loopErrorInfo{};
    loopErrorInfo.type = CcuErrorType::LOOP;
    loopErrorInfo.repType = CcuRepType::LOOP;
    loopErrorInfo.instrId = 2;
    loopErrorInfo.msg.loop.startInstrId = 7;
    loopErrorInfo.msg.loop.endInstrId = 17;
    loopErrorInfo.msg.loop.loopCnt = 10;
    loopErrorInfo.msg.loop.loopCurrentCnt = 8;
    loopErrorInfo.msg.loop.addrStride = 0xaabbcc;
    errorInfo.push_back(loopErrorInfo);

    CcuErrorInfo ccuErrorInfo{};
    ccuErrorInfo.type = CcuErrorType::WAIT_SIGNAL;
    ccuErrorInfo.repType = CcuRepType::LOC_WAIT_SEM;
    ccuErrorInfo.instrId = 3;
    ccuErrorInfo.msg.waitSignal.signalId = 0xb;
    ccuErrorInfo.msg.waitSignal.signalValue = 0xabc;
    ccuErrorInfo.msg.waitSignal.signalMask = 0x0010;
    errorInfo.push_back(ccuErrorInfo);

    return HcclResult::HCCL_SUCCESS;
}

void TaskExceptionHandlerTest::StubCcuExceptionProcess() const
{
    MOCKER(GetCcuErrorMsg).stubs().will(invoke(MockGetCcuErrorMsg));
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(0)));
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs();
    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();
}

TEST_F(TaskExceptionHandlerTest, test_process_ccu)
{
    // 打桩 GlobalMirrorTasks
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1); // diveceId 0
    shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
    dfxOpInfo->commIndex_ = 3;
    dfxOpInfo->op_.dataCount = 0xff;
    dfxOpInfo->op_.reduceOp = ReduceOp::PROD;
    dfxOpInfo->op_.dataType = DataType::FP64;
    dfxOpInfo->algType_ = AlgType{AlgType::RING}.Describe();
    dfxOpInfo->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    dfxOpInfo->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    CommunicatorImpl communicator{}; // Mock CommunicatorImpl
    communicator.id = "GroupName";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto exceptionInfo = PrepareNullDfxCcuException(mirrorTaskManager);
    TaskExceptionHandler::Process(&exceptionInfo);

    globalMirrorTasks.DestroyQueue(0, 0); // diveceId 0, streamId 0
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessException_When_DfxOpInfoNull_Expect_NoThrow)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);

    auto taskInfo = InitTaskInfo(0, 0);
    taskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(TaskExceptionHandler::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessCcuException_When_DfxOpInfoNull_Expect_NoThrow)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);

    auto exceptionInfo = PrepareNullDfxCcuException(mirrorTaskManager);
    EXPECT_NO_THROW(TaskExceptionHandler::Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(TaskExceptionHandlerTest, test_GetMC2AlgTaskParam)
{
    auto taskInfo = InitTaskInfo();

    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 0); // failed

    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo->dfxOpInfo_->comm_ = nullptr;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 0); // failed

    // Mock CommunicatorImpl
    CommunicatorImpl communicator{};
    communicator.collServices[AcceleratorState::CCU_SCHED] = std::make_shared<CollServiceDeviceMode>(&communicator);
    taskInfo->dfxOpInfo_->comm_ = &communicator;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 0); // failed

    // Mock collService
    communicator.collService = new CollServiceDeviceMode(&communicator);
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(communicator.GetCcuCollService());
    collServiceCcu->mc2Compont.ccuServerMap[10] = {1};
    CcuTaskParam ccuTaskParam{};
    std::vector<std::vector<CcuTaskParam>> ccuTaskParams{};
    ccuTaskParams.push_back({ccuTaskParam});
    ccuTaskParams.push_back({ccuTaskParam});
    collServiceCcu->mc2Compont.algoTemplateMap[1] = ccuTaskParams;
    taskInfo->taskParam_.taskPara.Ccu.executeId = 10;
    EXPECT_EQ(TaskExceptionHandler::GetMC2AlgTaskParam(*taskInfo).size(), 2); // success
    delete communicator.collService;
}

TEST_F(TaskExceptionHandlerTest, test_process_mc2)
{
    auto taskInfo1 = InitTaskInfo(); // for MC2 Server
    taskInfo1->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo1->taskParam_.taskPara.Ccu.dieId = 0;
    taskInfo1->taskParam_.taskPara.Ccu.missionId = 1;
    taskInfo1->taskParam_.taskPara.Ccu.instrId = 2;
    MC2GlobalMirrorTasks::GetInstance().AddTaskInfo(10, std::shared_ptr<TaskInfo>(std::move(taskInfo1)));
    auto taskInfo2 = InitTaskInfo(); // for Algo
    taskInfo2->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo2->taskParam_.taskPara.Ccu.dieId = 0;
    taskInfo2->taskParam_.taskPara.Ccu.missionId = 2;
    taskInfo2->taskParam_.taskPara.Ccu.instrId = 5;
    MC2GlobalMirrorTasks::GetInstance().AddTaskInfo(10, std::shared_ptr<TaskInfo>(std::move(taskInfo2)));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.expandInfo.type = RT_EXCEPTION_FUSION;
    exceptionInfo.expandInfo.u.fusionInfo.type = RT_FUSION_AICORE_CCU;
    exceptionInfo.deviceid = 10;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].dieId = 0;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].missionId = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].instrId = 2;

    MOCKER(GetCcuErrorMsg).stubs().will(returnValue(HcclResult::HCCL_SUCCESS)).then(invoke(MockGetCcuErrorMsg));
    CcuTaskParam ccuTaskParam{};
    ccuTaskParam.dieId = 0;
    ccuTaskParam.missionId = 2;
    ccuTaskParam.instStartId = 5;
    vector<CcuTaskParam> mockAlgTaskParams{ccuTaskParam};
    MOCKER(TaskExceptionHandler::GetMC2AlgTaskParam).stubs().will(returnValue(mockAlgTaskParams));

    // 打桩清除TaskKill状态, 清除表项, 清除CKE操作
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(0));
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs();

    TaskExceptionHandler::Process(&exceptionInfo);
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessAivException_When_Normal_Expect_PrintInfo)
{
    // 初始化AIV任务信息
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_AIV;
    taskInfo->taskParam_.taskPara.Aiv.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    taskInfo->taskParam_.taskPara.Aiv.tag = 2;
    taskInfo->taskParam_.taskPara.Aiv.rank = 3;
    taskInfo->taskParam_.taskPara.Aiv.rankSize = 4;
    taskInfo->taskParam_.taskPara.Aiv.count = 5;
    taskInfo->taskParam_.taskPara.Aiv.numBlocks = 6;
    taskInfo->taskParam_.taskPara.Aiv.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    taskInfo->taskParam_.beginTime = 123456789;

    // 模拟flag内存
    int32_t flagMemData[] = {10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0};
    size_t flagMemSize = sizeof(flagMemData);
    taskInfo->taskParam_.taskPara.Aiv.flagMem = flagMemData;
    taskInfo->taskParam_.taskPara.Aiv.flagMemSize = flagMemSize;

    // 打桩ACL函数
    void* mockFlagBuff = malloc(flagMemSize);
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtFreeHost).stubs().will(returnValue(ACL_SUCCESS));

    // 打桩PrintAivPreviousTaskException
    MOCKER(TaskExceptionHandler::PrintAivPreviousTaskException).stubs();

    // 构造异常信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    // 调用ProcessAivException（必须在AddTaskInfo之前，因为AddTaskInfo会move走taskInfo）
    TaskExceptionHandler handler(0);
    handler.ProcessAivException(&exceptionInfo, *taskInfo);

    // 打桩GlobalMirrorTasks
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 清理
    globalMirrorTasks.DestroyQueue(0, 0);
    if (mockFlagBuff) {
        free(mockFlagBuff);
    }
}

namespace Hccl {
void ReportErrorMsg(
    const TaskInfo& exceptionTaskInfo, const string& groupRankContent, const ErrorMessageReport& errorMessage,
    const rtExceptionInfo_t* exceptionInfo, const std::string& stageErrInfo);
}

namespace {
std::string g_capturedErrorCode;
std::vector<std::string> g_capturedValues;

void stub_RptInputErr_capture(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    g_capturedErrorCode = error_code;
    g_capturedValues = value;
}
} // namespace

TEST_F(TaskExceptionHandlerTest, Ut_ReportErrorMsg_When_TaskNotifyWait_Expect_EI0002WithGroupRankContent)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;

    ErrorMessageReport errorMessage{};
    strncpy(errorMessage.group, "test_group", sizeof(errorMessage.group) - 1);
    errorMessage.rankSize = 8;
    errorMessage.rankId = 3;

    rtExceptionInfo_t exceptionInfo{};
    const std::string testGroupRank = "group:[test_group], rankSize[8], localRank[3], remoteRank[0]";
    g_capturedErrorCode.clear();
    g_capturedValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_capture));
    EXPECT_NO_THROW(ReportErrorMsg(*taskInfo, testGroupRank, errorMessage, &exceptionInfo, ""));
    EXPECT_EQ(g_capturedErrorCode, "EI0002");
    ASSERT_EQ(g_capturedValues.size(), 4);
    EXPECT_EQ(g_capturedValues[3], testGroupRank);
}

TEST_F(TaskExceptionHandlerTest, Ut_ReportErrorMsg_When_TaskWriteWithNotify_Expect_EI0018Path)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;

    ErrorMessageReport errorMessage{};
    strncpy(errorMessage.group, "test_group", sizeof(errorMessage.group) - 1);
    errorMessage.rankSize = 8;
    errorMessage.rankId = 3;

    rtExceptionInfo_t exceptionInfo{};
    g_capturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_capture));
    EXPECT_NO_THROW(ReportErrorMsg(*taskInfo, "", errorMessage, &exceptionInfo, ""));
    EXPECT_EQ(g_capturedErrorCode, "EI0018");
}

TEST_F(TaskExceptionHandlerTest, Ut_ReportErrorMsg_When_TaskUbReduceInline_Expect_EI0018Path)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_UB_REDUCE_INLINE;

    ErrorMessageReport errorMessage{};
    strncpy(errorMessage.group, "test_group", sizeof(errorMessage.group) - 1);
    errorMessage.rankSize = 8;
    errorMessage.rankId = 3;

    rtExceptionInfo_t exceptionInfo{};
    g_capturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_capture));
    EXPECT_NO_THROW(ReportErrorMsg(*taskInfo, "", errorMessage, &exceptionInfo, ""));
    EXPECT_EQ(g_capturedErrorCode, "EI0018");
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessAivException_When_MallocFailure_Expect_ReturnEarly)
{
    // 初始化AIV任务信息
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_AIV;
    taskInfo->taskParam_.taskPara.Aiv.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    taskInfo->taskParam_.taskPara.Aiv.flagMemSize = 1024;

    // 打桩aclrtMallocHost失败
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(ACL_ERROR_BAD_ALLOC));

    // 构造异常信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    // 调用ProcessAivException（必须在AddTaskInfo之前，因为AddTaskInfo会move走taskInfo）
    TaskExceptionHandler handler(0);
    handler.ProcessAivException(&exceptionInfo, *taskInfo);

    // 打桩GlobalMirrorTasks
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 清理
    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessAivException_When_MemcpyFailure_Expect_ReturnEarly)
{
    // 初始化AIV任务信息
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_AIV;
    taskInfo->taskParam_.taskPara.Aiv.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    taskInfo->taskParam_.taskPara.Aiv.flagMemSize = 1024;

    // 打桩aclrtMallocHost成功，但aclrtMemcpy失败
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_ERROR_RT_MEMORY_FREE));
    MOCKER(aclrtFreeHost).stubs().will(returnValue(ACL_SUCCESS));

    // 构造异常信息
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    // 调用ProcessAivException（必须在AddTaskInfo之前，因为AddTaskInfo会move走taskInfo）
    TaskExceptionHandler handler(0);
    handler.ProcessAivException(&exceptionInfo, *taskInfo);

    // 打桩GlobalMirrorTasks
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1);
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 清理
    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(TaskExceptionHandlerTest, Ut_DisplayRPCMsg_When_TaskTypeNotCcu_Expect_ReturnEarly)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;

    MOCKER(HrtMallocHost).expects(never());
    EXPECT_NO_THROW(TaskExceptionHandler::DisplayRPCMsg(*taskInfo));
}

TEST_F(TaskExceptionHandlerTest, Ut_DisplayRPCMsg_When_DfxOpInfoNull_Expect_ReturnEarly)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo->dfxOpInfo_ = nullptr;

    MOCKER(HrtMallocHost).expects(never());
    EXPECT_NO_THROW(TaskExceptionHandler::DisplayRPCMsg(*taskInfo));
}

TEST_F(TaskExceptionHandlerTest, Ut_DisplayRPCMsg_When_CommNull_Expect_ReturnEarly)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    // dfxOpInfo_ 默认非空，comm_ 默认 nullptr

    MOCKER(HrtMallocHost).expects(never());
    EXPECT_NO_THROW(TaskExceptionHandler::DisplayRPCMsg(*taskInfo));
}

TEST_F(TaskExceptionHandlerTest, Ut_DisplayRPCMsg_When_CollServiceNull_Expect_ReturnEarly)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    CommunicatorImpl* communicator = new CommunicatorImpl{};
    communicator->collServices[AcceleratorState::CCU_SCHED] = nullptr;
    taskInfo->dfxOpInfo_->comm_ = communicator;

    MOCKER(HrtMallocHost).expects(never());
    EXPECT_NO_THROW(TaskExceptionHandler::DisplayRPCMsg(*taskInfo));
}

TEST_F(TaskExceptionHandlerTest, Ut_DisplayRPCMsg_When_Normal_Expect_InvokeDisplayRPCMsg)
{
    auto taskInfo = InitTaskInfo();
    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    auto communicator = std::make_unique<CommunicatorImpl>();
    communicator->collServices[AcceleratorState::CCU_SCHED]
        = std::make_shared<CollServiceDeviceMode>(communicator.get());
    taskInfo->dfxOpInfo_->comm_ = communicator.get();
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(communicator->GetCcuCollService());
    collServiceCcu->mc2Compont.comParamBuffer
        = std::make_shared<DevBuffer>(CCU_TASK_NUM_MAX * CCU_PARAM_NUM_MAX * sizeof(uint64_t));
    collServiceCcu->mc2Compont.comSyncBuffer = std::make_shared<DevBuffer>(CCU_TASK_NUM_MAX * 2 * sizeof(uint64_t));

    std::vector<uint64_t> hostBuf(CCU_TASK_NUM_MAX * CCU_PARAM_NUM_MAX, 0);
    MOCKER(HrtMallocHost).stubs().will(returnValue(static_cast<void*>(hostBuf.data())));
    MOCKER(HrtMemcpy).stubs();

    EXPECT_NO_THROW(TaskExceptionHandler::DisplayRPCMsg(*taskInfo));
}

TEST_F(TaskExceptionHandlerTest, Ut_ProcessCcuMC2Exception_When_StatusZero_Expect_SkipProcess)
{
    auto taskInfo1 = InitTaskInfo();
    taskInfo1->taskParam_.taskType = TaskParamType::TASK_CCU;
    taskInfo1->taskParam_.taskPara.Ccu.dieId = 0;
    taskInfo1->taskParam_.taskPara.Ccu.missionId = 1;
    taskInfo1->taskParam_.taskPara.Ccu.instrId = 2;
    MC2GlobalMirrorTasks::GetInstance().AddTaskInfo(20, std::shared_ptr<TaskInfo>(std::move(taskInfo1)));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.expandInfo.type = RT_EXCEPTION_FUSION;
    exceptionInfo.expandInfo.u.fusionInfo.type = RT_FUSION_AICORE_CCU;
    exceptionInfo.deviceid = 20;
    auto& ccuDetailMsg = exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg;
    ccuDetailMsg.ccuMissionNum = 1;
    ccuDetailMsg.missionInfo[0].dieId = 0;
    ccuDetailMsg.missionInfo[0].missionId = 1;
    ccuDetailMsg.missionInfo[0].instrId = 2;
    ccuDetailMsg.missionInfo[0].status = 0;
    ccuDetailMsg.missionInfo[0].subStatus = 0;

    // status==0 时跳过，不应调用 GetCcuErrorMsg
    MOCKER(GetCcuErrorMsg).expects(never());
    // 打桩循环后清理操作
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(0));
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs();
    MOCKER(TaskExceptionHandler::PrintAicpuErrorMessage).stubs();

    EXPECT_NO_THROW(TaskExceptionHandler::Process(&exceptionInfo));
}
