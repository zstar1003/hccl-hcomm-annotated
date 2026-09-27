/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hccl_comm_pub.h"
#define private public
#include "hcclCommTaskExceptionLite.h"
#include "aicpu_ts_thread.h"
#include "hcclCommTaskException.h"
#undef private
#include "hcomm_task_scheduler_error.h"
#include "aicpu_indop_env.h"
#include "comm_engine_res_aicpu_mgr.h"
#include "adapter_hal_pub.h"
#include "dlhal_function_v2.h"
#include "rtsq_base.h"
#include "kernel_entrance.h"
#include "dfx_profiling_handler_lite.h"
#include "adapter_error_manager_pub.h"
#include "task_info.h"
#include "hccp.h"
#include "hccp_ctx.h"

using namespace hccl;
using namespace hcomm;

constexpr u32 RT_UB_LOCAL_OPERATIOINERR = 0x2;
constexpr u32 RT_UB_REMOTE_OPERATIOINERR = 0x3;
constexpr u32 RT_UB_LINK_FAILEDERR = 0x5;

inline void InitCommEngineResMgr(CollCommAicpu& c)
{
    if (c.commEngineResMgr_ == nullptr) {
        c.commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(c.dfx_, [](bool) {
            return HCCL_SUCCESS;
        });
    }
}
class hcclCommTaskExceptionLiteTest : public testing::Test {
protected:
    virtual void SetUp() override
    {
        MOCKER(::getpid).stubs().will(returnValue(12345));
        MOCKER(HrtHalDrvQueryProcessHostPid).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HrtHalDrvGetDevIDByLocalDevID).stubs().will(returnValue(HCCL_SUCCESS));
        Hccl::DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent
            = [](unsigned int, struct event_summary*) -> drvError_t {
            return DRV_ERROR_NONE;
        };
        Hccl::DlHalFunctionV2::GetInstance().dlHalDrvQueryProcessHostPid
            = [](int, unsigned int*, unsigned int* vfid, unsigned int* hostpid, unsigned int*) -> drvError_t {
            *vfid = 0;
            *hostpid = 12345;
            return DRV_ERROR_NONE;
        };
        HcclCommTaskExceptionLite::GetInstance().Init(0);
        g_taskExpDevMemMap.clear();
    }

    virtual void TearDown() override
    {
        g_taskExpDevMemMap.clear();
        GlobalMockObject::verify();
    }

private:
    u32 notifyId = 1;
    u32 tsId = 2;
};

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchUBCqeErrCodeToTsErrCode_When_Normal_Expect_ReturnIsCorrect)
{
    uint16_t ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_LOCAL_OPERATIOINERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_DDRC_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_REMOTE_OPERATIOINERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_POISON_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(RT_UB_LINK_FAILEDERR);
    EXPECT_EQ(ret, TS_ERROR_HCCL_OP_UB_LINK_FAILED);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchUBCqeErrCodeToTsErrCode(static_cast<u32>(123));
    EXPECT_EQ(ret, TS_ERROR_HCCL_OTHER_ERROR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchSdmaCqeErrCodeToTsErrCode_When_Normal_Expect_ReturnIsCorrect)
{
    uint16_t ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_COMPERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_LINK_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_COMPDATAERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_POISON_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(RT_SDMA_DATAERR);
    EXPECT_EQ(ret, TS_ERROR_SDMA_DDRC_ERROR);

    ret = HcclCommTaskExceptionLite::GetInstance().SwitchSdmaCqeErrCodeToTsErrCode(static_cast<u32>(123));
    EXPECT_EQ(ret, TS_ERROR_HCCL_OTHER_ERROR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SwitchSdmaCqeErrCodeToTsErrCode_taskexception_disable)
{
    hcomm::SetTaskExceptionEnable(false);
    rtLogicCqReport_t exceptionInfo;
    dfx::CqeStatus cqeStatus = dfx::CqeStatus::kDefault;
    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    HcclResult ret
        = HcclCommTaskExceptionLite::GetInstance().ProcessCqe(nullptr, exceptionInfo, cqeStatus, aicpuCommInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::SetTaskExceptionEnable(true);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_UBSqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 9;
    exceptionInfo.errorCode = RT_UB_LOCAL_OPERATIOINERR;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_SDMASqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 11;
    exceptionInfo.errorCode = RT_SDMA_COMPERR;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_SendTaskExceptionByMBox_When_OtherSqeType_Expect_ReturnHCCL_SUCCESS)
{
    rtLogicCqReport_t exceptionInfo;
    exceptionInfo.sqeType = 8;
    exceptionInfo.errorCode = 123;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().SendTaskExceptionByMBox(notifyId, tsId, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintAllCommTaskException)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "taskException_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);

    // InitAicpuIndOp 被 mock，手动初始化 commEngineResMgr_ 以防 PrintAllCommTaskException 空指针
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    for (auto& kv : commInfo) {
        if (kv.second->commEngineResMgr_ == nullptr) {
            kv.second->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(kv.second->dfx_, [](bool) {
                return HCCL_SUCCESS;
            });
        }
    }

    EXPECT_EQ(hcomm::HcclCommTaskExceptionLite::GetInstance().PrintAllCommTaskException(), HCCL_SUCCESS);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().DestroyComm(commAicpuParam.hcomId), HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommNullptr_Expect_ReturnEmpty)
{
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(nullptr);
    EXPECT_EQ(result, "");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetGroupInfo_When_AicpuCommValid_Expect_ReturnGroupName)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_group_name";
    InitCommEngineResMgr(aicpuComm);
    std::string result = HcclCommTaskExceptionLite::GetInstance().GetGroupInfo(&aicpuComm);
    EXPECT_EQ(result, "group:[test_group_name], rankSize:[0], localRank:[0]");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_CommIdNotInMap_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_TaskexceptionVaNull_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = nullptr;

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_StopFlagIsOne_Expect_ClearFlagAndSetNullptr)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 1;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(shmem[0], 0);
    EXPECT_EQ(g_taskExpDevMemMap[testCommId], nullptr);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagZero_Expect_ReturnSuccess)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndDfxLiteNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest,
    Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndMirrorTaskMgrNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_ = HcclCommDfxLite();
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZeroAndDfxOpInfoNull_Expect_ReturnPtrNull)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.Init(0, testCommId, 0, 0);
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(
    hcclCommTaskExceptionLiteTest, Ut_HandleDpuTaskexception_When_ErrorFlagNonZero_Expect_SendTaskExceptionAndClearFlag)
{
    std::string testCommId = "dpuExpTest";
    std::vector<uint8_t> shmem(10, 0);
    shmem[0] = 0;
    uint16_t errVal = 1;
    memcpy(shmem.data() + 1, &errVal, sizeof(uint16_t));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = testCommId;
    aicpuComm.dfx_.Init(0, testCommId, 0, 0);
    Hccl::DfxDfxOpInfo dfxOpInfo;
    dfxOpInfo.cpuWaitAicpuNotifyId = 10;
    aicpuComm.dfx_.SetCurrDfxOpInfo(&dfxOpInfo);
    g_taskExpDevMemMap[testCommId] = shmem.data();

    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().HandleDpuTaskexception(&aicpuComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint16_t clearedFlag = 0xFFFF;
    memcpy(&clearedFlag, shmem.data() + 1, sizeof(uint16_t));
    EXPECT_EQ(clearedFlag, 0);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_Call_ReturnHCCL_SUCCESS_When_CommStatusSuSpending)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "taskException_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);
    MOCKER_CPP(&CollCommAicpu::GetCommmStatus).stubs().will(returnValue(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING));
    hcomm::HcclCommTaskExceptionLite::GetInstance().Call();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetConciseTaskName_When_NotifyWait_Expect_ReturnName)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    std::string name = HcclCommTaskExceptionLite::GetInstance().GetConciseTaskName(taskInfo);
    EXPECT_FALSE(name.empty());
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetConciseTaskName_When_Sdma_Expect_ReturnName)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    std::string name = HcclCommTaskExceptionLite::GetInstance().GetConciseTaskName(taskInfo);
    EXPECT_FALSE(name.empty());
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetNotifyInfo_When_UbDma_Expect_ReturnNotifyId)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE);
    taskInfo.taskPara.ubDma.notifyId = 42;
    std::string info = HcclCommTaskExceptionLite::GetInstance().GetNotifyInfo(taskInfo);
    EXPECT_EQ(info, "42");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetNotifyInfo_When_ReducWithNotify_Expect_ReturnNotifyId)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY);
    taskInfo.taskPara.Reduce.notifyId = 99;
    std::string info = HcclCommTaskExceptionLite::GetInstance().GetNotifyInfo(taskInfo);
    EXPECT_EQ(info, "99");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetNotifyInfo_When_DefaultTask_Expect_ReturnSlash)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
    std::string info = HcclCommTaskExceptionLite::GetInstance().GetNotifyInfo(taskInfo);
    EXPECT_EQ(info, "/");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_DfxOpInfoInvalid_Expect_ReturnInvalidRankId)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    u32 rankId = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rankId, Hccl::DFX_INVALID_RANKID);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GetRemoteRankId_When_OpInfoNull_Expect_ReturnInvalidRankId)
{
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.hcclCommDfxLite = nullptr;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    u32 rankId = HcclCommTaskExceptionLite::GetInstance().GetRemoteRankId(taskInfo);
    EXPECT_EQ(rankId, Hccl::DFX_INVALID_RANKID);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_PrintTaskContextInfo_When_QueueNull_Expect_ReturnParaError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_context";
    InitCommEngineResMgr(aicpuComm);
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().PrintTaskContextInfo(&aicpuComm, 0, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_CollectTaskContext_When_QueueNull_Expect_ReturnParaError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_collect";
    InitCommEngineResMgr(aicpuComm);
    std::vector<Hccl::DfxTaskInfo*> taskContext;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().CollectTaskContext(&aicpuComm, 0, 0, taskContext);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GenerateErrorMessageReport_When_DfxOpInfoInvalid_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_gen_err";
    InitCommEngineResMgr(aicpuComm);
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    rtLogicCqReport_t exceptionInfo{};
    Hccl::ErrorMessageReport errMsgInfo{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsgInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_CommNull_Expect_ReturnPtrError)
{
    rtLogicCqReport_t exceptionInfo{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(nullptr, exceptionInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

class MockThreadForReportErr : public Thread {
public:
    MockThreadForReportErr(void* streamPtr) : streamPtr_(streamPtr) {}
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult DeInit() override { return HCCL_SUCCESS; }
    std::string& GetUniqueId() override { return uniqueId_; }
    uint32_t GetNotifyNum() const override { return 0; }
    LocalNotify* GetNotify(uint32_t index) const override { return nullptr; }
    HcclResult SupplementNotify(uint32_t notifyNum) override { return HCCL_SUCCESS; }
    bool IsDeviceA5() const override { return true; }
    Stream* GetStream() const override { return nullptr; }
    void* GetStreamLitePtr() const override { return streamPtr_; }
    void LaunchTask() const override {}
    void TryLaunchTask() const override {}
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override { return HCCL_SUCCESS; }
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override { return HCCL_SUCCESS; }
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override
    {
        return HCCL_SUCCESS;
    }
    bool GetMaster() const override { return false; }
    void SetIsMaster(bool isMaster) override {}

private:
    void* streamPtr_;
    std::string uniqueId_{"mock_thread"};
};

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_FindDfxTaskInfoNull_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 99;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_DfxOpInfoInvalid_Expect_ReturnPtrError)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
    ASSERT_NE(slot, nullptr);
    slot->taskId = (1U << 16) | 0U;
    slot->dfxOpInfo = DFX_INVALID_U64;
    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());

    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.taskId = 1;
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 0;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrMsg_When_ErrorAlreadyReported_Expect_ReturnSuccess)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_report_err";
    InitCommEngineResMgr(aicpuComm);
    aicpuComm.isErrorReported_ = true;
    auto streamLite = std::make_shared<Hccl::StreamLite>(0, 0, 0, 0);
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(streamLite->taskInfos_.NextSlot());
    ASSERT_NE(slot, nullptr);
    slot->taskId = (1U << 16) | 0U;
    Hccl::DfxDfxOpInfo opInfo{};
    slot->dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    auto thread = std::make_shared<MockThreadForReportErr>(streamLite.get());

    aicpuComm.GetCommEngineResMgr()->threadMgr_->threads_.push_back(thread);
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.taskId = 1;
    exceptionInfo.streamId = 0;
    exceptionInfo.sqId = 0;
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().ReportErrMsg(&aicpuComm, exceptionInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_GenerateErrorMessageReport_When_UbTask_Expect_JettyFilled)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "test_gen_err_jetty";
    InitCommEngineResMgr(aicpuComm);

    const u64 jettyHandle = 0x1234567890abULL;
    const u32 jettyId = 7;

    Hccl::DfxDfxOpInfo opInfo{};
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB);
    taskInfo.taskPara.ubDma.jettyHandle = jettyHandle;
    taskInfo.taskPara.ubDma.jettyId = jettyId;

    rtLogicCqReport_t exceptionInfo{};
    Hccl::ErrorMessageReport errMsgInfo{};
    HcclResult ret = HcclCommTaskExceptionLite::GetInstance().GenerateErrorMessageReport(
        &aicpuComm, taskInfo, exceptionInfo, errMsgInfo);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(errMsgInfo.jettyHandle, jettyHandle);
    EXPECT_EQ(errMsgInfo.jettyId, jettyId);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_HandleHostErrorReport_When_DuplicateReport_Expect_SkipSecond)
{
    const s32 testDeviceId = 62;
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);

    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));
    handler->HandleHostErrorReport(&exceptionInfo, taskInfo);
    GlobalMockObject::verify();

    MOCKER(RptInputErr).expects(never());
    handler->HandleHostErrorReport(&exceptionInfo, taskInfo);
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_ReportErrorMsg_When_DuplicateReport_Expect_SkipSecond)
{
    const s32 testDeviceId = 61;
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    Hccl::TaskInfo taskInfo(0, 0, 1, taskParam, nullptr);

    Hccl::ErrorMessageReport errorMessage{};

    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));
    handler->ReportErrorMsg(taskInfo, "", errorMessage, &exceptionInfo, "");
    GlobalMockObject::verify();

    MOCKER(RptInputErr).expects(never());
    handler->ReportErrorMsg(taskInfo, "", errorMessage, &exceptionInfo, "");
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_Register_When_CommRegisterMapEmpty_Expect_RegisterCallbackAndUnregisterLegacy)
{
    const s32 testDeviceId = 62;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);
    handler->CommRegisterMap_.clear();

    HcclResult ret = handler->Register(0xABCD);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler->CommRegisterMap_.count(0xABCD), 1u);

    handler->CommRegisterMap_.clear();
}

// ============ NotifyControlPlaneOnUbError 测试 ============

TEST_F(hcclCommTaskExceptionLiteTest, Ut_NotifyControlPlaneOnUbError_When_UbCqeStatusZero_Expect_Return)
{
    const s32 testDeviceId = 0;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    Hccl::ErrorMessageReport errorMessage{};
    errorMessage.ubCqeStatus = 0;
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x100);

    MOCKER(RaCtxNotifyEvent).expects(never());

    EXPECT_NO_THROW(handler->NotifyControlPlaneOnUbError(0, rdmaHandle, errorMessage));
    GlobalMockObject::verify();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_NotifyControlPlaneOnUbError_When_CapabilityNotSupported_Expect_Return)
{
    const s32 testDeviceId = 0;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    Hccl::ErrorMessageReport errorMessage{};
    errorMessage.ubCqeStatus = 1;
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x100);

    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(false));
    MOCKER(RaCtxNotifyEvent).expects(never());

    EXPECT_NO_THROW(handler->NotifyControlPlaneOnUbError(0, rdmaHandle, errorMessage));
    GlobalMockObject::verify();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_NotifyControlPlaneOnUbError_When_RdmaHandleNull_Expect_Return)
{
    const s32 testDeviceId = 0;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    Hccl::ErrorMessageReport errorMessage{};
    errorMessage.ubCqeStatus = 1;

    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
    MOCKER(RaCtxNotifyEvent).expects(never());

    EXPECT_NO_THROW(handler->NotifyControlPlaneOnUbError(0, nullptr, errorMessage));
    GlobalMockObject::verify();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_NotifyControlPlaneOnUbError_When_Success_Expect_RaCtxNotifyEventCalled)
{
    const s32 testDeviceId = 0;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    Hccl::ErrorMessageReport errorMessage{};
    errorMessage.ubCqeStatus = 1;
    errorMessage.tpn = 5;
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x200);

    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
    MOCKER(RaCtxNotifyEvent).expects(once()).with(mockcpp::any(), mockcpp::any()).will(returnValue(0));

    EXPECT_NO_THROW(handler->NotifyControlPlaneOnUbError(0, rdmaHandle, errorMessage));
    GlobalMockObject::verify();
}

TEST_F(hcclCommTaskExceptionLiteTest, Ut_NotifyControlPlaneOnUbError_When_RaCtxNotifyEventFails_Expect_NoCrash)
{
    const s32 testDeviceId = 0;
    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(handler, nullptr);

    Hccl::ErrorMessageReport errorMessage{};
    errorMessage.ubCqeStatus = 1;
    errorMessage.tpn = 5;
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x300);

    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
    MOCKER(RaCtxNotifyEvent).expects(once()).with(mockcpp::any(), mockcpp::any()).will(returnValue(-1));

    EXPECT_NO_THROW(handler->NotifyControlPlaneOnUbError(0, rdmaHandle, errorMessage));
    GlobalMockObject::verify();
}
