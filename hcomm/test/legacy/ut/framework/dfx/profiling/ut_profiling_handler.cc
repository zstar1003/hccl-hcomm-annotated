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
#include <stdexcept>
#include <string>
#define private public
#define protected public
#include "profiling_handler.h"
#include "dlprof_function_v2.h"
#include "mirror_task_manager.h"
#include "communicator_impl.h"

#undef private
#undef protected

using namespace Hccl;

class ProfilingHandlerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ProfilingHandlerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ProfilingHandlerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in ProfilingHandlerTest SetUP" << std::endl;
        // 重置 DlProfFunction 为 stub 模式，避免加载 libprofapi.so 导致真实 profiling 函数返回错误
        Hccl::DlProfFunction::GetInstance().initializedFlag_ = false;
        Hccl::DlProfFunction::GetInstance().DlProfFunctionStubInit();
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ProfilingHandlerTest TearDown" << std::endl;
    }
};

// 全局状态为false：测试ReportHcclOpInfo接口
TEST_F(ProfilingHandlerTest, ReportHostApi_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    OpType opTyep = OpType::ALLREDUCE;
    uint64_t beginTime = 0;
    uint64_t endTime = 2;
    bool cachedReq = true;
    bool isAiCpu = 0;
    handler.ReportHostApi(opTyep, beginTime, endTime, cachedReq, isAiCpu);
}

TEST_F(ProfilingHandlerTest, ReportHcclOp_test)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    CommunicatorImpl* comm = new CommunicatorImpl;
    comm->rankSize = 2;
    dfxOpInfo->comm_ = comm;
    dfxOpInfo->op_ = op;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    bool cachedReq = true;
    u32 ranksize = comm->GetRankSize();
    handler.ReportHcclOp(*dfxOpInfo, cachedReq);
    delete comm;
}

TEST_F(ProfilingHandlerTest, ReportHcclTaskApi_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    uint64_t beginTime = 0;
    uint64_t endTime = 1;
    bool isMasterSream = true;
    bool igonreLevel = false;
    bool cachedReq = true;
    handler.ReportHcclTaskApi(taskParam.taskType, beginTime, endTime, isMasterSream, cachedReq, igonreLevel);
}

TEST_F(ProfilingHandlerTest, ReportHcclTaskApi1_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    uint64_t beginTime = 0;
    uint64_t endTime = 1;
    bool isMasterSream = true;
    bool igonreLevel = false;
    bool cachedReq = true;
    handler.enableHcclNode_ = true;
    handler.enableHcclL1_ = true;
    handler.ReportHcclTaskApi(taskParam.taskType, beginTime, endTime, isMasterSream, cachedReq, igonreLevel);
}

TEST_F(ProfilingHandlerTest, ReportHcclTaskApi2_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    uint64_t beginTime = 0;
    uint64_t endTime = 1;
    bool isMasterSream = true;
    bool igonreLevel = false;
    bool cachedReq = true;
    handler.enableHcclNode_ = true;
    handler.ReportHcclTaskApi(taskParam.taskType, beginTime, endTime, isMasterSream, cachedReq, igonreLevel);
}

TEST_F(ProfilingHandlerTest, ReportHcclTaskDetails_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    handler.ReportHcclTaskDetails(*taskInfo, cachedReq);
    delete comm;
}

TEST_F(ProfilingHandlerTest, ut_CommandHandle_When_ParamIsNullptr_Expect_ReturnHCCL_E_PARA)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint32_t rtType = 0;
    void* data = nullptr;
    uint32_t len = 0;
    auto ret = handler.CommandHandle(rtType, data, len);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ProfilingHandlerTest, ut_CommandHandle_When_TypeIsUnexpected_Expect_ReturnHCCL_SUCCESS)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint32_t rtType = 1;
    rtProfCommandHandle_t data{};
    data.type = 0;
    uint32_t len = 0;
    auto ret = handler.CommandHandle(rtType, static_cast<void*>(&data), len);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ProfilingHandlerTest, GetHCCLReportData_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_SDMA,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

TEST_F(ProfilingHandlerTest, GetHCCLReportData1_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_REDUCE_INLINE,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

TEST_F(ProfilingHandlerTest, GetHCCLReportData2_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

TEST_F(ProfilingHandlerTest, GetHCCLReportData3_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>();
    for (int i = 0; i < 3; ++i) {
        CcuProfilingInfo info;
        info.name = "StubTask" + std::to_string(i);
        info.type = i % 2; // 循环使用不同的类型
        info.dieId = i;
        info.missionId = i + 1;
        info.instrId = i + 2;
        info.reduceOpType = i + 3;
        info.inputDataType = i + 4;
        info.outputDataType = i + 5;
        info.dataSize = (i + 1) * 1024;
        info.ckeId = i + 6;
        info.mask = i + 7;
        for (int j = 0; j < CCU_MAX_CHANNEL_NUM; ++j) {
            info.channelId[j] = j;
            info.remoteRankId[j] = j + 1;
        }
        ccuDetailInfo->push_back(info);
    }
    taskParam.ccuDetailInfo = std::move(ccuDetailInfo);
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

TEST_F(ProfilingHandlerTest, GetHCCLReportData4_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};

    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>();
    for (int i = 0; i < 3; ++i) {
        CcuProfilingInfo info;
        info.name = "StubTask" + std::to_string(i);
        info.type = i % 2; // 循环使用不同的类型
        info.dieId = i;
        info.missionId = i + 1;
        info.instrId = i + 2;
        info.reduceOpType = i + 3;
        info.inputDataType = i + 4;
        info.outputDataType = i + 5;
        info.dataSize = (i + 1) * 1024;
        info.ckeId = i + 6;
        info.mask = i + 7;
        for (int j = 0; j < CCU_MAX_CHANNEL_NUM; ++j) {
            info.channelId[j] = j;
            info.remoteRankId[j] = j + 1;
        }
        ccuDetailInfo->push_back(info);
    }
    taskParam.ccuDetailInfo = std::move(ccuDetailInfo);

    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    CcuProfilingInfo info;
    handler.GetCcuGroupInfo(*taskInfo, info);
    delete comm;
}

// 测试 GetHCCLReportData - TASK_DPU_INLINE_WRITE 类型
TEST_F(ProfilingHandlerTest, GetHCCLReportData_DpuInlineWrite_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam - TASK_DPU_INLINE_WRITE 使用 DMA 参数
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_DPU_INLINE_WRITE,
           .beginTime = 100,
           .endTime = 200,
           .aicpuTaskId = 12345,
           .npuDevId = 0,
           .taskPara
           = {.DMA
              = {.src = (void*)0x1000,
                 .dst = (void*)0x2000,
                 .size = 1024,
                 .notifyID = 1,
                 .linkType = Hccl::DfxLinkType::PCIE}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

// 测试 GetHCCLReportData - TASK_DPU_WRITE_WITH_NOTIFY 类型
TEST_F(ProfilingHandlerTest, GetHCCLReportData_DpuWriteWithNotify_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam - TASK_DPU_WRITE_WITH_NOTIFY 使用 DMA 参数
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY,
           .beginTime = 100,
           .endTime = 200,
           .aicpuTaskId = 12346,
           .npuDevId = 1,
           .taskPara
           = {.DMA
              = {.src = (void*)0x3000,
                 .dst = (void*)0x4000,
                 .size = 2048,
                 .notifyID = 2,
                 .linkType = Hccl::DfxLinkType::ROCE}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(4, 1, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

// 测试 GetHCCLReportData - TASK_DPU_NOTIFY_WAIT 类型
TEST_F(ProfilingHandlerTest, GetHCCLReportData_DpuNotifyWait_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam - TASK_DPU_NOTIFY_WAIT 使用 Notify 参数
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_DPU_NOTIFY_WAIT,
           .beginTime = 100,
           .endTime = 200,
           .aicpuTaskId = 12347,
           .npuDevId = 2,
           .taskPara = {.Notify = {.notifyID = 3, .value = 456}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(5, 2, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

// 测试 GetHCCLReportData - TASK_DPU_CHANNEL_FENCE 类型
TEST_F(ProfilingHandlerTest, GetHCCLReportData_DpuChannelFence_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam - TASK_DPU_CHANNEL_FENCE 使用 Notify 参数
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_DPU_CHANNEL_FENCE,
           .beginTime = 100,
           .endTime = 200,
           .aicpuTaskId = 12348,
           .npuDevId = 3,
           .taskPara = {.Notify = {.notifyID = 4, .value = 789}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(6, 3, 0, taskParam, dfxOpInfo);
    HCCLReportData hcclReportData;
    handler.GetHCCLReportData(*taskInfo, hcclReportData);
    delete comm;
}

TEST_F(ProfilingHandlerTest, GetCcuWaitSignalInfo_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    bool cachedReq = true;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_CCU,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};

    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>();
    for (int i = 0; i < 3; ++i) {
        CcuProfilingInfo info;
        info.name = "StubTask" + std::to_string(i);
        info.type = i % 2; // 循环使用不同的类型
        info.dieId = i;
        info.missionId = i + 1;
        info.instrId = i + 2;
        info.reduceOpType = i + 3;
        info.inputDataType = i + 4;
        info.outputDataType = i + 5;
        info.dataSize = (i + 1) * 1024;
        info.ckeId = i + 6;
        info.mask = i + 7;
        for (int j = 0; j < CCU_MAX_CHANNEL_NUM; ++j) {
            info.channelId[j] = j;
            info.remoteRankId[j] = j + 1;
        }
        ccuDetailInfo->push_back(info);
    }
    taskParam.ccuDetailInfo = std::move(ccuDetailInfo);

    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    CcuProfilingInfo info;
    handler.GetCcuWaitSignalInfo(*taskInfo, info);
    delete comm;
}

TEST_F(ProfilingHandlerTest, ReportAclApi_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint32_t cmdType = 0;
    uint64_t beginTime = 0;
    uint64_t endTime = 1;
    uint64_t cmdItemId = 0;
    uint32_t threadId = 0;
    handler.ReportAclApi(cmdType, beginTime, endTime, cmdItemId, threadId, false);
}

TEST_F(ProfilingHandlerTest, ReportAclApi_CachedReq_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint32_t cmdType = 0;
    uint64_t beginTime = 0;
    uint64_t endTime = 1;
    uint64_t cmdItemId = 0;
    uint32_t threadId = 0;
    bool cachedReq = true;
    size_t beforeSize = handler.cachedAclApiInfo_.size();
    handler.ReportAclApi(cmdType, beginTime, endTime, cmdItemId, threadId, cachedReq);
    EXPECT_EQ(handler.cachedAclApiInfo_.size(), beforeSize + 1);
}

TEST_F(ProfilingHandlerTest, ReportNodeApi_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t beginTime = 0;
    uint64_t endTime = 1;
    uint64_t cmdItemId = 0;
    uint32_t threadId = 0;
    handler.enableHostApi_ = true;
    handler.ReportNodeApi(beginTime, endTime, cmdItemId, threadId, false);
}

TEST_F(ProfilingHandlerTest, ReportNodeBasicInfo_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t timeStamp = 0;
    uint64_t cmdItemId = 0;
    uint32_t threadId = 0;
    handler.enableHcclL1_ = true;
    handler.ReportNodeBasicInfo(timeStamp, cmdItemId, threadId, false);
}

TEST_F(ProfilingHandlerTest, ReportHcclOpInfo_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t timeStamp = 0;
    uint32_t threadId = 0;
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opTag = "StubTag";
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    CommunicatorImpl* comm = new CommunicatorImpl;
    comm->rankSize = 2;
    dfxOpInfo->comm_ = comm;
    dfxOpInfo->op_ = op;
    handler.enableHcclL0_ = true;
    u32 ranksize = comm->GetRankSize();
    handler.ReportHcclOpInfo(timeStamp, *dfxOpInfo, threadId, false);
    delete comm;
}

TEST_F(ProfilingHandlerTest, ReportHcclOpInfo_alltoallv_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t timeStamp = 0;
    uint32_t threadId = 0;
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opTag = "StubTag";
    op.opType = OpType::ALLTOALLV;
    op.staticAddr = false;
    CommunicatorImpl* comm = new CommunicatorImpl;
    comm->rankSize = 2;
    dfxOpInfo->comm_ = comm;
    dfxOpInfo->op_ = op;
    handler.enableHcclL0_ = true;
    u32 ranksize = comm->GetRankSize();
    handler.ReportHcclOpInfo(timeStamp, *dfxOpInfo, threadId, false);
    delete comm;
}

TEST_F(ProfilingHandlerTest, StartSubscribe_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t profconfig = 0;
    handler.StartSubscribe(profconfig);
    profconfig = PROF_ACL_API_MASK;
    handler.StartSubscribe(profconfig);
    profconfig = PROF_TASK_TIME_MASK;
    handler.StartSubscribe(profconfig);
    profconfig = PROF_TASK_TIME_L1_MASK;
    handler.StartSubscribe(profconfig);
}

// 跑完后看是否还可以继续添加接口调用
TEST_F(ProfilingHandlerTest, GetInitCacheData_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t profconfig = 0;
    handler.StartSubscribe(profconfig);
    handler.StartHostApiSubscribe();
    handler.StartHostHcclOpSubscribe();
    handler.StartTaskApiSubscribe();
    handler.StartAdditionInfoSubscribe();
    handler.StartCcuSubscribe();
    handler.StopSubscribe();
}

TEST_F(ProfilingHandlerTest, GetProfState_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    EXPECT_EQ(false, handler.GetHostApiState());
    EXPECT_EQ(false, handler.GetHcclNodeState());
    EXPECT_EQ(false, handler.GetHcclL0State());
    EXPECT_EQ(false, handler.GetHcclL1State());
}

TEST_F(ProfilingHandlerTest, Ut_ReportCcuInfo_When_ReportAllType)
{
    MOCKER_CPP(&Hccl::ProfilingHandler::GetCcuTaskInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&Hccl::ProfilingHandler::GetCcuWaitSignalInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&Hccl::ProfilingHandler::GetCcuGroupInfo).stubs().will(ignoreReturnValue());
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>();
    CcuProfilingInfo info0{};
    CcuProfilingInfo info1{};
    CcuProfilingInfo infoCcu{};
    info0.type = 0;
    info1.type = 1;
    infoCcu.type = 2;
    ccuDetailInfo->push_back(info0);
    ccuDetailInfo->push_back(info1);
    ccuDetailInfo->push_back(infoCcu);
    TaskParam fakeTaskParam{};
    TaskInfo fakeTaskInfo(0, 0, 0, fakeTaskParam, nullptr, true);
    fakeTaskInfo.taskParam_.ccuDetailInfo = ccuDetailInfo;
    handler.ReportCcuInfo(fakeTaskInfo);
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_DfxOpInfoNullptr_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    TaskParam taskParam{};
    TaskInfo taskInfo(0, 0, 0, taskParam, nullptr);
    bool cachedReq = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetails(taskInfo, cachedReq));
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_CommNullptr_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->comm_ = nullptr;
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    TaskParam taskParam{};
    TaskInfo taskInfo(0, 0, 0, taskParam, dfxOpInfo);
    bool cachedReq = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetails(taskInfo, cachedReq));
}

TEST_F(ProfilingHandlerTest, Ut_GetHCCLReportData_When_DfxOpInfoNullptr_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    TaskParam taskParam{};
    TaskInfo taskInfo(0, 0, 0, taskParam, nullptr);
    HCCLReportData hcclReportData;
    EXPECT_NO_THROW(handler.GetHCCLReportData(taskInfo, hcclReportData));
}

TEST_F(ProfilingHandlerTest, Ut_ReportCcuInfo_When_DfxOpInfoNullptr_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    TaskParam taskParam{};
    TaskInfo taskInfo(0, 0, 0, taskParam, nullptr);
    EXPECT_NO_THROW(handler.ReportCcuInfo(taskInfo));
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclMC2CommInfoLog_When_Normal_Expect_Success)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    u32 kfcStreamId = 0;
    std::vector<u32> aicpuStreamsId = {1, 2, 3};
    std::string id = "test_comm_group";
    RankId myRank = 0;
    u32 rankSize = 4;
    RankId rankInParentComm = 0;
    EXPECT_NO_THROW(
        handler.ReportHcclMC2CommInfoLog(kfcStreamId, aicpuStreamsId, id, myRank, rankSize, rankInParentComm));
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_TaskInfosEmpty_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    std::vector<TaskInfo*> taskInfos;
    bool cachedReq = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetailsBatch(taskInfos, cachedReq));
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_DfxOpInfoNullptr_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    TaskParam taskParam{};
    auto taskInfo = std::make_unique<TaskInfo>(0, 0, 0, taskParam, nullptr);
    std::vector<TaskInfo*> taskInfos{taskInfo.get()};
    bool cachedReq = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetailsBatch(taskInfos, cachedReq));
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_CommNullptr_Expect_ReturnNormally)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->comm_ = nullptr;
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    TaskParam taskParam{};
    auto taskInfo = std::make_unique<TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    std::vector<TaskInfo*> taskInfos{taskInfo.get()};
    bool cachedReq = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetailsBatch(taskInfos, cachedReq));
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_L1DisabledAndNotCached_Expect_ReturnEarly)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = false;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    TaskParam taskParam{};
    auto taskInfo = std::make_unique<TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    std::vector<TaskInfo*> taskInfos{taskInfo.get()};
    bool cachedReq = false;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetailsBatch(taskInfos, cachedReq));
    delete comm;
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_CachedReq_Expect_CacheTaskInfos)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = false;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    TaskParam taskParam{};
    auto taskInfo = std::make_unique<TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    std::vector<TaskInfo*> taskInfos{taskInfo.get()};
    bool cachedReq = true;
    size_t beforeSize = handler.cacheTaskInfos_.size();
    EXPECT_NO_THROW(handler.ReportHcclTaskDetailsBatch(taskInfos, cachedReq));
    EXPECT_EQ(handler.cacheTaskInfos_.size(), beforeSize + taskInfos.size());
    delete comm;
}

TEST_F(ProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_L1EnabledAndCached_Expect_ReportNormally)
{
    MOCKER_CPP(&Hccl::ProfilingHandler::CallAdditionInfo).stubs().will(ignoreReturnValue());
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    handler.enableHcclL1_ = true;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CommunicatorImpl* comm = new CommunicatorImpl;
    dfxOpInfo->comm_ = comm;
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    auto taskInfo1 = std::make_unique<TaskInfo>(1, 0, 0, taskParam, dfxOpInfo);
    auto taskInfo2 = std::make_unique<TaskInfo>(2, 1, 0, taskParam, dfxOpInfo);
    std::vector<TaskInfo*> taskInfos{taskInfo1.get(), taskInfo2.get()};
    bool cachedReq = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetailsBatch(taskInfos, cachedReq));
    delete comm;
}
