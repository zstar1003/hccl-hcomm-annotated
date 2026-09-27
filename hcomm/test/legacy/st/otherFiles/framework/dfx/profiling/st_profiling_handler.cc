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
#define protected public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>
#include <string>
#include "profiling_handler.h"
#include "mirror_task_manager.h"
#include "communicator_impl.h"

#undef private
#undef protected

using namespace Hccl;

class ProfilingHandlerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ProfilingHandlerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ProfilingHandlerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ProfilingHandlerTest SetUP" << std::endl; }

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
    std::auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    handler.ReportHcclTaskDetails(*taskInfo, cachedReq);
    delete comm;
}

TEST_F(ProfilingHandlerTest, CommandHandle_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint32_t rtType = 0;
    void* data = nullptr;
    uint32_t len = 0;
    auto ret = handler.CommandHandle(rtType, data, len);
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
    std::auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
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
    std::auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
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
    std::auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
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
    std::auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
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
    std::auto taskInfo = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    CcuProfilingInfo info;
    handler.GetCcuGroupInfo(*taskInfo, info);
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

TEST_F(ProfilingHandlerTest, StartSubscribe_test)
{
    ProfilingHandler& handler = Hccl::ProfilingHandler::GetInstance();
    uint64_t profconfig = 0;
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
