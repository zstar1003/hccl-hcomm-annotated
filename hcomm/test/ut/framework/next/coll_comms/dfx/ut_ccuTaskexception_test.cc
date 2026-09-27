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
#include "hccl_api_base_test.h"
#include "ccu_comp.h"
#include "hcomm_c_adpt.h"
#define private public
#define protected public
#include "hcclCommTaskException.h"
#include "ccuTaskException.h"
#include "ccu_urma_channel.h"
#include "urma_endpoint.h"
#include "ccu_channel_ctx_pool.h"
#include "ccu_jetty_.h"
#undef protected
#undef private

using namespace hccl;
using namespace hcomm;

constexpr size_t CCU_CTX_RAW_CAPACITY = 64;

class CcuTaskExceptionTest : public BaseInit {
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

TEST_F(CcuTaskExceptionTest, ProccessCcuException_Normal)
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

TEST_F(CcuTaskExceptionTest, GetGroupRankInfo_Normal)
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

TEST_F(CcuTaskExceptionTest, GetGroupRankInfo_Nullptr)
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

TEST_F(CcuTaskExceptionTest, PrintPanicLogInfo_Normal)
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

TEST_F(CcuTaskExceptionTest, PrintPaniclogInfo) { EXPECT_NO_THROW(CcuTaskException::PrintPanicLogInfo(nullptr)); }

TEST_F(CcuTaskExceptionTest, GetCcuCKEValue_Normal)
{
    // 模拟HccpRaTlvRequestForCustomChannel
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    uint16_t result = CcuTaskException::GetCcuCKEValue(0, 0, 0);
    EXPECT_EQ(result, 0); // INVALID_U16
}

TEST_F(CcuTaskExceptionTest, GetCcuCKEValue_GetDevicePhyIdFail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint16_t result = CcuTaskException::GetCcuCKEValue(0, 0, 0);
    EXPECT_EQ(result, 65535);
}

TEST_F(CcuTaskExceptionTest, GetCcuCKEValue_CustomChannelFail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint16_t result = CcuTaskException::GetCcuCKEValue(0, 0, 0);
    EXPECT_EQ(result, 65535);
}

TEST_F(CcuTaskExceptionTest, GetMissContectF)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    CcuMissionContext result{};
    ASSERT_EQ(memcpy_s(&result, sizeof(result), raw, sizeof(result)), EOK);
    EXPECT_EQ(result.part0.value, 0u);
}

TEST_F(CcuTaskExceptionTest, GetDevidFail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_E_PARA);
    CcuMissionContext result{};
    ASSERT_EQ(memcpy_s(&result, sizeof(result), raw, sizeof(result)), EOK);
    EXPECT_EQ(result.part0.value, 0u);
}

TEST_F(CcuTaskExceptionTest, GetMissContectFail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_NE(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    CcuMissionContext result{};
    ASSERT_EQ(memcpy_s(&result, sizeof(result), raw, sizeof(result)), EOK);
    EXPECT_EQ(result.part0.value, 0u);
}

constexpr uint64_t INVALID_U64_VAL = 18446744073709551615ULL;
TEST_F(CcuTaskExceptionTest, GetCcuGSAValue)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    uint64_t result = CcuTaskException::GetCcuGSAValue(0, 0, 0);
    EXPECT_EQ(result, 0u);
}

TEST_F(CcuTaskExceptionTest, GetCcuGSAValuegetdevidfailed)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuGSAValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(CcuTaskExceptionTest, GetCcuGSAValuechannelfail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuGSAValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(CcuTaskExceptionTest, GetCcuXnValue)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    uint64_t result = CcuTaskException::GetCcuXnValue(0, 0, 0);
    EXPECT_EQ(result, 0u);
}

TEST_F(CcuTaskExceptionTest, GetCcuXnValuedevidfailed)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuXnValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(CcuTaskExceptionTest, GetCcuXnValuechannelfail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    uint64_t result = CcuTaskException::GetCcuXnValue(0, 0, 0);
    EXPECT_EQ(result, INVALID_U64_VAL);
}

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsg)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(1024);
    EXPECT_EQ(result, "");
}

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgzero)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(0);
    EXPECT_FALSE(result.empty());
}

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgDefault)
{
    CcuErrorInfo ccuErrorInfo = {};
    string result = CcuTaskException::GetCcuErrorMsgDefault(ccuErrorInfo);
    EXPECT_FALSE(result.empty());
}

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgMission)
{
    CcuErrorInfo ccuErrorInfo = {};
    ccuErrorInfo.type = CcuErrorType::MISSION;
    string result = CcuTaskException::GetCcuErrorMsgMission(ccuErrorInfo);
    EXPECT_FALSE(result.empty());
}

TEST_F(CcuTaskExceptionTest, GetCcuLoopContext_Normal)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    CcuLoopContext result{};
    ASSERT_EQ(memcpy_s(&result, sizeof(result), raw, sizeof(result)), EOK);
    EXPECT_EQ(result.part0.value, 0u);
}

TEST_F(CcuTaskExceptionTest, GetCcuLoopContext_GetDevicePhyIdFail)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));
    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_E_PARA);
    CcuLoopContext result{};
    ASSERT_EQ(memcpy_s(&result, sizeof(result), raw, sizeof(result)), EOK);
    EXPECT_EQ(result.part0.value, 0u);
}

TEST_F(CcuTaskExceptionTest, GenStatusInfo_Normal)
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

TEST_F(CcuTaskExceptionTest, GetRankIdByChannelId_Normal)
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

TEST_F(CcuTaskExceptionTest, GetAddrPairByChannelId_Normal)
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

TEST_F(CcuTaskExceptionTest, GetMSIdPerDie_Normal)
{
    uint16_t result1 = CcuTaskException::GetMSIdPerDie(0x8001);
    EXPECT_EQ(result1, 0x0001);

    uint16_t result2 = CcuTaskException::GetMSIdPerDie(0x7FFF);
    EXPECT_EQ(result2, 0x7FFF);

    uint16_t result3 = CcuTaskException::GetMSIdPerDie(0);
    EXPECT_EQ(result3, 0);
}

TEST_F(CcuTaskExceptionTest, GenStatusInfo_UnsupportedOpcode)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_LocalOpError)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_RemoteOpError)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_TransactionRetryExceeded)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_TransactionAckTimeout)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_JettyWorkRequestFlushed)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_CCUAAlgTaskError)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_MemoryECCError)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_CCUMExecuteError)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_CCUAExecuteError)
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

TEST_F(CcuTaskExceptionTest, GenStatusInfo_HighPartNotInMap)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLoop_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLoopGroup_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLocPostSem_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLocWaitEvent_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLocWaitNotify_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgRemPostSem_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgRemWaitSem_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgRemPostVar_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgPostSharedSem_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgRead_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgWrite_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLocalCpy_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgLocalReduce_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgBufRead_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgBufWrite_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgBufLocRead_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgBufLocWrite_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgBufReduce_DefaultValues)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgByType_MISSION)
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

TEST_F(CcuTaskExceptionTest, GetCcuErrorMsgByType_DefaultType)
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

TEST_F(CcuTaskExceptionTest, GetCcuLenErrorMsg_LargeValue)
{
    constexpr uint64_t CCU_MSG_256MB_LEN = 256 * 1024 * 1024;
    string result = CcuTaskException::GetCcuLenErrorMsg(CCU_MSG_256MB_LEN);
    EXPECT_TRUE(result.empty());
}

TEST_F(CcuTaskExceptionTest, GetCcuLenErrorMsg_NormalValue)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(1024 * 1024);
    EXPECT_TRUE(result.empty());
}

TEST_F(CcuTaskExceptionTest, GetCcuLenErrorMsg_Exceed256MB)
{
    constexpr uint64_t CCU_MSG_256MB_LEN = 256 * 1024 * 1024;
    uint64_t exceedLen = static_cast<uint64_t>(CCU_MSG_256MB_LEN) + 1;
    string result = CcuTaskException::GetCcuLenErrorMsg(exceedLen);
    EXPECT_FALSE(result.empty());
}

TEST_F(CcuTaskExceptionTest, GetCcuLenErrorMsg_Zero)
{
    string result = CcuTaskException::GetCcuLenErrorMsg(0);
    EXPECT_FALSE(result.empty());
}

TEST_F(CcuTaskExceptionTest, ProcessCcuException_MultipleMission)
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

TEST_F(CcuTaskExceptionTest, ProcessCcuException_CleanTaskKillStateFail)
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

TEST_F(CcuTaskExceptionTest, ProcessCcuException_CleanDieCkesFail)
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

TEST_F(CcuTaskExceptionTest, GetGroupRankInfo_WithValidComm)
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

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_ExceptionInfo_Nullptr)
{
    rtExceptionInfo_t* exceptionInfo = nullptr;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->ProcessCallback(exceptionInfo));
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_NotFound_DeviceId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.DestroyQueue(0, 0);
    globalMirrorTasks.DestroyQueue(0, 1);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_NotFound_StreamId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);
    globalMirrorTasks.DestroyQueue(0, 1);

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 1;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_NotFound_TaskId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    auto taskInfo = make_unique<Hccl::TaskInfo>(0, 1, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_DfxOpInfo_Nullptr)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    auto taskInfo = make_unique<Hccl::TaskInfo>(0, 0, 0, taskParam, nullptr);
    globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_IsIndop_False)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = false;
    auto taskInfo = make_unique<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_TaskType_TaskCCU)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    auto taskInfo = make_unique<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_Process_FindTaskInfo_Success_TaskType_NotCCU)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    auto taskInfo = make_unique<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = 0;
    exceptionInfo.taskid = 0;
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->Process(&exceptionInfo));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_Queue_Nullptr)
{
    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->PrintTaskContextInfo(0, 0, 0, ""));
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_Task_Not_Found)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->isIndop_ = true;
    auto taskInfo = make_unique<Hccl::TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);
    globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));

    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->PrintTaskContextInfo(0, 0, 0, ""));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_TaskId_Greater_Then_ErrTaskId)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    for (u32 i = 0; i < 5; i++) {
        Hccl::TaskParam taskParam{};
        taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
        shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
        dfxOpInfo->isIndop_ = true;
        auto taskInfo = make_unique<Hccl::TaskInfo>(0, i + 10, 0, taskParam, dfxOpInfo);
        globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));
    }

    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->PrintTaskContextInfo(0, 0, 5, ""));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, TaskExceptionHost_PrintTaskContextInfo_Success)
{
    Hccl::GlobalMirrorTasks& globalMirrorTasks = Hccl::GlobalMirrorTasks::Instance();
    globalMirrorTasks.CreateQueue(0, 0, Hccl::QueueType::Circular_Queue);

    for (u32 i = 0; i < 5; i++) {
        Hccl::TaskParam taskParam{};
        taskParam.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
        shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = make_shared<Hccl::DfxOpInfo>();
        dfxOpInfo->isIndop_ = true;
        auto taskInfo = make_unique<Hccl::TaskInfo>(0, i, 0, taskParam, dfxOpInfo);
        globalMirrorTasks.GetQueue(0, 0)->Append(std::move(taskInfo));
    }

    EXPECT_NO_THROW(hcomm::TaskExceptionHost::GetInstance(0)->PrintTaskContextInfo(0, 0, 4, ""));

    globalMirrorTasks.DestroyQueue(0, 0);
}

TEST_F(CcuTaskExceptionTest, Ut_GetChannleIdByCcuErrorInfo)
{
    uint16_t remPostSemChannleId = 1;
    CcuErrorInfo remPostSem;
    remPostSem.repType = CcuRep::CcuRepType::REM_POST_SEM;
    remPostSem.msg.waitSignal.channelId[0] = remPostSemChannleId;
    EXPECT_EQ(CcuTaskException::GetChannleIdByCcuErrorInfo(remPostSem), remPostSemChannleId);

    uint16_t readChannleId = 2;
    CcuErrorInfo read;
    read.repType = CcuRep::CcuRepType::READ;
    read.msg.transMem.channelId = readChannleId;
    EXPECT_EQ(CcuTaskException::GetChannleIdByCcuErrorInfo(read), readChannleId);

    uint16_t bufReadChannleId = 3;
    CcuErrorInfo bufRead;
    bufRead.repType = CcuRep::CcuRepType::BUF_READ;
    bufRead.msg.bufTransMem.channelId = bufReadChannleId;
    EXPECT_EQ(CcuTaskException::GetChannleIdByCcuErrorInfo(bufRead), bufReadChannleId);

    CcuErrorInfo loop;
    bufRead.repType = CcuRep::CcuRepType::LOOP;
    EXPECT_EQ(CcuTaskException::GetChannleIdByCcuErrorInfo(loop), 65535);
}

// ============ PrintCcuUbRegisters 测试 ============

TEST_F(CcuTaskExceptionTest, PrintCcuUbRegisters_EmptyErrorInfos_ExpectSuccess)
{
    std::vector<CcuErrorInfo> errorInfos;
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};
    EXPECT_EQ(CcuTaskException::PrintCcuUbRegisters(errorInfos, 0, taskInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionTest, PrintCcuUbRegisters_RepTypeNotInRepWithChannel_ExpectSuccess)
{
    CcuErrorInfo errorInfo;
    errorInfo.repType = CcuRep::CcuRepType::LOOP;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};
    EXPECT_EQ(CcuTaskException::PrintCcuUbRegisters(errorInfos, 0, taskInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionTest, PrintCcuUbRegisters_InvalidChannelId_ExpectSuccess)
{
    CcuErrorInfo errorInfo;
    errorInfo.repType = CcuRep::CcuRepType::REM_POST_SEM;
    errorInfo.msg.waitSignal.channelId[0] = 65535;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};
    EXPECT_EQ(CcuTaskException::PrintCcuUbRegisters(errorInfos, 0, taskInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionTest, PrintCcuUbRegisters_MultipleInvalidChannelIds_ExpectSuccess)
{
    CcuErrorInfo errorInfo1;
    errorInfo1.repType = CcuRep::CcuRepType::REM_POST_SEM;
    errorInfo1.msg.waitSignal.channelId[0] = 65535;
    CcuErrorInfo errorInfo2;
    errorInfo2.repType = CcuRep::CcuRepType::READ;
    errorInfo2.msg.transMem.channelId = 65535;
    std::vector<CcuErrorInfo> errorInfos{errorInfo1, errorInfo2};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};
    EXPECT_EQ(CcuTaskException::PrintCcuUbRegisters(errorInfos, 0, taskInfo), HCCL_SUCCESS);
}

// Cover PrintCcuUbRegisters ctxGroups grouping logic added by PR4843:
// build non-empty ccuJettys, run ctxGroups -> HccpBatchQueryJettyStatus -> statusVec backfill.

TEST_F(CcuTaskExceptionTest, PrintCcuUbRegisters_When_JettysExist_Expect_QueryByCtxGroups)
{
    constexpr uint16_t channelId = 200;
    constexpr uint8_t dieId = 0;
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle[channelId] = 0xAAAA;
    }

    CcuJetty jetty(Hccl::IpAddress{}, CcuJettyInfo{});
    jetty.ctxHandle_ = reinterpret_cast<CtxHandle>(0x1);
    jetty.jettyHandlePtr_ = reinterpret_cast<void*>(0x2);
    jetty.rdmaHandle_ = reinterpret_cast<RdmaHandle>(0x3);
    jetty.jettyInfo_.taJettyId = 7;

    HcommChannelDesc chDesc{};
    CcuUrmaChannel channel(reinterpret_cast<EndpointHandle>(0x10), chDesc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    CcuChannelCtxPool pool(0);
    pool.channelJettyInfoMap_[{dieId, channelId}].second.push_back(&jetty);

    EndpointDesc epDesc{};
    UrmaEndpoint endpoint(epDesc);
    endpoint.ccuChannelCtxPool_.reset(&pool);
    void* endpointPtr = static_cast<Endpoint*>(&endpoint);

    MOCKER(HcommEndpointGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&endpointPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&UrmaEndpoint::GetCcuChannelCtxPool).stubs().will(returnValue(&pool));

    CcuErrorInfo errorInfo{};
    errorInfo.repType = CcuRep::CcuRepType::READ;
    errorInfo.dieId = dieId;
    errorInfo.msg.transMem.channelId = channelId;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    HcclResult ret = CcuTaskException::PrintCcuUbRegisters(errorInfos, 0, taskInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    pool.channelJettyInfoMap_.clear();
    (void)endpoint.ccuChannelCtxPool_.release();
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle.erase(channelId);
    }
}

// ============ NotifyControlPlaneOnUbError 测试 ============

static HcclResult StubBatchQueryJettyStatusError(
    const CtxHandle, const std::vector<JettyHandle>&, std::vector<JettyStatus>& statusVec, u32& num)
{
    statusVec.resize(num);
    for (u32 i = 0; i < num; ++i) {
        statusVec[i] = JettyStatus::ERROR;
    }
    return HCCL_SUCCESS;
}

static HcclResult StubBatchQueryJettyStatusReady(
    const CtxHandle, const std::vector<JettyHandle>&, std::vector<JettyStatus>& statusVec, u32& num)
{
    statusVec.resize(num);
    for (u32 i = 0; i < num; ++i) {
        statusVec[i] = JettyStatus::READY;
    }
    return HCCL_SUCCESS;
}

TEST_F(CcuTaskExceptionTest, NotifyControlPlaneOnUbError_When_EmptyErrorInfos_Expect_Return)
{
    std::vector<CcuErrorInfo> errorInfos;
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));

    EXPECT_NO_THROW(CcuTaskException::NotifyControlPlaneOnUbError(errorInfos, taskInfo, 0, 0x01));
    GlobalMockObject::verify();
}

TEST_F(CcuTaskExceptionTest, NotifyControlPlaneOnUbError_When_CapabilityNotSupported_Expect_Return)
{
    CcuErrorInfo errorInfo{};
    errorInfo.repType = CcuRep::CcuRepType::READ;
    errorInfo.msg.transMem.channelId = 200;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(false));

    EXPECT_NO_THROW(CcuTaskException::NotifyControlPlaneOnUbError(errorInfos, taskInfo, 0, 0x01));
    GlobalMockObject::verify();
}

TEST_F(CcuTaskExceptionTest, NotifyControlPlaneOnUbError_When_AllInvalidChannelIds_Expect_Return)
{
    CcuErrorInfo errorInfo1;
    errorInfo1.repType = CcuRep::CcuRepType::REM_POST_SEM;
    errorInfo1.msg.waitSignal.channelId[0] = 65535;
    CcuErrorInfo errorInfo2;
    errorInfo2.repType = CcuRep::CcuRepType::READ;
    errorInfo2.msg.transMem.channelId = 65535;
    std::vector<CcuErrorInfo> errorInfos{errorInfo1, errorInfo2};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));

    EXPECT_NO_THROW(CcuTaskException::NotifyControlPlaneOnUbError(errorInfos, taskInfo, 0, 0x01));
    GlobalMockObject::verify();
}

TEST_F(CcuTaskExceptionTest, NotifyControlPlaneOnUbError_When_JettysExist_Expect_NotifyEvent)
{
    constexpr uint16_t channelId = 200;
    constexpr uint8_t dieId = 0;
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle[channelId] = 0xAAAA;
    }

    CcuJetty jetty(Hccl::IpAddress{}, CcuJettyInfo{});
    jetty.ctxHandle_ = reinterpret_cast<CtxHandle>(0x1);
    jetty.jettyHandlePtr_ = reinterpret_cast<void*>(0x2);
    jetty.jettyInfo_.taJettyId = 7;

    HcommChannelDesc chDesc{};
    CcuUrmaChannel channel(reinterpret_cast<EndpointHandle>(0x10), chDesc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    CcuChannelCtxPool pool(0);
    pool.channelJettyInfoMap_[{dieId, channelId}].second.push_back(&jetty);

    EndpointDesc epDesc{};
    UrmaEndpoint endpoint(epDesc);
    endpoint.ccuChannelCtxPool_.reset(&pool);
    void* endpointPtr = static_cast<Endpoint*>(&endpoint);

    MOCKER(HcommEndpointGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&endpointPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&UrmaEndpoint::GetCcuChannelCtxPool).stubs().will(returnValue(&pool));

    MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
    MOCKER(HccpBatchQueryJettyStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(StubBatchQueryJettyStatusError));
    MOCKER(RaCtxNotifyEvent).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));

    CcuErrorInfo errorInfo{};
    errorInfo.repType = CcuRep::CcuRepType::READ;
    errorInfo.dieId = dieId;
    errorInfo.msg.transMem.channelId = channelId;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    EXPECT_NO_THROW(CcuTaskException::NotifyControlPlaneOnUbError(errorInfos, taskInfo, 0, 0x01));

    GlobalMockObject::verify();
    pool.channelJettyInfoMap_.clear();
    (void)endpoint.ccuChannelCtxPool_.release();
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle.erase(channelId);
    }
}

TEST_F(CcuTaskExceptionTest, NotifyControlPlaneOnUbError_When_RaCtxNotifyEventFails_Expect_NoCrash)
{
    constexpr uint16_t channelId = 201;
    constexpr uint8_t dieId = 0;
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle[channelId] = 0xBBBB;
    }

    CcuJetty jetty(Hccl::IpAddress{}, CcuJettyInfo{});
    jetty.ctxHandle_ = reinterpret_cast<CtxHandle>(0x1);
    jetty.jettyHandlePtr_ = reinterpret_cast<void*>(0x2);
    jetty.jettyInfo_.taJettyId = 8;

    HcommChannelDesc chDesc{};
    CcuUrmaChannel channel(reinterpret_cast<EndpointHandle>(0x10), chDesc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    CcuChannelCtxPool pool(0);
    pool.channelJettyInfoMap_[{dieId, channelId}].second.push_back(&jetty);

    EndpointDesc epDesc{};
    UrmaEndpoint endpoint(epDesc);
    endpoint.ccuChannelCtxPool_.reset(&pool);
    void* endpointPtr = static_cast<Endpoint*>(&endpoint);

    MOCKER(HcommEndpointGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&endpointPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&UrmaEndpoint::GetCcuChannelCtxPool).stubs().will(returnValue(&pool));

    MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
    MOCKER(HccpBatchQueryJettyStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(StubBatchQueryJettyStatusError));
    MOCKER(RaCtxNotifyEvent).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(-1));

    CcuErrorInfo errorInfo{};
    errorInfo.repType = CcuRep::CcuRepType::READ;
    errorInfo.dieId = dieId;
    errorInfo.msg.transMem.channelId = channelId;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    EXPECT_NO_THROW(CcuTaskException::NotifyControlPlaneOnUbError(errorInfos, taskInfo, 0, 0x01));

    GlobalMockObject::verify();
    pool.channelJettyInfoMap_.clear();
    (void)endpoint.ccuChannelCtxPool_.release();
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle.erase(channelId);
    }
}

TEST_F(CcuTaskExceptionTest, NotifyControlPlaneOnUbError_When_JettyStatusReady_Expect_Skip)
{
    constexpr uint16_t channelId = 202;
    constexpr uint8_t dieId = 0;
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle[channelId] = 0xCCCC;
    }

    CcuJetty jetty(Hccl::IpAddress{}, CcuJettyInfo{});
    jetty.ctxHandle_ = reinterpret_cast<CtxHandle>(0x1);
    jetty.jettyHandlePtr_ = reinterpret_cast<void*>(0x2);
    jetty.jettyInfo_.taJettyId = 9;

    HcommChannelDesc chDesc{};
    CcuUrmaChannel channel(reinterpret_cast<EndpointHandle>(0x10), chDesc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    CcuChannelCtxPool pool(0);
    pool.channelJettyInfoMap_[{dieId, channelId}].second.push_back(&jetty);

    EndpointDesc epDesc{};
    UrmaEndpoint endpoint(epDesc);
    endpoint.ccuChannelCtxPool_.reset(&pool);
    void* endpointPtr = static_cast<Endpoint*>(&endpoint);

    MOCKER(HcommEndpointGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&endpointPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&UrmaEndpoint::GetCcuChannelCtxPool).stubs().will(returnValue(&pool));

    MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER(RaHasCapability).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
    MOCKER(HccpBatchQueryJettyStatus)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(StubBatchQueryJettyStatusReady));
    MOCKER(RaCtxNotifyEvent).expects(never());

    CcuErrorInfo errorInfo{};
    errorInfo.repType = CcuRep::CcuRepType::READ;
    errorInfo.dieId = dieId;
    errorInfo.msg.transMem.channelId = channelId;
    std::vector<CcuErrorInfo> errorInfos{errorInfo};
    Hccl::TaskParam taskParam{.taskType = Hccl::TaskParamType::TASK_CCU};
    Hccl::TaskInfo taskInfo{0, 0, 0, taskParam, nullptr, false};

    EXPECT_NO_THROW(CcuTaskException::NotifyControlPlaneOnUbError(errorInfos, taskInfo, 0, 0x01));

    GlobalMockObject::verify();
    pool.channelJettyInfoMap_.clear();
    (void)endpoint.ccuChannelCtxPool_.release();
    {
        std::lock_guard<std::mutex> lock(g_channelMapMutex);
        g_channelIdToHandle.erase(channelId);
    }
}
