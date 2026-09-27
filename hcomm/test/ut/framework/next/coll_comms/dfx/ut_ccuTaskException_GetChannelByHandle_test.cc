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
#include "ccu_rep_read_v1.h"
#include "ccu_rep_write_v1.h"
#include "ccu_rep_rempostsem_v1.h"
#include "ccu_rep_remwaitsem_v1.h"
#include "ccu_rep_rempostvar_v1.h"
#include "ccu_rep_bufread_v1.h"
#include "ccu_rep_bufwrite_v1.h"
#include "ccu_ins_generator_v1.h"

#define private public
#define protected public
#include "ccu_urma_channel.h"
#include "ccuTaskException.h"
#undef protected
#undef private

using namespace hccl;
using namespace hcomm;

// 本测试组覆盖 PR4086 在 ccuTaskException.cc 中新增的三个私有静态方法：
//   GetChannelIdByHandle / GetSignalIdByHandle / GetVariableIdByHandle
// 这三个方法均通过 HcommChannelGet 解析 channel 指针，再 dynamic_cast 为
// CcuUrmaChannel 调用其成员方法。这里使用真实 CcuUrmaChannel 对象 +
// mockcpp 打桩 HcommChannelGet 与 CcuUrmaChannel 成员函数的方式覆盖各分支。
class GetChannelByHandleTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

// ----------------------------- GetChannelIdByHandle -----------------------------

TEST_F(GetChannelByHandleTest, Ut_GetChannelIdByHandle_When_Normal_Expect_Success)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xAAAA;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    const uint32_t expectedId = 101U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedId));

    uint32_t channelId = 0;
    HcclResult ret = CcuTaskException::GetChannelIdByHandle(handle, channelId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(channelId, expectedId);
}

TEST_F(GetChannelByHandleTest, Ut_GetChannelIdByHandle_When_HcommChannelGetFail_Expect_ReturnError)
{
    ChannelHandle handle = 0xBBBB;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_PARA)));

    uint32_t channelId = 0;
    HcclResult ret = CcuTaskException::GetChannelIdByHandle(handle, channelId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(GetChannelByHandleTest, Ut_GetChannelIdByHandle_When_ChannelPtrNull_Expect_ReturnPtrError)
{
    void* nullPtr = nullptr;
    ChannelHandle handle = 0xCCCC;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&nullPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    uint32_t channelId = 0;
    HcclResult ret = CcuTaskException::GetChannelIdByHandle(handle, channelId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);
}

// ----------------------------- GetSignalIdByHandle -----------------------------

TEST_F(GetChannelByHandleTest, Ut_GetSignalIdByHandle_When_RmtSigNormal_Expect_Success)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xDDDD;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    const uint32_t expectedRmtCkeId = 200U;
    MOCKER_CPP(&CcuUrmaChannel::GetRmtCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedRmtCkeId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    uint32_t signalId = 0;
    HcclResult ret = CcuTaskException::GetSignalIdByHandle(handle, 0, true, signalId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(signalId, expectedRmtCkeId);
}

TEST_F(GetChannelByHandleTest, Ut_GetSignalIdByHandle_When_LocSigNormal_Expect_Success)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xEEEE;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    const uint32_t expectedLocCkeId = 300U;
    MOCKER_CPP(&CcuUrmaChannel::GetLocCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedLocCkeId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    uint32_t signalId = 0;
    HcclResult ret = CcuTaskException::GetSignalIdByHandle(handle, 1, false, signalId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(signalId, expectedLocCkeId);
}

TEST_F(GetChannelByHandleTest, Ut_GetSignalIdByHandle_When_GetRmtCkeFail_Expect_ReturnUnavail)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xFFFF;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&CcuUrmaChannel::GetRmtCkeByIndex)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_INTERNAL));

    uint32_t signalId = 0;
    HcclResult ret = CcuTaskException::GetSignalIdByHandle(handle, 0, true, signalId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(GetChannelByHandleTest, Ut_GetSignalIdByHandle_When_GetLocCkeFail_Expect_ReturnUnavail)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0x1111;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&CcuUrmaChannel::GetLocCkeByIndex)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_INTERNAL));

    uint32_t signalId = 0;
    HcclResult ret = CcuTaskException::GetSignalIdByHandle(handle, 0, false, signalId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(GetChannelByHandleTest, Ut_GetSignalIdByHandle_When_ChannelPtrNull_Expect_ReturnPtrError)
{
    void* nullPtr = nullptr;
    ChannelHandle handle = 0x2222;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&nullPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    uint32_t signalId = 0;
    HcclResult ret = CcuTaskException::GetSignalIdByHandle(handle, 0, true, signalId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);
}

// ----------------------------- GetVariableIdByHandle -----------------------------

TEST_F(GetChannelByHandleTest, Ut_GetVariableIdByHandle_When_Normal_Expect_Success)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0x3333;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    const uint32_t expectedRmtXnId = 400U;
    MOCKER_CPP(&CcuUrmaChannel::GetRmtXnByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedRmtXnId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    uint32_t varId = 0;
    HcclResult ret = CcuTaskException::GetVariableIdByHandle(handle, 0, varId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(varId, expectedRmtXnId);
}

TEST_F(GetChannelByHandleTest, Ut_GetVariableIdByHandle_When_GetRmtXnFail_Expect_ReturnUnavail)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0x4444;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&CcuUrmaChannel::GetRmtXnByIndex)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_INTERNAL));

    uint32_t varId = 0;
    HcclResult ret = CcuTaskException::GetVariableIdByHandle(handle, 0, varId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(GetChannelByHandleTest, Ut_GetVariableIdByHandle_When_ChannelPtrNull_Expect_ReturnPtrError)
{
    void* nullPtr = nullptr;
    ChannelHandle handle = 0x5555;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&nullPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    uint32_t varId = 0;
    HcclResult ret = CcuTaskException::GetVariableIdByHandle(handle, 0, varId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRead_When_Normal_Expect_ChannelIdFromHandle)
{
    CcuRep::Address locAddr;
    CcuRep::Variable locToken;
    CcuRep::Address remAddr;
    CcuRep::Variable remToken;
    CcuRep::Variable len;
    CcuRep::CompletedEvent sem;
    locAddr.Reset(11);
    locToken.Reset(12);
    remAddr.Reset(13);
    remToken.Reset(14);
    len.Reset(15);
    sem.Reset(16);

    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRead>(
        &insGen, static_cast<ChannelHandle>(101), CcuRep::LocalAddr(locAddr, locToken),
        CcuRep::RemoteAddr(remAddr, remToken), len, sem, 0x5A);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedId = 7U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedId));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRead(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
    EXPECT_EQ(errorInfo[0].msg.transMem.channelId, expectedId);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoWrite_When_Normal_Expect_ChannelIdFromHandle)
{
    CcuRep::Address locAddr;
    CcuRep::Variable locToken;
    CcuRep::Address remAddr;
    CcuRep::Variable remToken;
    CcuRep::Variable len;
    CcuRep::CompletedEvent sem;
    locAddr.Reset(21);
    locToken.Reset(22);
    remAddr.Reset(23);
    remToken.Reset(24);
    len.Reset(25);
    sem.Reset(26);

    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepWrite>(
        &insGen, static_cast<ChannelHandle>(102), CcuRep::RemoteAddr(remAddr, remToken),
        CcuRep::LocalAddr(locAddr, locToken), len, sem, 0xA5);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedId = 8U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedId));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoWrite(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
    EXPECT_EQ(errorInfo[0].msg.transMem.channelId, expectedId);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRemPostSem_When_Normal_Expect_IdsFromHandle)
{
    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRemPostSem>(&insGen, static_cast<ChannelHandle>(201), 3, 0x70);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedChId = 11U;
    const uint32_t expectedSigId = 22U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedChId));
    MOCKER_CPP(&CcuUrmaChannel::GetRmtCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedSigId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRemPostSem(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, expectedSigId);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, 0x70u);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], expectedChId);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRemWaitSem_When_Normal_Expect_IdsFromHandle)
{
    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRemWaitSem>(&insGen, static_cast<ChannelHandle>(202), 4, 0x80);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedChId = 12U;
    const uint32_t expectedSigId = 33U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedChId));
    MOCKER_CPP(&CcuUrmaChannel::GetLocCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedSigId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRemWaitSem(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, expectedSigId);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, 0x80u);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], expectedChId);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRemPostVar_When_Normal_Expect_IdsFromHandle)
{
    CcuRep::Variable param;
    param.Reset(31);
    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRemPostVar>(&insGen, param, static_cast<ChannelHandle>(203), 5, 6, 0x90);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedChId = 13U;
    const uint32_t expectedSigId = 44U;
    const uint32_t expectedVarId = 55U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedChId));
    MOCKER_CPP(&CcuUrmaChannel::GetRmtCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedSigId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetRmtXnByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedVarId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRemPostVar(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, expectedSigId);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, 0x90u);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], expectedChId);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.paramId, expectedVarId);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoBufRead_When_Normal_Expect_ChannelIdFromHandle)
{
    CcuRep::Address srcAddr;
    CcuRep::Variable srcToken;
    CcuRep::CcuBuf dst;
    CcuRep::Variable len;
    CcuRep::CompletedEvent sem;
    srcAddr.Reset(31);
    srcToken.Reset(32);
    dst.Reset(0x8009);
    len.Reset(33);
    sem.Reset(34);

    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepBufRead>(
        &insGen, static_cast<ChannelHandle>(204), CcuRep::RemoteAddr(srcAddr, srcToken), dst, len, sem, 0x12);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedId = 14U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedId));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoBufRead(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 34u);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, 0x12u);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.channelId, expectedId);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoBufWrite_When_Normal_Expect_ChannelIdFromHandle)
{
    CcuRep::CcuBuf src;
    CcuRep::Address dstAddr;
    CcuRep::Variable dstToken;
    CcuRep::Variable len;
    CcuRep::CompletedEvent sem;
    src.Reset(0x800A);
    dstAddr.Reset(41);
    dstToken.Reset(42);
    len.Reset(43);
    sem.Reset(44);

    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepBufWrite>(
        &insGen, static_cast<ChannelHandle>(205), src, CcuRep::RemoteAddr(dstAddr, dstToken), len, sem, 0x34);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedId = 15U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedId));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoBufWrite(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 44u);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, 0x34u);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.channelId, expectedId);
}

// ----------------------------- 查询失败提前返回路径 -----------------------------

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRemPostSem_When_GetChannelFail_Expect_EmptyErrorInfo)
{
    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRemPostSem>(&insGen, static_cast<ChannelHandle>(206), 3, 0x70);

    MOCKER(HcommChannelGet).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(static_cast<HcommResult>(1)));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRemPostSem(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 0u);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRemPostVar_When_GetSignalFail_Expect_EmptyErrorInfo)
{
    CcuRep::Variable param;
    param.Reset(31);
    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRemPostVar>(&insGen, param, static_cast<ChannelHandle>(207), 5, 6, 0x90);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedChId = 16U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedChId));
    MOCKER_CPP(&CcuUrmaChannel::GetRmtCkeByIndex)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_UNAVAIL));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRemPostVar(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 0u);
}

TEST_F(GetChannelByHandleTest, Ut_GenErrorInfoRemPostVar_When_GetVariableFail_Expect_EmptyErrorInfo)
{
    CcuRep::Variable param;
    param.Reset(31);
    CcuRep::CcuInsGeneratorV1 insGen{};
    auto rep = std::make_shared<CcuRep::CcuRepRemPostVar>(&insGen, param, static_cast<ChannelHandle>(208), 5, 6, 0x90);

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    const uint32_t expectedChId = 17U;
    const uint32_t expectedSigId = 66U;
    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedChId));
    MOCKER_CPP(&CcuUrmaChannel::GetRmtCkeByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(expectedSigId))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetRmtXnByIndex)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_UNAVAIL));

    ErrorInfoBase baseInfo = {};
    baseInfo.dieId = 0;
    baseInfo.deviceId = 0;
    baseInfo.missionId = 1;
    std::vector<CcuErrorInfo> errorInfo;

    CcuTaskException::GenErrorInfoRemPostVar(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 0u);
}
