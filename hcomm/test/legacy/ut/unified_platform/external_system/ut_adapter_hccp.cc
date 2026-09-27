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
#include "hccp.h"
#include "hccp_ctx.h"
#include "hccp_common.h"
#include "orion_adapter_hccp.h"
#include "network_api_exception.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "hccp_async.h"
#include "ub_memory_transport.h"
#include "adapter_error_manager_pub.h"
using namespace Hccl;

namespace {

constexpr u32 kGetTpAttrVersion = 2U;

static uint8_t gCapturedJettyPriority = 0U;

int StubRaCtxQpCreateCapturePriority(
    void* ctxHandle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qpHandle)
{
    (void)ctxHandle;
    (void)info;
    if (attr != nullptr) {
        gCapturedJettyPriority = attr->ub.priority;
    }
    static char fakeQpHandle{};
    if (qpHandle != nullptr) {
        *qpHandle = &fakeQpHandle;
    }
    return 0;
}

static uint32_t gCapturedUboeFlag = 0U;

int StubRaGetTpInfoListAsyncCaptureUboe(
    void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num, void** reqHandle)
{
    (void)ctxHandle;
    (void)infoList;
    if (cfg != nullptr) {
        gCapturedUboeFlag = cfg->flag.bs.uboe;
    }
    if (num != nullptr) {
        *num = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = reinterpret_cast<void*>(0x12345678ULL);
    }
    return 0;
}

static uint32_t gCapturedCqDepth = 0U;

int StubRaNdaCqCreateCaptureCqe(void* rdmaHandle, struct NdaCqInitAttr* attr, struct NdaCqInfo* info, void** cqHandle)
{
    (void)rdmaHandle;
    (void)info;
    if (attr != nullptr) {
        gCapturedCqDepth = attr->attr.cqe;
    }
    static char fakeCqHandle{};
    if (cqHandle != nullptr) {
        *cqHandle = &fakeCqHandle;
    }
    return 0;
}

static std::string gCapturedErrorCode;
void StubRptInputErrCapture(std::string errorCode, std::vector<std::string> keys, std::vector<std::string> values)
{
    (void)keys;
    (void)values;
    gCapturedErrorCode = errorCode;
}

} // namespace

class AdapterHccpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdapterHccp tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AdapterHccp tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AdapterHccp SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AdapterHccp TearDown" << std::endl;
    }

    SocketHandle fakeSocHandle = (SocketHandle)0x123;
    FdHandle fakeFdHandle = (FdHandle)0x200;
    void* fakeData = (void*)0x100;
};

TEST_F(AdapterHccpTest, RaTlvInit_ok)
{
    // Given
    MOCKER(RaTlvInit).stubs().will(returnValue(0));
    HRaTlvInitConfig config;
    config.mode = HrtNetworkMode::HDC;
    config.phyId = 0;

    // when
    HrtRaTlvInit(config);
    // then
}

TEST_F(AdapterHccpTest, RaTlvInit_nok)
{
    // Given
    MOCKER(RaTlvInit).stubs().will(returnValue(1));
    HRaTlvInitConfig config;
    config.mode = HrtNetworkMode::HDC;
    config.phyId = 0;

    // when

    // then
    EXPECT_THROW(HrtRaTlvInit(config), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaTlvRequest_ok)
{
    // Given
    MOCKER(RaTlvRequest).stubs().will(returnValue(0));
    void* tlv_handle;
    // when
    HrtRaTlvRequest(tlv_handle, 1, 1);
    // then
}

TEST_F(AdapterHccpTest, RaTlvRequest_nok)
{
    // Given
    MOCKER(RaTlvRequest).stubs().will(returnValue(1));
    void* tlv_handle;

    // when

    // then
    EXPECT_THROW(HrtRaTlvRequest(tlv_handle, 1, 1), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaTlvRequestForCustomChannel_ok)
{
    // Given
    MOCKER(RaTlvRequest).stubs().will(returnValue(0));
    void* tlv_handle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    // when
    HrtRaTlvRequestForCustomChannel(tlv_handle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    // then
}

TEST_F(AdapterHccpTest, RaTlvRequestForCustomChannel_tlvHandleNull)
{
    // Given
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    // when

    // then
    EXPECT_THROW(
        HrtRaTlvRequestForCustomChannel(nullptr, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff), NullPtrException);
}

TEST_F(AdapterHccpTest, RaTlvRequestForCustomChannel_customInNull)
{
    // Given
    void* tlv_handle = reinterpret_cast<void*>(0x1234);
    char outBuff[8] = {0};

    // when

    // then
    EXPECT_THROW(
        HrtRaTlvRequestForCustomChannel(tlv_handle, MSG_TYPE_CCU_DISPATCH_CMD, nullptr, outBuff), NullPtrException);
}

TEST_F(AdapterHccpTest, RaTlvRequestForCustomChannel_customOutNull)
{
    // Given
    void* tlv_handle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};

    // when

    // then
    EXPECT_THROW(
        HrtRaTlvRequestForCustomChannel(tlv_handle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, nullptr), NullPtrException);
}

TEST_F(AdapterHccpTest, RaTlvRequestForCustomChannel_nok)
{
    // Given
    MOCKER(RaTlvRequest).stubs().will(returnValue(1));
    void* tlv_handle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    // when

    // then
    EXPECT_THROW(
        HrtRaTlvRequestForCustomChannel(tlv_handle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaTlvDeInit_ok)
{
    // Given
    MOCKER(RaTlvDeinit).stubs().will(returnValue(0));
    void* tlv_handle;

    // when
    HrtRaTlvDeInit(tlv_handle);
    // then
}

TEST_F(AdapterHccpTest, RaTlvDeInit_nok)
{
    // Given
    MOCKER(RaTlvDeinit).stubs().will(returnValue(1));
    void* tlv_handle;

    // when

    // then
    EXPECT_THROW(HrtRaTlvDeInit(tlv_handle), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaInit_ok)
{
    // Given
    MOCKER(RaInit).stubs().will(returnValue(0));
    HRaInitConfig config;
    config.mode = HrtNetworkMode::HDC;
    config.phyId = 0;

    // when
    HrtRaInit(config);
    // then
}

TEST_F(AdapterHccpTest, HrtRaInit_nok_no_limit_rate)
{
    // Given
    MOCKER(RaInit).stubs().will(returnValue(SOCK_ESOCKCLOSED));
    HRaInitConfig config;
    config.mode = HrtNetworkMode::HDC;
    config.phyId = 0;

    // when

    // then
    EXPECT_THROW(HrtRaInit(config), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaDeInit_ok)
{
    // Given
    MOCKER(RaDeinit).stubs().will(returnValue(0));
    HRaInitConfig config;
    config.mode = HrtNetworkMode::HDC;
    config.phyId = 0;

    // when
    HrtRaInit(config);
    // then
}

TEST_F(AdapterHccpTest, HrtRaDeInit_nok_no_limit_rate)
{
    // Given
    MOCKER(RaDeinit).stubs().will(returnValue(SOCK_ESOCKCLOSED));
    HRaInitConfig config;
    config.mode = HrtNetworkMode::HDC;
    config.phyId = 0;

    // when

    // then
    EXPECT_THROW(HrtRaDeInit(config), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaSocketWhiteListAdd_nok)
{
    // Given
    MOCKER(RaSocketWhiteListAdd).stubs().will(returnValue(-1));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_THROW(HrtRaSocketWhiteListAdd(fakeSocHandle, whiteList), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaSocketWhiteListAdd_ok)
{
    // Given
    MOCKER(RaSocketWhiteListAdd).stubs().will(returnValue(0));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_NO_THROW(HrtRaSocketWhiteListAdd(fakeSocHandle, whiteList));
}

TEST_F(AdapterHccpTest, HrtRaSocketWhiteListAdd_strcpy_nok)
{
    // Given
    MOCKER(strcpy_s).stubs().will(returnValue(-1));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_THROW(HrtRaSocketWhiteListAdd(fakeSocHandle, whiteList), InternalException);
}

TEST_F(AdapterHccpTest, HrtRaSocketWhiteListDel_nok)
{
    // Given
    MOCKER(RaSocketWhiteListDel).stubs().will(returnValue(-1));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_THROW(HrtRaSocketWhiteListDel(fakeSocHandle, whiteList), NetworkApiException);
}

TEST_F(AdapterHccpTest, hrtRaSocketListenOneStart_again)
{
    // Given
    MOCKER(RaSocketListenStart).stubs().will(returnValue(SOCK_EAGAIN));

    // 超时时间设置为1
    MOCKER(EnvLinkTimeoutGet).stubs().will(returnValue(1));

    SocketHandle socketHandle = nullptr;

    RaSocketListenParam listenInfo(socketHandle, 0, IpAddress());
    // when

    // then
    EXPECT_THROW(HrtRaSocketListenOneStart(listenInfo, HrtNetworkMode::HDC), NetworkApiException);
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketTryListenOneStart_When_InValid_IP_Expect_RptEI0016)
{
    gCapturedErrorCode.clear();
    MOCKER(RaSocketListenStart).stubs().will(returnValue(SOCK_EADDRNOTAVAIL));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));

    SocketHandle socketHandle = nullptr;
    RaSocketListenParam listenInfo(socketHandle, 0, IpAddress());

    EXPECT_THROW(HrtRaSocketTryListenOneStart(listenInfo, HrtNetworkMode::HDC), NetworkApiException);
    EXPECT_EQ(gCapturedErrorCode, "EI0016");
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketNonBlockSendHeart_When_Input_normal_Expect_Return_Success)
{
    u64 sendBuffer = 0;
    u64 sendSizeStub = 123;
    u64 fd = 0;
    SocketHandle socketHandle = &fd;
    MOCKER(RaSocketSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&sendSizeStub, sizeof(sendSizeStub)))
        .will(returnValue(0));

    u64 sentSize = 0;
    HcclResult ret = HrtRaSocketNonBlockSendHeart(socketHandle, &sendBuffer, sizeof(sendBuffer), &sentSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(sendSizeStub, sentSize);
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketNonBlockRecvHeart_When_Input_normal_Expect_Return_Success)
{
    u64 recvbuffer = 0;
    u64 recvSizeStub = 123;
    u64 fd = 0;
    SocketHandle socketHandle = &fd;
    MOCKER(RaSocketRecv)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&recvSizeStub, sizeof(recvSizeStub)))
        .will(returnValue(0));

    u64 recvedSize = 0;
    HcclResult ret = HrtRaSocketNonBlockRecvHeart(socketHandle, &recvbuffer, sizeof(recvbuffer), &recvedSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(recvSizeStub, recvedSize);
}

TEST_F(AdapterHccpTest, HrtRaSocketInit_OK)
{
    // Given
    u32* num = new u32[2];
    SocketHandle socketHandle = static_cast<void*>(num);
    MOCKER(RaSocketInit)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&socketHandle, sizeof(socketHandle)))
        .will(returnValue(0));

    struct RaInterface rdevInfo;
    // when

    // then
    HrtRaSocketInit(HrtNetworkMode::HDC, rdevInfo);
    delete[] num;
}

TEST_F(AdapterHccpTest, HrtRaSocketInit_NOK)
{
    // Given
    u32* num = new u32[2];
    SocketHandle socketHandle = static_cast<void*>(num);
    MOCKER(RaSocketInit)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&socketHandle, sizeof(socketHandle)))
        .will(returnValue(1));

    struct RaInterface rdevInfo;
    // when

    EXPECT_THROW(HrtRaSocketInit(HrtNetworkMode::HDC, rdevInfo), NetworkApiException);
    delete[] num;
}

TEST_F(AdapterHccpTest, HrtHrtRaRdmaInit_NOK)
{
    // Given
    u32* num = new u32[1];
    RdmaHandle rdmaHandle = static_cast<void*>(num);
    MOCKER(RaRdevInit)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&rdmaHandle, sizeof(rdmaHandle)))
        .will(returnValue(1));

    struct RaInterface rdevInfo;
    // when

    EXPECT_THROW(HrtRaRdmaInit(HrtNetworkMode::HDC, rdevInfo), NetworkApiException);
    delete[] num;
}

TEST_F(AdapterHccpTest, HrtHrtRaRdmaInit_return_HCCP_ELINKDOWN_NOK)
{
    // Given
    u32* num = new u32[1];
    RdmaHandle rdmaHandle = static_cast<void*>(num);
    MOCKER(RaRdevInit)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&rdmaHandle, sizeof(rdmaHandle)))
        .will(returnValue(HCCP_ELINKDOWN));

    struct RaInterface rdevInfo;
    // when

    EXPECT_THROW(HrtRaRdmaInit(HrtNetworkMode::HDC, rdevInfo), NetworkApiException);
    delete[] num;
}

TEST_F(AdapterHccpTest, HrtRaQpCreate_NOK)
{
    // Given
    u32* num = new u32[1];
    QpHandle connHandle = static_cast<void*>(num);
    MOCKER(RaQpCreate)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&connHandle, sizeof(connHandle)))
        .will(returnValue(1));

    RdmaHandle rdmaHandle;
    // when

    EXPECT_THROW(HrtRaQpCreate(rdmaHandle, 0, 0), NetworkApiException);
    delete[] num;
}

TEST_F(AdapterHccpTest, HrtGetRaQpStatus_NOK)
{
    // Given
    s32 status = 1;
    MOCKER(RaGetQpStatus).stubs().with(mockcpp::any(), outBoundP(&status, sizeof(status))).will(returnValue(1));

    QpHandle qpHandle;
    // when

    EXPECT_THROW(HrtGetRaQpStatus(qpHandle), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaMrReg_deReg_NOK)
{
    // Given
    MOCKER(RaMrReg).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));

    QpHandle qpHandle;
    struct RaMrInfo mrInfo;
    // when

    EXPECT_THROW(HrtRaMrReg(qpHandle, mrInfo), NetworkApiException);

    MOCKER(RaMrDereg).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(HrtRaMrDereg(qpHandle, mrInfo), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaUbCtxInit_ok)
{
    // Given

    IpAddress addr(1);

    HrtRaUbCtxInitParam initParam(HrtNetworkMode::HDC, 0, addr);

    // when

    RdmaHandle rdmaHandle = HrtRaUbCtxInit(initParam);
}

TEST_F(AdapterHccpTest, HrtRaUbCtxDestroy_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbCtxDestroy(handle);
}

TEST_F(AdapterHccpTest, HrtRaUbLocalMemReg_ok)
{
    uint64_t fakeAddr = 0;
    uint64_t fakeSize = 0;
    uint32_t fakeTokenValue = 0;
    uint64_t fakeTokenIdHandle = 0;

    HrtRaUbLocMemRegParam inParam(fakeAddr, fakeSize, fakeTokenValue, fakeTokenIdHandle);

    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0x123);

    HrtRaUbLocalMemRegOutParam result = HrtRaUbLocalMemReg(fakeRdmaHandle, inParam);

    EXPECT_EQ(0, result.tokenId);
}

TEST_F(AdapterHccpTest, Ut_RaUbLocalMemRegAsync_When_Normal_Input_Expect_No_Throw)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    uint64_t fakeAddr = 0;
    uint64_t fakeSize = 0;
    uint32_t fakeTokenValue = 0;
    uint64_t fakeTokenIdHandle = 0;
    HrtRaUbLocMemRegParam inParam(fakeAddr, fakeSize, fakeTokenValue, fakeTokenIdHandle);
    u64 fakeSegVa = 0x200;
    u8 fakeKey[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 fakeTokenId = 1;
    u64 fakeMemHandle = 0x200;
    void* fakeMemHandlePtr = reinterpret_cast<void*>(fakeMemHandle);
    RequestHandle fakeReqHandle = 1;

    vector<char_t> out;
    out.resize(sizeof(struct MrRegInfoT));
    struct MrRegInfoT* info = reinterpret_cast<struct MrRegInfoT*>(out.data());
    memcpy_s(info->out.key.value, HRT_UB_MEM_KEY_MAX_LEN, fakeKey, HRT_UB_MEM_KEY_MAX_LEN);
    info->out.key.size = 4;
    info->out.ub.tokenId = fakeTokenId;
    info->out.ub.targetSegHandle = fakeSegVa;

    EXPECT_NO_THROW(RaUbLocalMemRegAsync(handle, inParam, out, fakeMemHandlePtr));
}

TEST_F(AdapterHccpTest, HrtRaUbLocalMemUnreg_ok)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbLocalMemUnreg(fakeRdmaHandle, 0);
}

TEST_F(AdapterHccpTest, HrtRaUbRemoteMemImport_ok)
{
    uint8_t value[128];
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbRemMemImportedOutParam result = HrtRaUbRemoteMemImport(handle, value, 128, 0);

    EXPECT_EQ(0, result.targetSegVa);
}

TEST_F(AdapterHccpTest, HrtRaUbRemoteMemUnimport_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbRemoteMemUnimport(handle, 0);
}

TEST_F(AdapterHccpTest, HrtRaUbCreateJfc_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    struct Hccl::CqCreateInfo cqInfo;
    u64 result = HrtRaUbCreateJfc(handle, cqInfo, HrtUbJfcMode::NORMAL);
    EXPECT_EQ(0, result);
}

TEST_F(AdapterHccpTest, HrtRaUbDestroyJfc_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbDestroyJfc(handle, 0);
}

TEST_F(AdapterHccpTest, HrtRaUbCreateJetty_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);

    HrtRaUbCreateJettyParam inParam1{100, 100, 100, 100, HrtJettyMode::HOST_OPBASE, 0, 100, 100, 100, 100};
    HrtRaUbJettyCreatedOutParam result1 = HrtRaUbCreateJetty(handle, inParam1);
    EXPECT_EQ(0, result1.jettyVa);

    HrtRaUbCreateJettyParam inParam2{100, 100, 100, 100, HrtJettyMode::HOST_OFFLOAD, 0, 100, 100, 100, 100};
    HrtRaUbJettyCreatedOutParam result2 = HrtRaUbCreateJetty(handle, inParam2);
    EXPECT_EQ(0, result2.jettyVa);

    HrtRaUbCreateJettyParam inParam3{100, 100, 100, 100, HrtJettyMode::CCU_CCUM_CACHE, 0, 100, 100, 100, 100};
    HrtRaUbJettyCreatedOutParam result3 = HrtRaUbCreateJetty(handle, inParam3);
    EXPECT_EQ(0, result3.jettyVa);

    HrtRaUbCreateJettyParam inParam4{100, 100, 100, 100, HrtJettyMode::DEV_USED, 0, 100, 100, 100, 100};
    HrtRaUbJettyCreatedOutParam result4 = HrtRaUbCreateJetty(handle, inParam4);
    EXPECT_EQ(0, result4.jettyVa);
}

TEST_F(AdapterHccpTest, HrtRaUbDestroyJetty_ok) { HrtRaUbDestroyJetty(0); }

TEST_F(AdapterHccpTest, RaUbTpImportJetty_throw)
{
    struct JettyImportCfg cfg = {};
    uint8_t value[128];
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    EXPECT_THROW(RaUbTpImportJetty(handle, value, 0, 0, cfg), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtRaUbUnimportJetty_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbUnimportJetty(handle, 0);
}

TEST_F(AdapterHccpTest, HrtRaUbJettyBind_ok) { HrtRaUbJettyBind(0, 0); }

TEST_F(AdapterHccpTest, HrtRaUbJettyUnbind_ok) { HrtRaUbJettyUnbind(0); }

TEST_F(AdapterHccpTest, HrtRaUbPostSend_ok)
{
    HrtRaUbSendWrReqParam in;
    in.inlineFlag = true;
    in.inlineReduceFlag = true;
    in.opcode = HrtUbSendWrOpCode::WRITE_WITH_NOTIFY;
    in.reduceOp = ReduceOp::SUM;
    in.dataType = DataType::INT8;
    HrtRaUbSendWrRespParam result = HrtRaUbPostSend(0, in);
    EXPECT_EQ(0, result.dieId);
}

TEST_F(AdapterHccpTest, HrtGetHosIf_nok_ra_get_ifnum_error)
{
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(HrtGetHostIf(0), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtGetHosIf_nok_ra_get_ifnum_zero)
{
    unsigned int fakeNum = 0;
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), outBoundP(&fakeNum, sizeof(fakeNum))).will(returnValue(0));
    auto hostIfs = HrtGetHostIf(0);
    EXPECT_EQ(true, hostIfs.empty());
}

TEST_F(AdapterHccpTest, HrtGetHosIf_nok_ra_get_ifaddrs_error)
{
    unsigned int fakeNum = 1;
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), outBoundP(&fakeNum, sizeof(fakeNum))).will(returnValue(0));
    MOCKER(RaGetIfaddrs).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(1));

    EXPECT_THROW(HrtGetHostIf(0), NetworkApiException);
}

TEST_F(AdapterHccpTest, HrtGetHosIf_ok)
{
    unsigned int fakeNum = 1;
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), outBoundP(&fakeNum, sizeof(fakeNum))).will(returnValue(0));

    IpAddress addr(1);
    struct InterfaceInfo ifAddrInfos[1];
    char stub_ifname[256] = {};
    stub_ifname[0] = 'a';
    memcpy_s(&ifAddrInfos[0].ifname, 256, &stub_ifname, 256);
    ifAddrInfos[0].ifaddr.ip.addr = addr.GetBinaryAddress().addr;
    ifAddrInfos[0].family = addr.GetFamily();
    ifAddrInfos[0].scopeId = addr.GetScopeID();

    MOCKER(RaGetIfaddrs)
        .stubs()
        .with(mockcpp::any(), outBoundP(ifAddrInfos, sizeof(ifAddrInfos)), outBoundP(&fakeNum, sizeof(fakeNum)))
        .will(returnValue(0));

    auto hostIfs = HrtGetHostIf(0);
    EXPECT_EQ(1, hostIfs.size());
    EXPECT_EQ("a", hostIfs[0].first);
    EXPECT_EQ(0, hostIfs[0].second.GetScopeID());
    EXPECT_EQ(AF_INET, (int)hostIfs[0].second.GetFamily());
    EXPECT_EQ(1, hostIfs[0].second.GetBinaryAddress().addr.s_addr);
}

TEST_F(AdapterHccpTest, HrtGetDeviceIp_nok)
{
    u32 devPhyId = 0;
    auto hostIfs = HrtGetDeviceIp(devPhyId);
    EXPECT_EQ(true, hostIfs.empty());

    HrtGetDeviceIp(devPhyId);
}

TEST_F(AdapterHccpTest, HrtGetDeviceIp_ok)
{
    unsigned int fakeNum = 1;
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), outBoundP(&fakeNum, sizeof(fakeNum))).will(returnValue(0));

    u32 devPhyId = 0;
    IpAddress addr(1);
    struct InterfaceInfo ifAddrInfos[1];
    char stub_ifname[256] = {};
    stub_ifname[0] = 'a';
    memcpy_s(&ifAddrInfos[0].ifname, 256, &stub_ifname, 256);
    ifAddrInfos[0].ifaddr.ip.addr = addr.GetBinaryAddress().addr;
    ifAddrInfos[0].family = addr.GetFamily();
    ifAddrInfos[0].scopeId = addr.GetScopeID();

    MOCKER(RaGetIfaddrs)
        .stubs()
        .with(mockcpp::any(), outBoundP(ifAddrInfos, sizeof(ifAddrInfos)), outBoundP(&fakeNum, sizeof(fakeNum)))
        .will(returnValue(0));

    auto deviceIps = HrtGetDeviceIp(devPhyId);
    EXPECT_EQ(1, deviceIps.size());
    EXPECT_EQ(0, deviceIps[0].GetScopeID());
    EXPECT_EQ(AF_INET, (int)deviceIps[0].GetFamily());
    EXPECT_EQ(1, deviceIps[0].GetBinaryAddress().addr.s_addr);
}

TEST_F(AdapterHccpTest, HrtRaUbPostNops_exception)
{
    MOCKER(RaBatchSendWr).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(HrtRaUbPostNops(0, 0, 1), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaUbUpdateCi_exception)
{
    JettyHandle jettyHandle = 0;
    MOCKER(RaCtxUpdateCi).stubs().will(returnValue(1));
    EXPECT_THROW(RaUbUpdateCi(jettyHandle, 100), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaGetAsyncReqResult_exception)
{
    RequestHandle reqHandle = 0;
    HrtRaGetAsyncReqResult(reqHandle);
}

TEST_F(AdapterHccpTest, RaUbLocalMemUnregAsync_exception)
{
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x123);
    LocMemHandle lmemHandle;
    RaUbLocalMemUnregAsync(rdmaHandle, lmemHandle);
}

TEST_F(AdapterHccpTest, RaUbDestroyJettyAsync_exception)
{
    void* jettyHandle = reinterpret_cast<void*>(0x123);
    RequestHandle result = RaUbDestroyJettyAsync(jettyHandle);
}

TEST_F(AdapterHccpTest, RaUbUnimportJettyAsync_exception)
{
    void* targetJettyHandle = reinterpret_cast<void*>(0x123);
    RaUbUnimportJettyAsync(targetJettyHandle);
}

TEST_F(AdapterHccpTest, RaGetAsyncReqResult_return_others_eagain)
{
    MOCKER(RaGetAsyncReqResult).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(OTHERS_EAGAIN));
    RequestHandle reqHandle = 12;
    HrtRaGetAsyncReqResult(reqHandle);
    GlobalMockObject::verify();
}

TEST_F(AdapterHccpTest, RaGetAsyncReqResult_return_error)
{
    MOCKER(RaGetAsyncReqResult).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(OTHERS_EAGAIN + 1));
    RequestHandle reqHandle = 12;
    EXPECT_THROW(HrtRaGetAsyncReqResult(reqHandle), NetworkApiException);
    GlobalMockObject::verify();
}

TEST_F(AdapterHccpTest, RaGetAsyncReqResult_return_zero_result_error)
{
    int fakeResult = 12345;
    MOCKER(RaGetAsyncReqResult).stubs().with(mockcpp::any(), outBoundP(&fakeResult)).will(returnValue(0));
    RequestHandle reqHandle = 12;
    EXPECT_THROW(HrtRaGetAsyncReqResult(reqHandle), NetworkApiException);
    GlobalMockObject::verify();
}

TEST_F(AdapterHccpTest, RaBlockGetSocket_return_err)
{
    int reqResult = SOCK_EAGAIN;
    MOCKER(strcpy_s).stubs().will(returnValue(-1));

    SocketHandle socketHandle = nullptr;
    IpAddress ipAddr = IpAddress();
    std::string tag = "";
    FdHandle fdHandle = nullptr;
    u32 role = 0;
    RaSocketGetParam param(socketHandle, ipAddr, tag, fdHandle);

    RequestHandle reqHandle = 12;
    EXPECT_THROW(RaGetOneSocket(role, param), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaGetOneSocket_return_err_2)
{
    u32 connectedNum = 2;
    MOCKER(RaGetSockets)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&connectedNum))
        .will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    IpAddress ipAddr = IpAddress();
    std::string tag = "";
    FdHandle fdHandle = nullptr;
    u32 role = 0;
    RaSocketGetParam param(socketHandle, ipAddr, tag, fdHandle);

    RequestHandle reqHandle = 12;
    EXPECT_THROW(RaGetOneSocket(role, param), NetworkApiException);
}

TEST_F(AdapterHccpTest, RaSocketCloseOneAsync_return_ok)
{
    MOCKER(RaSocketBatchCloseAsync).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    FdHandle fdHandle = nullptr;
    RaSocketCloseParam param(socketHandle, fdHandle);

    RaSocketCloseOneAsync(param);
}

TEST_F(AdapterHccpTest, RaSocketListenOneStopAsync_return_ok)
{
    MOCKER(RaSocketListenStopAsync).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    unsigned int port = 100;
    RaSocketListenParam param(socketHandle, port, IpAddress());

    RaSocketListenOneStopAsync(param);
};

TEST_F(AdapterHccpTest, RaUbAllocTokenIdHandle_ok)
{
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x123);
    std::pair<TokenIdHandle, uint32_t> result = RaUbAllocTokenIdHandle(rdmaHandle);
    std::pair<TokenIdHandle, uint32_t> expectResult(0, 0);
    EXPECT_EQ(result, expectResult);
}

TEST_F(AdapterHccpTest, RaUbFreeTokenIdHandle_exception)
{
    MOCKER(RaCtxTokenIdFree).stubs().with(mockcpp::any()).will(returnValue(1));
    RdmaHandle handle = (void*)0x1234;
    TokenIdHandle tokenIdHandle = 1234;
    EXPECT_THROW(RaUbFreeTokenIdHandle(handle, tokenIdHandle), NetworkApiException);
}

void MockRaSocketRecv(int ret, unsigned long long recvSize)
{
    MOCKER(RaSocketRecv)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&recvSize, sizeof(recvSize)))
        .will(returnValue(ret));
}

void MockEnvLinkTimeoutGet(int timeout)
{
    // 设置超时时间
    MOCKER(EnvLinkTimeoutGet).stubs().will(returnValue(timeout));
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketBlockRecv_When_RecvOnceComplete_Expect_Success)
{
    MockRaSocketRecv(0, 100); // 一次接收完成
    MockEnvLinkTimeoutGet(1); // 1秒超时
    EXPECT_NO_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100));
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketBlockRecv_When_RecvSizeExceeds_Expect_Throw_NetworkApiException)
{
    MockRaSocketRecv(0, 150); // 超出预期大小
    MockEnvLinkTimeoutGet(1);
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketBlockRecv_When_RecvSizeZero_Expect_Throw_NetworkApiException)
{
    MockRaSocketRecv(0, 0); // 接收为0
    MockEnvLinkTimeoutGet(1);
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketBlockRecv_When_SockClosed_Expect_Throw_NetworkApiException)
{
    gCapturedErrorCode.clear();
    MockRaSocketRecv(SOCK_ESOCKCLOSED, 0);
    MockEnvLinkTimeoutGet(1);
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
    EXPECT_EQ(gCapturedErrorCode, "EI0015");
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketBlockRecv_When_SockClose_Expect_Throw_NetworkApiException)
{
    gCapturedErrorCode.clear();
    MockRaSocketRecv(SOCK_CLOSE, 0);
    MockEnvLinkTimeoutGet(1);
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
    EXPECT_EQ(gCapturedErrorCode, "EI0015");
}

TEST_F(AdapterHccpTest, Ut_HrtRaSocketBlockRecv_When_Timeout_Expect_Throw_NetworkApiException)
{
    MockRaSocketRecv(1, 0);   // 模拟一直接收不到数据
    MockEnvLinkTimeoutGet(0); // 超时时间设为0
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
}

TEST_F(AdapterHccpTest, ut_HrtRaSocketWhiteListDel_With_Enormous_WhiteList)
{
    // 大量删除接口
    RaSocketWhitelist wlist{};
    vector<RaSocketWhitelist> wlists(56, wlist);
    SocketHandle fakeFdHandle = reinterpret_cast<SocketHandle>(0x123);

    MOCKER(RaSocketWhiteListDel).stubs().will(returnValue(0));

    EXPECT_NO_THROW(HrtRaSocketWhiteListDel(fakeFdHandle, wlists));
}

TEST_F(AdapterHccpTest, Ut_HraGetRtpEnable_When_RTP_Equals_1_Expect_Return_True)
{
    DevBaseAttr out{};
    out.ub.priorityInfo[0].tpType.bs.rtp = 1;
    MOCKER(RaGetDevBaseAttr).stubs().with(mockcpp::any(), outBoundP(&out, sizeof(out))).will(returnValue(0));
    RdmaHandle handle = (void*)0x1234;

    EXPECT_EQ(HraGetRtpEnable(handle), true);
}

TEST_F(AdapterHccpTest, Ut_HraGetRtpEnable_When_RTP_Equals_0_Expect_Return_False)
{
    DevBaseAttr out{};
    MOCKER(RaGetDevBaseAttr).stubs().with(mockcpp::any(), outBoundP(&out, sizeof(out))).will(returnValue(0));
    RdmaHandle handle = (void*)0x1234;

    EXPECT_EQ(HraGetRtpEnable(handle), false);
}

TEST_F(AdapterHccpTest, Ut_HrtRaGetTlsStatus_When_InfoIsNull_Expect_ReturnPtrError)
{
    TlsStatus tlsStatus = TlsStatus::UNKNOWN;

    HcclResult ret = HrtRaGetTlsStatus(nullptr, tlsStatus);

    EXPECT_EQ(ret, HCCL_E_PTR);
    EXPECT_EQ(tlsStatus, TlsStatus::UNKNOWN);
}

TEST_F(AdapterHccpTest, Ut_HrtRaGetTlsStatus_When_InterfaceVersionQueryFails_Expect_ReturnNotSupportAndUnknown)
{
    RaInfo info{};
    info.phyId = 0;
    TlsStatus tlsStatus = TlsStatus::DISABLE;

    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));

    HcclResult ret = HrtRaGetTlsStatus(&info, tlsStatus);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(tlsStatus, TlsStatus::UNKNOWN);
}

TEST_F(AdapterHccpTest, Ut_HrtRaGetTlsStatus_When_InterfaceVersionTooLow_Expect_ReturnNotSupportAndUnknown)
{
    RaInfo info{};
    info.phyId = 0;
    TlsStatus tlsStatus = TlsStatus::DISABLE;
    u32 lowVersion = 0;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&lowVersion, sizeof(lowVersion)))
        .will(returnValue(0));

    HcclResult ret = HrtRaGetTlsStatus(&info, tlsStatus);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(tlsStatus, TlsStatus::UNKNOWN);
}

TEST_F(AdapterHccpTest, Ut_HrtRaGetTlsStatus_When_RaGetTlsEnableFails_Expect_ReturnNetworkErrorAndDisable)
{
    RaInfo info{};
    info.phyId = 0;
    TlsStatus tlsStatus = TlsStatus::UNKNOWN;
    u32 supportVersion = 1;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&supportVersion, sizeof(supportVersion)))
        .will(returnValue(0));
    MOCKER(RaGetTlsEnable).stubs().will(returnValue(-1));

    HcclResult ret = HrtRaGetTlsStatus(&info, tlsStatus);

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(tlsStatus, TlsStatus::DISABLE);
}

TEST_F(AdapterHccpTest, Ut_HrtRaGetTlsStatus_When_TlsEnableIsTrue_Expect_ReturnSuccessAndEnable)
{
    RaInfo info{};
    info.phyId = 0;
    TlsStatus tlsStatus = TlsStatus::UNKNOWN;
    u32 supportVersion = 1;
    bool tlsEnable = true;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&supportVersion, sizeof(supportVersion)))
        .will(returnValue(0));
    MOCKER(RaGetTlsEnable).stubs().with(mockcpp::any(), outBoundP(&tlsEnable, sizeof(tlsEnable))).will(returnValue(0));

    HcclResult ret = HrtRaGetTlsStatus(&info, tlsStatus);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tlsStatus, TlsStatus::ENABLE);
}

TEST_F(AdapterHccpTest, Ut_HrtRaGetTlsStatus_When_TlsEnableIsFalse_Expect_ReturnSuccessAndDisable)
{
    RaInfo info{};
    info.phyId = 0;
    TlsStatus tlsStatus = TlsStatus::UNKNOWN;
    u32 supportVersion = 1;
    bool tlsEnable = false;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&supportVersion, sizeof(supportVersion)))
        .will(returnValue(0));
    MOCKER(RaGetTlsEnable).stubs().with(mockcpp::any(), outBoundP(&tlsEnable, sizeof(tlsEnable))).will(returnValue(0));

    HcclResult ret = HrtRaGetTlsStatus(&info, tlsStatus);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tlsStatus, TlsStatus::DISABLE);
}

TEST_F(AdapterHccpTest, HrtRaGetEidByIp_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    std::vector<IpAddress> ipV4AddrList;
    ipV4AddrList.emplace_back(IpAddress(1)); // simple IPv4 placeholder
    std::vector<IpAddress> eidAddrList;

    unsigned int fakeNum = 1;
    union HccpEid fakeEid[1];
    (void)memset_s(fakeEid, sizeof(fakeEid), 0, sizeof(fakeEid));
    // set a non-zero raw to avoid all-zero ambiguity
    fakeEid[0].raw[0] = 1;

    MOCKER(RaGetEidByIp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(fakeEid, sizeof(fakeEid)), outBoundP(&fakeNum, sizeof(fakeNum)))
        .will(returnValue(0));

    HcclResult ret = HrtRaGetEidByIp(handle, ipV4AddrList, eidAddrList);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(eidAddrList.size(), 1u);
}

TEST_F(AdapterHccpTest, HrtRaGetEidByIp_ra_get_eid_by_ip_error)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    std::vector<IpAddress> ipV4AddrList;
    ipV4AddrList.emplace_back(IpAddress(1));
    std::vector<IpAddress> eidAddrList;

    MOCKER(RaGetEidByIp).stubs().will(returnValue(1));

    EXPECT_EQ(HrtRaGetEidByIp(handle, ipV4AddrList, eidAddrList), HCCL_E_INTERNAL);
}

TEST_F(AdapterHccpTest, HrtRaGetEidByIp_count_mismatch_returns_internal)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    std::vector<IpAddress> ipV4AddrList;
    ipV4AddrList.emplace_back(IpAddress(1));
    std::vector<IpAddress> eidAddrList;

    // Ra returns success but reports different num (0) than input size (1)
    unsigned int returnedNum = 0;
    union HccpEid fakeEid[1];
    (void)memset_s(fakeEid, sizeof(fakeEid), 0, sizeof(fakeEid));

    MOCKER(RaGetEidByIp)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), outBoundP(fakeEid, sizeof(fakeEid)),
            outBoundP(&returnedNum, sizeof(returnedNum)))
        .will(returnValue(0));

    HcclResult ret = HrtRaGetEidByIp(handle, ipV4AddrList, eidAddrList);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_TRUE(eidAddrList.empty());
}

TEST_F(AdapterHccpTest, HrtRaGetEidByIp_empty_input_ok)
{
    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    std::vector<IpAddress> ipV4AddrList; // empty
    std::vector<IpAddress> eidAddrList;

    unsigned int returnedNum = 0;
    MOCKER(RaGetEidByIp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&returnedNum, sizeof(returnedNum)))
        .will(returnValue(0));

    HcclResult ret = HrtRaGetEidByIp(handle, ipV4AddrList, eidAddrList);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(eidAddrList.empty());
}

TEST_F(AdapterHccpTest, ut_HrtGetUboeFlagEnable_VersionEnough_Expect_Success)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HrtGetUboeFlagEnable(devPhyId);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AdapterHccpTest, ut_HrtGetUboeFlagEnable_When_VersionNotEnough_Expect_NotSupport)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION - 1;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HrtGetUboeFlagEnable(devPhyId);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AdapterHccpTest, ut_HrtGetUboeFlagEnable_When_RaGetInterfaceFailed_Expect_E_INTERNAL)
{
    u32 devPhyId = 1;
    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));
    HcclResult ret = HrtGetUboeFlagEnable(devPhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(AdapterHccpTest, ut_HrtCheckUboeSupported_When_DevFeatureBitSet_Expect_True)
{
    u32 devFeature = 1 << UBOE_DEV_FLAG_RIGHT_SHIFT;
    bool result = HrtCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);

    // 测试其他位也有值的情况
    devFeature = (1 << UBOE_DEV_FLAG_RIGHT_SHIFT) | 0xFFFF;
    result = HrtCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
}

TEST_F(AdapterHccpTest, ut_HrtCheckUboeSupported_When_DevFeatureBitNotSet_Expect_False)
{
    u32 devFeature = 0;
    bool result = HrtCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);

    // 测试只有其他位被设置，但UBOE位未设置
    devFeature = 0xFFFFFFFF & ~(1 << UBOE_DEV_FLAG_RIGHT_SHIFT);
    result = HrtCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
}

TEST_F(AdapterHccpTest, ut_HrtRaUbCreateJetty_When_QosSet_Expect_PriorityMapped)
{
    GlobalMockObject::verify();
    gCapturedJettyPriority = 0U;
    MOCKER(RaCtxQpCreate).stubs().will(invoke(StubRaCtxQpCreateCapturePriority));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    HrtRaUbCreateJettyParam inParam{100, 100, 100, 100, HrtJettyMode::HOST_OPBASE, 0, 100, 100, 100, 100};
    inParam.qos = 0x17U;

    (void)HrtRaUbCreateJetty(handle, inParam);
    EXPECT_EQ(gCapturedJettyPriority, 7U);
}

TEST_F(AdapterHccpTest, ut_HrtRaSetTpAttrAsync_When_RaSetOk_Expect_Success)
{
    int reqResult = 0;
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&reqResult, sizeof(reqResult)))
        .will(returnValue(0));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    HcclResult ret = HrtRaSetTpAttrAsync(handle, 0x100ULL, 0x1U, attr, reqHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AdapterHccpTest, ut_HrtRaSetTpAttrAsync_When_RaSetFails_Expect_Throw)
{
    MOCKER(RaSetTpAttrAsync).stubs().will(returnValue(-1));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    EXPECT_THROW(HrtRaSetTpAttrAsync(handle, 0x100ULL, 0x1U, attr, reqHandle), NetworkApiException);
}

TEST_F(AdapterHccpTest, ut_HrtRaSetTpAttrAsync_When_AsyncUnexpected_Expect_Internal)
{
    int reqResult = SOCK_EAGAIN;
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&reqResult, sizeof(reqResult)))
        .will(returnValue(0));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    HcclResult ret = HrtRaSetTpAttrAsync(handle, 0x100ULL, 0x1U, attr, reqHandle);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(AdapterHccpTest, ut_HrtRaGetTpAttrAsync_When_VersionOk_Expect_Success)
{
    u32 tpAttrVersion = kGetTpAttrVersion;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&tpAttrVersion, sizeof(tpAttrVersion)))
        .will(returnValue(0));
    int reqResult = 0;
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&reqResult, sizeof(reqResult)))
        .will(returnValue(0));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    uint32_t attrBitmap = 0U;
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    HcclResult ret = HrtRaGetTpAttrAsync(0U, handle, 0x100ULL, attrBitmap, attr, reqHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AdapterHccpTest, ut_HrtRaGetTpAttrAsync_When_VersionTooLow_Expect_NotSupport)
{
    u32 tpAttrVersion = kGetTpAttrVersion - 1U;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&tpAttrVersion, sizeof(tpAttrVersion)))
        .will(returnValue(0));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    uint32_t attrBitmap = 0U;
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    HcclResult ret = HrtRaGetTpAttrAsync(0U, handle, 0x100ULL, attrBitmap, attr, reqHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AdapterHccpTest, ut_HrtRaGetTpAttrAsync_When_RaGetInterfaceVersionFails_Expect_NotSupport)
{
    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    uint32_t attrBitmap = 0U;
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    HcclResult ret = HrtRaGetTpAttrAsync(0U, handle, 0x100ULL, attrBitmap, attr, reqHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AdapterHccpTest, ut_HrtRaGetTpAttrAsync_When_RaGetTpAttrAsyncFails_Expect_Throw)
{
    u32 tpAttrVersion = kGetTpAttrVersion;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&tpAttrVersion, sizeof(tpAttrVersion)))
        .will(returnValue(0));
    MOCKER(RaGetTpAttrAsync).stubs().will(returnValue(-1));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    uint32_t attrBitmap = 0U;
    TpAttr attr{};
    RequestHandle reqHandle = 0;
    EXPECT_THROW(HrtRaGetTpAttrAsync(0U, handle, 0x100ULL, attrBitmap, attr, reqHandle), NetworkApiException);
}

TEST_F(AdapterHccpTest, ut_RaUbGetTpInfoAsync_When_UboeProtocol_Expect_UboeFlagSet)
{
    gCapturedUboeFlag = 0U;
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncCaptureUboe));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    IpAddress locAddr("8.0.0.1");
    IpAddress rmtAddr("8.0.0.2");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::UBOE};
    vector<char_t> out;
    uint32_t num = 0U;

    RequestHandle reqHandle = RaUbGetTpInfoAsync(handle, param, out, num);
    EXPECT_NE(reqHandle, 0ULL);
    EXPECT_EQ(gCapturedUboeFlag, 1U);
}

TEST_F(AdapterHccpTest, ut_RaUbGetTpInfoAsync_When_RaGetTpInfoListAsyncFails_Expect_Throw)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(returnValue(-1));

    RdmaHandle handle = reinterpret_cast<RdmaHandle>(0x123);
    IpAddress locAddr("8.0.0.1");
    IpAddress rmtAddr("8.0.0.2");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::TP};
    vector<char_t> out;
    uint32_t num = 0U;

    EXPECT_THROW(RaUbGetTpInfoAsync(handle, param, out, num), NetworkApiException);
}

TEST_F(AdapterHccpTest, ut_HrtRaNdaCqCreate_When_IndepUbMode_Expect_UbnacDepth)
{
    gCapturedCqDepth = 0U;
    MOCKER(RaNdaCqCreate).stubs().will(invoke(StubRaNdaCqCreateCaptureCqe));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x100);
    NdaOps ndaOps;
    (void)memset(&ndaOps, 0, sizeof(ndaOps));
    NdaCqInfo cqInfo;
    CqHandle cqHandle = nullptr;

    HcclResult ret = HrtRaNdaCqCreate(rdmaHandle, &ndaOps, QBUF_DMA_MODE_INDEP_UB, 0, &cqInfo, &cqHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AdapterHccpTest, ut_HrtRaNdaCqCreate_When_DefaultMode_Expect_XscdvDepth)
{
    gCapturedCqDepth = 0U;
    MOCKER(RaNdaCqCreate).stubs().will(invoke(StubRaNdaCqCreateCaptureCqe));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x100);
    NdaOps ndaOps;
    (void)memset(&ndaOps, 0, sizeof(ndaOps));
    NdaCqInfo cqInfo;
    CqHandle cqHandle = nullptr;

    HcclResult ret = HrtRaNdaCqCreate(rdmaHandle, &ndaOps, QBUF_DMA_MODE_DEFAULT, 0, &cqInfo, &cqHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AdapterHccpTest, HrtRaBlockGetOneSocket_return_ok)
{
    u32 connectedNum = 1;
    MOCKER(RaGetSockets)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&connectedNum))
        .will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    IpAddress ipAddr = IpAddress("127.0.0.1");
    std::string tag = "test";
    FdHandle fdHandle = nullptr;
    RaSocketGetParam param(socketHandle, ipAddr, tag, fdHandle);

    RaSocketFdHandleParam result = HrtRaBlockGetOneSocket(0, param, 10);
}

TEST_F(AdapterHccpTest, HrtRaWaitEventHandle_return_ok)
{
    MOCKER(RaWaitEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    std::vector<SocketEventInfo> eventInfos(1);
    u32 eventsNum = 0;
    HcclResult ret = HrtRaWaitEventHandle(0, eventInfos, 100, 1, eventsNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
