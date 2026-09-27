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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#define private public
#define protected public
#include "externalinput.h"
#include "adapter_rts.h"
#include "typical_qp_manager.h"
#include "typical_mr_manager.h"
#include "rdma_resource_manager.h"
#include "interface_hccl.h"
#include "adapter_hccp.h"
#include "adapter_rts_common.h"
#include "hccl_common.h"
#undef private
#undef protected

#include "llt_hccl_stub_pub.h"

using namespace std;
using namespace hccl;

/* ===== Stub functions for hrtGetDeviceRefresh ===== */
static HcclResult stub_hrtGetDeviceRefresh_success(s32* deviceLogicId)
{
    *deviceLogicId = 0;
    return HCCL_SUCCESS;
}

static HcclResult stub_hrtGetDeviceRefresh_fail(s32* deviceLogicId) { return HCCL_E_INTERNAL; }

/* ===== Stub for TypicalMrManager::RegisterMem ===== */
static HcclResult stub_RegisterMem_success(TypicalMrManager* self, struct MrInfoT& mrInfo)
{
    mrInfo.lkey = 0x12345678;
    mrInfo.rkey = 0x9ABCDEF0;
    return HCCL_SUCCESS;
}

/* ===== Stub for TypicalQpManager::CreateCq ===== */
static HcclResult stub_CreateCq_success(TypicalQpManager* self, AscendCQInfo& cqInfo)
{
    cqInfo.cqn = 42;
    return HCCL_SUCCESS;
}

/* ===== Stub for TypicalQpManager::CreateQpWithCQ ===== */
static HcclResult
stub_CreateQpWithCQ_success(TypicalQpManager* self, struct TypicalQp& qpInfo, const QpConfigWithCQInfo& qpConfig)
{
    qpInfo.qpn = 100;
    qpInfo.gidIdx = 1;
    qpInfo.psn = 0xDEADBEEF;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        qpInfo.gid[i] = static_cast<uint8_t>(i);
    }
    return HCCL_SUCCESS;
}

/* ===== Stub for TypicalQpManager::GetCqDepth ===== */
static HcclResult stub_GetCqDepth_success(TypicalQpManager* self, uint32_t cqn, uint32_t& cqDepth)
{
    cqDepth = 128;
    return HCCL_SUCCESS;
}

/* ===== Stub for TypicalQpManager::GetCqHandle ===== */
static HcclResult stub_GetCqHandle_success(TypicalQpManager* self, uint32_t cqn, void*& cqHandle)
{
    cqHandle = reinterpret_cast<void*>(0xDEAD0000 + cqn);
    return HCCL_SUCCESS;
}

/* ===== Stub for HrtRaPollTypicalCq ===== */
static s32 stub_HrtRaPollTypicalCq_success(void* cqHandle, u32 num, void* wc)
{
    struct AscendWc* wcArr = static_cast<struct AscendWc*>(wc);
    wcArr[0].wrId = 0xAAAA;
    wcArr[0].status = ASCEND_WC_SUCCESS;
    wcArr[0].opcode = ASCEND_WC_SEND;
    wcArr[0].byteLen = 1024;
    wcArr[0].qpNum = 100;
    return 2;
}

static s32 stub_HrtRaPollTypicalCq_fail(void* cqHandle, u32 num, void* wc) { return -1; }

/* ===== Stub for TypicalQpManager::GetQpHandleByQpn ===== */
static HcclResult stub_GetQpHandleByQpn_success(TypicalQpManager* self, u32 qpn, QpHandle& qpHandle)
{
    qpHandle = reinterpret_cast<QpHandle>(0xBEEF0000 + qpn);
    return HCCL_SUCCESS;
}

/* ===== Stub for TypicalQpManager::GetVerbsQpHandleByQpn ===== */
static HcclResult stub_GetVerbsQpHandleByQpn_success(TypicalQpManager* self, u32 qpn, QpHandle& qpHandle)
{
    qpHandle = reinterpret_cast<QpHandle>(0xBEEF0000 + qpn);
    return HCCL_SUCCESS;
}

/* ===== Stub for HrtRaSendWrVerbs ===== */
static HcclResult stub_HrtRaSendWrVerbs_success(QpHandle handle, struct SendWrVerbs* wr, struct SendWrRsp* opRsp)
{
    opRsp->db.dbIndex = 1;
    opRsp->db.dbInfo = 0x1000;
    return HCCL_SUCCESS;
}

/* ===== Stub for hrtRDMADBSend ===== */
static HcclResult stub_hrtRDMADBSend_success(uint32_t dbindex, uint64_t dbinfo, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

/* ===== Stub for HrtRaRecvWrVerbs ===== */
static HcclResult stub_HrtRaRecvWrVerbs_success(QpHandle handle, struct RecvWrVerbs* wr) { return HCCL_SUCCESS; }

/* ===== Helpers for TypicalQpManagerTest ===== */
static void ClearAllMaps(TypicalQpManager& instance)
{
    instance.qpMap_.clear();
    instance.verbsQpMap_.clear();
    instance.cqMap_.clear();
}

static void SetRdmaHandle(TypicalQpManager& instance)
{
    instance.rdmaHandle_ = reinterpret_cast<RdmaHandle>(0xCAFE0001);
}

/* ===== Stub for hrtGetDevice ===== */
static s32 g_deviceLogicId = 0;
static HcclResult stub_hrtGetDevice_success(s32* deviceLogicId)
{
    *deviceLogicId = g_deviceLogicId;
    return HCCL_SUCCESS;
}

/* ===== Stub for GetExternalInputRdmaRetryCnt / GetExternalInputRdmaTimeOut ===== */
static u32 stub_GetExternalInputRdmaRetryCnt() { return 7u; }

static u32 stub_GetExternalInputRdmaTimeOut() { return 14u; }

/* ===== Stub for RdmaResourceManager::GetRdmaHandle ===== */
static HcclResult stub_GetRdmaHandle_success(RdmaResourceManager* self, RdmaHandle& rdmaHandle)
{
    rdmaHandle = reinterpret_cast<RdmaHandle>(0xCAFE0001);
    return HCCL_SUCCESS;
}

static HcclResult stub_GetRdmaHandle_fail(RdmaResourceManager* self, RdmaHandle& rdmaHandle) { return HCCL_E_INTERNAL; }

/* ===== Stub for hrtRaTypicalQpModify ===== */
static HcclResult
stub_hrtRaTypicalQpModify_success(QpHandle qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo)
{
    return HCCL_SUCCESS;
}

static HcclResult
stub_hrtRaTypicalQpModify_fail(QpHandle qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo)
{
    return HCCL_E_INTERNAL;
}

/* ===== Stub for HrtRaQpDestroyWithoutCQ ===== */
static HcclResult stub_HrtRaQpDestroyWithoutCQ_success(QpHandle handle) { return HCCL_SUCCESS; }

static HcclResult stub_HrtRaQpDestroyWithoutCQ_fail(QpHandle handle) { return HCCL_E_INTERNAL; }

/* ===== Stub for HrtRaQpDestroy ===== */
static HcclResult stub_HrtRaQpDestroy_success(QpHandle handle) { return HCCL_SUCCESS; }

/* ===== Stub for CreateQpWithCQConfig ===== */
static HcclResult stub_CreateQpWithCQConfig_success(
    RdmaHandle rdmaHandle, s32 qpMode, const QpConfigWithCQInfo& qpConfig, QpHandle& qpHandle, struct TypicalQp& qpInfo)
{
    qpHandle = reinterpret_cast<QpHandle>(0xBEEF0001);
    qpInfo.qpn = 100;
    return HCCL_SUCCESS;
}

static HcclResult stub_CreateQpWithCQConfig_fail(
    RdmaHandle rdmaHandle, s32 qpMode, const QpConfigWithCQInfo& qpConfig, QpHandle& qpHandle, struct TypicalQp& qpInfo)
{
    return HCCL_E_INTERNAL;
}

/* ===== Stub for DestroyTypicalCq ===== */
static HcclResult stub_DestroyTypicalCq_success(RdmaHandle rdmaHandle, u32 cqn, void* cqHandle) { return HCCL_SUCCESS; }

/* ===== Stub for CreateTypicalCq ===== */
static HcclResult stub_CreateTypicalCq_success(RdmaHandle rdmaHandle, u32 cqDepth, u32& cqn, void** cqHandle)
{
    cqn = 42;
    *cqHandle = reinterpret_cast<void*>(0xDEAD0001);
    return HCCL_SUCCESS;
}

/* ==================================================================
 * Test Fixture: InterfaceHcclVerbsTest
 * ================================================================== */
class InterfaceHcclVerbsTest : public testing::Test {
protected:
    virtual void SetUp() { std::cout << "InterfaceHcclVerbsTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "InterfaceHcclVerbsTest TearDown" << std::endl;
    }
};

/* ==================================================================
 * 1. hcclAscendRdmaInitV2 Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, InitV2_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&RdmaResourceManager::Init).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(hcclAscendRdmaInitV2(), HCCL_SUCCESS);
}

TEST_F(InterfaceHcclVerbsTest, InitV2_GetDeviceRefreshFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_fail));

    EXPECT_EQ(hcclAscendRdmaInitV2(), HCCL_E_INTERNAL);
}

TEST_F(InterfaceHcclVerbsTest, InitV2_InitFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&RdmaResourceManager::Init).stubs().will(returnValue(HCCL_E_INTERNAL));

    EXPECT_EQ(hcclAscendRdmaInitV2(), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 2. hcclRdmaMemRegister Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, RdmaMemRegister_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalMrManager::RegisterMem)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(stub_RegisterMem_success));

    AscendMrAttr memInfo{};
    memInfo.addr = 0x1000;
    memInfo.size = 4096;

    EXPECT_EQ(hcclRdmaMemRegister(&memInfo), HCCL_SUCCESS);
    EXPECT_EQ(memInfo.lkey, 0x12345678u);
    EXPECT_EQ(memInfo.rkey, 0x9ABCDEF0u);
}

TEST_F(InterfaceHcclVerbsTest, RdmaMemRegister_NullPtr) { EXPECT_EQ(hcclRdmaMemRegister(nullptr), HCCL_E_PTR); }

TEST_F(InterfaceHcclVerbsTest, RdmaMemRegister_RegisterMemFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalMrManager::RegisterMem).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendMrAttr memInfo{};
    EXPECT_EQ(hcclRdmaMemRegister(&memInfo), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 3. hcclRdmaMemDeRegister Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, RdmaMemDeRegister_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalMrManager::DeRegisterMem).stubs().will(returnValue(HCCL_SUCCESS));

    AscendMrAttr memInfo{};
    memInfo.addr = 0x1000;
    memInfo.size = 4096;
    memInfo.lkey = 0x12345678;

    EXPECT_EQ(hcclRdmaMemDeRegister(&memInfo), HCCL_SUCCESS);
}

TEST_F(InterfaceHcclVerbsTest, RdmaMemDeRegister_NullPtr) { EXPECT_EQ(hcclRdmaMemDeRegister(nullptr), HCCL_E_PTR); }

TEST_F(InterfaceHcclVerbsTest, RdmaMemDeRegister_DeRegisterMemFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalMrManager::DeRegisterMem).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendMrAttr memInfo{};
    EXPECT_EQ(hcclRdmaMemDeRegister(&memInfo), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 4. hcclCreateAscendCQWithAttr Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, CreateAscendCQWithAttr_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::CreateCq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(stub_CreateCq_success));

    AscendCQInfo cqInfo{};
    cqInfo.cqDepth = 128;

    EXPECT_EQ(hcclCreateAscendCQWithAttr(&cqInfo), HCCL_SUCCESS);
    EXPECT_EQ(cqInfo.cqn, 42u);
}

TEST_F(InterfaceHcclVerbsTest, CreateAscendCQWithAttr_NullPtr)
{
    EXPECT_EQ(hcclCreateAscendCQWithAttr(nullptr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, CreateAscendCQWithAttr_DepthTooSmall)
{
    AscendCQInfo cqInfo{};
    cqInfo.cqDepth = 32;

    EXPECT_EQ(hcclCreateAscendCQWithAttr(&cqInfo), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, CreateAscendCQWithAttr_DepthTooLarge)
{
    AscendCQInfo cqInfo{};
    cqInfo.cqDepth = 65536;

    EXPECT_EQ(hcclCreateAscendCQWithAttr(&cqInfo), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, CreateAscendCQWithAttr_DepthNotPowerOf2)
{
    AscendCQInfo cqInfo{};
    cqInfo.cqDepth = 200;

    EXPECT_EQ(hcclCreateAscendCQWithAttr(&cqInfo), HCCL_E_PARA);
}

/* ==================================================================
 * 5. hcclDestroyAscendCQ Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, DestroyAscendCQ_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::DestroyCq).stubs().will(returnValue(HCCL_SUCCESS));

    AscendCQInfo cqInfo{};
    cqInfo.cqn = 42;

    EXPECT_EQ(hcclDestroyAscendCQ(&cqInfo), HCCL_SUCCESS);
}

TEST_F(InterfaceHcclVerbsTest, DestroyAscendCQ_NullPtr) { EXPECT_EQ(hcclDestroyAscendCQ(nullptr), HCCL_E_PTR); }

TEST_F(InterfaceHcclVerbsTest, DestroyAscendCQ_DestroyCqFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::DestroyCq).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendCQInfo cqInfo{};
    cqInfo.cqn = 42;

    EXPECT_EQ(hcclDestroyAscendCQ(&cqInfo), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 6. hcclCreateAscendQPWithCQWithAttr Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, CreateQPWithCQWithAttr_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::ValidateCq).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TypicalQpManager::GetCqDepth)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetCqDepth_success));
    MOCKER_CPP(&TypicalQpManager::CreateQpWithCQ)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_CreateQpWithCQ_success));
    MOCKER_CPP(&RdmaResourceManager::GetResvMemPoolIdByType).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendCQInfo sendCQ{};
    sendCQ.cqn = 1;
    AscendCQInfo recvCQ{};
    recvCQ.cqn = 2;

    AscendVerbsQPInfo qpInfo{};
    qpInfo.cap.maxSendWr = 256;
    qpInfo.cap.maxRecvWr = 256;
    qpInfo.cap.maxSendSge = 2;
    qpInfo.cap.maxRecvSge = 2;
    qpInfo.cap.maxInlineData = 64;
    qpInfo.qp_type = ASCEND_QPT_RC;
    qpInfo.sqSigAll = 1;

    EXPECT_EQ(hcclCreateAscendQPWithCQWithAttr(&sendCQ, &recvCQ, &qpInfo), HCCL_SUCCESS);
    EXPECT_EQ(qpInfo.qpn, 100u);
    EXPECT_EQ(qpInfo.gidIdx, 1u);
    EXPECT_EQ(qpInfo.psn, 0xDEADBEEFu);
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        EXPECT_EQ(qpInfo.gid[i], static_cast<uint8_t>(i));
    }
}

TEST_F(InterfaceHcclVerbsTest, CreateQPWithCQWithAttr_CapDefaults)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::ValidateCq).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TypicalQpManager::GetCqDepth)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetCqDepth_success));
    MOCKER_CPP(&TypicalQpManager::CreateQpWithCQ)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_CreateQpWithCQ_success));
    MOCKER_CPP(&RdmaResourceManager::GetResvMemPoolIdByType).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendCQInfo sendCQ{};
    sendCQ.cqn = 1;
    AscendCQInfo recvCQ{};
    recvCQ.cqn = 2;

    AscendVerbsQPInfo qpInfo{};

    EXPECT_EQ(hcclCreateAscendQPWithCQWithAttr(&sendCQ, &recvCQ, &qpInfo), HCCL_SUCCESS);
    EXPECT_EQ(qpInfo.qp_type, (u32)ASCEND_QPT_RC);
}

TEST_F(InterfaceHcclVerbsTest, CreateQPWithCQWithAttr_SendCQNull)
{
    AscendCQInfo recvCQ{};
    AscendVerbsQPInfo qpInfo{};

    EXPECT_EQ(hcclCreateAscendQPWithCQWithAttr(nullptr, &recvCQ, &qpInfo), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, CreateQPWithCQWithAttr_RecvCQNull)
{
    AscendCQInfo sendCQ{};
    AscendVerbsQPInfo qpInfo{};

    EXPECT_EQ(hcclCreateAscendQPWithCQWithAttr(&sendCQ, nullptr, &qpInfo), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, CreateQPWithCQWithAttr_QPInfoNull)
{
    AscendCQInfo sendCQ{};
    AscendCQInfo recvCQ{};

    EXPECT_EQ(hcclCreateAscendQPWithCQWithAttr(&sendCQ, &recvCQ, nullptr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, CreateQPWithCQWithAttr_ValidateCqFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::ValidateCq).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendCQInfo sendCQ{};
    AscendCQInfo recvCQ{};
    AscendVerbsQPInfo qpInfo{};

    EXPECT_EQ(hcclCreateAscendQPWithCQWithAttr(&sendCQ, &recvCQ, &qpInfo), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 7. hcclModifyAscendVerbsQPEx Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::ModifyVerbsQp).stubs().will(returnValue(HCCL_SUCCESS));

    AscendVerbsQPInfo localQP{};
    localQP.qpn = 1;
    AscendVerbsQPInfo remoteQP{};
    remoteQP.qpn = 2;
    AscendQPQos qos{};
    qos.sl = 4;
    qos.tc = 4;

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, &remoteQP, &qos), HCCL_SUCCESS);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_LocalNull)
{
    AscendVerbsQPInfo remoteQP{};
    AscendQPQos qos{};

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(nullptr, &remoteQP, &qos), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_RemoteNull)
{
    AscendVerbsQPInfo localQP{};
    AscendQPQos qos{};

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, nullptr, &qos), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_QosNull)
{
    AscendVerbsQPInfo localQP{};
    AscendVerbsQPInfo remoteQP{};

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, &remoteQP, nullptr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_SLTooHigh)
{
    AscendVerbsQPInfo localQP{};
    AscendVerbsQPInfo remoteQP{};
    AscendQPQos qos{};
    qos.sl = 8;
    qos.tc = 4;

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, &remoteQP, &qos), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_TCTooHigh)
{
    AscendVerbsQPInfo localQP{};
    AscendVerbsQPInfo remoteQP{};
    AscendQPQos qos{};
    qos.sl = 4;
    qos.tc = 256;

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, &remoteQP, &qos), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_TCNotMultipleOf4)
{
    AscendVerbsQPInfo localQP{};
    AscendVerbsQPInfo remoteQP{};
    AscendQPQos qos{};
    qos.sl = 4;
    qos.tc = 5;

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, &remoteQP, &qos), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQPEx_ModifyQpFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::ModifyVerbsQp).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendVerbsQPInfo localQP{};
    localQP.qpn = 1;
    AscendVerbsQPInfo remoteQP{};
    remoteQP.qpn = 2;
    AscendQPQos qos{};
    qos.sl = 4;
    qos.tc = 4;

    EXPECT_EQ(hcclModifyAscendVerbsQPEx(&localQP, &remoteQP, &qos), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 8. hcclModifyAscendVerbsQP Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQP_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1u));
    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(4u));
    MOCKER_CPP(&TypicalQpManager::ModifyVerbsQp).stubs().will(returnValue(HCCL_SUCCESS));

    AscendVerbsQPInfo localQP{};
    localQP.qpn = 1;
    AscendVerbsQPInfo remoteQP{};
    remoteQP.qpn = 2;

    EXPECT_EQ(hcclModifyAscendVerbsQP(&localQP, &remoteQP), HCCL_SUCCESS);
}

TEST_F(InterfaceHcclVerbsTest, ModifyVerbsQP_GetDeviceRefreshFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_fail));

    AscendVerbsQPInfo localQP{};
    AscendVerbsQPInfo remoteQP{};

    EXPECT_EQ(hcclModifyAscendVerbsQP(&localQP, &remoteQP), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 9. hcclPollAscendCQ Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, PollAscendCQ_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetCqHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetCqHandle_success));
    MOCKER(HrtRaPollTypicalCq).stubs().will(invoke(stub_HrtRaPollTypicalCq_success));

    AscendCQInfo cqInfo{};
    cqInfo.cqn = 42;
    uint32_t polledNum = 0;
    struct AscendWc wc[16]{};

    EXPECT_EQ(hcclPollAscendCQ(&cqInfo, 16, &polledNum, wc), HCCL_SUCCESS);
    EXPECT_EQ(polledNum, 2u);
}

TEST_F(InterfaceHcclVerbsTest, PollAscendCQ_CqInfoNull)
{
    uint32_t polledNum = 0;
    struct AscendWc wc[16]{};

    EXPECT_EQ(hcclPollAscendCQ(nullptr, 16, &polledNum, wc), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PollAscendCQ_PolledNumNull)
{
    AscendCQInfo cqInfo{};
    struct AscendWc wc[16]{};

    EXPECT_EQ(hcclPollAscendCQ(&cqInfo, 16, nullptr, wc), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PollAscendCQ_WcNull)
{
    AscendCQInfo cqInfo{};
    uint32_t polledNum = 0;

    EXPECT_EQ(hcclPollAscendCQ(&cqInfo, 16, &polledNum, nullptr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PollAscendCQ_PollFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetCqHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetCqHandle_success));
    MOCKER(HrtRaPollTypicalCq).stubs().will(invoke(stub_HrtRaPollTypicalCq_fail));

    AscendCQInfo cqInfo{};
    cqInfo.cqn = 42;
    uint32_t polledNum = 0;
    struct AscendWc wc[16]{};

    EXPECT_EQ(hcclPollAscendCQ(&cqInfo, 16, &polledNum, wc), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 10. hcclAscendPostSend Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, PostSend_Success_SingleWR)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));
    MOCKER(HrtRaSendWrVerbs).stubs().will(invoke(stub_HrtRaSendWrVerbs_success));
    MOCKER(hrtRDMADBSend).stubs().will(invoke(stub_hrtRDMADBSend_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendSge sge {};
    sge.addr = 0x1000;
    sge.len = 256;
    sge.lkey = 0x1234;

    struct AscendSendWr sendWr {};
    sendWr.wrId = 1;
    sendWr.numSge = 1;
    sendWr.sgList = &sge;
    sendWr.opcode = ASCEND_WR_RDMA_WRITE;
    sendWr.next = nullptr;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, &sendWr, stream, &badWr), HCCL_SUCCESS);
    EXPECT_EQ(badWr, nullptr);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_Success_LinkedListWRs)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));
    MOCKER(HrtRaSendWrVerbs).stubs().will(invoke(stub_HrtRaSendWrVerbs_success));
    MOCKER(hrtRDMADBSend).stubs().will(invoke(stub_hrtRDMADBSend_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendSge sge1 {
    }, sge2{}, sge3{};
    sge1.addr = 0x1000;
    sge1.len = 256;
    sge1.lkey = 0x1234;
    sge2.addr = 0x2000;
    sge2.len = 512;
    sge2.lkey = 0x5678;
    sge3.addr = 0x3000;
    sge3.len = 128;
    sge3.lkey = 0x9ABC;

    struct AscendSendWr wr1 {
    }, wr2{}, wr3{};
    wr1.wrId = 1;
    wr1.numSge = 1;
    wr1.sgList = &sge1;
    wr1.opcode = ASCEND_WR_RDMA_WRITE;
    wr1.next = &wr2;
    wr2.wrId = 2;
    wr2.numSge = 1;
    wr2.sgList = &sge2;
    wr2.opcode = ASCEND_WR_RDMA_WRITE;
    wr2.next = &wr3;
    wr3.wrId = 3;
    wr3.numSge = 1;
    wr3.sgList = &sge3;
    wr3.opcode = ASCEND_WR_RDMA_WRITE;
    wr3.next = nullptr;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, &wr1, stream, &badWr), HCCL_SUCCESS);
    EXPECT_EQ(badWr, nullptr);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_QpInfoNull)
{
    struct AscendSendWr sendWr {};
    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(nullptr, &sendWr, stream, &badWr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_SendWrNull)
{
    AscendVerbsQPInfo qpInfo{};
    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, nullptr, stream, &badWr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_BadWrNull)
{
    AscendVerbsQPInfo qpInfo{};
    struct AscendSendWr sendWr {};
    aclrtStream stream = (aclrtStream)0x87654321;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, &sendWr, stream, nullptr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_StreamNull)
{
    AscendVerbsQPInfo qpInfo{};
    struct AscendSendWr sendWr {};
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, &sendWr, nullptr, &badWr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_NumSgeExceedsMax)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendSendWr sendWr {};
    sendWr.numSge = 17;
    sendWr.sgList = (struct AscendSge*)0x1000;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, &sendWr, stream, &badWr), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, PostSend_NullSgList)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendSendWr sendWr {};
    sendWr.numSge = 1;
    sendWr.sgList = nullptr;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendSendWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostSend(&qpInfo, &sendWr, stream, &badWr), HCCL_E_PTR);
}

/* ==================================================================
 * 11. hcclAscendPostRecv Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, PostRecv_Success_SingleWR)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));
    MOCKER(HrtRaRecvWrVerbs).stubs().will(invoke(stub_HrtRaRecvWrVerbs_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendSge sge {};
    sge.addr = 0x1000;
    sge.len = 256;
    sge.lkey = 0x1234;

    struct AscendRecvWr recvWr {};
    recvWr.wrId = 1;
    recvWr.numSge = 1;
    recvWr.sgList = &sge;
    recvWr.next = nullptr;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, &recvWr, stream, &badWr), HCCL_SUCCESS);
    EXPECT_EQ(badWr, nullptr);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_Success_LinkedListWRs)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));
    MOCKER(HrtRaRecvWrVerbs).stubs().will(invoke(stub_HrtRaRecvWrVerbs_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendSge sge1 {
    }, sge2{}, sge3{};
    sge1.addr = 0x1000;
    sge1.len = 256;
    sge1.lkey = 0x1234;
    sge2.addr = 0x2000;
    sge2.len = 512;
    sge2.lkey = 0x5678;
    sge3.addr = 0x3000;
    sge3.len = 128;
    sge3.lkey = 0x9ABC;

    struct AscendRecvWr wr1 {
    }, wr2{}, wr3{};
    wr1.wrId = 1;
    wr1.numSge = 1;
    wr1.sgList = &sge1;
    wr1.next = &wr2;
    wr2.wrId = 2;
    wr2.numSge = 1;
    wr2.sgList = &sge2;
    wr2.next = &wr3;
    wr3.wrId = 3;
    wr3.numSge = 1;
    wr3.sgList = &sge3;
    wr3.next = nullptr;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, &wr1, stream, &badWr), HCCL_SUCCESS);
    EXPECT_EQ(badWr, nullptr);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_QpInfoNull)
{
    struct AscendRecvWr recvWr {};
    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(nullptr, &recvWr, stream, &badWr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_RecvWrNull)
{
    AscendVerbsQPInfo qpInfo{};
    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, nullptr, stream, &badWr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_BadWrNull)
{
    AscendVerbsQPInfo qpInfo{};
    struct AscendRecvWr recvWr {};
    aclrtStream stream = (aclrtStream)0x87654321;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, &recvWr, stream, nullptr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_StreamNull)
{
    AscendVerbsQPInfo qpInfo{};
    struct AscendRecvWr recvWr {};
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, &recvWr, nullptr, &badWr), HCCL_E_PTR);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_NumSgeExceedsMax)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendRecvWr recvWr {};
    recvWr.numSge = 17;
    recvWr.sgList = (struct AscendSge*)0x1000;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, &recvWr, stream, &badWr), HCCL_E_PARA);
}

TEST_F(InterfaceHcclVerbsTest, PostRecv_NullSgList)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::GetVerbsQpHandleByQpn)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(stub_GetVerbsQpHandleByQpn_success));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    struct AscendRecvWr recvWr {};
    recvWr.numSge = 1;
    recvWr.sgList = nullptr;

    aclrtStream stream = (aclrtStream)0x87654321;
    struct AscendRecvWr* badWr = nullptr;

    EXPECT_EQ(hcclAscendPostRecv(&qpInfo, &recvWr, stream, &badWr), HCCL_E_PTR);
}

/* ==================================================================
 * 12. hcclDestroyAscendVerbsQP Tests
 * ================================================================== */
TEST_F(InterfaceHcclVerbsTest, DestroyVerbsQP_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::DestroyQpWithoutCQ).stubs().will(returnValue(HCCL_SUCCESS));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    EXPECT_EQ(hcclDestroyAscendVerbsQP(&qpInfo), HCCL_SUCCESS);
}

TEST_F(InterfaceHcclVerbsTest, DestroyVerbsQP_NullPtr) { EXPECT_EQ(hcclDestroyAscendVerbsQP(nullptr), HCCL_E_PTR); }

TEST_F(InterfaceHcclVerbsTest, DestroyVerbsQP_DestroyQpFailure)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));
    MOCKER_CPP(&TypicalQpManager::DestroyQpWithoutCQ).stubs().will(returnValue(HCCL_E_INTERNAL));

    AscendVerbsQPInfo qpInfo{};
    qpInfo.qpn = 100;

    EXPECT_EQ(hcclDestroyAscendVerbsQP(&qpInfo), HCCL_E_INTERNAL);
}

/* ==================================================================
 * Test Fixture: TypicalQpManagerTest
 * ================================================================== */
class TypicalQpManagerTest : public testing::Test {
protected:
    virtual void SetUp() { std::cout << "TypicalQpManagerTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        TypicalQpManager& instance = TypicalQpManager::GetInstance();
        ClearAllMaps(instance);
        GlobalMockObject::verify();
        std::cout << "TypicalQpManagerTest TearDown" << std::endl;
    }
};

/* ==================================================================
 * 1. GetVerbsQpHandleByQpn Tests
 * ================================================================== */
TEST_F(TypicalQpManagerTest, GetVerbsQpHandleByQpn_Success)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 42;
    QpHandle qpHandle = reinterpret_cast<QpHandle>(0xABCD0001);
    instance.verbsQpMap_[42] = std::make_pair(qpInfo, qpHandle);

    QpHandle result = nullptr;
    EXPECT_EQ(instance.GetVerbsQpHandleByQpn(42, result), HCCL_SUCCESS);
    EXPECT_EQ(result, qpHandle);
}

TEST_F(TypicalQpManagerTest, GetVerbsQpHandleByQpn_QpnNotFound)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);
    ClearAllMaps(instance);

    QpHandle result = nullptr;
    EXPECT_EQ(instance.GetVerbsQpHandleByQpn(999, result), HCCL_E_NOT_FOUND);
}

TEST_F(TypicalQpManagerTest, GetVerbsQpHandleByQpn_NullQpHandleInMap)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 43;
    instance.verbsQpMap_[43] = std::make_pair(qpInfo, nullptr);

    QpHandle result = nullptr;
    EXPECT_EQ(instance.GetVerbsQpHandleByQpn(43, result), HCCL_E_NOT_FOUND);
}

TEST_F(TypicalQpManagerTest, GetVerbsQpHandleByQpn_RdmaHandleFail)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_fail));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    instance.rdmaHandle_ = nullptr;

    QpHandle result = nullptr;
    EXPECT_NE(instance.GetVerbsQpHandleByQpn(1, result), HCCL_SUCCESS);
}

/* ==================================================================
 * 2. ModifyVerbsQp Tests
 * ================================================================== */
TEST_F(TypicalQpManagerTest, ModifyVerbsQp_Success)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(invoke(stub_GetExternalInputRdmaRetryCnt));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(invoke(stub_GetExternalInputRdmaTimeOut));
    MOCKER(hrtRaTypicalQpModify).stubs().will(invoke(stub_hrtRaTypicalQpModify_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 1;
    QpHandle qpHandle = reinterpret_cast<QpHandle>(0xBEEF0001);
    instance.verbsQpMap_[1] = std::make_pair(qpInfo, qpHandle);

    struct TypicalQp localQp {};
    localQp.qpn = 1;
    struct TypicalQp remoteQp {};
    remoteQp.qpn = 2;

    EXPECT_EQ(instance.ModifyVerbsQp(localQp, remoteQp), HCCL_SUCCESS);
    EXPECT_EQ(localQp.retryCnt, 7u);
    EXPECT_EQ(localQp.retryTime, 14u);
}

TEST_F(TypicalQpManagerTest, ModifyVerbsQp_LocalQpnZero)
{
    TypicalQpManager& instance = TypicalQpManager::GetInstance();

    struct TypicalQp localQp {};
    localQp.qpn = 0;
    struct TypicalQp remoteQp {};
    remoteQp.qpn = 1;

    EXPECT_EQ(instance.ModifyVerbsQp(localQp, remoteQp), HCCL_E_PARA);
}

TEST_F(TypicalQpManagerTest, ModifyVerbsQp_RemoteQpnZero)
{
    TypicalQpManager& instance = TypicalQpManager::GetInstance();

    struct TypicalQp localQp {};
    localQp.qpn = 1;
    struct TypicalQp remoteQp {};
    remoteQp.qpn = 0;

    EXPECT_EQ(instance.ModifyVerbsQp(localQp, remoteQp), HCCL_E_PARA);
}

TEST_F(TypicalQpManagerTest, ModifyVerbsQp_GetVerbsQpHandleByQpnFails)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);
    ClearAllMaps(instance);

    struct TypicalQp localQp {};
    localQp.qpn = 42;
    struct TypicalQp remoteQp {};
    remoteQp.qpn = 2;

    EXPECT_EQ(instance.ModifyVerbsQp(localQp, remoteQp), HCCL_E_NOT_FOUND);
}

TEST_F(TypicalQpManagerTest, ModifyVerbsQp_HrtModifyFails)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(invoke(stub_GetExternalInputRdmaRetryCnt));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(invoke(stub_GetExternalInputRdmaTimeOut));
    MOCKER(hrtRaTypicalQpModify).stubs().will(invoke(stub_hrtRaTypicalQpModify_fail));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 1;
    QpHandle qpHandle = reinterpret_cast<QpHandle>(0xBEEF0001);
    instance.verbsQpMap_[1] = std::make_pair(qpInfo, qpHandle);

    struct TypicalQp localQp {};
    localQp.qpn = 1;
    struct TypicalQp remoteQp {};
    remoteQp.qpn = 2;

    EXPECT_EQ(instance.ModifyVerbsQp(localQp, remoteQp), HCCL_E_INTERNAL);
}

/* ==================================================================
 * 3. CreateQpWithCQ Tests (verbsQpMap_ insertion)
 * ================================================================== */
TEST_F(TypicalQpManagerTest, CreateQpWithCQ_InsertsIntoVerbsQpMap)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(CreateQpWithCQConfig).stubs().will(invoke(stub_CreateQpWithCQConfig_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);
    ClearAllMaps(instance);

    struct TypicalQp qpInfo {};
    QpConfigWithCQInfo qpConfig{};

    EXPECT_EQ(instance.CreateQpWithCQ(qpInfo, qpConfig), HCCL_SUCCESS);
    EXPECT_EQ(qpInfo.qpn, 100u);

    EXPECT_EQ(instance.verbsQpMap_.size(), 1u);
    EXPECT_EQ(instance.qpMap_.size(), 0u);
    EXPECT_NE(instance.verbsQpMap_.find(100), instance.verbsQpMap_.end());
}

TEST_F(TypicalQpManagerTest, CreateQpWithCQ_CreateFails)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(CreateQpWithCQConfig).stubs().will(invoke(stub_CreateQpWithCQConfig_fail));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);
    ClearAllMaps(instance);

    struct TypicalQp qpInfo {};
    QpConfigWithCQInfo qpConfig{};

    EXPECT_EQ(instance.CreateQpWithCQ(qpInfo, qpConfig), HCCL_E_INTERNAL);
    EXPECT_EQ(instance.verbsQpMap_.size(), 0u);
}

/* ==================================================================
 * 4. DestroyQpWithoutCQ Tests (verbsQpMap_ erase)
 * ================================================================== */
TEST_F(TypicalQpManagerTest, DestroyQpWithoutCQ_Success)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(HrtRaQpDestroyWithoutCQ).stubs().will(invoke(stub_HrtRaQpDestroyWithoutCQ_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 100;
    QpHandle qpHandle = reinterpret_cast<QpHandle>(0xBEEF0001);
    instance.verbsQpMap_[100] = std::make_pair(qpInfo, qpHandle);

    EXPECT_EQ(instance.verbsQpMap_.size(), 1u);

    EXPECT_EQ(instance.DestroyQpWithoutCQ(qpInfo), HCCL_SUCCESS);
    EXPECT_EQ(instance.verbsQpMap_.size(), 0u);
    EXPECT_EQ(instance.verbsQpMap_.find(100), instance.verbsQpMap_.end());
}

TEST_F(TypicalQpManagerTest, DestroyQpWithoutCQ_QpnZero)
{
    TypicalQpManager& instance = TypicalQpManager::GetInstance();

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 0;

    EXPECT_EQ(instance.DestroyQpWithoutCQ(qpInfo), HCCL_E_PARA);
}

TEST_F(TypicalQpManagerTest, DestroyQpWithoutCQ_QpnNotFound)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);
    ClearAllMaps(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 999;

    EXPECT_EQ(instance.DestroyQpWithoutCQ(qpInfo), HCCL_E_NOT_FOUND);
}

TEST_F(TypicalQpManagerTest, DestroyQpWithoutCQ_HrtDestroyFails)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(HrtRaQpDestroyWithoutCQ).stubs().will(invoke(stub_HrtRaQpDestroyWithoutCQ_fail));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);

    struct TypicalQp qpInfo {};
    qpInfo.qpn = 100;
    QpHandle qpHandle = reinterpret_cast<QpHandle>(0xBEEF0001);
    instance.verbsQpMap_[100] = std::make_pair(qpInfo, qpHandle);

    EXPECT_EQ(instance.DestroyQpWithoutCQ(qpInfo), HCCL_E_INTERNAL);
    EXPECT_EQ(instance.verbsQpMap_.size(), 1u);
}

/* ==================================================================
 * 5. CreateCq Tests
 * ================================================================== */
TEST_F(TypicalQpManagerTest, CreateCq_InsertsIntoCqMap)
{
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER_CPP(&RdmaResourceManager::GetRdmaHandle).stubs().will(invoke(stub_GetRdmaHandle_success));
    MOCKER(CreateTypicalCq).stubs().will(invoke(stub_CreateTypicalCq_success));

    TypicalQpManager& instance = TypicalQpManager::GetInstance();
    SetRdmaHandle(instance);
    ClearAllMaps(instance);

    AscendCQInfo cqInfo{};
    cqInfo.cqDepth = 128;

    EXPECT_EQ(instance.CreateCq(cqInfo), HCCL_SUCCESS);
    EXPECT_EQ(cqInfo.cqn, 42u);
    EXPECT_EQ(instance.cqMap_.size(), 1u);
}
