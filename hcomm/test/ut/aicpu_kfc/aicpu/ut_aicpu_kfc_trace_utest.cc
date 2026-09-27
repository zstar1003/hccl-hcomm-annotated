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

#ifndef private
#define private public
#define protected public
#endif
#include "mmpa_api.h"
#include "atrace_pub.h"
#include "dfx/mc2_trace_utils.h"
#include "common/aicpu_kfc_def.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

namespace {
char g_traceHandleStub;
TraceStructEntry g_traceStructEntryStub;
bool g_isError = false;

TraHandle TraceCreateWithAttr(TracerType tracerType, const char* objName, const TraceAttr* attr)
{
    if (g_isError) {
        return -1;
    }
    return 1;
}

TraHandle TraceGetHandle(TracerType tracerType, const char* objName) { return 1; }

TraStatus TraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize)
{
    if (buffer == nullptr) {
        return TRACE_FAILURE;
    }
    return TRACE_SUCCESS;
}

TraStatus TraceSave(TracerType tracerType, bool syncFlag) { return TRACE_SUCCESS; }

TraEventHandle TraceEventCreate(const char* eventName) { return 1; }

void TraceEventDestroy(TraEventHandle eventHandle) { return; }

void TraceDestroy(TraHandle handle) { return; }

TraStatus TraceEventBindTrace(TraEventHandle eventHandle, TraHandle handle) { return TRACE_SUCCESS; }

TraStatus TraceEventReport(TraEventHandle eventHandle) { return TRACE_SUCCESS; }

TraStatus TraceSetGlobalAttr(const TraceGlobalAttr* attr) { return TRACE_SUCCESS; }

TraceStructEntry* TraceStructEntryCreate(const char* name) { return &g_traceStructEntryStub; }

void TraceStructItemFieldSet(
    TraceStructEntry* en, const char* name, uint8_t type, uint8_t mode, uint8_t bytes, uint64_t length)
{
    return;
}

void TraceStructItemArraySet(
    TraceStructEntry* en, const char* name, uint8_t type, uint8_t mode, uint8_t bytes, uint64_t length)
{
    return;
}

void TraceStructSetAttr(TraceStructEntry* en, TraceAttr* attr) { return; }

void TraceStructEntryDestroy(TraceStructEntry* en) { return; }

std::map<std::string, void*> Mc2UtraceMap = {
    {"UtraceCreateWithAttr", (void*)TraceCreateWithAttr},
    {"UtraceGetHandle", (void*)TraceGetHandle},
    {"UtraceSubmit", (void*)TraceSubmit},
    {"UtraceEventCreate", (void*)TraceEventCreate},
    {"UtraceEventDestroy", (void*)TraceEventDestroy},
    {"UtraceDestroy", (void*)TraceDestroy},
    {"UtraceEventBindTrace", (void*)TraceEventBindTrace},
    {"UtraceEventReport", (void*)TraceEventReport},
    {"UtraceSetGlobalAttr", (void*)TraceSetGlobalAttr},
    {"UtraceStructEntryCreate", (void*)TraceStructEntryCreate},
    {"UtraceStructItemFieldSet", (void*)TraceStructItemFieldSet},
    {"UtraceStructItemArraySet", (void*)TraceStructItemArraySet},
    {"UtraceStructSetAttr", (void*)TraceStructSetAttr},
    {"UtraceStructEntryDestroy", (void*)TraceStructEntryDestroy},
};

std::map<std::string, void*> Mc2AtraceMap = {
    {"AtraceCreateWithAttr", (void*)TraceCreateWithAttr},
    {"AtraceGetHandle", (void*)TraceGetHandle},
    {"AtraceSubmit", (void*)TraceSubmit},
    {"AtraceEventCreate", (void*)TraceEventCreate},
    {"AtraceEventDestroy", (void*)TraceEventDestroy},
    {"AtraceDestroy", (void*)TraceDestroy},
    {"AtraceEventBindTrace", (void*)TraceEventBindTrace},
    {"AtraceEventReport", (void*)TraceEventReport},
    {"AtraceSetGlobalAttr", (void*)TraceSetGlobalAttr},
    {"AtraceStructEntryCreate", (void*)TraceStructEntryCreate},
    {"AtraceStructItemFieldSet", (void*)TraceStructItemFieldSet},
    {"AtraceStructItemArraySet", (void*)TraceStructItemArraySet},
    {"AtraceStructSetAttr", (void*)TraceStructSetAttr},
    {"AtraceStructEntryDestroy", (void*)TraceStructEntryDestroy},
};

void* DlopenStub(const char* filename, int flags)
{
    if (filename == nullptr) {
        return nullptr;
    }

    return (void*)&g_traceHandleStub;
}

void* DlsymStub(void* handle, const char* symbol)
{
    if (symbol == nullptr) {
        return nullptr;
    }
    std::string symbolt = symbol;
    auto it = Mc2UtraceMap.find(symbolt);
    if (it != Mc2UtraceMap.cend()) {
        return it->second;
    }
    it = Mc2AtraceMap.find(symbolt);
    if (it != Mc2AtraceMap.cend()) {
        return it->second;
    }
    return nullptr;
}

int DlcloseStub(void* handle)
{
    if (handle == nullptr) {
        return -1;
    } else if (handle == (void*)&g_traceHandleStub) {
        return 0;
    }

    return dlclose(handle);
}

void MockTraceDlopen()
{
    MOCKER(mmDlopen).stubs().will(invoke(DlopenStub));
    MOCKER(mmDlsym).stubs().will(invoke(DlsymStub));
}
} // namespace

class MC2Trace_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2Trace_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2Trace_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        g_stubDevType = DevType::DEV_TYPE_910B;
        MockTraceDlopen();
        std::cout << "MC2Trace_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2Trace_UT Test TearDown" << std::endl;
    }
};

TEST_F(MC2Trace_UT, Init_Error)
{
    g_isError = true;
    EXPECT_EQ(MC2TraceUtils::InitFuncHandle(), HCCL_SUCCESS);
    EXPECT_NE(MC2TraceUtils::InitTaskAndTilingDataHandle(), HCCL_SUCCESS);
    EXPECT_NE(MC2TraceUtils::InitAicpuComDataHandle(), HCCL_SUCCESS);
    EXPECT_NE(MC2TraceUtils::InitTraceStrHandle(), HCCL_SUCCESS);
    EXPECT_NE(MC2TraceUtils::InitMsgInfoHandle(), HCCL_SUCCESS);
    EXPECT_NE(MC2TraceUtils::InitSqeBatchInfoHandle(), HCCL_SUCCESS);
    g_isError = false;
}

TEST_F(MC2Trace_UT, Init)
{
    HcclResult ret = MC2TraceUtils::Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MC2Trace_UT, InitV2)
{
    HcclResult ret = MC2TraceUtils::InitFuncHandle();
    ret = MC2TraceUtils::InitTraceHandle();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MC2Trace_UT, SubmitTest)
{
    EXPECT_EQ(MC2TraceUtils::Init(), HCCL_SUCCESS);

    KFCTask task;
    HcclKFCTilingData tilingData = {0};
    EXPECT_EQ(MC2TraceUtils::Submit(&task, &tilingData), HCCL_SUCCESS);
    AicpuComContext ctx;
    EXPECT_EQ(MC2TraceUtils::Submit(&ctx), HCCL_SUCCESS);
    SqeBatchInfo sqeBatchInfo;
    EXPECT_EQ(MC2TraceUtils::Submit(&sqeBatchInfo), HCCL_SUCCESS);
    std::string traceStr = "test trace string";
    EXPECT_EQ(MC2TraceUtils::Submit(traceStr), HCCL_SUCCESS);
    EXPECT_EQ(MC2TraceUtils::Submit(traceStr.c_str()), HCCL_SUCCESS);
}

TEST_F(MC2Trace_UT, SaveTest)
{
    EXPECT_EQ(MC2TraceUtils::Init(), HCCL_SUCCESS);
    EXPECT_EQ(MC2TraceUtils::Save(), HCCL_SUCCESS);
}

TEST(MC2TraceUtilsTest, DestoryHandlesTest)
{
    EXPECT_EQ(MC2TraceUtils::Init(), HCCL_SUCCESS);
    MC2TraceUtils::DestoryHandles();
}
