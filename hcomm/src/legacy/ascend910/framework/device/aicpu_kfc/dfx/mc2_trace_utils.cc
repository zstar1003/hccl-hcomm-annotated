/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_trace_utils.h"

#include <cmath>
#include "log.h"
#include "mmpa_api.h"
#include "atrace_pub.h"
#include "common/aicpu_hccl_common.h"
#include "framework/aicpu_kfc_prof.h"
#include "common/aicpu_kfc_utils.h"
#include "adapter_hal_pub.h"

namespace {
constexpr uint32_t TRACE_RING_BUFF_SIZE = 128 * 1024;
constexpr uint32_t MAX_SQE_SUBMIT_NUM = 256U;
constexpr char UTRACE_SO[] = "libutrace.so";
constexpr char ATRACE_SO[] = "libascend_trace.so";

using TraceCreateWithAttrFunc = TraHandle (*)(TracerType, const char*, const TraceAttr*);
using TraceGetHandleFunc = TraHandle (*)(TracerType, const char*);
using TraceSubmitFunc = TraStatus (*)(TraHandle handle, const void*, uint32_t);
using TraceEventCreateFunc = TraEventHandle (*)(const char*);
using TraceEventDestroyFunc = void (*)(TraEventHandle);
using TraceHandleDestroyFunc = void (*)(TraHandle);
using TraceEventBindTraceFunc = TraStatus (*)(TraEventHandle, TraHandle);
using TraceEventReportFunc = TraStatus (*)(TraEventHandle);
using TraceSetGlobalAttrFunc = TraStatus (*)(const TraceGlobalAttr*);
using TraceStructEntryCreateFunc = TraceStructEntry* (*)(const char*);
using TraceStructItemFieldSetFunc = void (*)(TraceStructEntry*, const char*, uint8_t, uint8_t, uint64_t);
using TraceStructItemArraySetFunc = void (*)(TraceStructEntry*, const char*, uint8_t, uint8_t, uint64_t);
using TraceStructSetAttrFunc = void (*)(TraceStructEntry*, uint8_t, TraceAttr*);
using TraceStructEntryDestroyFunc = void (*)(TraceStructEntry*);

void* g_soHandle = nullptr;
bool g_isAtrace = true;
TraceCreateWithAttrFunc g_traceCreateWithAttr = nullptr;
TraceGetHandleFunc g_traceGetHandle = nullptr;
TraceSubmitFunc g_traceSubmit = nullptr;
TraceEventCreateFunc g_traceEventCreate = nullptr;
TraceEventDestroyFunc g_traceEventDestroy = nullptr;
TraceHandleDestroyFunc g_traceHandleDestroy = nullptr;
TraceEventBindTraceFunc g_traceEventBindTrace = nullptr;
TraceEventReportFunc g_traceEventReport = nullptr;
TraceSetGlobalAttrFunc g_traceSetGlobalAttr = nullptr;
TraceStructEntryCreateFunc g_traceStructEntryCreate = nullptr;
TraceStructItemFieldSetFunc g_traceStructItemFieldSet = nullptr;
TraceStructItemArraySetFunc g_traceStructItemArraySet = nullptr;
TraceStructSetAttrFunc g_traceStructSetAttr = nullptr;
TraceStructEntryDestroyFunc g_traceStructEntryDestroy = nullptr;

TraEventHandle g_eventHandle = -1;
TraHandle g_traceStrHandle = -1;
TraHandle g_traceTaskAndTilingDataHandle = -1;
TraHandle g_traceAicpuComDataHandle = -1;
TraHandle g_traceMsgInfoHandle = -1;
TraHandle g_traceSqeBatchInfoHandle = -1;
TraceStructEntry* g_traceStrSt = nullptr;
TraceStructEntry* g_traceAicpuComTraceSt = nullptr;
TraceStructEntry* g_traceKFCtaskAndTilingTraceDataSt = nullptr;
TraceStructEntry* g_traceMsgInfoSt = nullptr;
TraceStructEntry* g_traceSqeBatchInfoSt = nullptr;
} // namespace

// 当msgNum不为2的幂时，会自动向上取2的幂
uint16_t MC2TraceUtils::GetMsgNum(size_t msgSize)
{
    float logValue = log(TRACE_RING_BUFF_SIZE / (msgSize + 16)) / log(2); // 16 预留字节 2 幂次
    return static_cast<uint16_t>(pow(2, floor(logValue)));                // 2 幂次
}

HcclResult MC2TraceUtils::InitTraceStrHandle()
{
    // trace api存在限制: msgNum * msgSize <= 128k
    uint16_t strMsgNum = GetMsgNum(sizeof(TraceStr));
    TraceAttr attr = {};
    attr = {false, strMsgNum, sizeof(TraceStr), {nullptr}, 0, {0}};
    g_traceStrSt = g_traceStructEntryCreate("TraceStr");
    g_traceStructItemArraySet(
        g_traceStrSt, "transmit", TRACE_STRUCT_ARRAY_TYPE_CHAR, TRACE_STRUCT_SHOW_MODE_CHAR,
        sizeof(TraceStr::transmit));
    g_traceStructSetAttr(g_traceStrSt, 0, &attr);
    g_traceStrHandle = g_traceCreateWithAttr(TRACER_TYPE_SCHEDULE, "TraceStr", &attr);
    if (g_traceStrHandle < 0) {
        HCCL_ERROR("Create g_traceStrHandle failed, ret:%d", g_traceStrHandle);
        g_traceStructEntryDestroy(g_traceStrSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_traceEventBindTrace(g_eventHandle, g_traceStrHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceStrHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

void MC2TraceUtils::SetHcclKFCTilingDataOne()
{
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "sendOff", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "recvOff", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "tailSendOff", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "tailRecvOff", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "sendCnt", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "recvCnt", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "tailSendCnt", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "tailRecvCnt", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "totalCnt", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "turnNum", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "tailNum", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "stride", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "workspaceOff", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "notifyOff", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "notifyBeginCnt", TRACE_STRUCT_FIELD_TYPE_UINT16,
        TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "notifyEndCnt", TRACE_STRUCT_FIELD_TYPE_UINT16, TRACE_STRUCT_SHOW_MODE_DEC,
        2); // 2 uint16
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "useBufferType", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "funID", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "dataType", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "groupNum", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "reuseMode", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
}

void MC2TraceUtils::SetHcclKFCTilingDataTwo()
{
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "commType", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "reduceOp", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "commOrder", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "waitPolicy", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "rspPolicy", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "exitPolicy", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "commAlg", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "taskType", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "debugMode", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "stepSize", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "sendArgIndex", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "recvArgIndex", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "commOutArgIndex", TRACE_STRUCT_FIELD_TYPE_UINT8,
        TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "hasCommOut", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "reverse", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "reserve2", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        4); // 4 uint32
}

HcclResult MC2TraceUtils::InitTaskAndTilingDataHandle()
{
    uint16_t taskAndTilingDataMsgNum = GetMsgNum(sizeof(KFCtaskAndTilingTraceData));
    TraceAttr attr = {};
    attr = {false, taskAndTilingDataMsgNum, sizeof(KFCtaskAndTilingTraceData), {nullptr}, 0, {0}};
    g_traceKFCtaskAndTilingTraceDataSt = g_traceStructEntryCreate("KFCtaskAndTilingTraceData");
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "inputA", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "outputC", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "commOut", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "context", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "workSpace", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceKFCtaskAndTilingTraceDataSt, "tilingData", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    SetHcclKFCTilingDataOne();
    SetHcclKFCTilingDataTwo();
    g_traceStructSetAttr(g_traceKFCtaskAndTilingTraceDataSt, 0, &attr);
    g_traceTaskAndTilingDataHandle = g_traceCreateWithAttr(TRACER_TYPE_SCHEDULE, "KFCtaskAndTilingTraceData", &attr);
    if (g_traceTaskAndTilingDataHandle < 0) {
        HCCL_ERROR("Create g_traceTaskAndTilingDataHandle failed, ret:%d", g_traceTaskAndTilingDataHandle);
        g_traceStructEntryDestroy(g_traceKFCtaskAndTilingTraceDataSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_traceEventBindTrace(g_eventHandle, g_traceTaskAndTilingDataHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceTaskAndTilingDataHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitAicpuComDataHandle()
{
    uint16_t aicpuComMsgNum = GetMsgNum(sizeof(AicpuComTraceData));
    TraceAttr attr = {};
    attr = {false, aicpuComMsgNum, sizeof(AicpuComTraceData), {nullptr}, 0, {0}};
    g_traceAicpuComTraceSt = g_traceStructEntryCreate("AicpuComTraceData");
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "devId", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "ssid", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "rankId", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "rankNum", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "windowSize", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_DEC,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "workSpaceAddr", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "kfcNotifyId", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_DEC,
        8); // 8 uint64
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "eventIds", TRACE_STRUCT_ARRAY_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC,
        128); // 4 uint32 128 AC_MAX_RANK_NUM*uint32
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "windowIn", TRACE_STRUCT_ARRAY_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        256); // 8 uint64 256 AC_MAX_RANK_NUM*uint64
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "windowOut", TRACE_STRUCT_ARRAY_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        256); // 8 uint64 256 AC_MAX_RANK_NUM*uint64
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "actualStreamId", TRACE_STRUCT_ARRAY_TYPE_INT32, TRACE_STRUCT_SHOW_MODE_DEC,
        128); // 4 uint32 128 AC_MAX_RANK_NUM*uint32
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "sqId", TRACE_STRUCT_ARRAY_TYPE_INT32, TRACE_STRUCT_SHOW_MODE_DEC,
        128); // 4 uint32 128 AC_MAX_RANK_NUM*uint32
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "aicpuOpNotifyAddress", TRACE_STRUCT_ARRAY_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX,
        16); // 8 uint64 16 2*uint64
    g_traceStructItemArraySet(
        g_traceAicpuComTraceSt, "aicpuOpNotifyActualNotifyId", TRACE_STRUCT_ARRAY_TYPE_INT32,
        TRACE_STRUCT_SHOW_MODE_DEC, 8); // 4 uint32 8 2*uint32
    g_traceStructItemFieldSet(
        g_traceAicpuComTraceSt, "clusterId", TRACE_STRUCT_FIELD_TYPE_INT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructSetAttr(g_traceAicpuComTraceSt, 0, &attr);
    g_traceAicpuComDataHandle = g_traceCreateWithAttr(TRACER_TYPE_SCHEDULE, "AicpuComTraceData", &attr);
    if (g_traceAicpuComDataHandle < 0) {
        HCCL_ERROR("Create g_traceAicpuComDataHandle failed, ret:%d", g_traceAicpuComDataHandle);
        g_traceStructEntryDestroy(g_traceAicpuComTraceSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_traceEventBindTrace(g_eventHandle, g_traceAicpuComDataHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceAicpuComDataHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

void MC2TraceUtils::SetTraceMsgInfo()
{
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "commType", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "opType", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "sendBuffer", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "recvBuffer", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_HEX, 8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "count", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "strideLen", TRACE_STRUCT_FIELD_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "hcclDataType", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "valid", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "isLast", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "funID", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "sendCnt", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "rcvCnt", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "everyTurnRsp", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "everyTurnWait", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "totalTurnCnt", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "useBufferType", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceMsgInfoSt, "winOffset", TRACE_STRUCT_ARRAY_TYPE_UINT64, TRACE_STRUCT_SHOW_MODE_DEC, 8); // 8 uint64
    g_traceStructItemArraySet(
        g_traceMsgInfoSt, "res", TRACE_STRUCT_ARRAY_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 2); // 1 uint8 2 2*uint8
}

HcclResult MC2TraceUtils::InitMsgInfoHandle()
{
    uint16_t msgInfoNum = GetMsgNum(sizeof(AivAicpuOpParam));
    TraceAttr attr = {};
    attr = {false, msgInfoNum, sizeof(AivAicpuOpParam), {nullptr}, 0, {0}};
    g_traceMsgInfoSt = g_traceStructEntryCreate("AivAicpuOpParam");
    MC2TraceUtils::SetTraceMsgInfo();
    g_traceStructSetAttr(g_traceMsgInfoSt, 0, &attr);
    g_traceMsgInfoHandle = g_traceCreateWithAttr(TRACER_TYPE_SCHEDULE, "AivAicpuOpParam", &attr);
    if (g_traceMsgInfoHandle < 0) {
        HCCL_ERROR("Create g_traceMsgInfoHandle failed, ret:%d", g_traceMsgInfoHandle);
        g_traceStructEntryDestroy(g_traceMsgInfoSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_traceEventBindTrace(g_eventHandle, g_traceMsgInfoHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceMsgInfoHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

void MC2TraceUtils::SetTraceSqeBatchInfo()
{
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "addr1High", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "addr1Low", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "addr2High", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "addr2Low", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_HEX, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "sqeHeadIdx", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "notifyId", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "length", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "partId", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "remoteRank", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "dataType", TRACE_STRUCT_FIELD_TYPE_UINT32, TRACE_STRUCT_SHOW_MODE_DEC, 4); // 4 uint32
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "streamId", TRACE_STRUCT_FIELD_TYPE_UINT16, TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "eventId", TRACE_STRUCT_FIELD_TYPE_UINT16, TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "taskId", TRACE_STRUCT_FIELD_TYPE_UINT16, TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "condValue", TRACE_STRUCT_FIELD_TYPE_UINT16, TRACE_STRUCT_SHOW_MODE_DEC, 2); // 2 uint16
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "isLast", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "opCode", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "sqeNum", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "type", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "subType", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemFieldSet(
        g_traceSqeBatchInfoSt, "valid", TRACE_STRUCT_FIELD_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC, 1); // 1 uint8
    g_traceStructItemArraySet(
        g_traceSqeBatchInfoSt, "reverse", TRACE_STRUCT_ARRAY_TYPE_UINT8, TRACE_STRUCT_SHOW_MODE_DEC,
        10); // 1 uint8 10 10*uint8
}

HcclResult MC2TraceUtils::InitSqeBatchInfoHandle()
{
    uint16_t sqeBatchInfoMsgNum = GetMsgNum(sizeof(SqeBatchInfo));
    TraceAttr attr = {};
    attr = {false, sqeBatchInfoMsgNum, sizeof(SqeBatchInfo), {nullptr}, 0, {0}};
    g_traceSqeBatchInfoSt = g_traceStructEntryCreate("SqeBatchInfo");
    for (uint32_t i = 0; i < MAX_SQE_BATCH_SIZE; i++) {
        SetTraceSqeBatchInfo();
    }
    g_traceStructSetAttr(g_traceSqeBatchInfoSt, 0, &attr);
    g_traceSqeBatchInfoHandle = g_traceCreateWithAttr(TRACER_TYPE_SCHEDULE, "SqeBatchInfo", &attr);
    if (g_traceSqeBatchInfoHandle < 0) {
        HCCL_ERROR("Create g_traceSqeBatchInfoHandle failed, ret:%d", g_traceSqeBatchInfoHandle);
        g_traceStructEntryDestroy(g_traceSqeBatchInfoSt);
        return HCCL_E_INTERNAL;
    }
    TraStatus ret = g_traceEventBindTrace(g_eventHandle, g_traceSqeBatchInfoHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("Bind g_traceSqeBatchInfoHandle failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitTraceHandle()
{
    g_eventHandle = g_traceEventCreate("mc2_dfx");
    if (g_eventHandle < 0) {
        HCCL_ERROR("Create mc2_dfx event handle failed");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(InitTaskAndTilingDataHandle());
    CHK_RET(InitAicpuComDataHandle());
    CHK_RET(InitTraceStrHandle());
    CHK_RET(InitMsgInfoHandle());
    CHK_RET(InitSqeBatchInfoHandle());
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::GetTraceFunc(const std::string& traceName)
{
    g_traceEventBindTrace = reinterpret_cast<TraceEventBindTraceFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "EventBindTrace").c_str()));
    CHK_PRT_RET(
        g_traceEventBindTrace == nullptr, HCCL_ERROR("Get g_traceEventBindTrace error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_traceEventReport
        = reinterpret_cast<TraceEventReportFunc>(mmDlsym(g_soHandle, std::string(traceName + "EventReport").c_str()));
    CHK_PRT_RET(
        g_traceEventReport == nullptr, HCCL_ERROR("Get g_traceEventReport error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceSetGlobalAttr = reinterpret_cast<TraceSetGlobalAttrFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "SetGlobalAttr").c_str()));
    CHK_PRT_RET(
        g_traceSetGlobalAttr == nullptr, HCCL_ERROR("Get g_traceSetGlobalAttr error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceStructEntryCreate = reinterpret_cast<TraceStructEntryCreateFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "StructEntryCreate").c_str()));
    CHK_PRT_RET(
        g_traceStructEntryCreate == nullptr, HCCL_ERROR("Get g_traceStructEntryCreate error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_traceStructItemFieldSet = reinterpret_cast<TraceStructItemFieldSetFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "StructItemFieldSet").c_str()));
    CHK_PRT_RET(
        g_traceStructItemFieldSet == nullptr, HCCL_ERROR("Get g_traceStructItemFieldSet error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_traceStructItemArraySet = reinterpret_cast<TraceStructItemArraySetFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "StructItemArraySet").c_str()));
    CHK_PRT_RET(
        g_traceStructItemArraySet == nullptr, HCCL_ERROR("Get g_traceStructItemArraySet error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_traceStructSetAttr = reinterpret_cast<TraceStructSetAttrFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "StructSetAttr").c_str()));
    CHK_PRT_RET(
        g_traceStructSetAttr == nullptr, HCCL_ERROR("Get g_traceStructSetAttr error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceStructEntryDestroy = reinterpret_cast<TraceStructEntryDestroyFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "StructEntryDestroy").c_str()));
    CHK_PRT_RET(
        g_traceStructEntryDestroy == nullptr, HCCL_ERROR("Get g_traceStructEntryDestroy error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::InitFuncHandle()
{
    std::string traceName = g_isAtrace ? "Atrace" : "Utrace";
    HCCL_INFO("Start mmDlsym %s funcHandle", traceName.c_str());
    g_traceCreateWithAttr = reinterpret_cast<TraceCreateWithAttrFunc>(
        mmDlsym(g_soHandle, std::string(traceName + "CreateWithAttr").c_str()));
    CHK_PRT_RET(
        g_traceCreateWithAttr == nullptr, HCCL_ERROR("Get g_traceCreateWithAttr error: %s", mmDlerror()),
        HCCL_E_SYSCALL);
    g_traceGetHandle
        = reinterpret_cast<TraceGetHandleFunc>(mmDlsym(g_soHandle, std::string(traceName + "GetHandle").c_str()));
    CHK_PRT_RET(g_traceGetHandle == nullptr, HCCL_ERROR("Get g_traceGetHandle error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceSubmit = reinterpret_cast<TraceSubmitFunc>(mmDlsym(g_soHandle, std::string(traceName + "Submit").c_str()));
    CHK_PRT_RET(g_traceSubmit == nullptr, HCCL_ERROR("Get g_traceSubmit error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceEventCreate
        = reinterpret_cast<TraceEventCreateFunc>(mmDlsym(g_soHandle, std::string(traceName + "EventCreate").c_str()));
    CHK_PRT_RET(
        g_traceEventCreate == nullptr, HCCL_ERROR("Get g_traceEventCreate error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceEventDestroy
        = reinterpret_cast<TraceEventDestroyFunc>(mmDlsym(g_soHandle, std::string(traceName + "EventDestroy").c_str()));
    CHK_PRT_RET(
        g_traceEventDestroy == nullptr, HCCL_ERROR("Get g_traceEventDestroy error: %s", mmDlerror()), HCCL_E_SYSCALL);
    g_traceHandleDestroy
        = reinterpret_cast<TraceHandleDestroyFunc>(mmDlsym(g_soHandle, std::string(traceName + "Destroy").c_str()));
    CHK_PRT_RET(
        g_traceHandleDestroy == nullptr, HCCL_ERROR("Get g_traceHandleDestroy error: %s", mmDlerror()), HCCL_E_SYSCALL);
    CHK_RET(GetTraceFunc(traceName));
    return HCCL_SUCCESS;
}

__attribute__((constructor)) void DlopenTraceSo()
{
    if (g_soHandle == nullptr) {
        g_soHandle = mmDlopen(ATRACE_SO, RTLD_NOW | RTLD_GLOBAL);
        if (g_soHandle == nullptr) {
            g_soHandle = mmDlopen(UTRACE_SO, RTLD_NOW | RTLD_GLOBAL);
            g_isAtrace = false;
        }
        HCCL_INFO("mmDlopen success");
    }
}

HcclResult MC2TraceUtils::Init()
{
    if (g_soHandle == nullptr) {
        HCCL_WARNING("Cannot find so file: %s", UTRACE_SO);
        return HCCL_SUCCESS;
    }
    CHK_RET(InitFuncHandle());
    CHK_RET(InitTraceHandle());
    unsigned int hostpid = 0;
    unsigned int cpType = DEVDRV_PROCESS_CPTYPE_MAX;
    CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), nullptr, nullptr, &hostpid, &cpType));

    TraceGlobalAttr traceGlobalAttr = {};
    traceGlobalAttr.saveMode = 1;
    traceGlobalAttr.deviceId = AicpuGetComContext()->devId;
    traceGlobalAttr.pid = hostpid;
    TraStatus ret = g_traceSetGlobalAttr(&traceGlobalAttr);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("TraceSetGlobalAttrFunc failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult MC2TraceUtils::Submit(const T* const traceData)
{
    if (g_soHandle == nullptr || g_traceGetHandle == nullptr || g_traceSubmit == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(traceData);
    std::string typeName;
    if (std::is_same<T, KFCtaskAndTilingTraceData>::value) {
        typeName = "KFCtaskAndTilingTraceData";
    } else if (std::is_same<T, AicpuComTraceData>::value) {
        typeName = "AicpuComTraceData";
    } else if (std::is_same<T, AivAicpuOpParam>::value) {
        typeName = "AivAicpuOpParam";
    } else if (std::is_same<T, SqeBatchInfo>::value) {
        typeName = "SqeBatchInfo";
    } else {
        HCCL_ERROR("find typename: %s failed", typeid(T).name());
        return HCCL_E_INTERNAL;
    }

    auto traceHandle = g_traceGetHandle(TRACER_TYPE_SCHEDULE, typeName.c_str());
    if (traceHandle < 0) {
        HCCL_ERROR("getHandle %s failed, ret:%d", typeName.c_str(), traceHandle);
        return HCCL_E_INTERNAL;
    }
    auto traceRet = g_traceSubmit(traceHandle, reinterpret_cast<const void*>(traceData), sizeof(T));
    if (traceRet != TRACE_SUCCESS) {
        HCCL_ERROR("submit %s failed, ret:%d", typeName.c_str(), traceRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Submit(AicpuComContext* ctx)
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(ctx);
    uint64_t startUsec = GetCurCpuTimestamp();
    AicpuComTraceData comTraceData;
    comTraceData.devId = ctx->devId;
    comTraceData.ssid = ctx->ssid;
    comTraceData.rankId = ctx->rankId;
    comTraceData.rankNum = ctx->rankNum;
    comTraceData.windowSize = ctx->windowSize;
    comTraceData.workSpaceAddr = ctx->workSpaceAddr;
    comTraceData.kfcNotifyId = ctx->kfcNotifyId;
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        comTraceData.eventIds[i] = ctx->eventIds[i];
        comTraceData.windowIn[i] = ctx->rankInfo[i].window;
        comTraceData.windowOut[i] = ctx->rankInfo[i].windowOut;
        comTraceData.actualStreamId[i] = ctx->streamInfo[i].actualStreamId;
        comTraceData.sqId[i] = ctx->streamInfo[i].sqId;
    }
    for (uint32_t i = 0; i < 2; i++) { // 2 aicpuOpNotify size
        comTraceData.aicpuOpNotifyAddress[i] = ctx->aicpuOpNotify[i].address;
        comTraceData.aicpuOpNotifyActualNotifyId[i] = ctx->aicpuOpNotify[i].actualNotifyId;
    }
    comTraceData.clusterId = ctx->clusterId;
    CHK_RET(Submit<AicpuComTraceData>(&comTraceData));
    if (AicpuKfcUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        AicpuKfcProf::GetProInst(*ctx).traceCtxTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Submit(const KFCTask* const task, const HcclKFCTilingData* const tilingData)
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(task);
    CHK_PTR_NULL(tilingData);
    uint64_t startUsec = GetCurCpuTimestamp();
    KFCtaskAndTilingTraceData taskAndTilingData;
    error_t ret = memcpy_s(&taskAndTilingData, sizeof(KFCtaskAndTilingTraceData), task, sizeof(KFCTask));
    if (ret != EOK) {
        HCCL_ERROR("task memcpy_s fail ret:%d", ret);
        return HCCL_E_MEMORY;
    }
    ret = memcpy_s(
        reinterpret_cast<uint8_t*>(&taskAndTilingData) + sizeof(KFCTask),
        sizeof(KFCtaskAndTilingTraceData) - sizeof(KFCTask), tilingData, sizeof(HcclKFCTilingData));
    if (ret != EOK) {
        HCCL_ERROR("tilingData memcpy_s fail ret:%d", ret);
        return HCCL_E_MEMORY;
    }
    CHK_RET(Submit<KFCtaskAndTilingTraceData>(&taskAndTilingData));
    auto ctx = AicpuGetComContext();
    if (AicpuKfcUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        AicpuKfcProf::GetProInst(*ctx).traceSubmitTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Submit(const std::string& traceStr)
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    return Submit(traceStr.c_str());
}

HcclResult MC2TraceUtils::Submit(const char* traceStr)
{
    if (g_soHandle == nullptr || g_traceSubmit == nullptr) {
        return HCCL_SUCCESS;
    }
    uint64_t startUsec = GetCurCpuTimestamp();
    uint32_t strSize = strlen(traceStr);
    uint32_t posSize = 0U;
    TraceStr singleTraceStr;
    while (posSize < strSize) {
        uint32_t traceStrLen = std::min<uint32_t>(strSize - posSize, sizeof(singleTraceStr.transmit) - 1);
        CHK_SAFETY_FUNC_RET(memset_s(
            reinterpret_cast<void*>(&singleTraceStr.transmit), sizeof(singleTraceStr.transmit), 0,
            sizeof(singleTraceStr.transmit)));
        error_t ret = memcpy_s(singleTraceStr.transmit, traceStrLen, traceStr + posSize, traceStrLen);
        if (ret != EOK) {
            HCCL_ERROR("traceStr memcpy_s fail ret:%d", ret);
            return HCCL_E_MEMORY;
        }
        posSize += traceStrLen;
        auto traceStrRet
            = g_traceSubmit(g_traceStrHandle, reinterpret_cast<const void*>(&singleTraceStr), sizeof(singleTraceStr));
        if (traceStrRet != TRACE_SUCCESS) {
            HCCL_ERROR("submittraceStr failed, ret:%d", traceStrRet);
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("Submit string: %s", singleTraceStr.transmit);
    }
    auto ctx = AicpuGetComContext();
    if (AicpuKfcUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        AicpuKfcProf::GetProInst(*ctx).traceSubmitTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::SubmitBatchSqeInfo()
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    uint64_t startUsec = GetCurCpuTimestamp();
    SqeContext* context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    for (uint32_t streamId = 0U; streamId < AC_MAX_RANK_NUM; streamId++) {
        auto& buff = context->buffPtr[streamId];
        HCCL_INFO(
            "sqTail: %u, sqHead: %u, sqeCnt: %u, tailSqeTaskId: %u, tailSqeIdx: %u", buff.sqTail, buff.sqHead,
            buff.sqeCnt, buff.tailSqeTaskId, buff.tailSqeIdx);
        auto submitNum = std::min<uint32_t>(buff.tailSqeIdx - buff.sqeCnt, MAX_SQE_SUBMIT_NUM);
        for (uint32_t i = 0U; i <= submitNum / MAX_SQE_BATCH_SIZE; i++) {
            SqeBatchInfo sqeBatchInfo;
            for (uint32_t j = 0U; j < MAX_SQE_BATCH_SIZE; j++) {
                const auto idx = i * MAX_SQE_BATCH_SIZE + j;
                if (idx >= submitNum) {
                    break;
                }
                if (AicpuSqeContext::QuerySqeInfoByTaskId(
                        streamId, buff.tailSqeTaskId - submitNum + idx, &sqeBatchInfo.sqeInfos[j])
                    != HCCL_SUCCESS) {
                    HCCL_WARNING("Query sqe info failed, stream:%u, id:%u", streamId, idx);
                    break;
                }
            }
            CHK_RET(Submit<SqeBatchInfo>(&sqeBatchInfo));
        }
    }
    auto ctx = AicpuGetComContext();
    if (AicpuKfcUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN)) {
        AicpuKfcProf::GetProInst(*ctx).traceSqeTime += GetCurCpuTimestamp() - startUsec;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::Save()
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    TraStatus ret = g_traceEventReport(g_eventHandle);
    if (ret != TRACE_SUCCESS) {
        HCCL_ERROR("TraceEventReport failed, ret:%d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2TraceUtils::DestoryHandles()
{
    if (g_soHandle == nullptr) {
        return HCCL_SUCCESS;
    }
    g_traceStructEntryDestroy(g_traceStrSt);
    g_traceStructEntryDestroy(g_traceAicpuComTraceSt);
    g_traceStructEntryDestroy(g_traceKFCtaskAndTilingTraceDataSt);
    g_traceStructEntryDestroy(g_traceMsgInfoSt);
    g_traceStructEntryDestroy(g_traceSqeBatchInfoSt);
    g_traceHandleDestroy(g_traceStrHandle);
    g_traceHandleDestroy(g_traceTaskAndTilingDataHandle);
    g_traceHandleDestroy(g_traceAicpuComDataHandle);
    g_traceHandleDestroy(g_traceMsgInfoHandle);
    g_traceHandleDestroy(g_traceSqeBatchInfoHandle);
    g_traceEventDestroy(g_eventHandle);

    return HCCL_SUCCESS;
}

__attribute__((destructor)) void Finalize()
{
    if (g_soHandle != nullptr) {
        (void)mmDlclose(g_soHandle);
    }
    g_soHandle = nullptr;
}

// 显式实例化
template HcclResult MC2TraceUtils::Submit(const SqeBatchInfo* const);
