/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_TRACE_UTILS_H__
#define __MC2_TRACE_UTILS_H__

#include "common/aicpu_hccl_def.h"
#include "common/aicpu_kfc_def.h"
#include "common/aicpu_sqe_context.h"

constexpr uint32_t MAX_SQE_BATCH_SIZE = 16U;

struct TraceStr {
    char transmit[112] = {0}; // 112 = 128 - 16, 16为trace预留的字节大小
};

struct KFCtaskAndTilingTraceData {
    KFCTask singleKFCTask;
    HcclKFCTilingData singleKFCTilingData;
};

struct AicpuComTraceData {
    uint32_t devId;
    uint32_t ssid;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t windowSize;
    uint64_t workSpaceAddr;
    uint64_t kfcNotifyId;
    uint32_t eventIds[32];                  // 32最大rank数
    uint64_t windowIn[32];                  // 32最大rank数
    uint64_t windowOut[32];                 // 32最大rank数
    int32_t actualStreamId[32];             // 32最大rank数
    int32_t sqId[32];                       // 32最大rank数
    uint64_t aicpuOpNotifyAddress[2];       // 集合通信AICPU展开资源
    int32_t aicpuOpNotifyActualNotifyId[2]; // 集合通信AICPU展开资源
    int32_t clusterId;
};

struct SqeBatchInfo {
    SqeInfo sqeInfos[MAX_SQE_BATCH_SIZE];
};

class MC2TraceUtils {
public:
    static HcclResult Init();
    template <typename T>
    static HcclResult Submit(const T* const traceData);
    static HcclResult Submit(AicpuComContext* ctx);
    static HcclResult Submit(const KFCTask* const task, const HcclKFCTilingData* const tilingData);
    static HcclResult Submit(const std::string& traceStr);
    static HcclResult Submit(const char* traceStr);
    static HcclResult SubmitBatchSqeInfo();
    static HcclResult Save();
    static HcclResult DestoryHandles();

private:
    static uint16_t GetMsgNum(size_t msgSize);
    static HcclResult InitTraceStrHandle();
    static HcclResult InitTaskAndTilingDataHandle();
    static HcclResult InitAicpuComDataHandle();
    static HcclResult InitMsgInfoHandle();
    static HcclResult InitSqeBatchInfoHandle();
    static HcclResult InitTraceHandle();
    static HcclResult InitFuncHandle();
    static void SetHcclKFCTilingDataOne();
    static void SetHcclKFCTilingDataTwo();
    static void SetTraceMsgInfo();
    static void SetTraceSqeBatchInfo();
    static HcclResult GetTraceFunc(const std::string& traceName);
};
#endif // __MC2_TRACE_UTILS_H__
