/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_LOGIC_INFO_PUB_H
#define HCCL_TASK_LOGIC_INFO_PUB_H

#include <vector>
#include <memory>
#include <hccl/base.h>
#include "mem_device_pub.h"
#include "adapter_rts_common.h"

namespace hccl {
enum class UserMemType { INPUT_MEM, OUTPUT_MEM, MEM_RESERVED };

struct TxMemoryInfo {
    UserMemType dstMemType;
    u64 dstOffset;
    const void* src;
    u64 len;
};

struct RxMemoryInfo {
    UserMemType srcMemType;
    u64 srcOffset;
    void* dst;
    u64 len;
};

struct RxWithReduceMemoryInfo {
    UserMemType recvSrcMemType;
    u64 recvSrcOffset;
    void* recvDst;
    u64 recvLen;
    void* reduceSrc;
    void* reduceDst;
    u64 reduceDataCount;
};

enum class TaskLogicType { TRANSPORT_TYPE, DISPATCHER_TYPE, TYPE_RESERVED };

enum class TaskLogicFuncType {
    TRANSPORT_TXACK_TYPE, /* transport task type */
    TRANSPORT_RXACK_TYPE,
    TRANSPORT_TXASYNC_TYPE,
    TRANSPORT_RXASYNC_TYPE,
    TRANSPORT_TXDATASIGNAL_TYPE,
    TRANSPORT_RXDATASIGNAL_TYPE,

    DISPATCHER_SIGNALWAIT_TYPE = 100, /* dispatcher task type */
    DISPATCHER_SIGNALRECORD_TYPE,
    DISPATCHER_MEMCPYASYNC_TYPE,
    TYPE_RESERVED
};

struct ParaTxAck {
    void* stream;
};

struct ParaRxAck {
    void* stream;
};

struct ParaTxAsync {
    std::vector<TxMemoryInfo> txMems;
};

struct ParaRxAsync {
    std::vector<RxMemoryInfo> rxMems;
};

struct ParaTxDataSignal {
    void* stream;
};

struct ParaRxDataSignal {
    void* stream;
};

struct ParaSignalWait {
    void* signal;
    u32 userRank;
    u32 remoteRank;
    s32 stage;
};

struct ParaSignalRecord {
    void* signal;
    u32 userRank;
    u64 offset;
    s32 stage;
};

struct ParaMemAsync {
    void* dst;
    uint64_t destMax;
    void* src;
    u64 count;
    HcclRtMemcpyKind kind;
};

struct TaskLogicCmdInfo {
    TaskLogicType taskLogicType; /* logic task 操作类型：0: transport, 1: dispatcher */
    u32 index;                   /* 对应vtransport、vdispatcher的index信息 */
};

struct TaskLogicInfo {
    TaskLogicCmdInfo taskLogicCmd;  /* logic task 操作类型 */
    TaskLogicFuncType taskFuncType; /* logic task 具体执行方法 */
    union {
        union {
            ParaTxAck txAck;
            ParaRxAck rxAck;
            ParaTxDataSignal txDataSignal;
            ParaRxDataSignal rxDataSignal;
        } transportTaskLogicPara;

        union {
            ParaSignalWait signalWait;
            ParaSignalRecord signalRecord;
            ParaMemAsync memAsync;
        } dispatcherTaskLogicPara;
    } taskLogicPara;
    ParaTxAsync txAsync;
    ParaRxAsync rxAsync;
    TaskLogicInfo() {};
    TaskLogicInfo(u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType);
    TaskLogicInfo(
        u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, std::vector<TxMemoryInfo>& txMems);
    TaskLogicInfo(
        u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, std::vector<RxMemoryInfo>& rxMems);
    TaskLogicInfo(
        u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, void* signal, u32 userRank,
        u32 remoteUserRank, s32 stage);
    TaskLogicInfo(
        u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, void* signal, u32 userRank, u64 offset,
        s32 stage);
    TaskLogicInfo(
        u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, void* dst, uint64_t destMax, void* src,
        u64 count, HcclRtMemcpyKind kind);
};
} // namespace hccl
#endif /* HCCL_TASK_LOGIC_INFO_PUB_H */
