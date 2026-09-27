/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMON_DEFS_H
#define HCCL_COMMON_DEFS_H

#include <cstdint>
#include <cstring>
#include <vector>

#include "hccl_types.h"

#define RT_CCU_SQE_ARGS_LEN (13U)

enum class LinkProto {
    SDMA = 0,
    RDMA = 1,
    CCU = 2,
    INVALID_A = 3,
};

enum class HccLTaskMetaType : char {
    NOTIFY_WAIT,
    NOTIFY_RECORD,
    REDUCE,
    MEM_CPY,
    CCU_GRAPH,
    AIV_GRAPH,
    EVENT_WAIT,
    EVENT_RECORD
};

typedef enum {
    COMM_PROTOCOL_RESERVED = -1,
    COMM_PROTOCOL_HCCS = 0,
    COMM_PROTOCOL_TCP = 1,
    COMM_PROTOCOL_ROCE = 2,
    COMM_PROTOCOL_UB_CTP = 3,
    COMM_PROTOCOL_UB_TP = 4,
    COMM_PROTOCOL_PCIE = 5,
    COMM_PROTOCOL_SIO = 6,
} CommProtocol;

#pragma pack(push, 1)

typedef struct {
    uint32_t srcRankId;
    uint64_t srcOffset;
    uint32_t dstRankId;
    uint64_t dstOffset;
    uint64_t len;
    uint8_t protocol;
} TransMemTask;

typedef struct {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commId;
    uint64_t streamId;
    uint64_t inputAddr;
    uint64_t inputSize;
    uint64_t outputAddr;
    uint64_t outputSize;
    uint64_t cclAddr;
    uint64_t cclSize;
} OpStartTask;

typedef struct {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commId;
    uint64_t streamId;
} OpSyncTask;

typedef struct {
    uint32_t srcRankId;
    uint64_t srcOffset;
    uint32_t dstRankId;
    uint64_t dstOffset;
    uint64_t dataCount;
    uint8_t dataType;
    uint8_t reduceOp;
    uint8_t protocol;
    uint8_t resv;
} ReduceTask;

typedef struct {
    uint32_t srcRankId;
    uint64_t notifyId;
    uint32_t dstRankId;
    uint8_t notifyCount;
    uint8_t protocol;
} NotifyTask;

typedef struct {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t timeout;
    uint16_t instStartId;
    uint16_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[RT_CCU_SQE_ARGS_LEN];
} CcuTask;

typedef struct {
    uint64_t launchIdx;
} AivTask;

typedef struct HcclTaskMetaData {
    HccLTaskMetaType taskType;
    uint16_t commId;
    uint32_t rankId;
    uint64_t streamId;
    uint32_t jettyId;
    uint8_t rmEid[16];
    union {
        TransMemTask transMem;
        ReduceTask reduce;
        NotifyTask notify;
        CcuTask ccu;
        AivTask aiv;
        OpStartTask opStartTask;
        OpSyncTask opSyncTask;
    } taskData;
    HcclTaskMetaData()
    {
        jettyId = UINT32_MAX;
        std::memset(rmEid, 0, sizeof(rmEid));
    }
} HcclTaskMetaData;

typedef struct {
    uint64_t taskCid;
    uint16_t dispatchId;
} HcclTaskReq;

typedef struct {
    uint64_t taskCid;
    uint16_t status;
} HcclTaskRsp;

#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint64_t count;
    uint16_t dataType;
    uint32_t reduceOpType;
} OpInfoV1;

typedef struct {
    uint64_t sendCount;
    uint16_t sendDataType;
    uint64_t recvCount;
    uint16_t recvDataType;
    uint16_t extInfo;
} OpInfoV2;

typedef struct {
    uint16_t opType;
    uint16_t dataType;
    uint16_t reduceType;
    union {
        OpInfoV1 opV1;
        OpInfoV2 opV2;
    };
} OpDetails;
#pragma pack(pop)

#endif
