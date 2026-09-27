/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_GROUP_UTILS_H
#define HCCL_GROUP_UTILS_H

#include <mutex>
#include <string>
#include <hccl/hccl_types.h>
#include <hccl_inner.h>
#include "op_base.h"
#include "hccl_comm_pub.h"

namespace hccl {

using hcclGroupJobState_t = enum hcclGroupJobState {
    hcclGroupJobRunning = 0,
    hcclGroupJobDone = 1,
    hcclGroupJobJoined = 2,
};

struct hcclAsyncJob {
    struct hcclAsyncJob* next;
    std::unique_ptr<std::thread> thread; /*记录该job异步执行pthread_create 创建的thread句柄*/
    HcclResult result;                   /* 用于记录job->func的执行结果，job->result = job->func(job) */
    HcclResult (*func)(struct hcclAsyncJob*);
    hcclGroupJobState state;
    std::mutex mtx;
    HcclComm* comm;
};

struct hcclCommInitAsyncJob : public hcclAsyncJob {
    u32 nRanks;
    const HcclRootInfo* rootInfo;
    u32 rank;
    s32 devId;
    HcclComm* initComm;
    std::string identifier;
};

struct hcclCommInitConfigAsyncJob : public hcclAsyncJob {
    u32 nRanks;
    const HcclRootInfo* rootInfo;
    u32 rank;
    s32 devId;
    HcclComm* initComm;
    std::string identifier;
    const HcclCommConfig* config;
};

struct hcclCommInitScalableAsyncJob : public hcclAsyncJob {
    u32 nRanks;
    u32 nRoot;
    const HcclRootInfo* rootInfoList;
    u32 rank;
    u32 nExtRoot;
    s32 devId;
    HcclComm* initComm;
    const HcclCommConfig* config;
};

struct hcclCommInitRankTableAsyncJob : public hcclAsyncJob {
    const char* clusterInfo;
    u32 rank;
    HcclComm* initComm;
    s32 devId;
};

struct hcclCommInitRankTableConfigAsyncJob : public hcclAsyncJob {
    const char* clusterInfo;
    u32 rank;
    HcclComm* initComm;
    HcclCommConfig* config;
    s32 devId;
};

struct hcclCommDestroyAsyncJob : public hcclAsyncJob {
    HcclComm initComm;
    s32 devId;
};

struct hcclOpInfo { /*用于保存算子的入参。所有算子用同个结构体保存info */
    HcclCMDType coll;
    void* sendbuff;
    void* recvbuff;
    u64 sendCount; // for non-V operators
    u64 recvCount; // for non-V operators
    const void* sendCounts;
    const void* recvCounts;
    const void* sdispls;
    const void* rdispls;
    HcclDataType sendType;
    HcclDataType recvType;
    HcclReduceOp op;
    u32 root; // dstRank或root rank
    HcclComm comm;
    HcclRtStream stream;
};

struct hcclTaskP2p {
    struct hcclTaskP2p* next;
    HcclCMDType func;
    void* buff;
    u64 count;
    HcclDataType datatype;
    s32 root;  /*即peer或dstRank*/
    u64 bytes; /*即count*dataTypeSize*/
    HcclComm comm;
    HcclRtStream stream;
};

struct hcclKernelPlanner {
    s32 nTasksColl = -1;
    s32 nTasksP2p = -1; // 该plan中Coll和P2p task的个数
    u32 rankSize = 0;

    std::set<HcclRtStream> collStreams;
    HcclRtStream sendRecvMainStream; // sendRecv的主流，用于跟从流同步

    std::vector<HcclSendRecvItem> sendRecvInfo;

    std::deque<struct hcclOpInfo> collTaskQueue;
};

} // namespace hccl
#endif
