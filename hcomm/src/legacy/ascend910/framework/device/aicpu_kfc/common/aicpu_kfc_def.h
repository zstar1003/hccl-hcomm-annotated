/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_KFC_DEF_H__
#define __AICPU_KFC_DEF_H__

#include <cstdint>
#include <chrono>
#include "common/aicpu_hccl_def.h"
#include "hccl_common.h"
#include "hccl_msg.h"
#include "aicpu_operator_pub.h"

constexpr int32_t AICPU_CNT = 8;
constexpr int32_t CLUSTER_CNT = 2;
constexpr u64 HCCL_COPY_ALIGN = 16 * 1024;
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;
constexpr u32 AC_DEFAULT_ONE_SHOT_SIZE = 100 * 1024; // 缺省小于100K使用oneshot算法
constexpr s32 AC_ERROR_INVALID_PARAM = 0x011088;
constexpr u32 AICPU_OP_NOTIFY_NUM = 2U;
constexpr u32 AC_SQE_SIZE = 64U;
constexpr u32 AC_DEFAULT_WINDOW_DIM = 2; // 小数据量时分2片做swap window使用，可以提前把下一次内容拷贝到window中
constexpr u32 AC_DEFAULT_RANK_GROUP = 2;
constexpr u32 HCCL_SMALL_COUNT_1_M = 1024 * 1024;
constexpr u32 HCCL_SMALL_COUNT_256K = 256 * 1024; // 256KB: 256 * 1024
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;        // 只计算不通信
constexpr uint8_t MC2_DEBUG_PRINT_MSG = 2;
constexpr uint8_t MC2_DEBUG_PRINT_BUFF = 3;
constexpr uint8_t MC2_DEBUG_TIME_TAKEN = 4; // KFC算子自己统计各阶段耗时
constexpr uint8_t MC2_DEBUG_WAIT_COMM = 8;  // KFC算子等待通信结束
constexpr uint8_t MC2_DEBUG_PREPARE_TIMEOUT = 250;
constexpr uint8_t MC2_DEBUG_COMMIT_TIMEOUT = 251;
constexpr uint8_t MC2_DEBUG_NOTIFY_WAIT_TIMEOUT = 252;
constexpr uint8_t MC2_DEBUG_AICORE_WAIT_TIMEOUT = 253;
constexpr uint8_t MC2_DEBUG_FINALIZE_TIMEOUT = 254;
constexpr uint8_t MC2_DEBUG_SDMA_ERROR = 255;
constexpr uint64_t MC2_API_MSG_TIMEOUT = 20UL;
constexpr uint32_t MC2_API_XORCHECK_PRINT_NUM = 10000;
const uint16_t TAIL_TASK = 1;
const uint16_t HEAD_TASK = 0;
constexpr int32_t MAX_COMM_CTX_NUM = 3;
constexpr int32_t LOCAL = 0;
constexpr int32_t REMOTE = 1;
constexpr uint16_t MAX_BATCH_WRITE_THREAD_NUM = 2;

constexpr uint8_t FLAG_OFFSET = 1;
constexpr uint8_t FLAG_INTERVAL = 2;
constexpr uint8_t POST_SEND_FLAG_COUNT = 3;
constexpr u64 MAX_RDMA_WQE_SIZE = 2ULL * 1024 * 1024 * 1024; // RDMA最大WQE限制是2GB

using HcclHandle = int8_t;

enum CommAlgType {
    COMM_ALG_DEFAULT = 0,
    COMM_ALG_FULL_MESH = 1,
    COMM_ALG_DOUBLE_RING = 2,
    COMM_ALG_SWITCH_WING = 3,
    COMM_ALG_RESERVED
};

enum MC2_BUFFER_TYPE {
    MC2_BUFFER_TYPE_DEFAULT = 0,
    MC2_BUFFER_TYPE_OUTPUT,
    MC2_BUFFER_TYPE_WINDOW_IN,
    MC2_BUFFER_TYPE_WINDOW_OUT,
    MC2_BUFFER_TYPE_WORKSPACE,
    MC2_BUFFER_TYPE_INPUT,
    MC2_BUFFER_TYPE_COMMOUT,
    MC2_BUFFER_TYPE_END
};

enum AicpuTilingVer {
    TILING_DATA_VER_OLD_FOR_HOST = 0,
    TILING_DATA_VER_OLD_FOR_KERNEL,
    TILING_DATA_VER_OLD_FOR_KERNEL_V2,
    TILING_DATA_VER_FOR_TILING_API = 100
};

struct KFCGroupTilingDataAuto { // for grouped_mat_mul_all_reduce op
    HcclKFCTilingData msg[64];  // 64: same as the tiling data size
    uint32_t groupNum;
    uint32_t groupTilingMagicNum;
};

struct KFCGroupTilingData { // for grouped_mat_mul_all_reduce op
    uint32_t groupNum;
    uint32_t reserve;
    HcclKFCTilingData msg[64]; // 64: same as the tiling data size
};

struct KFCTask {
    u64 inputA;     // A矩阵地址，通信在前时为sendbuffer
    u64 outputC;    // 输出C矩阵地址
    u64 commOut;    // 双输出时，通信输出地址
    u64 context;    // HCCL通信context
    u64 workSpace;  // 通信结果不直接输出时，放到workspace中
    u64 tilingData; // 通信
};

struct KFCTaskV2 {
    u64 inputA;  // A矩阵地址，通信在前时为sendbuffer
    u64 outputC; // 输出C矩阵地址
    u64 commOut; // 双输出时，通信输出地址
    u64 ctxNum;
    u64 context[MAX_COMM_CTX_NUM]; // HCCL通信context
    u64 workSpace;                 // 通信结果不直接输出时，放到workspace中
    u64 tilingData;                // 通信
};

struct KFCResInitTask {
    u64 context; // A矩阵地址，通信在前时为sendbuffer
    bool isCustom;
};

struct PostSendTaskParam {
    // For DataCopy
    u32 lKey;
    u32 rKey;
    HcclQpInfoV2 qpInfo;
    u64 remoteAddr;
    u64 localAddr;
    u64 dataSize;

    u64 timeOut;

    // For Flag
    u64 localFlagAddr;
    u64 remoteFlagAddr;
    u32 lfKey;
    u32 rfKey;
};

struct CommonHcclMsg {
    HcclCMDType commType; // 通信原语类型，AllReduce/AllGather.../Finalize/InterHcclGroupSync
    HcclReduceOp opType;  // reduce操作类型，sum/prod/max/min
    uint64_t sendBuffer;  // 源数据buffer地址。
    uint64_t recvBuffer;  // 目的数据buffer地址
    uint64_t dataCnt;     // 参与操作的数据个数
    uint64_t
        strideCount; // 完整的数据结果一般是连续的，切分多轮后会导致需要加上stride，例如AllGather的stride是每个卡上的完整数据量
    HcclDataType hcclDataType; // 参与操作的数据类型
    uint32_t p2pSrcDestRankId; // 点对点通信send/recv对端的rankId，send中的destRank, recv中的srcRank
    uint32_t valid;            // 检查消息有效性
    uint8_t repeatCnt;         // 本消息需要重复的次数，默认是1
    uint8_t everyTurnRsp;      // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait;     // 每轮都需要等待work消息再执行
    HcclHandle
        commDepGroupID; // 本消息执行需要等待的通信域组id，默认是-1，表示不需要等待，用于设置notify监听的通信域组id
    HcclHandle commDepHandleID; // 本消息执行需要等待的通信域轮次，默认是-1，表示不需要等待，用于设置notify监听的地址
    HcclHandle selfHandleID;    // 通信消息对应的handleId值
    uint8_t seqNum;             // 消息序号
    HcclApi::HcclTilingVersion version; // 消息的版本信息，version=0使用hcclMsg
    uint32_t xorCheck;                  // xor checksum
    uint64_t ccOpTilingData;            // 消息的tiling信息
    void PrintMsg(const std::string& desc)
    {
        HCCL_INFO(
            "%s Msg[version %u, commType %u, opType %u, sendBuffer %p, recvBuffer %p, dataCnt %lu, strideLen %lu,"
            " hcclDataType %u, p2pSrcDestRankId %u, valid %u"
            " repeatCnt %u, everyTurnRsp %u, everyTurnWait %u, commDepGroupID %d,"
            " commDepHandleID %d, selfHandleID %d, seqNum %u, ccOpTilingData %#llx]",
            desc.c_str(), static_cast<uint32_t>(version), static_cast<uint32_t>(commType),
            static_cast<uint32_t>(opType), sendBuffer, recvBuffer, dataCnt, strideCount,
            static_cast<uint32_t>(hcclDataType), p2pSrcDestRankId, valid, repeatCnt, everyTurnRsp, everyTurnWait,
            commDepGroupID, commDepHandleID, selfHandleID, seqNum, ccOpTilingData);
    }
};

struct WqeSendSharedContect {
    volatile u32 startedThreadNum = 0;
    volatile u32 workedThreadNum = 0;
    std::atomic<bool> taskFinishFlag{false};
    volatile u32 curThreadIdsOnCpu[AICPU_CNT];
    volatile u32 sendWqeNum[MAX_BATCH_WRITE_THREAD_NUM];
};

struct DataBlock {
    uint32_t data[16];
};

constexpr uint32_t MAX_AICPU_NUM_BLOCKS = 6U;

// HCCL 代码直调时直接传此结构：
struct AivAicpuOpParam {
    HcclCMDType commType; // 32b
    HcclReduceOp opType;  // 32b
    u64 sendBuffer;
    u64 recvBuffer;
    u64 count;
    u64 strideLen;

    // offset 32B
    HcclDataType hcclDataType;

    uint32_t valid;        // 检查消息有效性
    uint8_t isLast;        // 是否最后一个下
    uint8_t funID;         // 功能ID，1地址消息；  2开始工作
    uint8_t sendCnt;       // 发送计数
    uint8_t rcvCnt;        // 执行结束轮次技术
    uint8_t everyTurnRsp;  // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait; // 每轮都需要等待work消息再执行
    uint8_t totalTurnCnt;  // 总轮次
    uint8_t useBufferType;
    uint64_t winOffset; // 发送数据偏移地址

    HcclOpIdentifier opId;
    uint8_t res[2]; // 整体消息64字节
    void PrintMsg(const std::string& desc)
    {
        HCCL_INFO(
            "%s Msg[commType %u, opType %u, sendBuffer %p, recvBuffer %p, count %lu, strideLen %lu,"
            " hcclDataType %s, valid %u, isLast %u, funID %u, sendCnt %u, rcvCnt %u,"
            " everyTurnRsp %u, everyTurnWait %u, totalTurnCnt %u, winOffset %lu]",
            desc.c_str(), static_cast<uint32_t>(commType), static_cast<uint32_t>(opType), sendBuffer, recvBuffer, count,
            strideLen, GetDataTypeEnumStr(hcclDataType).c_str(), valid, isLast, funID, sendCnt, rcvCnt, everyTurnRsp,
            everyTurnWait, totalTurnCnt, winOffset);
    }
};

constexpr uint32_t MAX_DEBUG_CNT = 128U;
struct AicDebugCntInfo {
    uint8_t cnt[MAX_DEBUG_CNT];
}; // 128

using AicpuAddOneNotifyWaitSqe
    = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*, const dfx::DfxTimeOutConfig&);
using AicpuAddOneRecordSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*);
using AicpuAddOneWriteValueRecordSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t*, uint8_t*);
using AicpuAddOneMemcpySqe = void (*)(
    uint16_t, uint16_t, const void*, uint32_t, const aclDataType, aclrtReduceKind, const void*, uint32_t, uint32_t,
    uint32_t, u64, uint8_t, const uint8_t*, uint8_t*, uint32_t);
using AicpuAddOneEventResetSqe = void (*)(uint16_t, int32_t, uint16_t, int64_t, int64_t, u64, const uint8_t*, uint8_t*);
using AicpuAddOneEventRecordSqe = void (*)(uint16_t, int32_t, uint16_t, const uint8_t*, uint8_t*);
using AicpuAddOneEventWaitSqe = void (*)(uint16_t, int32_t, uint16_t, const uint8_t*, uint8_t*);
using AicpuAddOneRdmaDbSendSqe
    = void (*)(uint16_t, uint16_t, uint64_t, uint64_t, uint32_t, uint8_t, const uint8_t*, uint8_t*);
using AicpuAddOneFlipPlaceHolderSqe = void (*)(uint16_t, uint16_t, uint16_t, const uint8_t*, uint8_t*);

extern AicpuAddOneNotifyWaitSqe AicpuGetAddOneNotifyWaitSqe();
extern AicpuAddOneRecordSqe AicpuGetAddOneRecordSqe();
extern AicpuAddOneWriteValueRecordSqe AicpuGetAddOneWriteValueRecordSqe();
extern AicpuAddOneMemcpySqe AicpuGetAddOneMemcpySqe();
extern AicpuAddOneEventResetSqe AicpuGetAddOneEventResetSqe();
extern AicpuAddOneEventRecordSqe AicpuGetAddOneEventRecordSqe();
extern AicpuAddOneEventWaitSqe AicpuGetAddOneEventWaitSqe();
extern AicpuAddOneRdmaDbSendSqe AicpuGetAddOneRdmaDbSendSqe();
extern AicpuAddOneFlipPlaceHolderSqe AicpuGetAddOneFlipPlaceHolderSqe();

struct RestartParam {
    // 重执行标记，表示是否发生异常需要重执行
    bool restartFlag = false;
    // 重执行次数
    uint32_t restartCnt = 0;
    // 是否所有通信域都重执行协商完成
    uint32_t consultationAllEnd = 0;
    // 通信域重执行协商情况
    bool consultationResult[MAX_COMM_CTX_NUM] = {false, false, false};
    // 通信域是否执行过changeLink
    bool linkChanged[MAX_COMM_CTX_NUM] = {false, false, false};
    // 通信域重执行协商初始化state
    HcclOpExecFSM fsmState[MAX_COMM_CTX_NUM]
        = {HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END, HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END,
           HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END};
    KfcError errorCode[MAX_COMM_CTX_NUM] = {KfcError::kNone, KfcError::kNone, KfcError::kNone};
    std::chrono::time_point<std::chrono::steady_clock> startTime[MAX_COMM_CTX_NUM];
};

enum class BarrierStatus : u8 { NO_BARRIER = 0U, SELF_BARRIER, INTER_BARRIER };

struct BarrierInfo {
    BarrierStatus status;
    u64 lastTimeStamp;
};

enum class AicpuServerRole { MASTER = 0, SLAVE = 1, INVALID = 2 };

#ifndef CCL_LLT
#define ANONYMOUS_NAMESPACE_BEGIN namespace {
#define ANONYMOUS_NAMESPACE_END }
#else
#define ANONYMOUS_NAMESPACE_BEGIN
#define ANONYMOUS_NAMESPACE_END
#endif

#endif
