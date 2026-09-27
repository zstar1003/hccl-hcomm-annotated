/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AIV_H
#define HCCL_AIV_H

#include <vector>
#include "string"

#include "hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_common.h"
#include "common.h"
#include "mem_device_pub.h"
#include "alg_profiling.h"

namespace hccl {
constexpr u64 ATTR_POS_AIV_COMM_BUFFER = 0x00;
constexpr u64 ATTR_POS_AIV_COMM_INFO_BUFFER = 0x01;
constexpr u64 AIV_COMM_BUFFER_BITMASK = 0x01;
constexpr u64 AIV_COMM_INFO_BUFFER_BITMASK = 0x02;

constexpr u64 AIV_ALL_REDUCE_BIG_SIZE = 16 * 1024 * 1024;
constexpr u64 AIV_ALL_REDUCE_A3_ENTRY_SIZE = 1 * 1024 * 1024; // AllReduce单张卡数据量A3
constexpr u64 AIV_ALL_REDUCE_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_DETER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_BIG_SIZE = 190 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_MID_SIZE = 2 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_ALL_GATHER_BIG_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_ALL_TO_ALL_BIG_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_TO_ALL_A3_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_BIG_SIZE = 256 * 1024 * 1024;
constexpr u64 AIV_ALL_REDUCE_DETER_SIZE = 1 * 1024 * 1024; // AllReduce确定性计算
constexpr u32 AIV_REDUCE_SCATTER_RANK_SIZE_8 = 8;

constexpr u64 AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE = 190 * 1024;
constexpr u64 AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr u64 AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr u64 AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE = 760 * 1024;

constexpr u64 AIV_REDUCE_SCATTER_A3_SMALL_RANKSIZE_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_MID_RANKSIZE_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_LARGE_RANKSIZE_ENTRY_SIZE = 128 * 1024;

constexpr u64 AIV_ALL_GATHER_A3_SMALL_RANKSIZE_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_MID_RANKSIZE_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_LARGE_RANKSIZE_ENTRY_SIZE = 32 * 1024;

constexpr u64 AIV_A3_CROSSNODE_TINY_SIZE = 28 * 1024;
constexpr u64 AIV_A3_CROSSNODE_SMALL_SIZE = 112 * 1024;
constexpr u64 AIV_A3_CROSSNODE_MID_SIZE = 448 * 1024;

constexpr u32 MAX_RANK_SIZE = 16;      // server内最大卡数
constexpr u32 MAX_RANK_SIZE_A3 = 768;  // 超节点内最大卡数
constexpr u32 MAX_RANK_SIZE_RDMA = 64; // 跨机支持的最大卡数

constexpr u32 NUM_BLOCKS_FACTOR_TWO = 2;
constexpr u32 NUM_BLOCKS_FACTOR_THREE = 3;
constexpr u32 NUM_BLOCKS_FACTOR_FOUR = 4;
constexpr u32 NUM_BLOCKS_FACTOR_SIX = 6;
constexpr u32 NUM_BLOCKS_FACTOR_EIGHT = 8;
constexpr u32 NUM_BLOCKS_THREE_PER_RANK_A3 = 3;
constexpr u32 NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr u32 MAX_NUM_BLOCKS = 48;
constexpr u32 HALF_MAX_NUM_BLOCKS = 24;
constexpr u32 ONE_THIRD_MAX_NUM_BLOCKS = 16;
constexpr u32 ONE_FOURTH_MAX_NUM_BLOCKS = 12;
constexpr u32 ONE_SIXTH_MAX_NUM_BLOCKS = 8;
constexpr u32 ONE_EIGHTH_MAX_NUM_BLOCKS = 6;

constexpr s32 TAG_INIT_VALUE = 1;
constexpr s32 TAG_RESET_COUNT = 1000;
constexpr s32 AIV_A2_ALL_REDUCE_RDMA_KERNEL_NUM = 2;

constexpr u32 TIME_S_TO_US = 1000000;
constexpr u32 AIV_TIMEOUT_DEFAULT = 1091;
constexpr u32 AIV_TIMEOUT_DEFAULT_US = 1091 * TIME_S_TO_US;
constexpr u32 AIV_TIMEOUT_MAX = 1091;
constexpr u32 AIV_TIMEOUT_MAX_US = 1091 * TIME_S_TO_US;

constexpr u32 DEV_TYPE_910_93 = 4;

constexpr u32 BUFFER_DIVIDE = 2;
constexpr u32 MAX_TARGET_NUM = 20;

enum class KernelArgsType {
    ARGS_TYPE_SERVER = 0,   // kernel参数为单机内
    ARGS_TYPE_SUPERPOD = 1, // kernel参数包含多机，当前仅A3 AlltoAllV跨机场景
    ARGS_TYPE_SIMPLE = 2,   // kernel参数为A3跨机
    ARGS_TYPE_DEFAULT
};

// AIV直驱Roce所需的rmaInfo信息
// Transport 内存类型
enum class HcclAiRMAMemType : u32 {
    LOCAL_INPUT = 0,
    REMOTE_INPUT,

    LOCAL_OUTPUT,
    REMOTE_OUTPUT,

    // 可透传更多的内存，可在MAX_NUM之前追加，例如：
    // LOCAL_EXP,
    // REMOTE_EXP,
    MAX_NUM
};

constexpr u32 GetAiMemTypeVal(HcclAiRMAMemType value) { return static_cast<u32>(value); }

constexpr u32 AiMemMaxNum = GetAiMemTypeVal(HcclAiRMAMemType::MAX_NUM);

// Transport 内存信息
struct HcclAiRMAMemInfo {
    uint32_t memMaxNum{0};        // 最大内存数量，等于 HcclAiRMAMemType::MAX_NUM
    uint32_t sizeOfMemDetails{0}; // sizeof(MemDetails)，用于内存校验和偏移计算
    uint64_t memDetailPtr{0};     // MemDetails数组首地址, 个数: HcclAiRMAMemType::MAX_NUM
    // 可往后追加字段
};

// 全部 Transport QP/Mem 信息
struct HcclRMAInfo {
    uint32_t curRankId{0}; // 当前rankId
    uint32_t rankNum{0};   // rank数量
    uint32_t qpNum{0};     // 单个Transport的QP数量

    uint32_t sizeOfRMAWQ{0};  // sizeof(HcclAiRMAWQ)
    uint32_t sizeOfRMACQ{0};  // sizeof(HcclAiRMACQ)
    uint32_t sizeOfRMAMem{0}; // sizeof(HcclAiRMAMemInfo)

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SQ指针：sqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t sqPtr{0};

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SCQ指针：scqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t scqPtr{0};

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RQ指针：rqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t rqPtr{0};

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RCQ指针: rcqPtr + (dstRankId * qpNum + qpIndex) * sizeOfRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t rcqPtr{0};

    // HcclAivMemInfo一维数组
    // 内存信息个数: rankNum
    // 计算偏移获取内存信息指针: memPtr + rankId * sizeOfRMAMem
    // srcRankId 获取自身内存信息，dstRankId 获取 Transport 内存信息
    uint64_t memPtr{0};
    // 可往后追加字段
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A2场景
using ExtraArgs = struct AlltoAllExtraArgs {
    u64 sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    u64 sendCounts[MAX_RANK_SIZE] = {};
    u64 sendDispls[MAX_RANK_SIZE] = {};
    u64 recvCounts[MAX_RANK_SIZE] = {};
    u64 recvDispls[MAX_RANK_SIZE] = {};
    u64 maxCount = 0;
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A3场景
struct ExtraArgsV2 {
    u64 sendCounts[MAX_RANK_SIZE_A3] = {};
    u64 sendDispls[MAX_RANK_SIZE_A3] = {};
    u64 recvCounts[MAX_RANK_SIZE_A3] = {};
    u64 recvDispls[MAX_RANK_SIZE_A3] = {};
};

// 表示算子属性的参数，相对固定
struct AivOpArgs {
    HcclCMDType cmdType;
    const void* input;
    const void* output;
    u64 count;
    HcclDataType dataType;
    HcclReduceOp op;
    u32 root;
    bool isOpBase;
};

// 表示拓扑信息的参数
struct AivTopoArgs {
    u32 rank;
    u32 rankSize;
    u32 devId;
    u32 serverId;
    u32 serverNum;
    DevType devType;
    std::string identify;

    AivTopoArgs(
        u32 rank, u32 rankSize, u32 devId = MAX_RANK_SIZE, u32 serverId = 0, u32 serverNum = 1,
        DevType devType = DevType::DEV_TYPE_910B, std::string identify = "INVALID_COMM")
        : rank(rank),
          rankSize(rankSize),
          devId(devId),
          serverId(serverId),
          serverNum(serverNum),
          devType(devType),
          identify(identify)
    {}
};

// 表示AIV所需要的资源参数
struct AivResourceArgs {
    std::string commTag;
    rtStream_t stream;
    void** buffersIn;  // 注册的CCLIN地址，所有卡可访问
    void** buffersOut; // 注册的CCLOUT地址，所有卡可访问
    u64 bufferSize;
    u32 numBlocks;
    s32 aivTag;
};

// 表示AIV算法流程控制的参数
struct AivAlgArgs {
    s32 step;
    bool isSmallCount;
    u32 deterministic;
    KernelArgsType argsType;
    s32 execTimeOut;
    bool execTimeOutSet; // true表示set by commConfig
    bool isNpuDirectRoce;
    u64 rmaInfo;

    explicit AivAlgArgs(
        s32 step = -1, bool isSmallCount = false, u32 deterministic = 0,
        KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER,
        s32 execTimeOut = static_cast<s32>(AIV_TIMEOUT_DEFAULT), bool execTimeOutSet = false,
        bool isNpuDirectRoce = false, u64 rmaInfo = 0)
        : step(step),
          isSmallCount(isSmallCount),
          deterministic(deterministic),
          argsType(argsType),
          execTimeOut(execTimeOut),
          execTimeOutSet(execTimeOutSet),
          isNpuDirectRoce(isNpuDirectRoce),
          rmaInfo(rmaInfo)
    {}
};

// 表示AIVProfiling所需要的参数
struct AivProfilingInfo {
    uint64_t beginTime = 0;
    OpCounterInfo counter;
};

struct HcclCacheInfo {
    bool isUseCache = false;
    AivOpArgs opArgs;
    AivTopoArgs topoArgs{0, 0};
    AivResourceArgs resourceArgs{"", nullptr, nullptr, nullptr, 0, 0, 0};
    AivAlgArgs algArgs;
    AivProfilingInfo profilingInfo;
    ExtraArgs extraArgs;
    void* buffersIn[MAX_RANK_SIZE] = {};  // 注册的CCLIN地址，所有卡可访问
    void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    AlgType algType;
    bool selectAivAlg = false;
    std::string newTag;
};

// 表示AIVSuperKernel所需要的参数
using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    void* buffersIn[MAX_RANK_SIZE] = {};  // 注册的CCLIN地址，所有卡可访问
    void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    u64 rank;
    u64 rankSize;
    u64 len;
    u64 dataType;
    u64 unitSize;
    u64 reduceOp;
    u64 numBlocks;
    s32 tag; // 第几次调用，定时重置成1
    s64 clearEnable;
    u32 devType;

    AivSuperKernelArgsDef(
        void** buffIn, void** buffOut, u32 rank, u32 rankSize, u64 len, u32 dataType, u32 unitSize, u32 reduceOp,
        u32 numBlocks = 0, s32 tag = 0, bool clearEnable = true, u32 devType = DEV_TYPE_910_93)
        : rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          unitSize(unitSize),
          reduceOp(reduceOp),
          numBlocks(numBlocks),
          tag(tag),
          clearEnable(clearEnable),
          devType(devType)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8*)buffIn[i];
            buffersOut[i] = (u8*)buffOut[i];
        }
    }
    AivSuperKernelArgsDef() {}
};

#ifdef OPEN_HCCL_TEST
enum class KernelLaunchMode {
    LAUNCH_MODE_ARGS_BASE = 0, // Launch模式，基础参数
    LAUNCH_MODE_ARGS_EXTRA,    // Launch模式，基础参数+ExtraArgs
    LAUNCH_MODE_ARGS_EXTRA_V2, // Launch模式，基础参数+ExtraArgsV2
    LAUNCH_MODE_ARGS_EXTRA_A3  // Launch模式，A3跨机
};

HcclResult ExecuteKernelLaunchImpl(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, AivProfilingInfo& aivProfilingInfo, KernelLaunchMode launchMode,
    void* extraArgsPtr = nullptr);
#endif

HcclResult RegisterKernel(DevType deviceType);
HcclResult UnRegisterAivKernel();

HcclResult
ClearAivSyncBufForMulServer(const AivResourceArgs& resourceArgs, const AivTopoArgs& topoArgs, void* args, u32 argsSize);

inline s32 GetNextAivTag(s32 curTag, s32 tagIncre = 1) { return (curTag + tagIncre - 1) % TAG_RESET_COUNT + 1; }

HcclResult ExecuteKernelLaunchInner(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, void* args, u32 argsSize, AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const ExtraArgs& extraArgs, AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const ExtraArgsV2& extraArgs, AivProfilingInfo& aivProfilingInfo);

HcclResult CacheTaskOpInfo(rtStream_t stream, const std::string& identify);

HcclResult ReadBinFile(const std::string& fileName, std::string& buffer);

HcclResult GetKernelFunc(aclrtFuncHandle& funcHandle, s8* stubFunc);

void SetAivProfilingInfoBeginTime(AivProfilingInfo& aivProfilingInfo);
void SetAivProfilingInfoBeginTime(uint64_t& beginTime);
} // namespace hccl

#endif // HCCL_AIV_H
