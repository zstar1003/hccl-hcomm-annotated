/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COMMUNICATION_BASE_H
#define AIV_COMMUNICATION_BASE_H

#include "kernel_operator.h"
#include "sync_interface.h"
#include "aiv_npu_direct_base.h"

using namespace AscendC;

#define EXPORT_AIV_META_INFO(kernel_name)                               \
    static const struct FunLevelKType kernel_name##_kernel_type_section \
        __attribute__((used, section(".ascend.meta." #kernel_name)))    \
        = {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

// AIV支持的Atomic数据类型
#define AIV_ATOMIC_DATA_TYPE_DEF(func) \
    func(float);                       \
    func(half);                        \
    func(int16_t);                     \
    func(int32_t);                     \
    func(int8_t);                      \
    func(bfloat16_t)

// AIV支持的DataCopy数据类型
#define AIV_COPY_DATA_TYPE_DEF(func) \
    func(half);                      \
    func(int16_t);                   \
    func(uint16_t);                  \
    func(float);                     \
    func(int32_t);                   \
    func(uint32_t);                  \
    func(int8_t);                    \
    func(uint8_t);                   \
    func(bfloat16_t);                \
    func(int64_t);                   \
    func(uint64_t);                  \
    func(double)

constexpr uint32_t ALIGNAS_LEN = 4;         // 对齐长度
constexpr uint32_t MAX_RANK_SIZE = 16;      // server内最大卡数
constexpr uint32_t MAX_RANK_SIZE_A3 = 768;  // 超节点内最大卡数
constexpr uint32_t MAX_TARGET_NUM = 20;     // 最大轮数
constexpr uint32_t MAX_RANK_SIZE_RDMA = 64; // A2跨机支持最大卡数

struct ExtraArgs {
    uint64_t sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    uint64_t sendCounts[MAX_RANK_SIZE] = {};
    uint64_t sendDispls[MAX_RANK_SIZE] = {};
    uint64_t recvCounts[MAX_RANK_SIZE] = {};
    uint64_t recvDispls[MAX_RANK_SIZE] = {};
    uint64_t maxCount = 0;
};

struct ExtraArgsV2 {
    uint64_t sendCounts[MAX_RANK_SIZE_A3] = {};
    uint64_t sendDispls[MAX_RANK_SIZE_A3] = {};
    uint64_t recvCounts[MAX_RANK_SIZE_A3] = {};
    uint64_t recvDispls[MAX_RANK_SIZE_A3] = {};
};

using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    GM_ADDR buffersIn[MAX_RANK_SIZE] = {};  // 注册的CCLIN地址，所有卡可访问
    GM_ADDR buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    uint64_t rank;
    uint64_t rankSize;
    uint64_t len;
    uint64_t dataType;
    uint64_t unitSize;
    uint64_t reduceOp;
    int64_t numBlocks;
    int32_t tag; // 第几次调用，定时重置成1
    int64_t clearEnable;
    uint32_t devType;
};

using AivRdmaArgs = struct AivRdmaArgsDef {
    GM_ADDR buffers[MAX_RANK_SIZE_RDMA * 2] = {}; // 注册的CCL地址，所有卡可访问
};

enum class AivNotifyType { ACK, DataSignal, Done, ClearACK, ClearDataSingal };

enum class CommPattern {
    // server间
    interRank,
    // server内
    intraRank
};

#define AIV_INFO(format, ...)                   \
    do {                                        \
        AscendC::PRINTF(format, ##__VA_ARGS__); \
    } while (0)

#define AIV_INFO_HINT AIV_INFO("Aiv log dump is enabled in %s\n", __func__)

#define KERNEL_ARGS_DEF                                                                                             \
    GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4, GM_ADDR buffIn5,           \
        GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11,     \
        GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1, \
        GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7, \
        GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12,                \
        GM_ADDR buffOut13, GM_ADDR buffOut14, GM_ADDR buffOut15, GM_ADDR input, GM_ADDR output, uint32_t rank,      \
        uint32_t rankSize, uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root, int32_t tag,          \
        uint32_t numBlocks, bool isOpBase, uint64_t bufferSize, int32_t aivRdmaStep, bool useAivRdmaSmall,          \
        int32_t serverNum, uint32_t devType, GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem,         \
        uint32_t counterMemSize, bool isEnableCounter, uint32_t deterministic, uint64_t rmaInfo

#define KERNEL_ARGS_CALL                                                                                               \
    buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, buffIn8, buffIn9, buffIn10, buffIn11,      \
        buffIn12, buffIn13, buffIn14, buffIn15, buffOut0, buffOut1, buffOut2, buffOut3, buffOut4, buffOut5, buffOut6,  \
        buffOut7, buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14, buffOut15, input, output, \
        rank, rankSize, len, dataType, reduceOp, root, tag, numBlocks, isOpBase, bufferSize, aivRdmaStep,              \
        useAivRdmaSmall, serverNum, devType, headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter,   \
        deterministic, rmaInfo

#define KERNEL_CLASS_INIT                                                                                             \
    buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, buffIn8, buffIn9, buffIn10, buffIn11,     \
        buffIn12, buffIn13, buffIn14, buffIn15, buffOut0, buffOut1, buffOut2, buffOut3, buffOut4, buffOut5, buffOut6, \
        buffOut7, buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14, buffOut15, rank,         \
        rankSize, dataType, reduceOp, root, tag, numBlocks, headCountMem, tailCountMem, addOneMem, counterMemSize,    \
        isEnableCounter

#define EXTERN_KERNEL_ARGS_DEF KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define EXTERN_KERNEL_ARGS_DEF_V2 KERNEL_ARGS_DEF, ExtraArgsV2 extraArgs

#define EXTERN_KERNEL_ARGS_CALL KERNEL_ARGS_CALL, extraArgs

#define SUPERKERNEL_LITE_ARGS_DEF uint64_t args_offset

#define SUPERKERNEL_LITE_ARGS_EXTRACT                                                            \
    GM_ADDR* param_base = (GM_ADDR*)get_para_base();                                             \
    GM_ADDR hiddenInput = param_base[args_offset++];                                             \
    GM_ADDR input = param_base[args_offset++];                                                   \
    GM_ADDR output = param_base[args_offset++];                                                  \
    __gm__ AivSuperKernelArgs* args = reinterpret_cast<__gm__ AivSuperKernelArgs*>(hiddenInput); \
    uint32_t devType = args->devType

#define SUPERKERNEL_ARGS_DEF GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output

#define SUPERKERNEL_ARGS_CALL hiddenInput, input, output

#define SUPERKERNEL_CLASS_INIT hiddenInput

constexpr uint64_t AIV_FLAG_BUFFER_SIZE = 3 * 1024 * 1024;    // aiv算子的flag区域大小
constexpr uint64_t INFO_EVEN_BUFFER_OFFSET = 3 * 1024 * 1024; // aiv算子偶数tag存放AIV_INFO区域偏移
constexpr uint64_t INFO_ODD_BUFFER_OFFSET = 4 * 1024 * 1024;  // aiv算子奇数tag存放AIV_INFO区域偏移
constexpr uint64_t LOG_LEVEL_OFFSET = 2 * 1024 * 1024;        // 存放AIV_INFO日志环境变量的偏移
constexpr uint64_t CLEAR_BUFFER_OFFSET = 1024 * 1024;         // 用于清空的aiv buffer的偏移
constexpr uint64_t SYNC_BUFFER_OFFSET = 2 * 1024 * 1024;      // 用于sync的aiv buffer的偏移
constexpr uint64_t BUFFER_AREA = 1024 * 1024;                 // aiv算子的单独功能flag区域大小
constexpr uint64_t GM_TMP_ARGS_OFFSET = 64 * 1024;

constexpr uint64_t AIV_ALL_REDUCE_BIG_SIZE = 16 * 1024 * 1024;
constexpr uint64_t AIV_ALL_REDUCE_SMALL_SIZE = 64 * 1024;
constexpr uint64_t AIV_INIT_OFFSET = 0;
constexpr uint64_t AIV_PING_PONG_SIZE = 16 * 1024 * 1024;
constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;

constexpr uint64_t AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_MID_SIZE = 2 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr uint64_t AIV_ALL_TO_ALL_BIG_SIZE = 512 * 1024;
constexpr uint32_t AIV_REDUCE_SCATTER_RANK_SIZE_8 = 8;

constexpr uint64_t AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE = 190 * 1024;
constexpr uint64_t AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_ALL_REDUCE_DETER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr uint64_t AIV_ALL_REDUCE_DETER_MID_SIZE = 8 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_DETER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_DETER_MID_SIZE = 8 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_BIG_SIZE = 190 * 1024;
constexpr uint32_t AIV_A3_CROSSNODE_TINY_SIZE = 28 * 1024;
constexpr uint32_t AIV_A3_CROSSNODE_SMALL_SIZE = 112 * 1024;
constexpr uint32_t AIV_A3_CROSSNODE_MID_SIZE = 448 * 1024;
constexpr uint32_t NUM_BLOCKS_THREE_PER_RANK_A3 = 3;
constexpr uint32_t NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_NUM_BLOCKS = 48;

constexpr uint32_t TAG_MOVE_LEFT_BITS = 15;

constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_7 = UB_FLAG_SIZE * 7;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t FLAG_SIZE = 32;
constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_INTERVAL = FLAG_SIZE * 2;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint64_t FLAG_FIVE_OFFSET = FLAG_SIZE * 4;
constexpr uint64_t FLAG_SIX_OFFSET = FLAG_SIZE * 5;
constexpr uint64_t FLAG_SEVEN_OFFSET = FLAG_SIZE * 6;
constexpr uint32_t HALF_MAX_NUM_BLOCKS = 24;
constexpr uint32_t ONE_THIRD_MAX_NUM_BLOCKS = 16;
constexpr uint32_t ONE_FOURTH_MAX_NUM_BLOCKS = 12;
constexpr uint32_t ONE_SIXTH_MAX_NUM_BLOCKS = 8;
constexpr uint32_t ONE_EIGHTH_MAX_NUM_BLOCKS = 6;
constexpr uint64_t DETERMINISTIC_RANKSIZE = 4;

constexpr uint64_t IDX_0 = 0;
constexpr uint64_t IDX_1 = 1;
constexpr uint64_t IDX_2 = 2;
constexpr uint64_t IDX_3 = 3;
constexpr uint64_t IDX_4 = 4;
constexpr uint64_t IDX_5 = 5;
constexpr uint64_t IDX_6 = 6;
constexpr uint64_t IDX_7 = 7;
constexpr uint64_t IDX_8 = 8;
constexpr uint64_t IDX_9 = 9;
constexpr uint64_t IDX_10 = 10;
constexpr uint64_t IDX_11 = 11;
constexpr uint64_t IDX_12 = 12;
constexpr uint64_t IDX_13 = 13;
constexpr uint64_t IDX_14 = 14;
constexpr uint64_t IDX_15 = 15;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;

// 当前每个kernel最多使用4组同步标记，这里预留6组
constexpr uint32_t MAX_FLAG_SIZE_PER_KERNEL = 6 * MAX_RANK_SIZE * FLAG_SIZE;

// 将__COUNTER__改为固定偏移，新执行器需添加新偏移
#define AIV_ALL_REDUCE_DETER_910B_SMALLDATA 0
#define AIV_REDUCE_SCATTER_DETER_910B_SMALLDATA 1
#define AIV_ALL_REDUCE_DETER_910B_MIDDATA 2
#define AIV_REDUCE_SCATTER_DETER_910B_MIDDATA 3
#define AIV_ALL_REDUCE_DETER_910B_BIGDATA 4
#define AIV_REDUCE_SCATTER_DETER_910B_BIGDATA 5
#define AIV_ALL_TO_ALL_910B_DIRECT_FULLMESH 6

#define BASE_FLAG_OFFSET (MAX_FLAG_SIZE_PER_KERNEL)

#define DEV_TYPE_910B 2
#define DEV_TYPE_910_93 4

// __sk__函数参数 A2
struct SkArgsStruct {
    GM_ADDR buffIn0;
    GM_ADDR buffIn1;
    GM_ADDR buffIn2;
    GM_ADDR buffIn3;
    GM_ADDR buffIn4;
    GM_ADDR buffIn5;
    GM_ADDR buffIn6;
    GM_ADDR buffIn7;
    GM_ADDR buffIn8;
    GM_ADDR buffIn9;
    GM_ADDR buffIn10;
    GM_ADDR buffIn11;
    GM_ADDR buffIn12;
    GM_ADDR buffIn13;
    GM_ADDR buffIn14;
    GM_ADDR buffIn15;
    GM_ADDR buffOut0;
    GM_ADDR buffOut1;
    GM_ADDR buffOut2;
    GM_ADDR buffOut3;
    GM_ADDR buffOut4;
    GM_ADDR buffOut5;
    GM_ADDR buffOut6;
    GM_ADDR buffOut7;
    GM_ADDR buffOut8;
    GM_ADDR buffOut9;
    GM_ADDR buffOut10;
    GM_ADDR buffOut11;
    GM_ADDR buffOut12;
    GM_ADDR buffOut13;
    GM_ADDR buffOut14;
    GM_ADDR buffOut15;
    GM_ADDR input;
    GM_ADDR output;
    uint32_t rank;
    uint32_t rankSize;
    uint64_t len;
    uint32_t dataType;
    uint32_t reduceOp;
    uint32_t root;
    int32_t tag;
    uint32_t numBlocks;
    alignas(ALIGNAS_LEN) bool isOpBase;
    uint64_t bufferSize;
    int32_t aivRdmaStep;
    alignas(ALIGNAS_LEN) bool useAivRdmaSmall;
    int32_t serverNum;
    uint32_t devType;
    GM_ADDR headCountMem;
    GM_ADDR tailCountMem;
    GM_ADDR addOneMem;
    uint32_t counterMemSize;
    alignas(ALIGNAS_LEN) bool isEnableCounter;
    uint32_t deterministic;
    uint64_t rmaInfo;
};

// __sk__定义的函数参数 A2
#define SK_BIND_FUNC_ARGS __gm__ struct SkArgsStruct* args

// 将__sk__参数转成__aicore__参数 A2
#define CONVERT_SK_PARAM_TO_KERNEL_ARGS_A2          \
    GM_ADDR buffIn0 = args->buffIn0;                \
    GM_ADDR buffIn1 = args->buffIn1;                \
    GM_ADDR buffIn2 = args->buffIn2;                \
    GM_ADDR buffIn3 = args->buffIn3;                \
    GM_ADDR buffIn4 = args->buffIn4;                \
    GM_ADDR buffIn5 = args->buffIn5;                \
    GM_ADDR buffIn6 = args->buffIn6;                \
    GM_ADDR buffIn7 = args->buffIn7;                \
    GM_ADDR buffIn8 = args->buffIn8;                \
    GM_ADDR buffIn9 = args->buffIn9;                \
    GM_ADDR buffIn10 = args->buffIn10;              \
    GM_ADDR buffIn11 = args->buffIn11;              \
    GM_ADDR buffIn12 = args->buffIn12;              \
    GM_ADDR buffIn13 = args->buffIn13;              \
    GM_ADDR buffIn14 = args->buffIn14;              \
    GM_ADDR buffIn15 = args->buffIn15;              \
    GM_ADDR buffOut0 = args->buffOut0;              \
    GM_ADDR buffOut1 = args->buffOut1;              \
    GM_ADDR buffOut2 = args->buffOut2;              \
    GM_ADDR buffOut3 = args->buffOut3;              \
    GM_ADDR buffOut4 = args->buffOut4;              \
    GM_ADDR buffOut5 = args->buffOut5;              \
    GM_ADDR buffOut6 = args->buffOut6;              \
    GM_ADDR buffOut7 = args->buffOut7;              \
    GM_ADDR buffOut8 = args->buffOut8;              \
    GM_ADDR buffOut9 = args->buffOut9;              \
    GM_ADDR buffOut10 = args->buffOut10;            \
    GM_ADDR buffOut11 = args->buffOut11;            \
    GM_ADDR buffOut12 = args->buffOut12;            \
    GM_ADDR buffOut13 = args->buffOut13;            \
    GM_ADDR buffOut14 = args->buffOut14;            \
    GM_ADDR buffOut15 = args->buffOut15;            \
    GM_ADDR input = args->input;                    \
    GM_ADDR output = args->output;                  \
    uint32_t rank = args->rank;                     \
    uint32_t rankSize = args->rankSize;             \
    uint64_t len = args->len;                       \
    uint32_t dataType = args->dataType;             \
    uint32_t reduceOp = args->reduceOp;             \
    uint32_t root = args->root;                     \
    int32_t tag = args->tag;                        \
    uint32_t numBlocks = args->numBlocks;           \
    bool isOpBase = args->isOpBase;                 \
    uint64_t bufferSize = args->bufferSize;         \
    int32_t aivRdmaStep = args->aivRdmaStep;        \
    bool useAivRdmaSmall = args->useAivRdmaSmall;   \
    int32_t serverNum = args->serverNum;            \
    uint32_t devType = args->devType;               \
    GM_ADDR headCountMem = args->headCountMem;      \
    GM_ADDR tailCountMem = args->tailCountMem;      \
    GM_ADDR addOneMem = args->addOneMem;            \
    uint32_t counterMemSize = args->counterMemSize; \
    bool isEnableCounter = args->isEnableCounter;   \
    uint32_t deterministic = args->deterministic;   \
    uint64_t rmaInfo = args->rmaInfo

// sk 绑定函数 A2
#define SuperKernelBindA2(kernel_name)                         \
    extern "C" __sk__ void kernel_name##_1(SK_BIND_FUNC_ARGS); \
    extern "C" __sk__ void kernel_name##_2(SK_BIND_FUNC_ARGS); \
    extern "C" __sk__ void kernel_name##_3(SK_BIND_FUNC_ARGS); \
    extern "C" __sk__ void kernel_name##_4(SK_BIND_FUNC_ARGS); \
    SK_BIND(kernel_name, 8, kernel_name##_1, kernel_name##_2, kernel_name##_3, kernel_name##_4)

// A2 sk 导出函数
#define _SK_BIND_FUNC_DEF_A2(kernel_name, postfix)                    \
    extern "C" __sk__ void kernel_name##_##postfix(SK_BIND_FUNC_ARGS) \
    {                                                                 \
        CONVERT_SK_PARAM_TO_KERNEL_ARGS_A2;                           \
        kernel_name##_inner(KERNEL_ARGS_CALL);                        \
    }
#define SK_BIND_FUNC_DEF_A2(kernel_name, postfix) _SK_BIND_FUNC_DEF_A2(kernel_name, postfix)

// A2 Global 导出函数
#define GLOBAL_FUNC_DEF_A2(kernel_name)                                                                           \
    extern "C" __global__ __aicore__ void kernel_name(KERNEL_ARGS_DEF) { kernel_name##_inner(KERNEL_ARGS_CALL); } \
    EXPORT_AIV_META_INFO(kernel_name)

class AivCommBase {
public:
    __aicore__ inline AivCommBase() {}

    __aicore__ inline void Init(
        GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4, GM_ADDR buffIn5,
        GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11,
        GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
        GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7,
        GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
        GM_ADDR buffOut14, GM_ADDR buffOut15, uint32_t rank, uint32_t rankSize, uint32_t dataType, uint32_t reduceOp,
        uint32_t root, int32_t tag, int32_t numBlocks, GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem,
        uint32_t counterMemSize, bool isEnableCounter, bool useDoubleBuffer)
    {
        InitBuffArray(
            buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, buffIn8, buffIn9, buffIn10,
            buffIn11, buffIn12, buffIn13, buffIn14, buffIn15, buffOut0, buffOut1, buffOut2, buffOut3, buffOut4,
            buffOut5, buffOut6, buffOut7, buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14,
            buffOut15);

        rank_ = rank;
        rankSize_ = rankSize;
        reduceOp_ = reduceOp;
        tag_ = tag;

        useDoubleBuffer_ = useDoubleBuffer;
        numBlocks_ = numBlocks;

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        separateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;
        logLevel_ = GetLogLevel();
        uint64_t offset = (logLevel_ == 1) ? (tag_ & 1 ? INFO_EVEN_BUFFER_OFFSET : INFO_ODD_BUFFER_OFFSET) :
                                             INFO_EVEN_BUFFER_OFFSET;

        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_4);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);

        pipe.InitBuffer(
            flagBatchSetQue, 1, UB_FLAG_SIZE_8); // 最多支持同时set8个flag值，256B可存放32个u64，最多2组16rank
        pipe.InitBuffer(flagBatchCheckQue, 1, UB_FLAG_SIZE_8); // 最多支持同时check8个flag值

        if (useDoubleBuffer) {
            pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE); // double buffer
        } else {
            pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }

        pipe.InitBuffer(flagInQue, AIV_PING_PONG_FACTOR_TWO, UB_FLAG_SIZE);
        InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
        if (tag_ == 1) {
            ClearSyncBuf();
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void Init(GM_ADDR hiddenInput, uint64_t threshold, bool useDoubleBuffer = false)
    {
        __gm__ AivSuperKernelArgs* args = reinterpret_cast<__gm__ AivSuperKernelArgs*>(hiddenInput);

        for (int32_t i = 0; i < MAX_RANK_SIZE; i++) {
            GM_IN[i] = args->buffersIn[i];
            GM_OUT[i] = args->buffersOut[i];
        }
        rank_ = args->rank;
        rankSize_ = args->rankSize;
        reduceOp_ = args->reduceOp;
        len_ = args->len;
        tag_ = args->tag;
        dataType_ = args->dataType;
        unitSize_ = args->unitSize;
        numBlocks_ = args->numBlocks;

        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_4);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        separateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        useDoubleBuffer_ = useDoubleBuffer;
        if ((args->devType == DEV_TYPE_910_93) && (len_ * (unitSize_) > threshold)) {
            useDoubleBuffer_ = true;
        }
        if (useDoubleBuffer_) {
            pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE);
        } else {
            pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }

        if (args->tag == 1) {
            ClearSyncBuf();
        }
    }

    __aicore__ inline void InitBuffArray(
        GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4, GM_ADDR buffIn5,
        GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11,
        GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
        GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7,
        GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
        GM_ADDR buffOut14, GM_ADDR buffOut15)
    {
        GM_IN[IDX_0] = buffIn0;
        GM_IN[IDX_1] = buffIn1;
        GM_IN[IDX_2] = buffIn2;
        GM_IN[IDX_3] = buffIn3;
        GM_IN[IDX_4] = buffIn4;
        GM_IN[IDX_5] = buffIn5;
        GM_IN[IDX_6] = buffIn6;
        GM_IN[IDX_7] = buffIn7;
        GM_IN[IDX_8] = buffIn8;
        GM_IN[IDX_9] = buffIn9;
        GM_IN[IDX_10] = buffIn10;
        GM_IN[IDX_11] = buffIn11;
        GM_IN[IDX_12] = buffIn12;
        GM_IN[IDX_13] = buffIn13;
        GM_IN[IDX_14] = buffIn14;
        GM_IN[IDX_15] = buffIn15;

        GM_OUT[IDX_0] = buffOut0;
        GM_OUT[IDX_1] = buffOut1;
        GM_OUT[IDX_2] = buffOut2;
        GM_OUT[IDX_3] = buffOut3;
        GM_OUT[IDX_4] = buffOut4;
        GM_OUT[IDX_5] = buffOut5;
        GM_OUT[IDX_6] = buffOut6;
        GM_OUT[IDX_7] = buffOut7;
        GM_OUT[IDX_8] = buffOut8;
        GM_OUT[IDX_9] = buffOut9;
        GM_OUT[IDX_10] = buffOut10;
        GM_OUT[IDX_11] = buffOut11;
        GM_OUT[IDX_12] = buffOut12;
        GM_OUT[IDX_13] = buffOut13;
        GM_OUT[IDX_14] = buffOut14;
        GM_OUT[IDX_15] = buffOut15;
    }

    __aicore__ inline void InitForRDMA(
        GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4, GM_ADDR buffIn5,
        GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11,
        GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
        GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7,
        GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
        GM_ADDR buffOut14, GM_ADDR buffOut15, uint32_t rank, uint32_t rankSize, uint32_t dataType, uint32_t reduceOp,
        uint32_t root, int32_t tag, int32_t numBlocks, GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem,
        uint32_t counterMemSize, bool isEnableCounter, bool useDoubleBuffer)
    {
        rank_ = rank;
        rankSize_ = rankSize;
        reduceOp_ = reduceOp;
        useDoubleBuffer_ = useDoubleBuffer;
        numBlocks_ = numBlocks;
        tag_ = tag;

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        separateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_7);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);
        ubLocal = localFlagBuf.GetWithOffset<uint64_t>(UB_FLAG_PAD_COUNT, FLAG_FIVE_OFFSET);
        ubLocalHead = localFlagBuf.GetWithOffset<uint32_t>(UB_FLAG_PAD_COUNT, FLAG_SIX_OFFSET);
        bufferArgsTensor = localFlagBuf.GetWithOffset<uint64_t>(UB_FLAG_PAD_COUNT, FLAG_SEVEN_OFFSET);

        pipe.InitBuffer(
            flagBatchSetQue, 1, UB_FLAG_SIZE_8); // 最多支持同时set8个flag值，256B可存放32个u64，最多2组16rank
        pipe.InitBuffer(flagBatchCheckQue, 1, UB_FLAG_SIZE_8); // 最多支持同时check8个flag值

        if (useDoubleBuffer) {
            pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE); // double buffer
        } else {
            pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }

        pipe.InitBuffer(flagInQue, AIV_PING_PONG_FACTOR_TWO, UB_FLAG_SIZE);
        InitBuffArrayForRMDA(buffOut1);
        InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
    }

    __aicore__ inline void InitBuffArrayForRMDA(GM_ADDR commInfoAddr)
    {
        __gm__ AivRdmaArgs* args = reinterpret_cast<__gm__ AivRdmaArgs*>(commInfoAddr);
        for (uint32_t i = 0; i < MAX_RANK_SIZE_RDMA; i++) {
            GM_IN_RDMA[i] = args->buffers[2 * i];
            GM_OUT_RDMA[i] = args->buffers[2 * i + 1];
        }
    }

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
    {}

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    __aicore__ inline uint64_t
    CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength);

    __aicore__ inline void CalBlockCountAndOffset(
        uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup, uint32_t padCount, uint64_t& count,
        uint64_t& blockOffset);

    template <typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    template <typename T>
    __aicore__ inline void
    DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount);

    template <typename T>
    __aicore__ inline void
    DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount);

    template <typename T>
    __aicore__ inline void
    CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, bool atomic = false, uint32_t atomicOp = 0);

    template <typename T>
    __aicore__ inline void CpGM2GMWithFlagWrap(
        __gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, int32_t rank, uint64_t flushFrequency = 8,
        int32_t tag = 0);

    template <typename T>
    __aicore__ inline void CpGM2GMWithFlagWrap(
        __gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, __gm__ int32_t* ctrlFlagGM, uint64_t flushFrequency = 8,
        int32_t tag = 0);

    __aicore__ inline void
    Record(uint32_t tag, int32_t waitRank, AivNotifyType notifyType, int32_t blockGroup = 0, bool ifPingpong = false);

    __aicore__ inline void
    LocalRecord(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifPingpong = false);

    __aicore__ inline void Record1vN(
        uint32_t tag, CommPattern pattern, AivNotifyType notifyType = AivNotifyType::ACK, int32_t block = 0,
        bool ifPingpong = false);

    __aicore__ inline void RecordNv1(
        uint32_t tag, int32_t waitRank, AivNotifyType notifyType = AivNotifyType::ACK, int32_t block = 0,
        bool ifPingpong = false);

    __aicore__ inline void CountRecord(uint32_t tag, int64_t count, int32_t waitRank);

    __aicore__ inline void
    Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType, int32_t blockGroup = 0, bool ifpingpong = false);

    __aicore__ inline void LocalWait(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifpingpong = false);

    __aicore__ inline void WaitNv1(
        uint32_t tag, int32_t recordRank, AivNotifyType notifyType = AivNotifyType::ACK, int32_t block = 0,
        bool ifPingpong = false);

    __aicore__ inline void Wait1vN(
        uint32_t tag, CommPattern pattern, bool ifClear = true, AivNotifyType notifyType = AivNotifyType::ACK,
        int32_t block = 0, bool ifPingpong = false);

    __aicore__ inline int32_t CountWait(int32_t recordRank, int32_t index);

    __aicore__ inline void AIVRDMAPostSend(
        GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclRMAInfo* QpInfo,
        bool isLocalOutput, bool isRemoteOutput);

    __aicore__ inline void Barrier(uint32_t step);

    __aicore__ inline void ClearFlag();

    __aicore__ inline void BlockSync();

    __aicore__ inline void ClearSyncBuf();

    __aicore__ inline int32_t GetLogLevel();

    __aicore__ inline void InitOpCounter(
        GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter)
    {
        headCountMem_ = headCountMem;
        tailCountMem_ = tailCountMem;
        addOneMem_ = addOneMem;
        counterMemSize_ = counterMemSize;
        isEnableCounter_ = isEnableCounter;
    }

    __aicore__ inline void HeadCounter()
    {
        if (blockIdx_ == 0 && isEnableCounter_) {
            CpGM2GM(
                (__gm__ int32_t*)headCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t), true,
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }

    __aicore__ inline void TailCounter()
    {
        if (blockIdx_ == 0 && isEnableCounter_) {
            CpGM2GM(
                (__gm__ int32_t*)tailCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t), true,
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }
    // protected:
    GM_ADDR GM_IN[MAX_RANK_SIZE];
    GM_ADDR GM_OUT[MAX_RANK_SIZE];
    GM_ADDR GM_IN_RDMA[MAX_RANK_SIZE_RDMA];
    GM_ADDR GM_OUT_RDMA[MAX_RANK_SIZE_RDMA];

    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t reduceOp_;
    uint32_t dataType_;
    uint32_t unitSize_;

    uint64_t len_;
    int32_t tag_;
    int32_t numBlocks_;
    int32_t logLevel_;
    uint32_t blockIdx_ = GetBlockIdx(); // 在构造函数中初始化，以免漏初始化

    bool useDoubleBuffer_;

    TPipe pipe;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localCheckGETensor;
    LocalTensor<int32_t> localGetTensor;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;
    LocalTensor<uint64_t> bufferArgsTensor;

    TQue<QuePosition::VECOUT, 1> flagBatchSetQue;
    TQue<QuePosition::VECIN, 1> flagBatchCheckQue;

    TQue<QuePosition::VECIN, 1> flagInQue;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;

    GM_ADDR headCountMem_;
    GM_ADDR tailCountMem_;
    GM_ADDR addOneMem_;
    uint32_t counterMemSize_;
    bool isEnableCounter_;

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
    uint32_t separateOffset;
};

__aicore__ inline void AivCommBase::Barrier(uint32_t step)
{
    // 用10个flag
    uint32_t flagOffset = 2 * 1024 * 1024 + (step % 2) * FLAG_SIZE * rankSize_;
    __gm__ int32_t* ctrlFlagsGM;
    if (blockIdx_ == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[targetRank] + flagOffset + rank_ * FLAG_SIZE);
            SetSignalValue(ctrlFlagsGM, localSetTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE);
            WaitSignalValue(ctrlFlagsGM, localCheckTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE);
            SetSignalValue(ctrlFlagsGM, localSetTensor, 0);
        }
    }
}

__aicore__ inline void AivCommBase::ClearFlag()
{
    // 用10个flag
    __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_]);
    __gm__ int32_t* emtpyGM = (__gm__ int32_t*)(GM_OUT[rank_] + CLEAR_BUFFER_OFFSET);
    if (blockIdx_ == 0) {
        CpGM2GM(ctrlFlagsGM, emtpyGM, BUFFER_AREA / sizeof(int32_t));
    }
}

__aicore__ inline void AivCommBase::BlockSync()
{
    uint32_t flagOffset = SYNC_BUFFER_OFFSET + 2 * FLAG_SIZE * numBlocks_;
    __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset);
    if (blockIdx_ == 0) {
        // 通知其他核
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < numBlocks_; i++) {
            SetSignalValue(ctrlFlagsGM + i * FLAG_SIZE, localSetTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
    } else {
        // 接收通知并清零
        WaitSignalValue(ctrlFlagsGM + blockIdx_ * FLAG_SIZE, localCheckTensor, 1);
        SetSignalValue(ctrlFlagsGM + blockIdx_ * FLAG_SIZE, localSetTensor, 0);
        pipe_barrier(PIPE_ALL);
    }
}

__aicore__ inline void AivCommBase::ClearSyncBuf()
{
    // 用10个flag
    Barrier(1);
    ClearFlag();
    Barrier(DOUBLE);
    BlockSync();
}

__aicore__ inline int32_t AivCommBase::GetLogLevel()
{
#ifndef OPEN_HCCL_TEST
    int32_t tmpLogLevel = *((__gm__ int32_t*)(GM_OUT[rank_] + LOG_LEVEL_OFFSET - sizeof(int32_t)));
    return tmpLogLevel;
#else
    return 0;
#endif
}

__aicore__ inline uint64_t AivCommBase::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t
AivCommBase::CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength)
{
    if (sliceIdx == sliceCount - 1) {
        return tailLength;
    } else if (sliceIdx < sliceCount - 1) {
        return avgLengthPerSlice;
    } else {
        return 0;
    }
}

__aicore__ inline void AivCommBase::CalBlockCountAndOffset(
    uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup, uint32_t padCount, uint64_t& count,
    uint64_t& blockOffset)
{
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    blockOffset = blockIdxInGroup * avgLengthPerSlice;
}

template <typename T>
__aicore__ inline void AivCommBase::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
            SetAtomicAdd<T>();
            break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
            SetAtomicMax<T>();
            break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
            SetAtomicMin<T>();
            break;
        default:
            SetAtomicNone();
            break;
    }
}

template <typename T>
__aicore__ inline void
AivCommBase::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, calCount);
    } else {
        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void
AivCommBase::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

template <typename T>
__aicore__ inline void
AivCommBase::CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, bool atomic, uint32_t atomicOp)
{
    AIV_INFO("[CpGM2GM]outputGM is [%p], inputGM is [%p], count is [%llu]\n", outputGM, inputGM, count);
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    if (atomic) {
        SetAtomicOp<T>(atomicOp);
    }

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }

    if (atomic) {
        SetAtomicNone();
    }
    return;
}

template <typename T>
__aicore__ inline void AivCommBase::CpGM2GMWithFlagWrap(
    __gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, int32_t index, uint64_t flushFrequency, int32_t tag)
{
    AIV_INFO(
        "[AivCommBase::CpGM2GMWithFlagWrap][CpGM2GMWithFlagWrap]outputGM is [%p], inputGM is [%p], count is [%llu], "
        "index is [%d], flushFrequency is [%llu], tag is [%d]\n",
        outputGM, inputGM, count, index, flushFrequency, tag_);
    uint64_t curBatchCount = 0;

    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;

        curBatchCount += 1;

        if (curBatchCount % flushFrequency == 0 || count == 0) {
            SyncFunc<HardEvent::MTE3_S>();
            CountRecord(tag, curBatchCount, index);
        }
    }
}

template <typename T>
__aicore__ inline void AivCommBase::CpGM2GMWithFlagWrap(
    __gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, __gm__ int32_t* ctrlFlagGM, uint64_t flushFrequency,
    int32_t tag)
{
    AIV_INFO(
        "[AivCommBase::CpGM2GMWithFlagWrap][CpGM2GMWithFlagWrap]outputGM is [%p], inputGM is [%p], count is [%llu], "
        "ctrlFlagGM is [%p], flushFrequency is [%llu], tag is [%d]\n",
        outputGM, inputGM, count, ctrlFlagGM, flushFrequency, tag_);
    uint64_t curBatchCount = 0;

    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;

        curBatchCount += 1;

        if (curBatchCount % flushFrequency == 0 || count == 0) {
            SyncFunc<HardEvent::MTE3_S>();
            SetSignalValue(ctrlFlagGM, localSetTensor, curBatchCount + tag);
        }
    }
}

__aicore__ inline void
AivCommBase::Record(uint32_t tag, int32_t waitRank, AivNotifyType notifyType, int32_t blockGroup, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset
        = (blockGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + rank_) * FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[waitRank] + OffSet + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void
AivCommBase::LocalRecord(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = localOffset + (int32_t(notifyType) * MAX_NUM_BLOCKS + blockIdx) * FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[rank_] + OffSet + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void
AivCommBase::Record1vN(uint32_t tag, CommPattern pattern, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = multiOffset
                           + (int32_t(pattern) * 2 * NUM_BLOCKS_FOUR_PER_RANK_A3
                              + int32_t(notifyType) * NUM_BLOCKS_FOUR_PER_RANK_A3 + block)
                                 * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[rank_] + OffSet + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void
AivCommBase::RecordNv1(uint32_t tag, int32_t waitRank, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = multiOffset + 2 * 2 * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE
                           + (int32_t(waitRank == rank_) * NUM_BLOCKS_FOUR_PER_RANK_A3 * 2
                              + int32_t(notifyType) * NUM_BLOCKS_FOUR_PER_RANK_A3 + block)
                                 * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[waitRank] + OffSet + recordOffset);
    AddSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCommBase::CountRecord(uint32_t tag, int64_t count, int32_t index)
{
    int32_t OffSet = countOffset;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[rank_] + OffSet + index * FLAG_SIZE);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag + count);
}

__aicore__ inline void
AivCommBase::Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType, int32_t blockGroup, bool ifpingpong)
{
    int32_t OffSet = ifpingpong ? pingpongOffset : 0;
    int32_t waitOffset
        = (blockGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + recordRank) * FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[rank_] + OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void AivCommBase::LocalWait(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifpingpong)
{
    int32_t OffSet = ifpingpong ? pingpongOffset : 0;
    int32_t waitOffset = localOffset + (int32_t(notifyType) * MAX_NUM_BLOCKS + blockIdx) * FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[rank_] + OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void
AivCommBase::WaitNv1(uint32_t tag, int32_t recordRank, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t waitOffset = multiOffset
                         + (int32_t(recordRank == rank_) * NUM_BLOCKS_FOUR_PER_RANK_A3 * 2
                            + int32_t(notifyType) * NUM_BLOCKS_FOUR_PER_RANK_A3 + block)
                               * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[recordRank] + OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}
// 是否直接清零
__aicore__ inline void AivCommBase::Wait1vN(
    uint32_t tag, CommPattern pattern, bool ifClear, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t waitOffset = multiOffset + 2 * 2 * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE
                         + (int32_t(pattern) * NUM_BLOCKS_FOUR_PER_RANK_A3 * 2
                            + int32_t(notifyType) * NUM_BLOCKS_FOUR_PER_RANK_A3 + block)
                               * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[rank_] + OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();
    if (ifClear) {
        SetSignalValue(ctrlFlagGM, localSetTensor, 0);
    }
}

__aicore__ inline int32_t AivCommBase::CountWait(int32_t recordRank, int32_t index)
{
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[recordRank] + countOffset + index * FLAG_SIZE);
    LocalTensor<int32_t> flag = flagInQue.AllocTensor<int32_t>();
    int32_t flagValue = GetSignalValue(ctrlFlagGM, flag);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    flagInQue.FreeTensor(flag);
    return flagValue;
}

__aicore__ inline void AivCommBase::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclRMAInfo* QpInfo,
    bool isLocalOutput, bool isRemoteOutput)
{
    auto qpNum = ((__gm__ HcclRMAInfo*)QpInfo)->qpNum;
    auto qp_ctx_entry
        = (__gm__ HcclAiRMAWQ*)(((__gm__ HcclRMAInfo*)QpInfo)->sqPtr
                                + destRankId * qpNum * (uint64_t)(((__gm__ HcclRMAInfo*)QpInfo)->sizeOfRMAWQ));
    auto mem_info_table = ((__gm__ HcclRMAInfo*)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclRMAInfo*)QpInfo)->sizeOfRMAMem;
    auto cur_rank_id = (((__gm__ HcclRMAInfo*)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t*)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t*)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    uint64_t shift = 15U;
    auto QP_DEPTH = qp_ctx_entry->depth;
    PipeBarrier<PIPE_ALL>();

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while (1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                     // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U); // [7] owner_bit
    byte_4 |= 1U << 8U;                       // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t*)(wqeAddr) = byte_4;         // Control set by local parameter see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen; // message size
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0;          // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U; // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0;         // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    HcclAiRMAMemType remoteType = isRemoteOutput ? HcclAiRMAMemType::REMOTE_OUTPUT : HcclAiRMAMemType::REMOTE_INPUT;
    *(__gm__ uint32_t*)(wqeAddr + 20)
        = ((__gm__ MemDetails*)(memDetail->memDetailPtr
                                + memDetail->sizeOfMemDetails * static_cast<uint32_t>(remoteType)))
              ->key;
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr; // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    HcclAiRMAMemType localType = isLocalOutput ? HcclAiRMAMemType::LOCAL_OUTPUT : HcclAiRMAMemType::LOCAL_INPUT;
    *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t))
        = ((__gm__ MemDetails*)(memDetail->memDetailPtr
                                + memDetail->sizeOfMemDetails * static_cast<uint32_t>(localType)))
              ->key;                                                            // L_Key
    *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr; // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;                    // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL;                          // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL;          // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL; // [48:50] DB_SL = qp.sl

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t*)(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

#endif /* AIV_COMMUNICATION_BASE_H */
