/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_CROSSNODE_91093_BASE_H
#define AIV_CROSSNODE_91093_BASE_H

#include "aiv_communication_base.h"

using namespace AscendC;

#define KERNEL_ARGS_DEF_A3                                                                                          \
    GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffOut0, GM_ADDR buffOut1, GM_ADDR bufferSize, GM_ADDR headCountMem, \
        GM_ADDR tailCountMem, GM_ADDR addOneMem, GM_ADDR isEnableCounter, GM_ADDR input, GM_ADDR output,            \
        uint32_t rank, uint32_t rankSize, uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root,        \
        int32_t tag, uint32_t numBlocks, bool isOpBase, int32_t step, uint32_t deterministic

#define KERNEL_ARGS_CALL_A3                                                                                          \
    buffIn0, buffIn1, buffOut0, buffOut1, bufferSize, headCountMem, tailCountMem, addOneMem, isEnableCounter, input, \
        output, rank, rankSize, len, dataType, reduceOp, root, tag, numBlocks, isOpBase, step, deterministic

// __sk__函数参数 A3
struct SkArgsStructA3 {
    GM_ADDR buffIn0;
    GM_ADDR buffIn1;
    GM_ADDR buffOut0;
    GM_ADDR buffOut1;
    GM_ADDR bufferSize;
    GM_ADDR headCountMem;
    GM_ADDR tailCountMem;
    GM_ADDR addOneMem;
    GM_ADDR isEnableCounter;
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
    int32_t step;
    uint32_t deterministic;
};

// __sk__定义的函数参数
#define SK_BIND_FUNC_ARGS_A3 __gm__ struct SkArgsStructA3* args

// 将__sk__参数转成__aicore__参数 A3
#define CONVERT_SK_PARAM_TO_KERNEL_ARGS_A3           \
    GM_ADDR buffIn0 = args->buffIn0;                 \
    GM_ADDR buffIn1 = args->buffIn1;                 \
    GM_ADDR buffOut0 = args->buffOut0;               \
    GM_ADDR buffOut1 = args->buffOut1;               \
    GM_ADDR bufferSize = args->bufferSize;           \
    GM_ADDR headCountMem = args->headCountMem;       \
    GM_ADDR tailCountMem = args->tailCountMem;       \
    GM_ADDR addOneMem = args->addOneMem;             \
    GM_ADDR isEnableCounter = args->isEnableCounter; \
    GM_ADDR input = args->input;                     \
    GM_ADDR output = args->output;                   \
    uint32_t rank = args->rank;                      \
    uint32_t rankSize = args->rankSize;              \
    uint64_t len = args->len;                        \
    uint32_t dataType = args->dataType;              \
    uint32_t reduceOp = args->reduceOp;              \
    uint32_t root = args->root;                      \
    int32_t tag = args->tag;                         \
    uint32_t numBlocks = args->numBlocks;            \
    bool isOpBase = args->isOpBase;                  \
    int32_t step = args->step;                       \
    uint32_t deterministic = args->deterministic

// sk 绑定函数 A3
#define SuperKernelBindA3(kernel_name)                            \
    extern "C" __sk__ void kernel_name##_1(SK_BIND_FUNC_ARGS_A3); \
    extern "C" __sk__ void kernel_name##_2(SK_BIND_FUNC_ARGS_A3); \
    extern "C" __sk__ void kernel_name##_3(SK_BIND_FUNC_ARGS_A3); \
    extern "C" __sk__ void kernel_name##_4(SK_BIND_FUNC_ARGS_A3); \
    SK_BIND(kernel_name, 8, kernel_name##_1, kernel_name##_2, kernel_name##_3, kernel_name##_4)

// A3 sk 导出函数
#define _SK_BIND_FUNC_DEF_A3(kernel_name, postfix)                       \
    extern "C" __sk__ void kernel_name##_##postfix(SK_BIND_FUNC_ARGS_A3) \
    {                                                                    \
        CONVERT_SK_PARAM_TO_KERNEL_ARGS_A3;                              \
        kernel_name##_inner(KERNEL_ARGS_CALL_A3);                        \
    }
#define SK_BIND_FUNC_DEF_A3(kernel_name, postfix) _SK_BIND_FUNC_DEF_A3(kernel_name, postfix)

// A3 Global 导出函数
#define GLOBAL_FUNC_DEF_A3(kernel_name)                                   \
    extern "C" __global__ __aicore__ void kernel_name(KERNEL_ARGS_DEF_A3) \
    {                                                                     \
        kernel_name##_inner(KERNEL_ARGS_CALL_A3);                         \
    }                                                                     \
    EXPORT_AIV_META_INFO(kernel_name)

constexpr uint32_t SIZE_OF_INT32 = 4;

class AivCrossNode91093Base {
public:
    __aicore__ inline AivCrossNode91093Base() {}

    __aicore__ inline void Init(
        GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, int32_t tag, uint32_t numBlocks,
        bool isOpBase, bool useDoubleBuffer); // ALL2ALL的init

    template <typename T>
    __aicore__ inline void InitDeter(
        GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, uint32_t reduceOp, int32_t tag,
        uint32_t numBlocks, bool useDoubleBuffer);

    __aicore__ inline void InitSuperKernel(GM_ADDR hiddenInput, bool useDoubleBuffer);

    template <typename T>
    __aicore__ inline void Init(
        GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, uint64_t perRankBufferCount, uint64_t len,
        uint32_t reduceOp, int32_t tag, int32_t step, uint32_t numBlocks, bool useDoubleBuffer); // AG、RS单算子的init

    template <typename T>
    __aicore__ inline void Init(
        GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, uint64_t len, uint32_t reduceOp,
        int32_t tag, int32_t step, uint32_t numBlocks, bool useDoubleBuffer); // AG、RS图模式的init

    __aicore__ inline void InitOffset(); // 初始化offset

    __aicore__ inline void InitSetCheckClearArgsTensor();

    __aicore__ inline void CalcNumTargetsAndTargetRanks();

    __aicore__ inline void CalcNumTargetsAndTargetRanksGroup();

    __aicore__ inline void ClearGM();

    __aicore__ inline void Barrier(GM_ADDR* buffersOut, int32_t curTag);

    __aicore__ inline void GetTargetBuffer(bool isOpBase = true);

    __aicore__ inline void ClearCycle();

    __aicore__ inline void SyncAllCycle(AivNotifyType notifyType, int32_t blockGroup, bool ifSyncCore);

    template <typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    __aicore__ inline uint64_t
    CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength);

    __aicore__ inline void CalCountAndBlockOffset(
        uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup, uint32_t padCount, uint64_t& count,
        uint64_t& blockOffset);

    template <typename T>
    __aicore__ inline void
    DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount);

    template <typename T>
    __aicore__ inline void
    DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount);

    template <typename T>
    __aicore__ inline void
    CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, bool atomic = false, uint32_t atomicOp = 0);

    template <HardEvent event>
    __aicore__ inline void SyncFunc();

    __aicore__ inline void
    SingleRecordBatchWaitCoreLevel(int32_t curTag, bool isTheSingleCore, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    BatchRecordSingleWaitCoreLevel(int32_t curTag, bool isTheSingleCore, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void SingleRecordBatchWait(
        int32_t curTag, GM_ADDR* buffersOut, bool isTheSingleCore, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    BatchRecordWait(int32_t curTag, GM_ADDR* buffersOut, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    LocalMultiWaitRecord(uint32_t tag, AivNotifyType notifyType, int32_t blockGroup, bool ifClear);

    __aicore__ inline void localMultiRecord(uint32_t tag, int32_t blockGroup, AivNotifyType notifyType);

    __aicore__ inline void LocalRecord(uint32_t tag, uint32_t waitBlock, AivNotifyType notifyType, bool ifSet = true);

    __aicore__ inline void LocalWait(uint32_t tag, AivNotifyType notifyType, bool ifClear = false);

    __aicore__ inline void Record(uint32_t tag, GM_ADDR waitAddr, AivNotifyType notifyType);

    __aicore__ inline void Record1vN(uint32_t tag, CommPattern pattern, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    RecordNv1(uint32_t tag, GM_ADDR waitAddr, bool ifCoreLevel, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    WaitNv1(uint32_t tag, GM_ADDR recordAddr, bool ifCoreLevel, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    Wait1vN(uint32_t tag, CommPattern pattern, bool ifClear = true, AivNotifyType notifyType = AivNotifyType::ACK);

    __aicore__ inline void
    SetSyncRecord(int32_t value, GM_ADDR setAddr, int32_t offAddr, int32_t setBlock, bool ifPingpong = false);

    __aicore__ inline void
    WaitSyncFlag(int32_t value, GM_ADDR waitAddr, int32_t offAddr, int32_t waitBlock, bool ifPingpong = false);

    __aicore__ inline void CountRecord(uint64_t count, int32_t index);

    __aicore__ inline void CountWaitGE(GM_ADDR waitAddr, uint64_t count, int32_t index);

    __aicore__ inline void IntraSync(int32_t curTag, int32_t offset, int32_t blockIdx, bool ifPingpong = false);

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

    uint32_t baseFlagOffset_ = 0;
    GM_ADDR flagAddrSelf_;
    GM_ADDR dataAddrSelf_;
    GM_ADDR commAddr_;
    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t reduceOp_;
    uint32_t usedBlockNum_;
    uint32_t blockGroup_;
    bool useDoubleBuffer_;
    int32_t logLevel_;
    int32_t tag_;
    bool localCopyCores = false;
    int32_t clearEnable_ = 0;
    uint32_t dataType_;
    uint32_t unitSize_;

    uint64_t len_;
    uint32_t numBlocks_;
    uint32_t blockIdx_ = GetBlockIdx(); // 在构造函数中初始化，以免漏初始化

    TPipe pipe;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localClearTensor;
    LocalTensor<int32_t> localCheckGETensor;
    TBuf<> bufferArgsBuf;
    LocalTensor<uint64_t> bufferArgsTensor; // buffer地址GM-UB
    TBuf<> offsetArgsBuf;
    LocalTensor<uint64_t> offsetArgsTensor; // count参数UB-GM，类似做allgather

    // 每个aiv核的数据搬运参数，用于多核并行优化方案
    uint32_t numTargets; // 每个aiv需要顺序与几个对端通信，ranksize太大时，aiv不够用，需要多次
    uint32_t targetRanks[MAX_TARGET_NUM] = {}; // 最多768/48 = 16 次（一次代表服务48张卡）
    uint32_t blockNumPerGroup = 1;             // 多少个aiv服务一个rank
    uint32_t blockIdxInGroup = 0;              // 同一组中的aiv编号
    uint32_t blockGroupIdx = blockIdx_;        // 同一组中的aiv编号
    uint64_t groupMid_ = 0;                    // 一组aiv负责搬运的数据量, 区分中间轮和尾轮
    uint64_t groupTail_ = 0;
    uint64_t groupTailLast_ = 0; // 尾轮对应lastRank的一组aiv负责的数据量
    uint64_t countMid;           // 中间轮一个aiv负责搬运的数据量（一轮代表一次ccl buffer装满）
    uint64_t countTail;          // 尾轮一个aiv负责搬运的数据量
    uint64_t countTailLast_ = 0;
    uint64_t blockOffsetMid; // 数据块offset，区分中间轮和尾轮
    uint64_t blockOffsetTail;
    uint64_t blockOffsetTailLast_ = 0;
    uint32_t flagOffsetInGroup; // 标志位offset，不区分中间轮和尾轮
    uint64_t blockOffset;       // 数据块offset，不区分中间轮和尾轮
    uint64_t countPerCore;      // 每个核负责的数据块大小，不区分中间轮和尾轮

    // 维测相关
    GM_ADDR headCountMem_;
    GM_ADDR tailCountMem_;
    GM_ADDR addOneMem_;
    uint32_t counterMemSize_;
    bool isEnableCounter_;

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
    uint32_t syncAllOffset;

    GM_ADDR buffersIn[MAX_TARGET_NUM] = {};
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};
};

__aicore__ inline uint64_t AivCrossNode91093Base::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t AivCrossNode91093Base::CalActualCount(
    uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength)
{
    if (sliceIdx == sliceCount - 1) {
        return tailLength;
    }

    if (sliceIdx < sliceCount - 1) {
        return avgLengthPerSlice;
    }

    return 0;
}

__aicore__ inline void AivCrossNode91093Base::CalCountAndBlockOffset(
    uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup, uint32_t padCount, uint64_t& count,
    uint64_t& blockOffset)
{
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice; // 多核并行搬数据，最后一核搬运的数据量

    count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    blockOffset = blockIdxInGroup * avgLengthPerSlice;
    AIV_INFO("count %llu, blockOffset %llu\n", count, blockOffset);
}

__aicore__ inline void AivCrossNode91093Base::CalcNumTargetsAndTargetRanks()
{
    // 计算本core的numTargets和targetsList
    // 前concurrentSize/2个aiv负责与左边rank号的通信，后concurrentSize/2个负责与右边rank号的通信
    uint32_t halfConcurrent = usedBlockNum_ / 2;  // usedBlockNum_需要为偶数
    numTargets = (rankSize_ - 1) / usedBlockNum_; // 除去本rank，可能需要补上一个
    uint32_t tailRankSize = (rankSize_ - 1) % usedBlockNum_;
    uint32_t leftTailRankSize = 0;
    uint32_t rightTailRankSize = 0;
    if (tailRankSize > 0) {
        if (tailRankSize <= halfConcurrent) {
            leftTailRankSize = tailRankSize;
        } else {
            leftTailRankSize = halfConcurrent;
            rightTailRankSize = tailRankSize - halfConcurrent;
        }
        if (blockIdx_ < halfConcurrent && (halfConcurrent - blockIdx_) <= leftTailRankSize) {
            numTargets += 1;
        }
        if (blockIdx_ >= halfConcurrent && (blockIdx_ - halfConcurrent + 1) <= rightTailRankSize) {
            numTargets += 1;
        }
    }

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank;
        if (blockIdx_ < halfConcurrent) {
            targetRank = (rank_ + rankSize_ - (halfConcurrent - blockIdx_) - i * halfConcurrent) % rankSize_; // left
        } else {
            targetRank = (rank_ + (blockIdx_ - halfConcurrent + 1) + i * halfConcurrent) % rankSize_; // right
        }
        targetRanks[i] = targetRank;
    }
}

__aicore__ inline void AivCrossNode91093Base::CalcNumTargetsAndTargetRanksGroup()
{
    blockNumPerGroup = numBlocks_ / blockGroup_; // 多少个aiv服务同一个对端
    blockIdxInGroup = (blockIdx_ / blockGroup_) % blockNumPerGroup;
    blockGroupIdx = blockIdx_ % blockGroup_;
    numTargets = (rankSize_) / blockGroup_;
    uint32_t tailRankSize = (rankSize_) % blockGroup_;
    if (tailRankSize > 0 && blockIdx_ < tailRankSize) {
        numTargets += 1;
    }

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = (blockIdx_ % blockGroup_ + i * blockGroup_) % rankSize_;
        targetRanks[i] = targetRank;
        if (targetRank == rank_) {
            localCopyCores = true;
        }
    }
}

__aicore__ inline void AivCrossNode91093Base::InitSetCheckClearArgsTensor()
{
    logLevel_ = GetLogLevel();
    uint64_t offset
        = (logLevel_ == 1) ? (tag_ & 1 ? INFO_EVEN_BUFFER_OFFSET : INFO_ODD_BUFFER_OFFSET) : INFO_EVEN_BUFFER_OFFSET;
    pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE * FLAG_BUF_NUM);
    localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, 0);
    localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE);
    localClearTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE * IDX_2);
    localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE * IDX_3);
    localClearTensor.SetValue(0, 0);
    pipe.InitBuffer(bufferArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    bufferArgsTensor = bufferArgsBuf.Get<uint64_t>();
    if (useDoubleBuffer_) {
        pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE);
    } else {
        pipe.InitBuffer(inOutQue, 1, UB_DB_DATA_BATCH_SIZE);
    }
}

// ALL2ALL的init
__aicore__ inline void AivCrossNode91093Base::Init(
    GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, int32_t tag, uint32_t numBlocks,
    bool isOpBase, bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;
    rank_ = rank;
    tag_ = tag;
    rankSize_ = rankSize;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = numBlocks;
    numBlocks_ = numBlocks;
    blockGroup_ = numBlocks_;
    commAddr_ = buffOut1;

    InitSetCheckClearArgsTensor();
    pipe.InitBuffer(offsetArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    offsetArgsTensor = offsetArgsBuf.Get<uint64_t>();

    CalcNumTargetsAndTargetRanks();
    GetTargetBuffer(isOpBase);
    InitOffset();
    if (tag_ == 1) {
        ClearCycle();
    }
}

__aicore__ inline void AivCrossNode91093Base::InitOffset()
{
    int32_t notifyArea = MAX_RANK_SIZE_A3;
    if (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 < MAX_RANK_SIZE_A3) {
        notifyArea = rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3;
    }
    localOffset = (notifyArea * FLAG_BUF_NUM) * FLAG_SIZE;
    multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
    pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
    countOffset = DOUBLE * pingpongOffset;
    syncAllOffset = countOffset + notifyArea * FLAG_SIZE;
}

template <typename T>
__aicore__ inline void AivCrossNode91093Base::InitDeter(
    GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, uint32_t reduceOp, int32_t tag,
    uint32_t numBlocks, bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;
    rank_ = rank;
    tag_ = tag;
    rankSize_ = rankSize;
    reduceOp_ = reduceOp;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = numBlocks;
    numBlocks_ = numBlocks;
    pingpongOffset = 0;
    blockGroup_ = numBlocks_;
    commAddr_ = buffOut1;

    InitSetCheckClearArgsTensor();
    if (rankSize > numBlocks_) {
        blockNumPerGroup = 1;
    } else {
        blockNumPerGroup = numBlocks_ / rankSize_; // 多少个aiv服务一个rank
    }
    CalcNumTargetsAndTargetRanksGroup();
    GetTargetBuffer();
    InitOffset();
    if (tag_ == 1) {
        ClearCycle();
    }
}

__aicore__ inline void AivCrossNode91093Base::InitSuperKernel(GM_ADDR hiddenInput, bool useDoubleBuffer)
{
    __gm__ AivSuperKernelArgs* args = reinterpret_cast<__gm__ AivSuperKernelArgs*>(hiddenInput);
    rank_ = args->rank;
    rankSize_ = args->rankSize;
    reduceOp_ = args->reduceOp;
    len_ = args->len;
    tag_ = args->tag;
    dataType_ = args->dataType;
    unitSize_ = args->unitSize;
    numBlocks_ = args->numBlocks;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = numBlocks_;
    flagAddrSelf_ = args->buffersOut[0];
    dataAddrSelf_ = args->buffersIn[0];
    commAddr_ = args->buffersOut[1];

    blockGroup_ = rankSize_ > numBlocks_ ? numBlocks_ : rankSize_;

    InitSetCheckClearArgsTensor();
    pipe.InitBuffer(offsetArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    offsetArgsTensor = offsetArgsBuf.Get<uint64_t>();
    CalcNumTargetsAndTargetRanksGroup();
    GetTargetBuffer(false);
    InitOffset();
    if (tag_ == 1) {
        ClearCycle();
    }
}

// AG、RS单算子的Init
template <typename T>
__aicore__ inline void AivCrossNode91093Base::Init(
    GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, uint64_t perRankBufferCount, uint64_t len,
    uint32_t reduceOp, int32_t tag, int32_t step, uint32_t numBlocks, bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;
    blockGroup_ = step;
    rank_ = rank;
    tag_ = tag;
    rankSize_ = rankSize;
    reduceOp_ = reduceOp;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = numBlocks;
    numBlocks_ = numBlocks;
    commAddr_ = buffOut1;

    InitSetCheckClearArgsTensor();

    CalcNumTargetsAndTargetRanksGroup();

    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    if (len <= perRankBufferCount) { // ccl够用，只需要搬一轮的情况
        countMid = 0;
        blockOffsetMid = 0;
        CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
    } else if (len % perRankBufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
        CalCountAndBlockOffset(
            perRankBufferCount, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);
        countTail = countMid;
        blockOffsetTail = blockOffsetMid;
    } else { // ccl不够用，要搬多轮的情况2: 不能整除
        CalCountAndBlockOffset(
            perRankBufferCount, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);
        uint64_t remainLen = len % perRankBufferCount;
        CalCountAndBlockOffset(remainLen, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
    }
    flagOffsetInGroup = blockIdxInGroup * FLAG_SIZE;
    CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    InitOffset();
    GetTargetBuffer();
    if (tag_ == 1) {
        ClearCycle();
    }
}

// AG、RS图模式的Init
template <typename T>
__aicore__ inline void AivCrossNode91093Base::Init(
    GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, uint64_t len, uint32_t reduceOp, int32_t tag,
    int32_t step, uint32_t numBlocks, bool useDoubleBuffer)
{
    flagAddrSelf_ = buffOut0;
    blockGroup_ = step;
    rank_ = rank;
    tag_ = tag;
    rankSize_ = rankSize;
    reduceOp_ = reduceOp;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = numBlocks;
    numBlocks_ = numBlocks;
    commAddr_ = buffOut1;

    InitSetCheckClearArgsTensor();
    CalcNumTargetsAndTargetRanksGroup();
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    InitOffset();
    GetTargetBuffer(false);
    if (tag_ == 1) {
        ClearCycle();
    }
}

template <typename T>
__aicore__ inline void AivCrossNode91093Base::SetAtomicOp(uint32_t atomicOp)
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
__aicore__ inline void AivCrossNode91093Base::DataCopyGM2UB(
    const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount)
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
__aicore__ inline void AivCrossNode91093Base::DataCopyUB2GM(
    const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
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
AivCrossNode91093Base::CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, bool atomic, uint32_t atomicOp)
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

template <HardEvent event>
__aicore__ inline void AivCrossNode91093Base::SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

__aicore__ inline void
AivCrossNode91093Base::SingleRecordBatchWaitCoreLevel(int32_t curTag, bool isTheSingleCore, AivNotifyType notifyType)
{
    if (isTheSingleCore) {
        Record1vN(curTag, CommPattern::intraRank, notifyType);
    } else {
        WaitNv1(curTag, flagAddrSelf_, true, notifyType);
    }
}

__aicore__ inline void
AivCrossNode91093Base::BatchRecordSingleWaitCoreLevel(int32_t curTag, bool isTheSingleCore, AivNotifyType notifyType)
{
    // 负责localcopy的核去查该flag，等所有其他核已经完成写（原子累加）
    if (isTheSingleCore) {
        Wait1vN(curTag * (rankSize_ - 1), CommPattern::intraRank);
        // 其他核去写该flag，做原子累加达到核间同步的目的
    } else {
        RecordNv1(curTag, flagAddrSelf_, true);
    }
}

__aicore__ inline void AivCrossNode91093Base::SingleRecordBatchWait(
    int32_t curTag, GM_ADDR* buffersOut, bool isTheSingleCore, AivNotifyType notifyType)
{
    if (isTheSingleCore) {
        Record1vN(curTag, CommPattern::interRank, notifyType);
    }
    for (uint32_t i = 0; i < numTargets; i++) {
        WaitNv1(curTag, buffersOut[i], false, notifyType);
    }
}

__aicore__ inline void
AivCrossNode91093Base::BatchRecordWait(int32_t curTag, GM_ADDR* buffersOut, AivNotifyType notifyType)
{
    // 写所有对端的flag
    for (uint32_t i = 0; i < numTargets; i++) {
        Record(curTag, buffersOut[i], notifyType);
    }
    // 读自己的所有flag
    for (uint32_t i = 0; i < numTargets; i++) {
        Wait(curTag, targetRanks[i], notifyType);
    }
}

__aicore__ inline void AivCrossNode91093Base::Record(uint32_t tag, GM_ADDR waitAddr, AivNotifyType notifyType)
{
    AIV_INFO("[Record]tag is [%u], waitAddr is [%p], notifyType is [%d]\n", tag, waitAddr, notifyType);
    int32_t recordOffset
        = (blockIdxInGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + rank_) * FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(waitAddr + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void
AivCrossNode91093Base::LocalRecord(uint32_t tag, uint32_t waitBlock, AivNotifyType notifyType, bool ifSet)
{
    AIV_INFO("[LocalRecord]tag is [%u], waitBlock is [%p], notifyType is [%d]\n", tag, waitBlock, notifyType);
    int32_t recordOffset = localOffset + waitBlock * FLAG_SIZE + (int32_t(notifyType) % 3) * MAX_NUM_BLOCKS * FLAG_SIZE
                           + (int32_t(notifyType) / 3) * 2560 * 1024;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag, ifSet);
}

__aicore__ inline void AivCrossNode91093Base::LocalWait(uint32_t tag, AivNotifyType notifyType, bool ifClear)
{
    int32_t waitOffset = localOffset + blockIdx_ * FLAG_SIZE + (int32_t(notifyType) % 3) * MAX_NUM_BLOCKS * FLAG_SIZE
                         + (int32_t(notifyType) / 3) * 2560 * 1024;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
    pipe_barrier(PIPE_ALL);
    if (ifClear) {
        SetSignalValue(ctrlFlagGM, localSetTensor, 0);
    }
}

__aicore__ inline void AivCrossNode91093Base::Record1vN(uint32_t tag, CommPattern pattern, AivNotifyType notifyType)
{
    AIV_INFO("[Record1vN]tag is [%u], pattern is [%d], notifyType is [%d]\n", tag, pattern, notifyType);
    int32_t recordOffset
        = multiOffset
          + (int32_t(pattern) * 2 * blockNumPerGroup + int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup)
                * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void
AivCrossNode91093Base::RecordNv1(uint32_t tag, GM_ADDR waitAddr, bool ifCoreLevel, AivNotifyType notifyType)
{
    AIV_INFO(
        "[RecordNv1]tag is [%d], waitAddr is [%p], ifCoreLevel is [%d], notifyType is [%d]\n", tag, waitAddr,
        ifCoreLevel, notifyType);
    int32_t recordOffset
        = multiOffset + 2 * 2 * blockNumPerGroup * ATOMIC_FLAG_SIZE
          + (int32_t(ifCoreLevel) * blockNumPerGroup * 2 + int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup)
                * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(waitAddr + recordOffset);
    AddSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void
AivCrossNode91093Base::localMultiRecord(uint32_t tag, int32_t blockGroup, AivNotifyType notifyType)
{
    // todo
#ifndef OPEN_HCCL_TEST
    localSetTensor.SetValue(0, tag);
#endif
    SyncFunc<HardEvent::S_MTE3>();
    for (int32_t i = 0; i < blockGroup; i++) {
        LocalRecord(tag, blockIdx_ + i, notifyType, false);
    }
}

__aicore__ inline void
AivCrossNode91093Base::LocalMultiWaitRecord(uint32_t tag, AivNotifyType notifyType, int32_t blockGroup, bool ifClear)
{
    // todo
    int32_t waitOffset = localOffset + (int32_t(notifyType) % 3) * MAX_NUM_BLOCKS * FLAG_SIZE
                         + (int32_t(notifyType) / 3) * 2560 * 1024;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + waitOffset);
    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT * blockGroup);

    while (true) {
        DataCopy(localCheckTensor, globalTensor, UB_FLAG_PAD_COUNT * blockGroup);
        SyncFunc<HardEvent::MTE2_S>();
        int32_t sum = 0;
        for (int32_t i = 1; i < blockGroup; i++) {
            sum += localCheckTensor.GetValue(UB_FLAG_PAD_COUNT * i);
        }
        if (sum == (blockGroup - 1) * tag) {
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
    if (!ifClear) {
        localMultiRecord(tag + 1, blockGroup, notifyType);
    } else {
        localMultiRecord(0, blockGroup, notifyType);
    }
    return;
}

__aicore__ inline void AivCrossNode91093Base::Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType)
{
    AIV_INFO("[Wait]tag is [%u], recordRank is [%d], notifyType is [%d]\n", tag, recordRank, notifyType);
    int32_t waitOffset
        = (blockIdxInGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + recordRank) * FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void
AivCrossNode91093Base::WaitNv1(uint32_t tag, GM_ADDR recordAddr, bool ifCoreLevel, AivNotifyType notifyType)
{
    AIV_INFO(
        "[WaitNv1]tag is [%u], recordAddr is [%p], ifCoreLevel is [%d], notifyType is [%d]\n", tag, recordAddr,
        ifCoreLevel, notifyType);
    int32_t waitOffset
        = multiOffset
          + (int32_t(ifCoreLevel) * blockNumPerGroup * 2 + int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup)
                * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(recordAddr + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void
AivCrossNode91093Base::Wait1vN(uint32_t tag, CommPattern pattern, bool ifClear, AivNotifyType notifyType)
{
    AIV_INFO(
        "[Wait1vN]tag is [%u], pattern is [%d], ifClear is [%d], notifyType is [%d]\n", tag, pattern, ifClear,
        notifyType);
    int32_t waitOffset
        = multiOffset + 2 * 2 * blockNumPerGroup * ATOMIC_FLAG_SIZE
          + (int32_t(pattern) * blockNumPerGroup * 2 + int32_t(notifyType) * blockNumPerGroup + blockIdxInGroup)
                * ATOMIC_FLAG_SIZE;
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();
    if (ifClear) {
        SetSignalValue(ctrlFlagGM, localSetTensor, 0);
    }
}

// 卡内全Aiv同步
__aicore__ inline void AivCrossNode91093Base::IntraSync(int32_t tag, int32_t offset, int32_t blockIdx, bool ifPingpong)
{
    AIV_INFO(
        "[IntraSync]tag is [%d], offset is [%d], blockIdx is [%d], ifPingpong is [%d]\n", tag, offset, blockIdx,
        ifPingpong);
    SetSyncRecord(tag, flagAddrSelf_, offset, blockIdx, ifPingpong);
    for (uint32_t i = 0; i < usedBlockNum_; i++) {
        if (i == blockIdx) {
            continue;
        }
        WaitSyncFlag(tag, flagAddrSelf_, offset, i, ifPingpong);
    }
}

__aicore__ inline int32_t AivCrossNode91093Base::GetLogLevel()
{
#ifndef OPEN_HCCL_TEST
    int32_t tmpLogLevel = *((__gm__ int32_t*)(flagAddrSelf_ + LOG_LEVEL_OFFSET - sizeof(int32_t)));
    return tmpLogLevel;
#else
    return 0;
#endif
}

__aicore__ inline void AivCrossNode91093Base::SetSyncRecord(
    int32_t value, GM_ADDR setAddr, int32_t highOrderOff, int32_t lowOrderOff, bool ifPingpong)
{
    AIV_INFO(
        "[SetSyncRecord]value is [%d], setAddr is [%p], highOrderOff is [%d], "
        "lowOrderOff is [%d], ifPingpong is [%d]\n",
        value, setAddr, highOrderOff, lowOrderOff, ifPingpong);
    int32_t ppOffset = ifPingpong ? pingpongOffset : 0;

    int32_t recordOffset = (highOrderOff * MAX_NUM_BLOCKS + lowOrderOff) * FLAG_SIZE;

    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(setAddr + localOffset + ppOffset + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, value);
}

__aicore__ inline void AivCrossNode91093Base::WaitSyncFlag(
    int32_t value, GM_ADDR waitAddr, int32_t highOrderOff, int32_t lowOrderOff, bool ifPingpong)
{
    AIV_INFO(
        "[WaitSyncFlag]value is [%d], waitAddr is [%p], highOrderOff is [%d], "
        "lowOrderOff is [%d], ifPingpong is [%d]\n",
        value, waitAddr, highOrderOff, lowOrderOff, ifPingpong);
    int32_t ppOffset = ifPingpong ? pingpongOffset : 0;

    int32_t waitOffset = (highOrderOff * MAX_NUM_BLOCKS + lowOrderOff) * FLAG_SIZE;

    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(waitAddr + localOffset + ppOffset + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, value);
}

__aicore__ inline void AivCrossNode91093Base::Barrier(GM_ADDR* buffersOut, int32_t curTag)
{
    uint32_t flagOffset = SYNC_BUFFER_OFFSET;
    flagOffset += ((curTag % AIV_PING_PONG_FACTOR_TWO == 0) ? 0 : rankSize_ * FLAG_SIZE);
    // tx
    localSetTensor.SetValue(0, curTag);
    GlobalTensor<int32_t> globalTag;
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t i = 0; i < numTargets; i++) {
        GM_ADDR flagAddrOther = buffersOut[i];
        globalTag.SetGlobalBuffer((__gm__ int32_t*)(flagAddrOther + flagOffset + rank_ * FLAG_SIZE), UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }
    // rx and clear
    for (uint32_t i = 0; i < numTargets; i++) {
        globalTag.SetGlobalBuffer(
            (__gm__ int32_t*)(flagAddrSelf_ + flagOffset + targetRanks[i] * FLAG_SIZE), UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localCheckTensor, globalTag, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localCheckTensor.GetValue(0) == curTag) {
                break;
            }
        }
        DataCopy(globalTag, localClearTensor, UB_FLAG_PAD_COUNT); // 清零
    }
}

__aicore__ inline void AivCrossNode91093Base::ClearGM()
{
    uint32_t emptyOffset = 1 * 1024 * 1024;
    uint32_t blockOffset = 1 * 1024 * 1024 / blockGroup_ * blockGroupIdx;
    uint32_t blockCount = 1 * 1024 * 1024 / blockGroup_;
    CpGM2GM(flagAddrSelf_ + blockOffset, flagAddrSelf_ + blockOffset + emptyOffset, blockCount);
}

__aicore__ inline void
AivCrossNode91093Base::SyncAllCycle(AivNotifyType notifyType, int32_t blockGroup, bool ifSyncCore)
{
    // todo
    LocalRecord(1, blockIdx_, notifyType);
    if (ifSyncCore) {
        LocalMultiWaitRecord(1, notifyType, blockGroup, false);
    }
    LocalWait(IDX_2, notifyType, true);
}

__aicore__ inline void AivCrossNode91093Base::ClearCycle()
{
    // todo
    if (blockIdxInGroup == 0) {
        Barrier(buffersOut, 1);
        pipe_barrier(PIPE_ALL);
        SyncAllCycle(AivNotifyType::ClearACK, blockGroup_, blockIdx_ == 0);
        pipe_barrier(PIPE_ALL);
        ClearGM();
        Barrier(buffersOut, IDX_2);
        pipe_barrier(PIPE_ALL);
    }
    SyncAllCycle(AivNotifyType::ClearDataSingal, numBlocks_, blockIdx_ == 0);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void AivCrossNode91093Base::GetTargetBuffer(bool isOpBase)
{
    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t* buffersGmAddr = (__gm__ uint64_t*)(commAddr_);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    // 准备参数，buffer地址
    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * IDX_4], bufferArgsGT[2 * targetRank], IDX_4); // buffersIn buffersOut
    }

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersIn[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx));
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    if (!isOpBase) {
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void AivCrossNode91093Base::CountRecord(uint64_t count, int32_t index)
{
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(flagAddrSelf_ + countOffset + index * FLAG_SIZE);
    SetSignalValue(ctrlFlagGM, localSetTensor, count);
}

__aicore__ inline void AivCrossNode91093Base::CountWaitGE(GM_ADDR waitAddr, uint64_t count, int32_t index)
{
    __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(waitAddr + countOffset + index * FLAG_SIZE);
    WaitSignalGEValue(ctrlFlagGM, localCheckGETensor, count);
}

#endif /* AIV_CROSSNODE_91093_BASE_H */
