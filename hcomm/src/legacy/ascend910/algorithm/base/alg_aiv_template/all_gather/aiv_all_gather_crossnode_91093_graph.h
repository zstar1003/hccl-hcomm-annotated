/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"
#include "aiv_crossnode_91093_base.h"

using namespace AscendC;

class AivAllGatherCrossNodeGraph91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivAllGatherCrossNodeGraph91093() {}

    template <typename T>
    __aicore__ inline void
    Process(GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t len);
};

template <typename T>
__aicore__ inline void AivAllGatherCrossNodeGraph91093::Process(
    GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t len)
{
    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;

    // 首次卡间同步
    BatchRecordWait(tag, buffersOut);

    PipeBarrier<PIPE_ALL>();

    // 读对端userin到usrout
    for (uint32_t i = 0; i < numTargets; i++) {
        __gm__ T* inputGMOther = (__gm__ T*)(buffersIn[i]);
        CpGM2GM(outputGM + targetRanks[i] * len + blockOffset, inputGMOther + blockOffset, countPerCore);
    }

    PipeBarrier<PIPE_ALL>();

    // 结尾卡间同步
    BatchRecordWait(tag, buffersOut, AivNotifyType::DataSignal);
}

template <typename T>
__aicore__ inline void aiv_all_gather_crossnode_91093_graph(KERNEL_ARGS_DEF_A3)
{
    AivAllGatherCrossNodeGraph91093 op;
    op.Init<T>(buffOut0, buffOut1, rank, rankSize, len, reduceOp, tag, step, numBlocks, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, SIZE_OF_INT32, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffOut0, buffOut1, input, output, tag, len);
    op.TailCounter();
}

__aicore__ inline void sk_all_gather_crossnode(SUPERKERNEL_ARGS_DEF)
{
    AivAllGatherCrossNodeGraph91093 op;
    op.InitSuperKernel(hiddenInput, true);
    uint32_t padCount = UB_ALIGN_SIZE / op.unitSize_;
    op.CalCountAndBlockOffset(
        op.len_, op.blockNumPerGroup, op.blockIdxInGroup, padCount, op.countPerCore, op.blockOffset);

    if (op.dataType_ == HcclDataType::HCCL_DATA_TYPE_INT8) {
        op.Process<int8_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HcclDataType::HCCL_DATA_TYPE_INT16) {
        op.Process<int16_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_INT32) {
        op.Process<int32_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_FP16) {
        op.Process<half>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_FP32) {
        op.Process<float>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_BFP16) {
        op.Process<bfloat16_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_UINT8) {
        op.Process<uint8_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_UINT16) {
        op.Process<uint16_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_UINT32) {
        op.Process<uint32_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_INT64) {
        op.Process<int64_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_UINT64) {
        op.Process<uint64_t>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_FP64) {
        op.Process<double>(op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, op.len_);
    }
}
