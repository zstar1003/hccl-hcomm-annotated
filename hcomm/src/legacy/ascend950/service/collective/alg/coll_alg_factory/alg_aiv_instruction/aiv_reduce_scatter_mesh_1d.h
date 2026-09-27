/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template <typename T>
// todo 简化参数
class AivReduceScatterMesh1D : public AivCommBase {
    constexpr static uint64_t stageNum = 2; // 生产者 消费者
    constexpr static uint64_t DATA_SLICE_NUM = 64 * 1024 / sizeof(T);

public:
    __aicore__ inline AivReduceScatterMesh1D() {}

    __aicore__ inline void InitCoreInfo(uint64_t len, uint64_t inputStride)
    {
        coreNumPerStage = rankSize_;
        outputCoreNum = (len_ + DATA_SLICE_NUM - 1) / DATA_SLICE_NUM;
        if (outputCoreNum > rankSize_) {
            outputCoreNum = rankSize_;
        }
        consumProcessNum = (len_ + DATA_SLICE_NUM - 1) / DATA_SLICE_NUM / outputCoreNum * DATA_SLICE_NUM;
        if (GetBlockIdx() < coreNumPerStage) {
            // input->远端ipc,一个block负责一个rank input
            targetRank = GetBlockIdx();
            int64_t outerOffset = targetRank * inputStride; // inputStride是整个算子的输入size
            int64_t ipcRankOffset = rank_ * len_ * sizeof(T);
            inputOffset = input_ + outerOffset; // 这里的input是已经偏移过前面处理完数据量的地址了
            outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + ipcRankOffset;
        } else if (GetBlockIdx() < (coreNumPerStage + outputCoreNum)) {
            // ipc->output,一个block负责一个卡的数据
            int64_t targetPart = GetBlockIdx() - coreNumPerStage;
            int64_t outerOffset = targetPart * consumProcessNum * sizeof(T);
            outputOffset = output_ + outerOffset;
            int64_t consumInOffset;
            for (int index = 0; index < rankSize_; index++) { // 轮询每个rank的数据，拉取过来，做顺序累加
                consumInOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + len_ * index * sizeof(T)
                                 + outerOffset; // 本卡的数据都在ipc
                inputOffVec[index] = consumInOffset;
            }
            if (targetPart == (outputCoreNum - 1)) {
                consumProcessNum = len_ - consumProcessNum * (outputCoreNum - 1);
            }
        }
    }

    __aicore__ inline void Producer()
    {
        CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)inputOffset, len_); // 本卡数据拷贝到ipc上
        pipe_barrier(PIPE_ALL);
        uint64_t flag_offset = rank_;
        Record(targetRank, flag_offset, curTag);
    }

    __aicore__ inline void Consumer()
    {
        if (GetBlockIdx() >= (coreNumPerStage)) {
            WaitFlag(rank_, 0, curTag);
            CpGM2GM((__gm__ T*)(outputOffset), (__gm__ T*)(inputOffVec[0]), consumProcessNum);
            pipe_barrier(PIPE_ALL);
            for (int index = 1; index < rankSize_; index++) {
                WaitFlag(rank_, index, curTag);
                CpGM2GM((__gm__ T*)(outputOffset), (__gm__ T*)(inputOffVec[index]), consumProcessNum, reduceOp_);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    __aicore__ inline void Process(uint32_t curTag)
    {
        this->curTag = static_cast<int32_t>(curTag);
        if (GetBlockIdx() < coreNumPerStage) { // 0-1
            Producer();
        } else if (GetBlockIdx() < (coreNumPerStage + outputCoreNum)) { // 2-3
            Consumer();
        }
    }

private:
    uint32_t outputCoreNum;
    uint32_t targetRank;
    uint64_t inputOffset;
    uint64_t outputOffset;
    int32_t curTag;
    uint64_t consumProcessNum;
    int64_t inputOffVec[MAX_RANK_SIZE];
    uint32_t coreNumPerStage;
};

template <typename T>
__aicore__ inline void AivReduceScatterV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivReduceScatterMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCoreInfo(len, inputSliceStride);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process(tag);
    op.BarrierAll();
}

template <typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1D<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = (op.tag_ << 15);

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.len_ = curCount;
        op.InitCoreInfo(curCount, op.inputSliceStride_);
        op.Process(loopTag);
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_rs_mesh_1d(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivReduceScatterV2Mesh1DSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivReduceScatterV2Mesh1DSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivReduceScatterV2Mesh1DSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivReduceScatterV2Mesh1DSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivReduceScatterV2Mesh1DSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivReduceScatterV2Mesh1DSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}
