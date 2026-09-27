/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_reduce_mesh_oneshot.h"

namespace hccl {
AllReduceMeshDirectOneshot::AllReduceMeshDirectOneshot(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceMeshDirectOneshot::~AllReduceMeshDirectOneshot() {}

HcclResult AllReduceMeshDirectOneshot::Prepare(
    u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank,
    HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    localRank_ = interRank;
    localRankSize_ = interRankSize;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    opInfo_ = opInfo;
    return HCCL_SUCCESS;
}

// ring allreduce算法的函数入口
HcclResult AllReduceMeshDirectOneshot::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllReduceMeshDirectOneshot run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceMeshDirectOneshot][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank,
            links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (opInfo_->inputAddr != opInfo_->outputAddr) {
            DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * SIZE_TABLE[dataType_]);
            DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * SIZE_TABLE[dataType_]);
            ret = HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceMeshOneshot][RunAsync]rank[%u] memcpy async failed", rank),
                ret);
        }
        return ret;
    }

    ret = RunAllReduceOne(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceMeshOneshot][RunAsync]rank[%u] count[%llu] failed in step", rank, count_), ret);

    HCCL_INFO("AllReduceMeshDirectOneshot finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirectOneshot::RunAllReduceOne(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "RunAllReduceOne run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = unitSize * count_;

    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, totalSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);

    DeviceMem src = DeviceMem::create(static_cast<char*>(opInfo_->inputAddr), totalSize);
    DeviceMem dst = DeviceMem::create(static_cast<char*>(opInfo_->outputAddr), totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    if (opInfo_->outputAddr != outputMem_.ptr()) {
        src = DeviceMem::create(static_cast<char*>(opInfo_->inputAddr), totalSize);
        dst = commMemOut.range(0, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        CHK_RET(links[dstRank]->TxAck(stream_));
        CHK_RET(links[dstRank]->RxAck(stream_));

        void* remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

        src = DeviceMem::create(static_cast<char*>(remMemPtr), totalSize);
        dst = userMemOut.range(0, totalSize);
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, stream_,
            static_cast<void*>(dst.ptr()), links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(links[dstRank]->TxDataSignal(stream_));
        CHK_RET(links[dstRank]->RxDataSignal(stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_MESH_DIRECT_ONESHOT, AllReduceMeshDirectOneshot);
} // namespace hccl
