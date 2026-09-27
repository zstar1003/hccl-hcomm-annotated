/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_nb.h"
#include "alg_template_register.h"

namespace hccl {

ReduceScatterNB::ReduceScatterNB(const HcclDispatcher dispatcher) : NBBase(dispatcher) {}

ReduceScatterNB::~ReduceScatterNB() {}

HcclResult ReduceScatterNB::Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo)
{
    (void)opInfo;
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 参数校验
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "ReduceScatterNB run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[ReduceScatterRing][RunAsync] rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    std::vector<Slice> outputSlices(slices_);

    // 处理和检查Slices
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        outputSlices.resize(rankSize);

        // 生成std::vector<Slice> slices_
        u64 sliceSize = count_ * unitSize;
        HCCL_DEBUG("[ReduceScatterNB][RunAsync]sliceSize is %llu", sliceSize);

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);

            outputSlices[i].size = sliceSize;
            outputSlices[i].offset = (inputMem_.size() > outputMem_.size()) ? 0 : (i * sliceSize);
            HCCL_DEBUG(
                "rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu] outputSlices[%u].offset=[%llu], "
                "outputSlices[%u].size=[%llu] count_[%llu] unitSize[%llu]",
                rank, i, slices_[i].offset, i, slices_[i].size, i, outputSlices[i].offset, i, outputSlices[i].size,
                count_, unitSize);
        }
    }

    CHK_RET(CheckSlices(slices_, rankSize));

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);

    // 运行reduce-scatter, NB 算法
    CHK_RET(RunReduceScatterNB(rank, rankSize, links, slices_, outputSlices));

    HCCL_INFO("ReduceScatterNB finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_, HCCL_ERROR("[ReduceScatterNB]rank[%u] inputmem or outputmem is null", rank),
        HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[ReduceScatterNB]rank[%u] link size[%llu] is less than "
            "rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::CheckSlices(const std::vector<Slice>& checkSlices, const u32 rankSize)
{
    CHK_PRT_RET(
        checkSlices.size() % rankSize != 0,
        HCCL_ERROR("[ReduceScatterNB]slices.size[%u] should be divided by rankSize[%u]", checkSlices.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::RunReduceScatterNB(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    bool bRetSize = (inputSlices.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[Run][ReduceScatter]rank[%u] inputslice size[%llu] is less than rank size[%u]", rank, inputSlices.size(),
            rankSize),
        HCCL_E_INTERNAL);

    bRetSize = (outputSlices.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[Run][ReduceScatter]rank[%u] outputslice size[%llu] is less than rank size[%u]", rank, outputSlices.size(),
            rankSize),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;

    // 计算通信步数：ceiling(log2(rankSize))
    u32 nSteps = CalcCeilLog2(rankSize);
    u32 sliceSize = inputSlices.size() / rankSize;
    HCCL_DEBUG(
        "ReduceScatter debug-1: rank[%u] rankSize[%u] nSteps[%u] sliceSize[%u]", rank, rankSize, nSteps, sliceSize);
    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 recvFrom = (rankSize + rank - deltaRank) % rankSize;
        u32 sendTo = (rank + deltaRank) % rankSize;

        // 数据份数和数据编号增量
        u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 txSliceIdx = (rank + (1 << step)) % rankSize;
        u32 rxSliceIdx = rank;

        LINK linkLeft = links[recvFrom];
        CHK_SMART_PTR_NULL(linkLeft);

        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        // 当前每个数据块发送一次ACK、reduce一次、同步一次
        HCCL_DEBUG(
            "ReduceScatter debug-2: recvFrom[%u] sendTo[%u] step[%u] nSlices[%u] deltaSliceIndex[%u] "
            "rxSliceIdx[%u] txSliceIdx[%u]",
            recvFrom, sendTo, step, nSlices, deltaSliceIndex, rxSliceIdx, txSliceIdx);

        u32 txCount = 0;
        u32 txSliceIdxTmp = txSliceIdx;
        for (u32 i = 0; i < nSlices; i++) {
            for (u32 j = 0; j < sliceSize; j++) {
                if (inputSlices[txSliceIdxTmp * sliceSize + j].size > 0) {
                    txCount++;
                }
            }
            txSliceIdxTmp = (txSliceIdxTmp + deltaSliceIndex) % rankSize;
        }

        u32 rxCount = 0;
        u32 rxSliceIdxTmp = rxSliceIdx;
        for (u32 i = 0; i < nSlices; i++) {
            for (u32 j = 0; j < sliceSize; j++) {
                if (inputSlices[rxSliceIdxTmp * sliceSize + j].size > 0) {
                    rxCount++;
                }
            }
            rxSliceIdxTmp = (rxSliceIdxTmp + deltaSliceIndex) % rankSize;
        }
        if (rxCount > 0) {
            CHK_RET(linkLeft->TxAck(stream_));
        }
        if (txCount > 0) {
            CHK_RET(linkRight->RxAck(stream_));
            RunSrcReducerNB(
                step, nSlices, sliceSize, txSliceIdx, deltaSliceIndex, linkRight, rank, rankSize, inputSlices,
                outputSlices);
        }
        if (rxCount > 0) {
            RunDestReducerNB(
                step, nSteps, sliceSize, nSlices, rxSliceIdx, deltaSliceIndex, linkLeft, rank, rankSize, inputSlices,
                outputSlices);
            ret = linkLeft->RxWaitDone(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Run][ReduceScatterNB]rank[%u] step[%u] blocknum[%u] rx wait done failed", rank, step, nSlices),
                ret);
            ret = linkLeft->PostFinAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]PostFinAck failed"), ret);
        }
        if (txCount > 0) {
            ret = linkRight->TxWaitDone(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Run][ReduceScatterNB]rank[%u] step[%u] blocknum[%u] tx wait done failed", rank, step, nSlices),
                ret);
            ret = linkRight->WaitFinAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]WaitFinAck failed"), ret);
        }
        if (linkRight->IsSpInlineReduce() || linkLeft->IsSpInlineReduce()) {
            // SDMA场景同步
            CHK_RET(ExecuteBarrier(linkLeft, linkRight));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::RunSrcReducerNB(
    const u32 step, const u32 nSlices, const u32 sliceSize, u32 txSliceIdx, const u32 deltaSliceIndex,
    const LINK linkRight, const u32 rank, const u32 rankSize, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    HcclResult ret = HCCL_SUCCESS;

    std::vector<Slice> txSlices;
    std::vector<Slice> txSlicestemp;
    for (u32 i = 0; i < nSlices; i++) {
        for (u32 j = 0; j < sliceSize; j++) {
            u32 txIndex = txSliceIdx * sliceSize + j;
            if (inputSlices[txIndex].size > 0) {
                txSlices.push_back(inputSlices[txIndex]);
                txSlicestemp.push_back(outputSlices[txIndex]);
            }
        }
        txSliceIdx = (txSliceIdx + deltaSliceIndex) % rankSize;
    }

    ret = RunSourceReducer(linkRight, txSlices, txSlicestemp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][ReduceScatterNB]rank[%u] step[%u] blocknum[%u] tx multi blocks failed", rank, step, nSlices),
        ret);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::RunDestReducerNB(
    const u32 step, const u32 nSteps, const u32 sliceSize, const u32 nSlices, u32 rxSliceIdx, const u32 deltaSliceIndex,
    const LINK linkLeft, const u32 rank, const u32 rankSize, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    HcclResult ret = HCCL_SUCCESS;

    if (step == (nSteps - 1)) {
        std::vector<ReducerMemoryInfo> rxReduceMems;
        for (u32 i = 0; i < nSlices; i++) {
            for (u32 j = 0; j < sliceSize; j++) {
                u32 rxIndex = rxSliceIdx * sliceSize + j;
                if (inputSlices[rxIndex].size > 0) {
                    DeviceMem dstMem = outputMem_.range(outputSlices[rxIndex].offset, outputSlices[rxIndex].size);
                    DeviceMem srcMem = inputMem_.range(inputSlices[rxIndex].offset, inputSlices[rxIndex].size);
                    DeviceMem scratchMem = scratchMem_.range(outputSlices[rxIndex].offset, outputSlices[rxIndex].size);
                    HCCL_DEBUG(
                        "final reduce rxSliceIdx[%u] will reduce with inputMem_ offset[%llu] to output_mem_ "
                        "offset[%llu] size[%llu]",
                        rxIndex, inputSlices[rxIndex].offset, outputSlices[rxIndex].offset, outputSlices[rxIndex].size);

                    rxReduceMems.emplace_back(
                        ReducerMemoryInfo{baseOffset_ + inputSlices[rxIndex].offset, srcMem, dstMem, scratchMem});
                }
            }
            rxSliceIdx = (rxSliceIdx + deltaSliceIndex) % rankSize;
        }

        ret = reducerInfo_->run(dispatcher_, linkLeft, rxReduceMems, stream_);
    } else {
        std::vector<Slice> rxSlices;
        std::vector<Slice> rxSlicestemp;
        for (u32 i = 0; i < nSlices; i++) {
            for (u32 j = 0; j < sliceSize; j++) {
                u32 rxIndex = rxSliceIdx * sliceSize + j;
                if (inputSlices[rxIndex].size > 0) {
                    rxSlices.push_back(inputSlices[rxIndex]);
                    rxSlicestemp.push_back(outputSlices[rxIndex]);
                }
            }
            rxSliceIdx = (rxSliceIdx + deltaSliceIndex) % rankSize;
        }

        ret = RunDestReducer(linkLeft, rxSlices, rxSlicestemp);
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][ReduceScatterNB]rank[%u] step[%u] blocknum[%u] rx multi blocks failed", rank, step, nSlices),
        ret);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::RunDestReducer(
    const LINK& link, const std::vector<Slice>& rxSlices, const std::vector<Slice>& rxSlicestemp)
{
    std::vector<ReducerMemoryInfo> rxReduceMems;

    for (u64 i = 0; i < rxSlices.size(); i++) {
        DeviceMem dstMem = inputMem_.range(rxSlices[i].offset, rxSlices[i].size);
        DeviceMem srcMemTemp = scratchMem_.range(rxSlicestemp[i].offset, rxSlicestemp[i].size);
        HCCL_DEBUG(
            "rcv offset[%llu], size[%llu] ,then reduce with "
            "offset[%llu] size[%llu] ",
            rxSlicestemp[i].offset, rxSlicestemp[i].size, rxSlices[i].offset, rxSlices[i].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSlices[i].offset, dstMem, dstMem, srcMemTemp});
    }
    CHK_RET(reducerInfo_->run(dispatcher_, link, rxReduceMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNB::RunSourceReducer(
    const LINK& link, const std::vector<Slice>& txSlices, const std::vector<Slice>& txSlicestemp)
{
    std::vector<SenderMemoryInfo> txMems;

    for (u64 i = 0; i < txSlices.size(); i++) {
        DeviceMem srcMem = inputMem_.range(txSlices[i].offset, txSlices[i].size);
        HCCL_DEBUG(
            " send inputmem range[%llu], size[%llu] tx dstmem offset[%llu]", txSlices[i].offset, txSlices[i].size,
            txSlicestemp[i].offset);
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSlicestemp[i].offset, srcMem});
    }
    CHK_RET(senderInfo_->run(link, txMems, stream_));
    return HCCL_SUCCESS;
}
HcclResult ReduceScatterNB::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (links.size() < rankSize) {
        return HCCL_SUCCESS;
    }
    u32 nSteps = 0;
    for (u32 temp = rankSize - 1; temp != 0; temp >>= 1, ++nSteps) {
    }

    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRank = 1 << step;
        u32 sendTo = (rank + deltaRank) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nSteps;
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_NB, ReduceScatterNB);
} // namespace hccl
