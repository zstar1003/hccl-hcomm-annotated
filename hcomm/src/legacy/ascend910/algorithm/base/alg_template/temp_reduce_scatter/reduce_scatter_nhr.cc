/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_nhr.h"
#include "alg_template_register.h"

namespace hccl {

ReduceScatterNHR::ReduceScatterNHR(const HcclDispatcher dispatcher) : NHRBase(dispatcher) {}

ReduceScatterNHR::~ReduceScatterNHR() {}

HcclResult ReduceScatterNHR::Prepare(u64 reduceAttrBitMap, bool needMerge)
{
    reduceAttr_ = reduceAttrBitMap;
    isNeedMerge = needMerge;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "[ReduceScatterNHR][RunAsync] rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (isNeedMerge == true) {
        // 获取tree映射，存储到类对象的成员变量中
        GetSliceMap(rankSize);
    }

    // 判断rank_size == 1
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[ReduceScatterNHR][RunAsync] rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    std::vector<Slice> outputSlices(slices_);

    // 处理和检查Slices
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        outputSlices.resize(rankSize);

        // 生成std::vector<Slice> slices_
        u64 sliceSize = count_ * unitSize;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);

            outputSlices[i].size = sliceSize;
            outputSlices[i].offset = (inputMem_.size() > outputMem_.size()) ? 0 : (i * sliceSize);
            HCCL_DEBUG(
                "[ReduceScatterNHR][RunAsync] rank[%u], slices[%u].offset=[%llu] slices[%u].size=[%llu] "
                "outputSlices[%u].offset=[%llu], outputSlices[%u].size=[%llu] count_[%llu] unitSize[%llu]",
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

    if (sliceMap_.size() != rankSize) {
        GetRankMapping(rankSize, true); // 没有初始化过，说明不是由allreduce或者bcast调入，需要保序
    }

    // 运行reduce-scatter, NHR 算法
    CHK_RET(RunReduceScatterNHR(rank, rankSize, links, slices_, outputSlices));

    HCCL_INFO("[ReduceScatterNHR][RunAsync] ReduceScatterNHR finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

void ReduceScatterNHR::GetSliceMap(const u32 rankSize)
{
    std::vector<u32> tree;
    for (u32 i = 0; i < rankSize; i++) {
        tree.push_back(i);
    }

    // 其他的再进行计算
    std::vector<u32> tmp(rankSize);
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    u32 len = rankSize;

    for (u32 step = 0; step < nSteps; step++) {
        u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
        if (nSlices <= 1) {
            break;
        }

        bool endFlag = false;

        for (u32 part = 0; part * len < rankSize; part++) {
            u32 start = part * len;
            u32 end = std::min(start + len, rankSize);
            Reorder(start, end, len, tree, tmp);

            if (((end - start) & 1) == 1) {
                endFlag = true;
            }
        }

        for (u32 i = 0; i < rankSize; i++) {
            tree[i] = tmp[i];
        }

        if (endFlag) {
            break;
        }

        len >>= 1;
    }

    // 因为取的是tree中rank的idx，所以直接返回反向的映射
    sliceMap_.resize(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        sliceMap_[tree[i]] = i;
    }

    return;
}

void ReduceScatterNHR::Reorder(u32 start, u32 end, u32 len, std::vector<u32>& tree, std::vector<u32>& tmp)
{
    const u32 idxTwo = 2;

    for (u32 i = start; i < end; i++) {
        u32 offset = i - start;
        if ((offset & 1) == 0) {
            tmp[start + offset / idxTwo] = tree[i];
        } else {
            tmp[start + (offset + len) / idxTwo] = tree[i];
        }
    }
}

HcclResult ReduceScatterNHR::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_,
        HCCL_ERROR("[ReduceScatterNHR][RunAsync] rank[%u] inputmem or outputmem is null", rank), HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[ReduceScatterNHR][RunAsync] rank[%u] link size[%llu] is "
            "less than rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::CheckSlices(const std::vector<Slice>& checkSlices, const u32 rankSize)
{
    CHK_PRT_RET(
        checkSlices.size() % rankSize != 0,
        HCCL_ERROR(
            "[ReduceScatterNHR][RunAsync] slices.size[%u] should be divided by rankSize[%u]", checkSlices.size(),
            rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::InlineReducer(const LINK& linkLeft, const std::vector<ReducerMemoryInfo>& rxReduceMems)
{
    HcclResult ret = HCCL_SUCCESS;
    void* remoteMem = nullptr;
    CHK_RET(linkLeft->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
    for (ReducerMemoryInfo reduceMem : rxReduceMems) {
        const u64 dataBytes = reduceMem.remoteRcvTemp.size();
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<s8*>(remoteMem) + reduceMem.remoteMemOffset, dataBytes / SIZE_TABLE[dataType_],
            dataType_, reductionOp_, stream_, reduceMem.localsrc.ptr(), linkLeft->GetRemoteRank(),
            linkLeft->GetLinkType(), INLINE_REDUCE_BIT));

        if (reduceMem.localsrc != reduceMem.localdst) {
            ret = HcclD2DMemcpyAsync(dispatcher_, reduceMem.localdst, reduceMem.localsrc, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Reducer][Run]memcpy_async localSrc[%p] localDst[%p] failed", reduceMem.localsrc.ptr(),
                    reduceMem.localdst.ptr()),
                ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterNHR::InlineReduceRx(const LINK& linkLeft, std::vector<Slice>& rxSlices, std::vector<Slice>& rxSlicestemp)
{
    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u64 i = 0; i < rxSlices.size(); i++) {
        DeviceMem dstMem = inputMem_.range(rxSlices[i].offset, rxSlices[i].size);
        DeviceMem srcMemTemp = scratchMem_.range(rxSlicestemp[i].offset, rxSlicestemp[i].size);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunDestReducer] rcv offset[%llu], size[%llu] ,then reduce with "
            "offset[%llu] size[%llu] ",
            rxSlicestemp[i].offset, rxSlicestemp[i].size, rxSlices[i].offset, rxSlices[i].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSlices[i].offset, dstMem, dstMem, srcMemTemp});
    }
    CHK_RET(InlineReducer(linkLeft, rxReduceMems));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::InlineReduceRxLastStep(
    const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u32 i = 0; i < stepInfo.nSlices; i++) { // rst算法的reduce scatter最后一步是一个slice，暂不用合并
        u32 rxSliceIdx = stepInfo.rxSliceIdxs[i];
        DeviceMem dstMem = outputMem_.range(outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);
        DeviceMem srcMem = inputMem_.range(inputSlices[rxSliceIdx].offset, inputSlices[rxSliceIdx].size);
        DeviceMem tmpMem = scratchMem_.range(outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunReduceScatterNHR] final reduce rxSliceIdx[%u] will reduce with "
            "inputMem_ offset[%llu] to output_mem_ offset[%llu] size[%llu]",
            rxSliceIdx, inputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);

        rxReduceMems.emplace_back(
            ReducerMemoryInfo{baseOffset_ + inputSlices[rxSliceIdx].offset, srcMem, dstMem, tmpMem});
    }
    CHK_RET(InlineReducer(linkLeft, rxReduceMems));
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterNHR::TbeReduceRx(const LINK& linkLeft, std::vector<Slice>& rxSlices, std::vector<Slice>& rxSlicestemp)
{
    void* srcMemPtr = nullptr;
    CHK_RET(linkLeft->GetRemoteMem(UserMemType::INPUT_MEM, &srcMemPtr));
    std::vector<RxWithReduceMemoryInfo> rxWithReduceMems;
    for (u64 i = 0; i < rxSlices.size(); i++) {
        DeviceMem dstMem = inputMem_.range(rxSlices[i].offset, rxSlices[i].size);
        DeviceMem srcMem(static_cast<s8*>(srcMemPtr) + baseOffset_ + rxSlices[i].offset, rxSlices[i].size);
        DeviceMem dstMemScratch = scratchMem_.range(rxSlicestemp[i].offset, rxSlicestemp[i].size);
        u64 dataCount = dstMem.size() / SIZE_TABLE[dataType_];
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunDestReducer] rcv offset[%llu], size[%llu] ,then reduce with "
            "offset[%llu] size[%llu] ",
            rxSlicestemp[i].offset, rxSlicestemp[i].size, rxSlices[i].offset, rxSlices[i].size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstMemScratch, srcMem, stream_,
            linkLeft->GetRemoteRank(), // left的inputMem拷到本端的scratchMem
            linkLeft->GetLinkType()));
        rxWithReduceMems.emplace_back(RxWithReduceMemoryInfo{
            UserMemType::INPUT_MEM, baseOffset_ + rxSlices[i].offset, dstMemScratch.ptr(), dstMemScratch.size(),
            dstMemScratch.ptr(), dstMem.ptr(), dataCount});
    }
    for (RxWithReduceMemoryInfo rxReduceMem : rxWithReduceMems) {
        CHK_RET(HcclReduceAsync(
            dispatcher_, rxReduceMem.reduceSrc, rxReduceMem.reduceDataCount,
            dataType_, // 本端scratchMem localReduce到 本端inputMem
            reductionOp_, stream_, rxReduceMem.reduceDst, INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::TbeReduceRxLastStep(
    const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    void* srcMemPtr = nullptr;
    CHK_RET(linkLeft->GetRemoteMem(UserMemType::INPUT_MEM, &srcMemPtr));
    std::vector<RxWithReduceMemoryInfo> rxWithReduceMems;
    for (u32 i = 0; i < stepInfo.nSlices; i++) { // rst算法的reduce scatter最后一步是一个slice，暂不用合并
        u32 rxSliceIdx = stepInfo.rxSliceIdxs[i];
        DeviceMem srcMemRemote(
            static_cast<s8*>(srcMemPtr) + baseOffset_ + inputSlices[rxSliceIdx].offset,
            inputSlices[rxSliceIdx].size); // 对端inputMem
        DeviceMem dstMem
            = outputMem_.range(outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size); // 本端outputMem
        DeviceMem srcMem
            = inputMem_.range(inputSlices[rxSliceIdx].offset, inputSlices[rxSliceIdx].size); // 本端inputMem
        DeviceMem tmpMem
            = scratchMem_.range(outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size); // 本端scratchMem
        u64 dataCount = dstMem.size() / SIZE_TABLE[dataType_];
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunReduceScatterNHR] final reduce rxSliceIdx[%u] will reduce with "
            "inputMem_ offset[%llu] to output_mem_ offset[%llu] size[%llu]",
            rxSliceIdx, inputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, tmpMem, srcMemRemote, stream_, linkLeft->GetRemoteRank(), // left的inputMem拷到本端的scratchMem
            linkLeft->GetLinkType()));
        DeviceMem reduceSrc = (srcMem == dstMem) ? tmpMem : srcMem;
        rxWithReduceMems.emplace_back(RxWithReduceMemoryInfo{
            UserMemType::INPUT_MEM, baseOffset_ + inputSlices[rxSliceIdx].offset, tmpMem.ptr(), tmpMem.size(),
            reduceSrc.ptr(), dstMem.ptr(), dataCount});
    }
    for (RxWithReduceMemoryInfo rxReduceMem : rxWithReduceMems) {
        CHK_RET(HcclReduceAsync(
            dispatcher_, rxReduceMem.reduceSrc, rxReduceMem.reduceDataCount,
            dataType_, // 本端inputMem localReduce到 本端outputMem(之前拷到本端scratch的数据呢？)
            reductionOp_, stream_, rxReduceMem.reduceDst, INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::RunDestReducerLastStep(
    const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    HcclResult ret = HCCL_SUCCESS;
    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u32 i = 0; i < stepInfo.nSlices; i++) { // rst算法的reduce scatter最后一步是一个slice，暂不用合并
        u32 rxSliceIdx = stepInfo.rxSliceIdxs[i];
        DeviceMem dstMem = outputMem_.range(outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);
        DeviceMem srcMem = inputMem_.range(inputSlices[rxSliceIdx].offset, inputSlices[rxSliceIdx].size);
        DeviceMem tmpMem = scratchMem_.range(outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunReduceScatterNHR] final reduce rxSliceIdx[%u] will reduce with "
            "inputMem_ offset[%llu] to output_mem_ offset[%llu] size[%llu]",
            rxSliceIdx, inputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].offset, outputSlices[rxSliceIdx].size);

        rxReduceMems.emplace_back(
            ReducerMemoryInfo{baseOffset_ + inputSlices[rxSliceIdx].offset, srcMem, dstMem, tmpMem});
    }

    ret = reducerInfo_->run(dispatcher_, linkLeft, rxReduceMems, stream_);
    return ret;
}

HcclResult ReduceScatterNHR::GetRxSlices(
    std::vector<Slice>& rxSlices, std::vector<Slice>& rxSlicestemp, InterServerAlgoStep& stepInfo,
    const std::vector<Slice>& inputSlices, const std::vector<Slice>& outputSlices)
{
    for (u32 i = 0; i < stepInfo.nSlices; i++) {
        rxSlices.push_back(inputSlices[stepInfo.rxSliceIdxs[i]]);
        rxSlicestemp.push_back(outputSlices[stepInfo.rxSliceIdxs[i]]);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunDestReducer] i[%u] rxSliceIndex[%u] rx offset[%llu] size[%llu]", i,
            stepInfo.rxSliceIdxs[i], outputSlices[stepInfo.rxSliceIdxs[i]].offset,
            outputSlices[stepInfo.rxSliceIdxs[i]].size);
    }

    HCCL_DEBUG(
        "[ReduceScatterNHR][RunDestReducer] rxslices size [%u], rxslices temp size [%u]", rxSlices.size(),
        rxSlicestemp.size());

    // 合并连续slices
    MergeSlices(rxSlices);
    MergeSlices(rxSlicestemp);
    HCCL_DEBUG(
        "[ReduceScatterNHR][RunDestReducer] merged rxslices size [%u], merged rxslices temp size [%u]", rxSlices.size(),
        rxSlicestemp.size());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::SdmaReducer(
    const u32 nSteps, const LINK& linkLeft, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    HcclResult ret = HCCL_SUCCESS;
    std::vector<Slice> rxSlices;
    std::vector<Slice> rxSlicestemp;
    if ((INLINE_REDUCE_BITMASK & reduceAttr_) == 1) { // InlineReduce
        if (stepInfo.step == (nSteps - 1)) {
            ret = InlineReduceRxLastStep(linkLeft, stepInfo, inputSlices, outputSlices);
        } else {
            CHK_RET(GetRxSlices(rxSlices, rxSlicestemp, stepInfo, inputSlices, outputSlices));
            ret = InlineReduceRx(linkLeft, rxSlices, rxSlicestemp);
        }
    } else { // TbeReduce
        if (stepInfo.step == (nSteps - 1)) {
            ret = TbeReduceRxLastStep(linkLeft, stepInfo, inputSlices, outputSlices);
        } else {
            CHK_RET(GetRxSlices(rxSlices, rxSlicestemp, stepInfo, inputSlices, outputSlices));
            ret = TbeReduceRx(linkLeft, rxSlices, rxSlicestemp);
        }
    }
    return ret;
}

HcclResult ReduceScatterNHR::RunReduceScatterNHR(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    bool bRetSize = (inputSlices.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[ReduceScatterNHR][RunReduceScatterNHR] rank[%u] inputslice size[%llu] is less "
            "than rank size[%u]",
            rank, outputSlices.size(), rankSize),
        HCCL_E_INTERNAL);

    bRetSize = (outputSlices.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[ReduceScatterNHR][RunReduceScatterNHR] rank[%u] outputslice size[%llu] is less "
            "than rank size[%u]",
            rank, outputSlices.size(), rankSize),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;

    // 计算通信步数
    u32 nSteps = GetStepNumInterServer(rankSize);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        InterServerAlgoStep stepInfo;
        GetStepInfo(step, nSteps, rank, rankSize, stepInfo);

        // 链的关系没有变化，区别的是发送的slice编号，因为重排tree不影响每棵树节点间的连接关系
        LINK linkLeft = links[stepInfo.fromRank];
        CHK_SMART_PTR_NULL(linkLeft);

        LINK linkRight = links[stepInfo.toRank];
        CHK_SMART_PTR_NULL(linkRight);

        // 当前每个数据块发送一次ACK、reduce一次、同步一次
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunReduceScatterNHR] rank[%u] rankSize[%u] from[%u] to[%u] step[%u] nSteps[%u] "
            "nSlices[%u]",
            rank, rankSize, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        if (linkLeft->IsSpInlineReduce() && linkRight->IsSpInlineReduce()) { // SDMA
            CHK_RET(linkRight->TxAck(stream_));
            CHK_RET(linkLeft->RxAck(stream_));
            CHK_RET(SdmaReducer(nSteps, linkLeft, stepInfo, inputSlices, outputSlices));
            CHK_RET(linkLeft->TxDataSignal(stream_));  // 告知left我读完了
            CHK_RET(linkRight->RxDataSignal(stream_)); // 等right读完
        } else {                                       // RDMA
            CHK_RET(linkLeft->TxAck(stream_));
            CHK_RET(linkRight->RxAck(stream_));
            // tx
            ret = RunSourceSender(linkRight, stepInfo, inputSlices, outputSlices);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterNHR][RunReduceScatterNHR] Tx failed"), ret);

            // rx
            if (step == (nSteps - 1)) {
                ret = RunDestReducerLastStep(linkLeft, stepInfo, inputSlices, outputSlices);
            } else {
                ret = RunDestReducer(linkLeft, stepInfo, inputSlices, outputSlices);
            }

            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterNHR][RunReduceScatterNHR] Rx failed"), ret);
            ret = linkLeft->PostFinAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterNHR][RunReduceScatterNHR] PostFinAck failed"), ret);

            ret = linkRight->WaitFinAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterNHR][RunReduceScatterNHR] WaitFinAck failed"), ret);

            if (barrierSwitchOn_) {
                CHK_RET(ExecuteBarrier(linkLeft, linkRight));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::RunSourceSender(
    const LINK& link, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    std::vector<Slice> txSlices;
    std::vector<Slice> txSlicestemp;
    for (u32 i = 0; i < stepInfo.nSlices; i++) {
        txSlices.push_back(inputSlices[stepInfo.txSliceIdxs[i]]);
        txSlicestemp.push_back(outputSlices[stepInfo.txSliceIdxs[i]]);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunSourceSender] i[%u] txSliceIndex[%u] tx data offset[%llu] size[%llu]", i,
            stepInfo.txSliceIdxs[i], outputSlices[stepInfo.txSliceIdxs[i]].offset,
            outputSlices[stepInfo.txSliceIdxs[i]].size);
    }
    HCCL_DEBUG(
        "[ReduceScatterNHR][RunSourceSender] txSlices size [%u], txSlices temp size [%u]", txSlices.size(),
        txSlicestemp.size());

    // 合并连续slices
    MergeSlices(txSlices);
    MergeSlices(txSlicestemp);
    HCCL_DEBUG(
        "[ReduceScatterNHR][RunSourceSender] merged txSlices size [%u], merged txSlices temp size [%u]",
        txSlices.size(), txSlicestemp.size());

    std::vector<SenderMemoryInfo> txMems;
    for (u64 i = 0; i < txSlices.size(); i++) {
        DeviceMem srcMem = inputMem_.range(txSlices[i].offset, txSlices[i].size);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunSourceSender] send inputmem range[%llu], size[%llu] tx dstmem offset[%llu]",
            txSlices[i].offset, txSlices[i].size, txSlicestemp[i].offset);
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSlicestemp[i].offset, srcMem});
    }

    CHK_RET(senderInfo_->run(link, txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::RunDestReducer(
    const LINK& link, InterServerAlgoStep& stepInfo, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    std::vector<Slice> rxSlices;
    std::vector<Slice> rxSlicestemp;
    CHK_RET(GetRxSlices(rxSlices, rxSlicestemp, stepInfo, inputSlices, outputSlices));

    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u64 i = 0; i < rxSlices.size(); i++) {
        DeviceMem dstMem = inputMem_.range(rxSlices[i].offset, rxSlices[i].size);
        DeviceMem srcMemTemp = scratchMem_.range(rxSlicestemp[i].offset, rxSlicestemp[i].size);
        HCCL_DEBUG(
            "[ReduceScatterNHR][RunDestReducer] rcv offset[%llu], size[%llu] ,then reduce with "
            "offset[%llu] size[%llu] ",
            rxSlicestemp[i].offset, rxSlicestemp[i].size, rxSlices[i].offset, rxSlices[i].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSlices[i].offset, dstMem, dstMem, srcMemTemp});
    }

    CHK_RET(reducerInfo_->run(dispatcher_, link, rxReduceMems, stream_));
    return HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult ReduceScatterNHR::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo)
{
    (void)nSteps;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    u32 sliceSize = slices_.size() / rankSize;
    stepInfo.step = step;
    stepInfo.myRank = rank;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (rank + rankSize - deltaRank) % rankSize;
    u32 recvFrom = (rank + deltaRank) % rankSize;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 txSliceIdx = sendTo; // 第一片rank
    u32 rxSliceIdx = rank;

    for (u32 i = 0; i < nSlices; i++) {
        for (u32 j = 0; j < sliceSize; j++) {
            u32 targetTxSliceIdx = sliceMap_[txSliceIdx];
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx * sliceSize + j);

            u32 targetRxSliceIdx = sliceMap_[rxSliceIdx];
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx * sliceSize + j);

            HCCL_DEBUG(
                "[ReduceScatterNHR][GetStepInfo] i[%u] txSliceIdx[%u]->targetTxSliceIdx[%u] rxSliceIdx[%u]->"
                "targetRxSliceIdx[%u]",
                i, txSliceIdx, targetTxSliceIdx, rxSliceIdx, targetRxSliceIdx);
        }
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }

    stepInfo.nSlices = nSlices * sliceSize;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHR::GetNslbAdjInfo(
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
        u32 sendTo = (rank + rankSize - deltaRank) % rankSize;
        ;
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
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_NHR, ReduceScatterNHR);
} // namespace hccl
