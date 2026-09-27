/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_nhr_v1.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherNHRV1::AllGatherNHRV1(const HcclDispatcher dispatcher) : NHRV1Base(dispatcher) {}

AllGatherNHRV1::~AllGatherNHRV1() {}

// 服务器间allgather的入口函数
HcclResult AllGatherNHRV1::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "[AllGatherNHRV1] run_async rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 判断rank_size == 1
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        links.size() < rankSize, HCCL_ERROR("[AllGatherNHRV1][RunAsync]rank[%u] linkSize is less than rankSize", rank),
        HCCL_E_INTERNAL);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[AllGatherNHRV1][RunAsync]unitSize is zero"), HCCL_E_INTERNAL);

    // 处理和检查Slices
    if (slices_.size() != 0) {
        HCCL_WARNING("[AllGatherNHRV1][RunAsync]AllGatherNHRV1 not supported passing in parameter slice_, "
                     "otherwise will be cleared");
        slices_.clear();
    }
    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices.resize(rankSize);
        u64 sliceSize = count_ * unitSize;
        HCCL_DEBUG("[AllGatherNHRV1][RunAsync]sliceSize is %llu, rankSize is %u", sliceSize, rankSize);
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG(
                "rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, i, slices_[i].offset, i,
                slices_[i].size);
        }
    }

    // 双buffer下, 先将input拷贝到output的合适位置
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(inputSlices[rank].offset, inputSlices[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    HcclResult ret;
    // 获取通信关系
    RingInfo info = GetRingInfo(rankSize);

    // 水平方向做ring
    ret = RunAllGatherOnHorizontal(rank, links, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherNHRV1][RunAsync]rank[%u] count[%llu] failed in "
            "RunAllGatherOnHorizontal step",
            rank, count_),
        ret);

    // 垂直方向做ring
    ret = RunAllGatherOnVertical(rank, links, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherNHRV1][RunAsync]rank[%u] count[%llu] failed in "
            "RunAllGatherOnVertical step",
            rank, count_),
        ret);

    HCCL_INFO("[AllGatherNHRV1] finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRV1::RunAllGatherOnHorizontal(u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 ringRank = info.GetHIndex(rank);      // 查找自己位于第几列，也即处于Ring中的第几个rank
    u32 ringSize = info.GetHSizeByRank(rank); // 查找自己所处的行长度，也即Ring的大小
    u32 vIndex = info.GetVIndex(rank);        // 查找自己位于第几行

    // 收集本列各rank号，构建新的links、slices数组
    std::vector<Slice> hSlices;
    std::vector<LINK> hLinks;
    for (u32 hIdx = 0; hIdx < ringSize; hIdx++) {
        u32 oldRank = info.GetRank(vIndex, hIdx);
        CHK_PRT_RET(
            oldRank >= links.size(),
            HCCL_ERROR(
                "[AllGatherNHRV1] rank[%u] out of range, "
                "oldRank=%u, links.size=%u",
                rank, oldRank, links.size()),
            HCCL_E_INTERNAL);
        hSlices.push_back(slices_[oldRank]);
        hLinks.push_back(links[oldRank]);
    }

    // 长度不足2，直接跳过
    if (hLinks.size() < 2) {
        return HCCL_SUCCESS;
    }

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    HCCL_INFO(
        "rank[%u] tempAlg AllGathering inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), inputMem_.size(), count_, profilerInput_.planeID);

    // 判断是否关闭AllGather的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }

    // 调用AllGather ring的算法执行
    CHK_RET(tempAlg->Prepare(
        outputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, hSlices, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    HCCL_DEBUG("[AllGatherNHRV1][Horizontal] rank[%u], ringRank=%u, ringSize=%u", rank, ringRank, ringSize);
    return tempAlg->RunAsync(ringRank, ringSize, hLinks);
}

HcclResult AllGatherNHRV1::RunAllGatherOnVertical(u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 hIndex = info.GetHIndex(rank); // 查找自己位于第几列

    u32 hIndexForRing = (hIndex < info.GetRowSize()) ? hIndex : info.GetVIndex(rank); // 属于第几个垂直Ring
    u32 vSizeForRing = info.GetVSizeByHIndex(hIndexForRing);                          // 所属垂直Ring的大小

    // 收集本列各rank号，构建新的links、slices数组
    std::vector<LINK> vLinks;
    std::vector<Slice> vSlices;
    for (u32 vIdx = 0; vIdx < vSizeForRing; vIdx++) {
        u32 oldLRank = info.GetRank(vIdx, hIndexForRing);
        CHK_PRT_RET(
            oldLRank >= links.size(),
            HCCL_ERROR(
                "[AllGatherNHRV1] rank[%u] out of range, "
                "oldLRank=%u, links.size=%u",
                rank, oldLRank, links.size()),
            HCCL_E_INTERNAL);
        vLinks.push_back(links[oldLRank]);
        Slice slice;
        slice.size = slices_[vIdx].size * info.GetHSizeByVIndex(vIdx);
        u32 oldSRank = info.GetRank(vIdx, 0);
        CHK_PRT_RET(
            oldSRank >= links.size(),
            HCCL_ERROR(
                "[AllGatherNHRV1] rank[%u] out of range, "
                "oldSRank=%u, links.size=%u",
                rank, oldSRank, links.size()),
            HCCL_E_INTERNAL);
        slice.offset = slices_[oldSRank].offset;
        vSlices.push_back(slice);
    }

    // -- 可能还涉及跳跃的一个链接，比如8节点
    // ---- 0   1   2
    // ---- 3   4   5
    // ---- 6   7
    // -- 两个垂直Ring分别是{0,3,6,2}和{1,4,7,5}，而不是{0,3,6}和{1,4,7}
    if (info.GetHSizeByVIndex(hIndexForRing) > info.GetRowSize()) {
        u32 oldLRank = info.GetRank(hIndexForRing, info.GetSqrtRankSize());
        CHK_PRT_RET(
            oldLRank >= links.size(),
            HCCL_ERROR(
                "[AllGatherNHRV1] rank[%u] out of range, "
                "oldLRank=%u, links.size=%u",
                rank, oldLRank, links.size()),
            HCCL_E_INTERNAL);
        vLinks.push_back(links[oldLRank]);
        Slice slice;
        slice.offset = 0;
        slice.size = 0;
        vSlices.push_back(slice);
    }

    // 长度不足2，直接跳过
    if (vLinks.size() < 2) {
        return HCCL_SUCCESS;
    }

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    HCCL_INFO(
        "rank[%u] tempAlg AllGathering inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), inputMem_.size(), count_, profilerInput_.planeID);
    // 判断是否关闭allgather的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    // 调用allgather ring的算法执行
    CHK_RET(tempAlg->Prepare(
        outputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, vSlices, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    // 计算在垂直Ring中的rank号
    u32 subRank = (hIndex == hIndexForRing) ? info.GetVIndex(rank) : vSizeForRing;

    HCCL_DEBUG("[AllGatherNHR][Vertical] rank[%u], subRank=%u, ringSize=%u", rank, subRank, vLinks.size());
    return tempAlg->RunAsync(subRank, vLinks.size(), vLinks);
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_NHRV1, AllGatherNHRV1);
} // namespace hccl
