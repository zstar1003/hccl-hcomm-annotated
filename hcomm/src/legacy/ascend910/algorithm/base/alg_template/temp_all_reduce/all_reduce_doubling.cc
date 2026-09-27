/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <math.h>
#include "alg_template_register.h"
#include "all_reduce_doubling.h"

namespace hccl {

// Doubling算法实现AllReduce，只用于server内通信
AllReduceDoubling::AllReduceDoubling(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceDoubling::~AllReduceDoubling() {}

HcclResult AllReduceDoubling::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoubling::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "[AllReduceDoubling] runAsync rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));

    // 判断rank_size == 1
    if (rankSize == 1) {
        // 对于Doubling，input和output必须是两块不同的内存
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        return HCCL_SUCCESS;
    }

    // 设置Slices
    if (slices_.size() != 0) {
        HCCL_WARNING("[AllReduceDoubling] slices_ will be not used in executor.");
    }

    // 执行算法
    CHK_RET(RunAllReduce(rank, rankSize, links));

    HCCL_INFO("AllReduceDoubling finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoubling::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 判断Memory是否为空
    CHK_PRT_RET(!inputMem_, HCCL_ERROR("[AllReduceDoubling] rank[%u] inputmem is null", rank), HCCL_E_PTR);
    CHK_PRT_RET(!outputMem_, HCCL_ERROR("[AllReduceDoubling] rank[%u] outputmem is null", rank), HCCL_E_PTR);

    // 必须有两块memory
    CHK_PRT_RET(
        inputMem_ == outputMem_,
        HCCL_ERROR("[AllReduceDoubling] rank[%u] inputMem and outputMem should be different", rank), HCCL_E_PARA);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[AllReduceDoubling] rank[%u] link size[%llu] is less than "
            "rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_PARA);

    // 判断rankSize是否为2的幂次
    CHK_PRT_RET(
        (rankSize & (rankSize - 1)) != 0,
        HCCL_ERROR(
            "[AllReduceDoubling] rankSize must be power of 2, "
            "but get rankSize=%u",
            rankSize),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoubling::RunAllReduce(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem cclInMem = inputMem_.range(0, totalSize);
    DeviceMem cclOutMem = outputMem_.range(0, totalSize);

    u32 nSteps = static_cast<u32>(log2(rankSize));
    for (u32 step = 0; step < nSteps; step++) {
        // 计算邻居并获取link
        u32 neighbor = rank ^ (1 << step);
        const LINK& link = links[neighbor];
        CHK_PTR_NULL(link);

        // 拷贝数据，避免读写冲突
        if (step == 0) { // 把本端的cclIn拷到cclOut（cclIn是整个template的入口）
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, cclOutMem, cclInMem, stream_));
        } else { // 把本端的cclOut拷到cclIn（cclOut是上一步的结果）
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, cclInMem, cclOutMem, stream_));
        }

        // Ack
        CHK_RET(link->TxAck(stream_));
        CHK_RET(link->RxAck(stream_));

        // 从对端的cclIn读到本端的cclOut
        void* remMemPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
        DeviceMem remoteCclInMem = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
        CHK_RET(HcclReduceAsync(
            dispatcher_, remoteCclInMem.ptr(), count_, dataType_, reductionOp_, stream_, cclOutMem.ptr(),
            link->GetRemoteRank(), link->GetLinkType(), INLINE_REDUCE_BIT));

        // DataSignal
        CHK_RET(link->TxDataSignal(stream_));
        CHK_RET(link->RxDataSignal(stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING, AllReduceDoubling);
} // namespace hccl
