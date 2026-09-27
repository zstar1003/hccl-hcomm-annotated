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
#include "all_reduce_doubling_direct.h"

namespace hccl {

// Doubling算法实现AllReduce，只用于server内通信
AllReduceDoublingDirect ::AllReduceDoublingDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceDoublingDirect::~AllReduceDoublingDirect() {}

HcclResult AllReduceDoublingDirect::Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    opInfo_ = opInfo;
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoublingDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "[AllReduceDoublingDirect] runAsync rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));

    // 判断 ranksize == 1 场景，把数据从userIn直接拷到userOut
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, totalSize);
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);
    if (rankSize == 1) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_));
        return HCCL_SUCCESS;
    }

    // 设置Slices
    if (slices_.size() != 0) {
        HCCL_WARNING("[AllReduceDoublingDirect] slices_ will be not used in executor.");
    }

    // 执行算法
    CHK_RET(RunAllReduce(rank, rankSize, links));

    HCCL_INFO("[AllReduceDoublingDirect] finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoublingDirect::RunAllReduce(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    u32 nSteps = static_cast<u32>(log2(rankSize));
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem commMemIn = DeviceMem::create(inputMem_.ptr(), totalSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), totalSize);
    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, totalSize);
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);

    DeviceMem src;
    DeviceMem dst;

    // 第一步：把本端的数据从userIn拷到cclIn
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMemIn, userMemIn, stream_));

    u32 neighbor = rank ^ (1 << 0);
    CHK_PTR_NULL(links[neighbor]);

    const u32 ONE_STEPS = 1;
    if (nSteps == ONE_STEPS) {
        // 把本端的数据从userIn拷到userOut
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_));
        // 把数据从远端的cclIn读到本端的userOut
        void* remMemPtr = nullptr;
        CHK_RET(links[neighbor]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
        CHK_RET(RunInlineReduce(userMemOut, src, links[neighbor]));
        return HCCL_SUCCESS;
    }
    // 把数据从本端的userIn写到对端的cclIn
    void* remMemPtr = nullptr;
    CHK_RET(links[neighbor]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
    dst = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
    CHK_RET(RunInlineReduce(dst, userMemIn, links[neighbor]));

    // 只需要一块cclbuffer
    const u32 TWO_STEPS = 2;
    if (nSteps == TWO_STEPS) {
        // 把数据从本端的cclIn拷到本端的userOut
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, commMemIn, stream_));

        // 从远端的cclIn读到本端的userOut
        neighbor = rank ^ (1 << 1);
        CHK_PTR_NULL(links[neighbor]);
        CHK_RET(links[neighbor]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
        CHK_RET(RunInlineReduce(userMemOut, src, links[neighbor]));
    }
    // 必须有两块memory
    const u32 THREE_STEPS = 3;
    if (nSteps == THREE_STEPS) {
        // 把数据从本端的cclIn拷到本端的cclOut
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMemOut, commMemIn, stream_));

        // 从远端的cclIn读到本端的cclOut
        neighbor = rank ^ (1 << 1);
        CHK_PTR_NULL(links[neighbor]);
        CHK_RET(links[neighbor]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
        CHK_RET(RunInlineReduce(commMemOut, src, links[neighbor]));

        // 把数据从本端的cclOut拷到本端的userOut
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, commMemOut, stream_));

        // 最后一步：从远端的cclOut读到本端的userOut
        const u32 LAST_STEPS = 2;
        neighbor = rank ^ (1 << LAST_STEPS);
        CHK_PTR_NULL(links[neighbor]);
        CHK_RET(links[neighbor]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
        CHK_RET(RunInlineReduce(userMemOut, src, links[neighbor]));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoublingDirect::RunInlineReduce(hccl::DeviceMem& dst, const hccl::DeviceMem& src, const LINK& link)
{
    CHK_RET(link->TxAck(stream_));
    CHK_RET(link->RxAck(stream_));

    CHK_RET(HcclReduceAsync(
        dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, stream_,
        static_cast<void*>(dst.ptr()), link->GetRemoteRank(), link->GetLinkType(), INLINE_REDUCE_BIT));

    CHK_RET(link->TxDataSignal(stream_));
    CHK_RET(link->RxDataSignal(stream_));

    HCCL_INFO("[AllReduceDoublingDirect] RunInlineReduce finished");
    return HCCL_SUCCESS;
}

HcclResult AllReduceDoublingDirect::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 当前只支持ranksize <= 8
    CHK_PRT_RET(rankSize > 8, HCCL_ERROR("[AllReduceDoublingDirect] only support rankSize <= 8"), HCCL_E_PTR);

    // 判断Memory是否为空
    CHK_PRT_RET(!inputMem_, HCCL_ERROR("[AllReduceDoublingDirect] rank[%u] inputmem is null", rank), HCCL_E_PTR);
    CHK_PRT_RET(!outputMem_, HCCL_ERROR("[AllReduceDoublingDirect] rank[%u] outputmem is null", rank), HCCL_E_PTR);
    CHK_PRT_RET(
        inputMem_ == outputMem_, HCCL_ERROR("rank[%u] inputMem and outputMem should be different", rank), HCCL_E_PARA);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[AllReduceDoublingDirect] rank[%u] link size[%llu] is less than "
            "rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_PARA);

    // 判断rankSize是否为2的幂次
    CHK_PRT_RET(
        (rankSize & (rankSize - 1)) != 0,
        HCCL_ERROR(
            "[AllReduceDoublingDirect] rankSize must be power of 2, "
            "but get rankSize=%u",
            rankSize),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING_DIRECT, AllReduceDoublingDirect);
} // namespace hccl
