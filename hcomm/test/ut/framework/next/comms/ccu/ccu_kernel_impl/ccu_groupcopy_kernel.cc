/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_groupcopy_demo.h"

static CcuResult AgInitResource(AllGatherContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        return CCU_E_PARA;
    }

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], AG_OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], AG_TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.resourceAllocated = false;
    ctx.groupCopyRegistered = false;
    return CCU_SUCCESS;
}

static CcuResult AgLoadArgs(AllGatherContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tmpRepeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    return CCU_SUCCESS;
}

static CcuResult AgPreSync(AllGatherContext& ctx)
{
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.output[arg->rankId], AG_OUTPUT_XN_ID, AG_CKE_IDX_0, 1 << AG_OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], AG_TOKEN_XN_ID, AG_CKE_IDX_0, 1 << AG_TOKEN_XN_ID));
    }

    const uint32_t allBit = (1 << AG_OUTPUT_XN_ID) | (1 << AG_TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], AG_CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult AgPostSync(AllGatherContext& ctx)
{
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], AG_CKE_IDX_0, 1 << AG_POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], AG_CKE_IDX_0, 1 << AG_POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult AgDoAllGather(
    AllGatherContext& ctx, const ccu::LocalAddr& src, std::vector<ccu::RemoteAddr>& dst, const ccu::Variable& sliceSize)
{
    const auto* arg = ctx.arg;
    uint32_t channelId = 0;

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        const uint16_t rankMask = 1 << rankIdx;
        if (rankIdx == arg->rankId) {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
        } else {
            CCU_CHK_RET(ccu::Write(arg->channels[channelId], dst[rankIdx], src, sliceSize, ctx.event, rankMask));
            channelId++;
        }
    }

    CCU_IF(ctx.isInputOutputEqual == 0) { CCU_CHK_RET(AgGroupCopy(ctx, ctx.localDst, ctx.srcLocCopy, ctx.goSize)); }

    const uint16_t allRankMask = (1 << arg->rankSize) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, allRankMask));
    return CCU_SUCCESS;
}

static CcuResult AgDoRepeatAllGather(AllGatherContext& ctx)
{
    const auto* arg = ctx.arg;

    ccu::LocalAddr src;
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(arg->rankSize);

    src.addr = ctx.input;
    src.addr += ctx.currentRankSliceInputOffset;
    src.token = ctx.token[arg->rankId];

    ctx.srcLocCopy.addr = ctx.input;
    ctx.srcLocCopy.addr += ctx.currentRankSliceInputOffset;
    ctx.srcLocCopy.token = ctx.token[arg->rankId];

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.localDst.addr = ctx.output[arg->rankId];
            ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.localDst.token = ctx.token[arg->rankId];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }
    }

    ccu::Variable constVar1;
    ccu::Variable repeatTimeflag;
    constVar1 = 1;
    repeatTimeflag = 0;

    CCU_WHILE(ctx.tmpRepeatNum != UINT64_MAX)
    {
        ctx.tmpRepeatNum += constVar1;

        CCU_IF(repeatTimeflag != 0)
        {
            src.addr += ctx.inputRepeatStride;
            for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx == arg->rankId) {
                    ctx.localDst.addr += ctx.outputRepeatStride;
                } else {
                    dst[rankIdx].addr += ctx.outputRepeatStride;
                }
            }
        }

        CCU_IF(ctx.normalSliceSize != 0) { CCU_CHK_RET(AgDoAllGather(ctx, src, dst, ctx.normalSliceSize)); }

        repeatTimeflag = 1;
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherMesh1dMem2MemKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<AllGatherKernelArg*>(arg);

    AllGatherContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.groupCopyRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    CCU_CHK_RET(AgInitResource(ctx));
    CCU_CHK_RET(AgLoadArgs(ctx));

    CCU_CHK_RET(AgPreSync(ctx));

    CCU_CHK_RET(AgDoRepeatAllGather(ctx));

    CCU_CHK_RET(AgPostSync(ctx));

    return CCU_SUCCESS;
}
