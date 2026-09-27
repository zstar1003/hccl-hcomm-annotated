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

CcuResult AgCreateMultiOpCopy(AllGatherContext& ctx)
{
    CCU_CHK_RET(AgAllocGoResource(
        ctx.moConfig, ctx.moRes, ctx.resourceAllocated, AG_CCU_MS_LOCAL_COPY_LOOP_COUNT, AG_LOCAL_COPY_MS_PER_LOOP));

    if (ctx.groupCopyRegistered) {
        return CCU_SUCCESS;
    }

    for (uint32_t index = 0; index < 2; index++) {
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];
        ccu::CcuBuffer& buf = ctx.moRes.ccuBuf[index * ctx.moConfig.msInterleave];

        ctx.copyBody[index].reset(new ccu::Func([&ctx, index, buf, loopEvt]() {
            ccu::LocalCopy(buf, ctx.loopSrc[index], ctx.loopLen[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
            ccu::LocalCopy(ctx.loopDst[index], buf, ctx.loopLen[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
        }));
        ctx.copyLoops[index].reset(new ccu::Loop(ctx.copyLoopParam[index], *ctx.copyBody[index]));
    }

    ctx.groupCopyRegistered = true;
    return CCU_SUCCESS;
}

CcuResult AgGroupCopy(AllGatherContext& ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupCopyGoSizeVars& goSize)
{
    CCU_CHK_RET(AgCreateMultiOpCopy(ctx));

    // 第一个 LoopGroup：搬运 m 部分（256K 整数倍部分），单 loop 模板按 loopCount 展开。
    CCU_IF(goSize.addrOffset != 0)
    {
        ccu::Variable loopParam;
        loopParam = AgGetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;

        ctx.loopSrc[0].addr = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = sliceSize;

        ccu::Variable paraCfg;
        paraCfg = AgGetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = AgGetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        ctx.copyLoopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{*ctx.copyLoops[0]};
        ccu::LoopGroup group(paraCfg, offsetCfg, /*maxLoopNum=*/1, grpLoops);
    }

    // 第二个 LoopGroup：搬运 n + p 部分。
    CCU_IF(goSize.parallelParam != 0)
    {
        src.addr += goSize.addrOffset;
        dst.addr += goSize.addrOffset;

        ctx.loopSrc[0].addr = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = goSize.residual;

        src.addr += goSize.residual;
        dst.addr += goSize.residual;

        ctx.loopSrc[1].addr = src.addr;
        ctx.loopSrc[1].token = src.token;
        ctx.loopDst[1].addr = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1] = ctx.moConfig.memSlice;

        ccu::Variable loopCfg0;
        loopCfg0 = AgGetLoopParam(0, 0, 1);

        ccu::Variable loopCfg1;
        loopCfg1 = AgGetLoopParam(0, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = AgGetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        ctx.copyLoopParam[0] = loopCfg0;
        ctx.copyLoopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{*ctx.copyLoops[0], *ctx.copyLoops[1]};
        ccu::LoopGroup group(goSize.parallelParam, offsetCfg, /*maxLoopNum=*/2, grpLoops);
    }

    return CCU_SUCCESS;
}
