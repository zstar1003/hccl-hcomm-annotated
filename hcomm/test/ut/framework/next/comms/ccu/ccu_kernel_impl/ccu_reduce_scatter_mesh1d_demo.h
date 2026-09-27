/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REDUCE_SCATTER_MESH1D_DEMO_H
#define CCU_REDUCE_SCATTER_MESH1D_DEMO_H

#include "ccu_primitives.hpp"
#include "ccu_types.h"

#include <cstdint>
#include <climits>
#include <memory>
#include <vector>

namespace ccu = ::AscendC::ccu;
#include <string>
#include <set>

// ============================================================================
// 常量定义
// ============================================================================

constexpr uint32_t RS_MAX_RANK_SIZE = 16;

constexpr int RS_INPUT_XN_ID = 0;
constexpr int RS_SCRATCH_XN_ID = 1;
constexpr int RS_TOKEN_XN_ID = 2;
constexpr int RS_POST_SYNC_ID = 3;

constexpr int RS_CKE_IDX_0 = 0;

constexpr uint64_t RS_CCU_MS_SIZE = 4096;
constexpr uint64_t RS_CCU_MS_INTERLEAVE = 8;
constexpr uint64_t RS_CCU_MS_DEFAULT_LOOP_COUNT = 64;

// ============================================================================
// 参数结构体
// ============================================================================

struct ReduceScatterKernelArg {
    uint64_t rankSize;
    uint32_t rankId;
    ChannelHandle channels[RS_MAX_RANK_SIZE];
    uint32_t channelCount;
    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

// ============================================================================
// 运行时配置（对应 CcuKernelAlgBase::moConfig / moRes）
// ============================================================================

struct LoopGroupConfig {
    uint64_t msInterleave;
    uint64_t loopCount;
    uint64_t memSlice;
};

struct LoopGroupResource {
    // 真正分配延后到 AllocGoResource：默认构造为空 Array(0)，运行期再 move-assign 出
    // 实际容量，避免在 ReduceScatterContext 默认构造时就一次性占用大量底层资源。
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t eventCount{0};
    uint32_t bufCount{0};
};

// ============================================================================
// Bit-packing 辅助函数（对应 ccu_kernel_utils.cc）
// ============================================================================

static inline constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

static inline uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return SetBits(loopNumBitNum);
}

static inline uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t ctxIdBitNum = 8;
    constexpr uint16_t ctxIdShiftBit = 45;
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 13;
    constexpr uint16_t loopNumBitNum = 13;
    constexpr uint16_t loopNumShiftBit = 0;
    return ((loopCtxId & SetBits(ctxIdBitNum)) << ctxIdShiftBit) | ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit)
           | ((loopIterNum & SetBits(loopNumBitNum)) << loopNumShiftBit);
}

static inline uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitNum = 7;
    constexpr uint16_t repeatNumShiftBit = 55;
    constexpr uint16_t repeatLoopBitNum = 7;
    constexpr uint16_t repeatLoopShiftBit = 48;
    constexpr uint16_t totalLoopBitNum = 7;
    constexpr uint16_t totalLoopShiftBit = 41;
    return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShiftBit)
           | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
           | ((totalLoopNum & SetBits(totalLoopBitNum)) << totalLoopShiftBit);
}

static inline uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 21;
    constexpr uint16_t msBitNum = 11;
    constexpr uint16_t msShiftBit = 10;
    constexpr uint16_t ckeBitNum = 10;
    constexpr uint16_t ckeShiftBit = 0;
    return ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit) | ((msOffset & SetBits(msBitNum)) << msShiftBit)
           | ((ckeOffset & SetBits(ckeBitNum)) << ckeShiftBit);
}

static inline uint64_t GetExpansionParam(uint64_t expansionNum)
{
    constexpr uint64_t expansionNum2 = 2;
    constexpr uint64_t expansionNumShiftBit = 53;
    return (expansionNum == expansionNum2 ? uint64_t(1) : uint64_t(2)) << expansionNumShiftBit;
}

static inline uint32_t GetReduceExpansionNum(HcclReduceOp reduceOp, HcclDataType dataType, HcclDataType outputDataType)
{
    (void)reduceOp;
    (void)dataType;
    (void)outputDataType;
    // 简化实现：大多数场景 expansionNum = 1
    // 完整实现需要 DataTypeSizeGet(outputDataType) / DataTypeSizeGet(dataType)
    return 1;
}

// ============================================================================
// CalGoSize（对应 CcuKernelAlgBase::CalGoSize，host 侧调用）
// ============================================================================

static inline std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig& config)
{
    uint64_t loopSize = config.loopCount * config.memSlice;
    uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);

    uint64_t m = size / loopSize;
    uint64_t n = (size - m * loopSize) / config.memSlice;
    uint64_t p = size - m * loopSize - n * config.memSlice;

    if (size == maxSize) {
        m = GetMaxLoopIterNum();
        n = config.loopCount - 1;
        p = config.memSlice;
    }

    uint64_t offset = config.memSlice * config.loopCount * m;
    uint64_t loopIterNum = m;

    uint64_t loopExtendNum = 0;
    uint64_t tailSize = 0;

    if (n == 0 && p == 0) {
        loopExtendNum = 0;
        tailSize = 0;
    } else if (n != 0 && p == 0) {
        loopExtendNum = GetParallelParam(n - 1, 0, 1);
        tailSize = config.memSlice;
    } else if (n == 0 && p != 0) {
        loopExtendNum = GetParallelParam(0, 0, 1);
        tailSize = p;
    } else {
        loopExtendNum = GetParallelParam(n - 1, 1, 2);
        tailSize = p;
    }

    return {offset, loopIterNum, loopExtendNum, tailSize};
}

// ============================================================================
// AllocGoResource（对应 CcuKernelAlgBase::AllocGoResource）
// ============================================================================

static inline CcuResult AllocGoResource(
    LoopGroupConfig& config, LoopGroupResource& res, bool& allocated,
    uint32_t parallelDim = RS_CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1)
{
    if (allocated) {
        return CCU_SUCCESS;
    }

    config.msInterleave = RS_CCU_MS_INTERLEAVE;
    config.loopCount = parallelDim;
    config.memSlice = msPerLoop * RS_CCU_MS_SIZE;

    res.eventCount = config.loopCount;
    res.completedEvent = ccu::Array<ccu::Event>(res.eventCount);

    res.bufCount = config.loopCount * config.msInterleave;
    res.ccuBuf = ccu::Array<ccu::CcuBuffer>(res.bufCount);

    allocated = true;
    return CCU_SUCCESS;
}

// ============================================================================
// InitResource
// ============================================================================

struct ReduceScatterContext {
    const ReduceScatterKernelArg* arg;

    ccu::Variable input[RS_MAX_RANK_SIZE];
    ccu::Variable scratch[RS_MAX_RANK_SIZE];
    ccu::Variable token[RS_MAX_RANK_SIZE];
    ccu::Variable output;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable repeatNum;
    ccu::Variable flag;
    GroupOpSizeVars goSize;

    uint16_t selfBit;
    uint16_t allBit;

    ccu::LocalAddr myInput;
    ccu::RemoteAddr remoteInput[RS_MAX_RANK_SIZE];
    ccu::LocalAddr scratchMem[RS_MAX_RANK_SIZE];
    ccu::Event event;

    LoopGroupConfig moConfig;
    LoopGroupResource moRes;
    bool resourceAllocated;

    // 同一 ccu::Loop 跨 LoopGroup 复用：body 只翻译一次（在 ccu::Loop ctor 里），
    // 每个 group 进入前通过 reduceLoopParam[i] 注入不同的 var-based 参数。
    std::unique_ptr<ccu::Func> reduceBody[2];
    std::unique_ptr<ccu::Loop> reduceLoops[2];
    ccu::Variable reduceLoopParam[2];
    bool loopRegistered;

    // Loop body 中的外部 LocalAddr（每个 loop index 各两组）
    ccu::LocalAddr loopDst[2];
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopScratch[2][RS_MAX_RANK_SIZE];
    ccu::Variable loopLen[2];
    ccu::Variable loopLenExp[2];
};

static CcuResult InitResource(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        return CcuResult::CCU_E_PARA;
    }

    // ctx 默认构造时 Variable / LocalAddr / RemoteAddr / Event 成员已自动 alloc，
    // 这里仅需把"远端 rank"对应的 Variable 通过 GetResByChannel 重绑定即可
    // (右值赋值走 Variable 的 move-assign，直接覆盖 handle)。
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], RS_INPUT_XN_ID);
            ctx.scratch[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], RS_SCRATCH_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], RS_TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.selfBit = 1 << arg->rankId;
    ctx.allBit = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId));

    // LoopEngine 池由各 ccu::LoopGroup 在 ctor 第二个参数 maxLoopNum 处自报，
    // kernel 按需扩容（取最大值，跨组复用）。下面两个 group 分别填 1 / 2。

    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;

    return CCU_SUCCESS;
}

// ============================================================================
// LoadArgs
// ============================================================================

static CcuResult LoadArgs(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    return CCU_SUCCESS;
}

// ============================================================================
// PreSync / PostSync
// ============================================================================

static void PreSync(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.input[arg->rankId], RS_INPUT_XN_ID, RS_CKE_IDX_0, 1 << RS_INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.scratch[arg->rankId], RS_SCRATCH_XN_ID, RS_CKE_IDX_0, 1 << RS_SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], RS_TOKEN_XN_ID, RS_CKE_IDX_0, 1 << RS_TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << RS_INPUT_XN_ID) | (1 << RS_SCRATCH_XN_ID) | (1 << RS_TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
}

// ============================================================================
// CreateReduceLoop
// ============================================================================

static CcuResult CreateReduceLoop(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;
    const uint32_t size = arg->rankSize;

    constexpr uint32_t LOOP_NUM = 16;
    CCU_CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM));

    if (ctx.loopRegistered) {
        return CCU_SUCCESS;
    }

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;
    (void)expansionNum;
    (void)usedBufNum;

    for (int32_t index = 0; index < 2; index++) {
        // ctx.loopDst / loopSrc / loopScratch / loopLen / loopLenExp 已在 ctx 默认构造时 alloc
        uint32_t bufBase = static_cast<uint32_t>(index * ctx.moConfig.msInterleave);

        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        // 用对象式 API：Func 承载 body 闭包，Loop ctor 立即把 body 翻译为 LoopBlock；
        // var-based 参数预留 ctx.reduceLoopParam[index]，由 ReduceLoopGroup 在加入
        // 不同 LoopGroup 之前赋值。
        ctx.reduceBody[index].reset(new ccu::Func([&ctx, index, bufBase, loopEvt, size, arg]() {
            // mask 已与 Event 解耦：由各个数据 / 同步 API 末尾参数显式传入。
            for (uint32_t i = 0; i < size; i++) {
                const uint32_t copyMask = 1 << i;
                if (i == arg->rankId) {
                    ccu::LocalCopy(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopSrc[index], ctx.loopLen[index], loopEvt, copyMask);
                } else {
                    ccu::LocalCopy(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt,
                        copyMask);
                }
            }
            ccu::EventWait(loopEvt, (1 << size) - 1);

            if (size > 1) {
                ccu::LocalReduce(
                    &ctx.moRes.ccuBuf[bufBase], size, arg->dataType, arg->outputDataType, arg->reduceOp,
                    ctx.loopLen[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }

            ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
        }));
        ctx.reduceLoops[index].reset(new ccu::Loop(ctx.reduceLoopParam[index], *ctx.reduceBody[index]));
    }

    ctx.loopRegistered = true;
    return CCU_SUCCESS;
}

// ============================================================================
// ReduceLoopGroup
// ============================================================================

static CcuResult ReduceLoopGroup(
    ReduceScatterContext& ctx, ccu::LocalAddr outDstOrg, ccu::LocalAddr srcOrg, ccu::LocalAddr* scratchOrg,
    uint32_t scratchCount, GroupOpSizeVars& goSize)
{
    const auto* arg = ctx.arg;
    const uint32_t size = scratchCount;

    ccu::LocalAddr dst;
    dst.addr = outDstOrg.addr;
    dst.token = outDstOrg.token;

    ccu::LocalAddr src;
    src.addr = srcOrg.addr;
    src.token = srcOrg.token;

    ccu::LocalAddr scratch[RS_MAX_RANK_SIZE];
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch[idx].addr = scratchOrg[idx].addr;
        scratch[idx].token = scratchOrg[idx].token;
    }

    CCU_CHK_RET(CreateReduceLoop(ctx));

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    ccu::Variable sliceSizeExpansion;

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m 部分
    CCU_IF(goSize.loopParam != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定 loop0 的外部 LocalAddr 和 Variable
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopSrc[0].addr = src.addr;
        ctx.loopSrc[0].token = src.token;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopLen[0] = sliceSize;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        // 把本组要用的 var-based loop 参数注入：reduceLoopParam[0] 已被 reduceLoops[0]
        // 持引用，给它赋值即在当前位置 emit 一条 LoadImd/AddVar，并在 LoopGroup 翻译时
        // 转化为对应 LoopInstr 的低位参数。
        ctx.reduceLoopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{*ctx.reduceLoops[0]};
        ccu::LoopGroup group(paraCfg, offsetCfg, /*maxLoopNum=*/1, grpLoops);
    }

    // n+p 部分
    CCU_IF(goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.addrOffset;
        }
        src.addr += goSize.addrOffset;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + goSize.residual;
        }

        // 绑定 loop0 参数 (p 部分)
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopSrc[0].addr = src.addr;
        ctx.loopSrc[0].token = src.token;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopLen[0] = goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        // n 部分偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.residual;
        }
        src.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定 loop1 参数 (n 部分)
        ctx.loopDst[1].addr = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopSrc[1].addr = src.addr;
        ctx.loopSrc[1].token = src.token;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopLen[1] = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);

        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        ctx.reduceLoopParam[0] = loopCfg0;
        ctx.reduceLoopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{*ctx.reduceLoops[0], *ctx.reduceLoops[1]};
        ccu::LoopGroup group(goSize.parallelParam, offsetCfg, /*maxLoopNum=*/2, grpLoops);
    }

    return CCU_SUCCESS;
}

// ============================================================================
// DoReduceScatter
// ============================================================================

static CcuResult DoReduceScatter(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t channelId = 0;

    ccu::LocalAddr myOutput;
    myOutput.addr = ctx.output;
    myOutput.addr += ctx.currentRankSliceOutputOffset;
    myOutput.token = ctx.token[arg->rankId];

    ccu::Variable sliceSize;
    sliceSize = (arg->rankId == (arg->rankSize - 1)) ? ctx.lastSliceSize : ctx.normalSliceSize;

    CCU_IF(sliceSize != 0)
    {
        // mask 已与 Event 解耦：record/read 用 (1 << rankIdx)，等待用全位 mask。
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            const uint32_t rankMask = 1 << rankIdx;
            if (rankIdx == arg->rankId) {
                ccu::EventRecord(ctx.event, rankMask);
            } else {
                ccu::Read(
                    arg->channels[channelId], ctx.scratchMem[rankIdx], ctx.remoteInput[rankIdx], sliceSize, ctx.event,
                    rankMask);
                channelId++;
            }
        }

        ccu::EventWait(ctx.event, (1 << arg->rankSize) - 1);

        ReduceLoopGroup(ctx, myOutput, ctx.myInput, ctx.scratchMem, arg->rankSize, ctx.goSize);
    }

    return CCU_SUCCESS;
}

// ============================================================================
// DoRepeatReduceScatter
// ============================================================================

static CcuResult DoRepeatReduceScatter(ReduceScatterContext& ctx)
{
    const auto* arg = ctx.arg;

    ccu::Variable scratchOffset;
    scratchOffset = 0;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.myInput.addr = ctx.input[rankIdx];
            ctx.myInput.addr += ctx.currentRankSliceInputOffset;
            ctx.myInput.token = ctx.token[rankIdx];
        } else {
            ctx.remoteInput[rankIdx].addr = ctx.input[rankIdx];
            ctx.remoteInput[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.remoteInput[rankIdx].token = ctx.token[rankIdx];
        }

        ctx.scratchMem[rankIdx].addr = ctx.scratch[arg->rankId];
        ctx.scratchMem[rankIdx].addr += scratchOffset;
        scratchOffset = scratchOffset + ctx.normalSliceSize;
        ctx.scratchMem[rankIdx].token = ctx.token[arg->rankId];
    }

    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.flag = 0;

    CCU_DO
    {
        ctx.repeatNum = ctx.repeatNum + repeatNumAdd;

        CCU_IF(ctx.flag == 1)
        {
            for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx == arg->rankId) {
                    ctx.myInput.addr += ctx.inputRepeatStride;
                } else {
                    ctx.remoteInput[rankIdx].addr += ctx.inputRepeatStride;
                }
            }
            ctx.output = ctx.output + ctx.outputRepeatStride;
        }

        DoReduceScatter(ctx);
        ctx.flag = 1;
    }
    CCU_WHILE(ctx.repeatNum != UINT64_MAX);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================

CcuResult CcuReduceScatterMesh1dKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<ReduceScatterKernelArg*>(arg);

    ReduceScatterContext ctx;
    ctx.arg = kernelArg;
    ctx.selfBit = 0;
    ctx.allBit = 0;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    // LoopEngine 池由各 LoopGroup ctor 自动按 maxLoopNum 扩容，无需在此显式申请。

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_CHK_RET(DoRepeatReduceScatter(ctx));

    PostSync(ctx);

    return CCU_SUCCESS;
}

#endif // CCU_REDUCE_SCATTER_MESH1D_DEMO_H
