/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REDUCE_SCATTER_MESH1D_A6_DEMO_H
#define CCU_REDUCE_SCATTER_MESH1D_A6_DEMO_H

#include "ccu_primitives.hpp"
#include "ccu_types.h"

#include <cstdint>
#include <climits>
#include <memory>
#include <vector>
#include <string>
#include <set>

namespace ccu = ::AscendC::ccu;

namespace rs_v2 {

constexpr uint32_t RS_MAX_RANK_SIZE = 16;

constexpr int RS_INPUT_XN_ID = 0;
constexpr int RS_SCRATCH_XN_ID = 1;
constexpr int RS_TOKEN_XN_ID = 2;
constexpr int RS_POST_SYNC_ID = 3;

constexpr int RS_CKE_IDX_0 = 0;

constexpr uint64_t RS_CCU_MS_SIZE = 4096;
constexpr uint64_t RS_CCU_MS_INTERLEAVE = 8;
constexpr uint64_t RS_CCU_MS_DEFAULT_LOOP_COUNT = 64;

struct ReduceScatterKernelArgV2 {
    uint64_t rankSize;
    uint32_t rankId;
    ChannelHandle channels[RS_MAX_RANK_SIZE];
    uint32_t channelCount;
    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;
};

struct GroupOpSizeVarsV2 {
    ccu::Variable addrOffset;
    ccu::Variable loopParam; // host 端 CalGoSize 返回的裸迭代次数 m
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct LoopGroupConfigV2 {
    uint64_t msInterleave;
    uint64_t loopCount;
    uint64_t memSlice;
};

struct LoopGroupResourceV2 {
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t eventCount{0};
    uint32_t bufCount{0};
};

static inline constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

static inline uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return SetBits(loopNumBitNum);
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
    return 1;
}

static inline std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfigV2& config)
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

static inline CcuResult AllocGoResource(
    LoopGroupConfigV2& config, LoopGroupResourceV2& res, bool& allocated,
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

struct ReduceScatterContextV2 {
    const ReduceScatterKernelArgV2* arg;

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
    GroupOpSizeVarsV2 goSize;

    uint16_t selfBit;
    uint16_t allBit;

    ccu::LocalAddr myInput;
    ccu::RemoteAddr remoteInput[RS_MAX_RANK_SIZE];
    ccu::LocalAddr scratchMem[RS_MAX_RANK_SIZE];
    ccu::Event event;

    LoopGroupConfigV2 moConfig;
    LoopGroupResourceV2 moRes;
    bool resourceAllocated;

    // Loop body 复用：body 只翻译一次；每次进入不同 LoopGroup 前，通过
    // reduceIterNum / reduceGsaOffset 分别注入 V2 loop 参数（ctxId 由 ccu::Loop 内部维护）。
    std::unique_ptr<ccu::Func> reduceBody[2];
    std::unique_ptr<ccu::Loop> reduceLoops[2];
    ccu::Variable reduceIterNum[2];
    ccu::Variable reduceGsaOffset[2];
    bool loopRegistered;

    // Loop body 中的外部 LocalAddr / 长度参数（每个 loop index 各一组）
    ccu::LocalAddr loopDst[2];
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopScratch[2][RS_MAX_RANK_SIZE];
    ccu::Variable loopLen[2];
    ccu::Variable loopLenExp[2];
};

static CcuResult InitResource(ReduceScatterContextV2& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        return CcuResult::CCU_E_PARA;
    }

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

    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterContextV2& ctx)
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

static void PreSync(ReduceScatterContextV2& ctx)
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

static void PostSync(ReduceScatterContextV2& ctx)
{
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
}

static CcuResult CreateReduceLoop(ReduceScatterContextV2& ctx)
{
    const auto* arg = ctx.arg;
    const uint32_t size = arg->rankSize;

    constexpr uint32_t LOOP_NUM = 16;
    CCU_CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM));

    if (ctx.loopRegistered) {
        return CCU_SUCCESS;
    }

    // 显式 rs_v2:: 限定：避免与 v1 头里全局作用域同名函数经由 HcclReduceOp 等
    // 全局枚举的 ADL 一起参与重载解析导致的二义性。
    uint32_t expansionNum = rs_v2::GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;
    (void)expansionNum;
    (void)usedBufNum;

    for (int32_t index = 0; index < 2; index++) {
        uint32_t bufBase = static_cast<uint32_t>(index * ctx.moConfig.msInterleave);

        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        ctx.reduceBody[index].reset(new ccu::Func([&ctx, index, bufBase, loopEvt, size, arg]() {
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

        // V2 Loop：iterNum / gsaOffset 两个 Variable 独立注入，ctxId 由 ccu::Loop
        // 内部自动创建维护，由 ReduceLoopGroup 在加入不同 LoopGroup 之前对 iterNum /
        // gsaOffset 赋值。
        ctx.reduceLoops[index].reset(
            new ccu::Loop(ctx.reduceIterNum[index], ctx.reduceGsaOffset[index], *ctx.reduceBody[index]));
    }

    ctx.loopRegistered = true;
    return CCU_SUCCESS;
}

// m 部分：goSize.loopParam != 0 时发起单 loop 的 LoopGroup。
// 由 ReduceLoopGroup 在 CCU_IF(goSize.loopParam != 0) 作用域内调用。
static void EmitReduceLoopGroupM(
    ReduceScatterContextV2& ctx, const ccu::LocalAddr& dst, const ccu::LocalAddr& src, const ccu::LocalAddr* scratch,
    uint32_t size, uint32_t expansionNum, GroupOpSizeVarsV2& goSize)
{
    ccu::Variable sliceSize;
    sliceSize = ctx.moConfig.memSlice;
    ccu::Variable sliceSizeExpansion;
    sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

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

    ccu::Variable xnOffsetCfg;
    xnOffsetCfg = 0;

    // V2 loop 参数：gsaOffset=memSlice*loopCount, iterNum=m（ctxId 由 Loop 内部维护，默认 0）。
    // m 对应 host 侧 CalGoSize 返回的裸 loopIterNum（存放于 goSize.loopParam）。
    ctx.reduceIterNum[0] = goSize.loopParam;
    ctx.reduceGsaOffset[0] = ctx.moConfig.memSlice * ctx.moConfig.loopCount;

    std::vector<ccu::Loop> grpLoops{*ctx.reduceLoops[0]};
    ccu::LoopGroup group(paraCfg, offsetCfg, xnOffsetCfg, /*maxLoopNum=*/1, grpLoops);
}

// 将单个 loop slot 的 dst/src/scratch/len 参数绑定到 ctx。
// loopIdx 为 ctx.loop* 数组下标；len/lenExp 为对应 slice 长度变量。
static void BindReduceLoopSlot(
    ReduceScatterContextV2& ctx, uint32_t loopIdx, const ccu::LocalAddr& dst, const ccu::LocalAddr& src,
    const ccu::LocalAddr* scratch, uint32_t size, const ccu::Variable& len, const ccu::Variable& lenExp)
{
    ctx.loopDst[loopIdx].addr = dst.addr;
    ctx.loopDst[loopIdx].token = dst.token;
    ctx.loopSrc[loopIdx].addr = src.addr;
    ctx.loopSrc[loopIdx].token = src.token;
    for (uint32_t i = 0; i < size; i++) {
        ctx.loopScratch[loopIdx][i].addr = scratch[i].addr;
        ctx.loopScratch[loopIdx][i].token = scratch[i].token;
    }
    ctx.loopLen[loopIdx] = len;
    ctx.loopLenExp[loopIdx] = lenExp;

    // V2 loop 参数：gsaOffset=0, iterNum=1（ctxId 由 Loop 内部维护，默认 0）—— 与 v1 中
    // GetLoopParam(0, 0, 1) 位打包语义等价。
    ctx.reduceIterNum[loopIdx] = 1;
    ctx.reduceGsaOffset[loopIdx] = 0;
}

// n+p 部分：goSize.parallelParam != 0 时发起双 loop 的 LoopGroup。
// dst/src 按值传入, 内部偏移不影响调用方。
// scratch 为指针传入, 内部对 scratch[i].addr 的偏移会写回调用方数组 (与重构前内联行为一致)。
// 由 ReduceLoopGroup 在 CCU_IF(goSize.parallelParam != 0) 作用域内调用。
static void EmitReduceLoopGroupNP(
    ReduceScatterContextV2& ctx, ccu::LocalAddr dst, ccu::LocalAddr src, ccu::LocalAddr* scratch, uint32_t size,
    uint32_t expansionNum, GroupOpSizeVarsV2& goSize)
{
    for (uint32_t i = 0; i < size; i++) {
        scratch[i].addr += goSize.addrOffset;
    }
    src.addr += goSize.addrOffset;
    for (uint32_t i = 0; i < expansionNum; i++) {
        dst.addr += goSize.addrOffset;
    }

    ccu::Variable sliceSizeExpansion;
    sliceSizeExpansion = 0;
    for (uint32_t i = 0; i < expansionNum; i++) {
        sliceSizeExpansion = sliceSizeExpansion + goSize.residual;
    }

    // 绑定 loop0 参数 (p 部分)
    BindReduceLoopSlot(ctx, 0, dst, src, scratch, size, goSize.residual, sliceSizeExpansion);

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
    BindReduceLoopSlot(ctx, 1, dst, src, scratch, size, sliceSize, sliceSizeExpansion);

    ccu::Variable offsetCfg;
    offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

    ccu::Variable xnOffsetCfg;
    xnOffsetCfg = 0;

    std::vector<ccu::Loop> grpLoops{*ctx.reduceLoops[0], *ctx.reduceLoops[1]};
    ccu::LoopGroup group(goSize.parallelParam, offsetCfg, xnOffsetCfg, /*maxLoopNum=*/2, grpLoops);
}

static CcuResult ReduceLoopGroup(
    ReduceScatterContextV2& ctx, ccu::LocalAddr outDstOrg, ccu::LocalAddr srcOrg, ccu::LocalAddr* scratchOrg,
    uint32_t scratchCount, GroupOpSizeVarsV2& goSize)
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

    uint32_t expansionNum = rs_v2::GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m 部分
    CCU_IF(goSize.loopParam != 0) { EmitReduceLoopGroupM(ctx, dst, src, scratch, size, expansionNum, goSize); }

    // n+p 部分
    CCU_IF(goSize.parallelParam != 0) { EmitReduceLoopGroupNP(ctx, dst, src, scratch, size, expansionNum, goSize); }

    return CCU_SUCCESS;
}

static CcuResult DoReduceScatter(ReduceScatterContextV2& ctx)
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

static CcuResult DoRepeatReduceScatter(ReduceScatterContextV2& ctx)
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

} // namespace rs_v2

using ReduceScatterKernelArgV2 = rs_v2::ReduceScatterKernelArgV2;

inline CcuResult CcuReduceScatterMesh1dV2Kernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<rs_v2::ReduceScatterKernelArgV2*>(arg);

    rs_v2::ReduceScatterContextV2 ctx;
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

    CCU_CHK_RET(rs_v2::InitResource(ctx));
    CCU_CHK_RET(rs_v2::LoadArgs(ctx));

    rs_v2::PreSync(ctx);

    CCU_CHK_RET(rs_v2::DoRepeatReduceScatter(ctx));

    rs_v2::PostSync(ctx);

    return CCU_SUCCESS;
}

#endif // CCU_REDUCE_SCATTER_MESH1D_A6_DEMO_H
