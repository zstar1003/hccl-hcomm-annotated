/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_GROUPCOPY_DEMO_H
#define CCU_GROUPCOPY_DEMO_H

#include "ccu_primitives.hpp"
#include "ccu_types.h"
#include "ccu_log.h" // demo演示使用，hccl仓需要另外实现

#include <cstdint>
#include <climits>
#include <memory>
#include <vector>
#include <string>

namespace ccu = ::AscendC::ccu;

constexpr uint32_t AG_MAX_RANK_SIZE = 16;

constexpr int AG_OUTPUT_XN_ID = 1;
constexpr int AG_TOKEN_XN_ID = 2;
constexpr int AG_CKE_IDX_0 = 0;
constexpr int AG_POST_SYNC_ID = 3;

constexpr uint64_t AG_CCU_MS_SIZE = 4096;
constexpr uint64_t AG_CCU_MS_INTERLEAVE = 8;
constexpr uint64_t AG_CCU_MS_DEFAULT_LOOP_COUNT = 64;

constexpr uint64_t AG_LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint64_t AG_CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

// =============================================================================
// 参数结构体
// =============================================================================

struct AllGatherKernelArg {
    uint64_t rankSize;
    uint32_t rankId;
    ChannelHandle channels[AG_MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct GroupCopyGoSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

// =============================================================================
// 运行时配置
// =============================================================================

struct GroupCopyLoopGroupConfig {
    uint64_t msInterleave;
    uint64_t loopCount;
    uint64_t memSlice;
};

struct GroupCopyLoopGroupResource {
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t eventCount{0};
    uint32_t bufCount{0};
};

// =============================================================================
// Bit-packing 辅助（纯 host 计算，无 IR 副作用）
// =============================================================================

static inline constexpr uint64_t AgSetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

static inline uint64_t AgGetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return AgSetBits(loopNumBitNum);
}

static inline uint64_t AgGetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t ctxIdBitNum = 8;
    constexpr uint16_t ctxIdShiftBit = 45;
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 13;
    constexpr uint16_t loopNumBitNum = 13;
    constexpr uint16_t loopNumShiftBit = 0;
    return ((loopCtxId & AgSetBits(ctxIdBitNum)) << ctxIdShiftBit) | ((gsaOffset & AgSetBits(gsaBitNum)) << gsaShiftBit)
           | ((loopIterNum & AgSetBits(loopNumBitNum)) << loopNumShiftBit);
}

static inline uint64_t AgGetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitNum = 7;
    constexpr uint16_t repeatNumShiftBit = 55;
    constexpr uint16_t repeatLoopBitNum = 7;
    constexpr uint16_t repeatLoopShiftBit = 48;
    constexpr uint16_t totalLoopBitNum = 7;
    constexpr uint16_t totalLoopShiftBit = 41;
    return ((repeatNum & AgSetBits(repeatBitNum)) << repeatNumShiftBit)
           | ((repeatLoopIndex & AgSetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
           | ((totalLoopNum & AgSetBits(totalLoopBitNum)) << totalLoopShiftBit);
}

static inline uint64_t AgGetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 21;
    constexpr uint16_t msBitNum = 11;
    constexpr uint16_t msShiftBit = 10;
    constexpr uint16_t ckeBitNum = 10;
    constexpr uint16_t ckeShiftBit = 0;
    return ((gsaOffset & AgSetBits(gsaBitNum)) << gsaShiftBit) | ((msOffset & AgSetBits(msBitNum)) << msShiftBit)
           | ((ckeOffset & AgSetBits(ckeBitNum)) << ckeShiftBit);
}

// =============================================================================
// Kernel 上下文（纯 POD，没有任何带 IR 副作用的代码）
// =============================================================================

struct AllGatherContext {
    const AllGatherKernelArg* arg;

    ccu::Variable input;
    ccu::Variable output[AG_MAX_RANK_SIZE];
    ccu::Variable token[AG_MAX_RANK_SIZE];

    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable tmpRepeatNum;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable isInputOutputEqual;

    GroupCopyGoSizeVars goSize;

    ccu::Event event;
    ccu::LocalAddr srcLocCopy;
    ccu::LocalAddr localDst;

    GroupCopyLoopGroupConfig moConfig;
    GroupCopyLoopGroupResource moRes;
    bool resourceAllocated;
    bool groupCopyRegistered;

    std::unique_ptr<ccu::Func> copyBody[2];
    std::unique_ptr<ccu::Loop> copyLoops[2];
    ccu::Variable copyLoopParam[2];

    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable loopLen[2];
};

// =============================================================================
// AllocGoResource —— 纯资源分配，不展开 CCU_IF，放在头里 inline 安全
// =============================================================================

static inline CcuResult AgAllocGoResource(
    GroupCopyLoopGroupConfig& config, GroupCopyLoopGroupResource& res, bool& allocated,
    uint32_t parallelDim = AG_CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1)
{
    if (allocated) {
        return CCU_SUCCESS;
    }

    config.msInterleave = AG_CCU_MS_INTERLEAVE;
    config.loopCount = parallelDim;
    config.memSlice = msPerLoop * AG_CCU_MS_SIZE;

    res.eventCount = config.loopCount;
    res.completedEvent = ccu::Array<ccu::Event>(res.eventCount);

    res.bufCount = config.loopCount * config.msInterleave;
    res.ccuBuf = ccu::Array<ccu::CcuBuffer>(res.bufCount);

    allocated = true;
    return CCU_SUCCESS;
}

CcuResult AgGroupCopy(AllGatherContext& ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupCopyGoSizeVars& goSize);

CcuResult CcuAllGatherMesh1dMem2MemKernel(CcuKernelArg arg);

#endif // CCU_GROUPCOPY_DEMO_H
