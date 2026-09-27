/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIFROST_CNCOI_BUILD_MULTI_LAYER_H
#define BIFROST_CNCOI_BUILD_MULTI_LAYER_H
#include "simpo_multi_layer_common_builder.h"
#include "bifrost_cncoi_reader.h"
#ifdef __cplusplus
extern "C" {
#endif

static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryPacketNumAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryPacketNumAdd, 0);
    *((uint16_t *)(builder->curWriteAddr + 0)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryResvAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryResvAdd, 1);
    *((uint16_t *)(builder->curWriteAddr + 2)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntrySendRankInfoIgnored(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 4;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntrySendRankInfoDeprecated, 2);
    *((uint32_t *)(builder->curWriteAddr)) = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntrySendRankInfoStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 4;
    builder->structVecLengthAddr = builder->curWriteAddr;
    builder->structVecLength = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntrySendRankInfoStart, 2);
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoEntrySendRankInfoEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    *((uint32_t *)(builder->structVecLengthAddr)) = builder->structVecLength;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoEntrySendRankInfoPush(SimpoBuilderT *builder,
    BifrostCncoiSendRankInfoT *input)
{
    if (builder->curWriteAddr + sizeof(BifrostCncoiSendRankInfoT) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }
    *((BifrostCncoiSendRankInfoT *)(builder->curWriteAddr)) = *input;
    builder->curWriteAddr += sizeof(BifrostCncoiSendRankInfoT);
    builder->structVecLength++;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryMd5sumCreate(SimpoBuilderT *builder,
    BifrostCncoiCommMd5sumT *value)
{
    if (builder->curWriteAddr + 20 > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryMd5sumCreate, 3);
    *((BifrostCncoiCommMd5sumT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryRankTotalNumAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryRankTotalNumAdd, 4);
    *((uint16_t *)(builder->curWriteAddr + 16)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryResv2Add(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryResv2Add, 5);
    *((uint16_t *)(builder->curWriteAddr + 18)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryRankInfoIgnored(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 20;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryRankInfoDeprecated, 6);
    *((uint32_t *)(builder->curWriteAddr)) = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryRankInfoStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 20;
    builder->structVecLengthAddr = builder->curWriteAddr;
    builder->structVecLength = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoEntryRankInfoStart, 6);
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryRankInfoEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    *((uint32_t *)(builder->structVecLengthAddr)) = builder->structVecLength;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoEntryRankInfoPush(SimpoBuilderT *builder,
    BifrostCncoiCommInfoRankInfoT *input)
{
    if (builder->curWriteAddr + sizeof(BifrostCncoiCommInfoRankInfoT) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }
    *((BifrostCncoiCommInfoRankInfoT *)(builder->curWriteAddr)) = *input;
    builder->curWriteAddr += sizeof(BifrostCncoiCommInfoRankInfoT);
    builder->structVecLength++;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiOperatorEntryTrafficCntAdd(SimpoBuilderT *builder, uint64_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiOperatorEntryTrafficCntAdd, 0);
    ((BifrostCncoiOperatorEntryT *)(builder->curWriteAddr))->trafficCnt = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorEntryL4SPortIdAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiOperatorEntryL4SPortIdAdd, 1);
    ((BifrostCncoiOperatorEntryT *)(builder->curWriteAddr))->l4SPortId = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorEntryMaskLenAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiOperatorEntryMaskLenAdd, 2);
    ((BifrostCncoiOperatorEntryT *)(builder->curWriteAddr))->maskLen = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorEntryResvAdd(SimpoBuilderT *builder, uint32_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiOperatorEntryResvAdd, 3);
    ((BifrostCncoiOperatorEntryT *)(builder->curWriteAddr))->resv = value;
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiAdjacencyEntryAdjInfoIgnored(SimpoBuilderT *builder)
{
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiAdjacencyEntryAdjInfoDeprecated, 0);
    *((uint32_t *)(builder->curWriteAddr)) = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiAdjacencyEntryAdjInfoStart(SimpoBuilderT *builder)
{
    builder->structVecLengthAddr = builder->curWriteAddr;
    builder->structVecLength = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiAdjacencyEntryAdjInfoStart, 0);
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiAdjacencyEntryAdjInfoEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    *((uint32_t *)(builder->structVecLengthAddr)) = builder->structVecLength;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiAdjacencyEntryAdjInfoPush(SimpoBuilderT *builder, BifrostCncoiAdjInfoT *input)
{
    if (builder->curWriteAddr + sizeof(BifrostCncoiAdjInfoT) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }
    *((BifrostCncoiAdjInfoT *)(builder->curWriteAddr)) = *input;
    builder->curWriteAddr += sizeof(BifrostCncoiAdjInfoT);
    builder->structVecLength++;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRankEntryPacketNumAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntryPacketNumAdd, 0);
    *((uint16_t *)(builder->curWriteAddr + 0)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntryResvAdd(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntryResvAdd, 1);
    *((uint16_t *)(builder->curWriteAddr + 2)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntrySendRankInfoIgnored(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 4;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntrySendRankInfoDeprecated, 2);
    *((uint32_t *)(builder->curWriteAddr)) = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntrySendRankInfoStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 4;
    builder->structVecLengthAddr = builder->curWriteAddr;
    builder->structVecLength = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntrySendRankInfoStart, 2);
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRankEntrySendRankInfoEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    *((uint32_t *)(builder->structVecLengthAddr)) = builder->structVecLength;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRankEntrySendRankInfoPush(SimpoBuilderT *builder,
    BifrostCncoiSendRankInfoT *input)
{
    if (builder->curWriteAddr + sizeof(BifrostCncoiSendRankInfoT) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }
    *((BifrostCncoiSendRankInfoT *)(builder->curWriteAddr)) = *input;
    builder->curWriteAddr += sizeof(BifrostCncoiSendRankInfoT);
    builder->structVecLength++;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntryMd5sumCreate(SimpoBuilderT *builder, BifrostCncoiCommMd5sumT *value)
{
    if (builder->curWriteAddr + 20 > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntryMd5sumCreate, 3);
    *((BifrostCncoiCommMd5sumT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntryRankTotalNumAdd(SimpoBuilderT *builder, uint32_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntryRankTotalNumAdd, 4);
    *((uint32_t *)(builder->curWriteAddr + 16)) = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntryRankInfoIgnored(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 20;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntryRankInfoDeprecated, 5);
    *((uint32_t *)(builder->curWriteAddr)) = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiRankEntryRankInfoStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 20;
    builder->structVecLengthAddr = builder->curWriteAddr;
    builder->structVecLength = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankEntryRankInfoStart, 5);
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRankEntryRankInfoEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    *((uint32_t *)(builder->structVecLengthAddr)) = builder->structVecLength;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRankEntryRankInfoPush(SimpoBuilderT *builder, BifrostCncoiRankInfoT *input)
{
    if (builder->curWriteAddr + sizeof(BifrostCncoiRankInfoT) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }
    *((BifrostCncoiRankInfoT *)(builder->curWriteAddr)) = *input;
    builder->curWriteAddr += sizeof(BifrostCncoiRankInfoT);
    builder->structVecLength++;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRankDistributeEntryServerIpAdd(SimpoBuilderT *builder, uint32_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeEntryServerIpAdd, 0);
    ((BifrostCncoiRankDistributeEntryT *)(builder->curWriteAddr))->serverIp = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeEntryNodeIdAdd(SimpoBuilderT *builder, uint32_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeEntryNodeIdAdd, 1);
    ((BifrostCncoiRankDistributeEntryT *)(builder->curWriteAddr))->nodeId = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeEntryLocalRankNumAdd(SimpoBuilderT *builder, uint8_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeEntryLocalRankNumAdd, 2);
    ((BifrostCncoiRankDistributeEntryT *)(builder->curWriteAddr))->localRankNum = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeEntryPad1Add(SimpoBuilderT *builder, uint8_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeEntryPad1Add, 3);
    ((BifrostCncoiRankDistributeEntryT *)(builder->curWriteAddr))->pad1 = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeEntryPad2Add(SimpoBuilderT *builder, uint16_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeEntryPad2Add, 4);
    ((BifrostCncoiRankDistributeEntryT *)(builder->curWriteAddr))->pad2 = value;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeEntryRankTotalNumAdd(SimpoBuilderT *builder, uint32_t value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeEntryRankTotalNumAdd, 5);
    ((BifrostCncoiRankDistributeEntryT *)(builder->curWriteAddr))->rankTotalNum = value;
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRootRankEntryRootRankInfoIgnored(SimpoBuilderT *builder)
{
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRootRankEntryRootRankInfoDeprecated, 0);
    *((uint32_t *)(builder->curWriteAddr)) = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiRootRankEntryRootRankInfoStart(SimpoBuilderT *builder)
{
    builder->structVecLengthAddr = builder->curWriteAddr;
    builder->structVecLength = 0;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + SIMPO_BUILD_LENGTH_SIZE > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRootRankEntryRootRankInfoStart, 0);
    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRootRankEntryRootRankInfoEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    *((uint32_t *)(builder->structVecLengthAddr)) = builder->structVecLength;

    return SIMPO_BUILD_OK;
}

static SIMPO_INLINE int32_t BifrostCncoiRootRankEntryRootRankInfoPush(SimpoBuilderT *builder,
    BifrostCncoiRootRankInfoT *input)
{
    if (builder->curWriteAddr + sizeof(BifrostCncoiRootRankInfoT) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMATCH;
        return builder->errorCode;
    }
    *((BifrostCncoiRootRankInfoT *)(builder->curWriteAddr)) = *input;
    builder->curWriteAddr += sizeof(BifrostCncoiRootRankInfoT);
    builder->structVecLength++;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoUpdateStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 3);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoUpdateEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 3, BifrostCncoiComminfoUpdateEndAsRoot, 1);

    builder->curWriteAddr += 0;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoUpdateKeyCreate(SimpoBuilderT *builder, BifrostCncoiComminfoKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiComminfoUpdateKeyCreate, 0);
    *((BifrostCncoiComminfoKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoUpdateEntryStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 152;
    builder->tempWriteAddr[1] = builder->curWriteAddr;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + 4 > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiComminfoUpdateEntryStart, 1);
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoUpdateEntryEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiComminfoUpdateEntryEnd, 6);
    uint32_t len = builder->curWriteAddr - builder->tempWriteAddr[1] - SIMPO_BUILD_LENGTH_SIZE;
    *((uint32_t *)(builder->tempWriteAddr[1])) = len;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoDeleteStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiComminfoDeleteEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiComminfoDeleteEndAsRoot, 0);

    builder->curWriteAddr += 152;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiComminfoDeleteKeyCreate(SimpoBuilderT *builder, BifrostCncoiComminfoKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiComminfoDeleteKeyCreate, 0);
    *((BifrostCncoiComminfoKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorUpdateStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 3);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiOperatorUpdateEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 3, BifrostCncoiOperatorUpdateEndAsRoot, 1);

    builder->curWriteAddr += 0;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorUpdateKeyCreate(SimpoBuilderT *builder, BifrostCncoiOperatorKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiOperatorUpdateKeyCreate, 0);
    *((BifrostCncoiOperatorKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorUpdateEntryStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 152;
    if (builder->curWriteAddr + (SIMPO_BUILD_LENGTH_SIZE + 16) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiOperatorUpdateEntryStart, 1);
    SIMPO_OPER_INIT(builder, 1);
    *((uint32_t *)(builder->curWriteAddr)) = 16;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiOperatorUpdateEntryEnd(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 16;
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiOperatorUpdateEntryEnd, 3);

    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorDeleteStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiOperatorDeleteEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiOperatorDeleteEndAsRoot, 0);

    builder->curWriteAddr += 152;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiOperatorDeleteKeyCreate(SimpoBuilderT *builder, BifrostCncoiOperatorKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiOperatorDeleteKeyCreate, 0);
    *((BifrostCncoiOperatorKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiAdjacencyUpdateStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 160)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 3);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiAdjacencyUpdateEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 3, BifrostCncoiAdjacencyUpdateEndAsRoot, 1);

    builder->curWriteAddr += 0;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiAdjacencyUpdateKeyCreate(SimpoBuilderT *builder,
    BifrostCncoiAdjacencyKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiAdjacencyUpdateKeyCreate, 0);
    *((BifrostCncoiAdjacencyKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiAdjacencyUpdateEntryStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 160;
    builder->tempWriteAddr[1] = builder->curWriteAddr;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiAdjacencyUpdateEntryStart, 1);
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiAdjacencyUpdateEntryEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiAdjacencyUpdateEntryEnd, 0);
    uint32_t len = builder->curWriteAddr - builder->tempWriteAddr[1] - SIMPO_BUILD_LENGTH_SIZE;
    *((uint32_t *)(builder->tempWriteAddr[1])) = len;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiAdjacencyDeleteStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 160)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiAdjacencyDeleteEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiAdjacencyDeleteEndAsRoot, 0);

    builder->curWriteAddr += 160;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiAdjacencyDeleteKeyCreate(SimpoBuilderT *builder,
    BifrostCncoiAdjacencyKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiAdjacencyDeleteKeyCreate, 0);
    *((BifrostCncoiAdjacencyKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankUpdateStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 3);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRankUpdateEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 3, BifrostCncoiRankUpdateEndAsRoot, 1);

    builder->curWriteAddr += 0;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiRankUpdateKeyCreate(SimpoBuilderT *builder, BifrostCncoiRankKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiRankUpdateKeyCreate, 0);
    *((BifrostCncoiRankKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankUpdateEntryStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 152;
    builder->tempWriteAddr[1] = builder->curWriteAddr;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    if (builder->curWriteAddr + 4 > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
    }

    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiRankUpdateEntryStart, 1);
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRankUpdateEntryEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiRankUpdateEntryEnd, 5);
    uint32_t len = builder->curWriteAddr - builder->tempWriteAddr[1] - SIMPO_BUILD_LENGTH_SIZE;
    *((uint32_t *)(builder->tempWriteAddr[1])) = len;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDeleteStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRankDeleteEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiRankDeleteEndAsRoot, 0);

    builder->curWriteAddr += 152;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDeleteKeyCreate(SimpoBuilderT *builder, BifrostCncoiRankKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDeleteKeyCreate, 0);
    *((BifrostCncoiRankKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeUpdateStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 16)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 3);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRankDistributeUpdateEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 3, BifrostCncoiRankDistributeUpdateEndAsRoot, 1);

    builder->curWriteAddr += 0;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeUpdateKeyCreate(SimpoBuilderT *builder,
    BifrostCncoiRankDistributeKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiRankDistributeUpdateKeyCreate, 0);
    *((BifrostCncoiRankDistributeKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeUpdateEntryStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 16;
    if (builder->curWriteAddr + (SIMPO_BUILD_LENGTH_SIZE + 16) > builder->bufferEnd) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiRankDistributeUpdateEntryStart, 1);
    SIMPO_OPER_INIT(builder, 1);
    *((uint32_t *)(builder->curWriteAddr)) = 16;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRankDistributeUpdateEntryEnd(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 16;
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiRankDistributeUpdateEntryEnd, 5);

    return builder->errorCode;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeDeleteStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 16)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRankDistributeDeleteEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiRankDistributeDeleteEndAsRoot, 0);

    builder->curWriteAddr += 16;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiRankDistributeDeleteKeyCreate(SimpoBuilderT *builder,
    BifrostCncoiRankDistributeKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRankDistributeDeleteKeyCreate, 0);
    *((BifrostCncoiRankDistributeKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRootRankUpdateStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 3);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRootRankUpdateEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 3, BifrostCncoiRootRankUpdateEndAsRoot, 1);

    builder->curWriteAddr += 0;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiRootRankUpdateKeyCreate(SimpoBuilderT *builder, BifrostCncoiRootRankKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiRootRankUpdateKeyCreate, 0);
    *((BifrostCncoiRootRankKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRootRankUpdateEntryStart(SimpoBuilderT *builder)
{
    builder->curWriteAddr += 152;
    builder->tempWriteAddr[1] = builder->curWriteAddr;
    builder->curWriteAddr += SIMPO_BUILD_LENGTH_SIZE;
    SIMPO_CHECK_OPER_ORDER(builder, 3, BifrostCncoiRootRankUpdateEntryStart, 1);
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRootRankUpdateEntryEnd(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiRootRankUpdateEntryEnd, 0);
    uint32_t len = builder->curWriteAddr - builder->tempWriteAddr[1] - SIMPO_BUILD_LENGTH_SIZE;
    *((uint32_t *)(builder->tempWriteAddr[1])) = len;
    return SIMPO_BUILD_OK;
}
static SIMPO_INLINE int32_t BifrostCncoiRootRankDeleteStartAsRoot(SimpoBuilderT *builder, uint8_t *buffer,
    uint32_t bufferLen)
{
    builder->buffer = buffer;
    builder->bufferEnd = buffer + bufferLen;
    builder->curWriteAddr = buffer + SIMPO_BUILD_LENGTH_SIZE;
    if (bufferLen < (SIMPO_BUILD_LENGTH_SIZE + 152)) {
        builder->errorCode = SIMPO_BUILD_NOMEM;
        return builder->errorCode;
    }
    builder->errorCode = SIMPO_BUILD_OK;
    *((uint32_t *)buffer) = builder->msgHead;
    SIMPO_OPER_INIT(builder, 1);
    return builder->errorCode;
}

static SIMPO_INLINE int32_t BifrostCncoiRootRankDeleteEndAsRoot(SimpoBuilderT *builder)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_END(builder, 1, BifrostCncoiRootRankDeleteEndAsRoot, 0);

    builder->curWriteAddr += 152;
    uint32_t size = builder->curWriteAddr - builder->buffer;

    return size;
}
static SIMPO_INLINE int32_t BifrostCncoiRootRankDeleteKeyCreate(SimpoBuilderT *builder, BifrostCncoiRootRankKeyT *value)
{
    if (builder->errorCode != SIMPO_BUILD_OK) {
        return builder->errorCode;
    }
    SIMPO_CHECK_OPER_ORDER(builder, 1, BifrostCncoiRootRankDeleteKeyCreate, 0);
    *((BifrostCncoiRootRankKeyT *)(builder->curWriteAddr + 0)) = *value;

    return SIMPO_BUILD_OK;
}
#ifdef __cplusplus
}
#endif

#endif /* BIFROST_CNCOI_BUILD_MULTI_LAYER_H */
