/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_dc_pri.h"
#include "bkf_dc_itor_pri.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define LOG_ID_DC_TUPLE_EXCEED 0x089611b8
/* DC_TUPLE_EXCEED(D):(tableType=%u,%s, tupleCount=%d, tupleCountMax=%d) */
#define DIAG_ID_FMT_DC_TUPLE_EXCEED LOG_ID_DC_TUPLE_EXCEED, "%u%s%d%d"

BkfDcTuple *BkfDcAddTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey, void *tupleValOrNull)
{
    uint32_t hashIdx = BKF_DC_GET_TUPLE_HASH_IDX(tupleKey);
    BkfDcTuple *tuple = VOS_NULL;
    BkfDcTableType *tableType = table->tableType;
    uint32_t len;
    void *valDest = VOS_NULL;
    BOOL insOk = VOS_FALSE;
    errno_t err;

    len = sizeof(BkfDcTuple) + tableType->vTbl.tupleKeyLen + tableType->vTbl.tupleValLen;
    tuple = BKF_MALLOC(dc->argInit.memMng, len);
    if (tuple == VOS_NULL) {
        goto error;
    }
    (void)memset_s(tuple, len, 0, len);
    BKF_DL_NODE_INIT(&tuple->dlNodeBySeq);
    BKF_SIGN_SET(tuple->sign, BKF_DC_TUPLE_SIGN);
    tuple->isAddUpd = VOS_TRUE;
    tuple->seq = BKF_DC_SEQ_INVALID;
    err = memcpy_s(tuple->keyVal, tableType->vTbl.tupleKeyLen, tupleKey, tableType->vTbl.tupleKeyLen);
    BKF_ASSERT(err == EOK);
    valDest = BKF_DC_TUPLE_GET_VAL(tuple, tableType);
    if ((valDest != VOS_NULL) && (tupleValOrNull != VOS_NULL)) {
        err = memcpy_s(valDest, tableType->vTbl.tupleValLen, tupleValOrNull, tableType->vTbl.tupleValLen);
        BKF_ASSERT(err == EOK);
    }
    VOS_AVLL_INIT_NODE(tuple->avlNode);
    insOk = VOS_AVLL_INSERT(table->tupleSet, tuple->avlNode);
    if (!insOk) {
        goto error;
    }

    tableType->tupleCntAddUpd++;
    table->tupleCache[hashIdx] = tuple;
    return tuple;

error:

    if (tuple != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(dc->argInit.memMng, tuple);
    }
    return VOS_NULL;
}

uint32_t BkfDcUpdTuple(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple, void *tupleValOrNull)
{
    (void)dc;
    uint32_t hashIdx = BKF_DC_GET_TUPLE_HASH_IDX(tuple->keyVal);
    BkfDcTableType *tableType = table->tableType;
    void *valDest = VOS_NULL;
    errno_t err;

    if (!BKF_SIGN_IS_VALID(tuple->sign, BKF_DC_TUPLE_SIGN)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    if (!tuple->isAddUpd) {
        tuple->isAddUpd = VOS_TRUE;
        tableType->tupleCntAddUpd++;
    }
    valDest = BKF_DC_TUPLE_GET_VAL(tuple, tableType);
    if ((valDest != VOS_NULL) && (tupleValOrNull != VOS_NULL)) {
        err = memcpy_s(valDest, tableType->vTbl.tupleValLen, tupleValOrNull, tableType->vTbl.tupleValLen);
        BKF_ASSERT(err == EOK);
    }

    table->tupleCache[hashIdx] = tuple;
    return BKF_OK;
}

void BkfDcSetTupleDelState(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple)
{
    (void)dc;
    BkfDcTableType *tableType = table->tableType;
    void *valDest = VOS_NULL;

    if (!BKF_SIGN_IS_VALID(tuple->sign, BKF_DC_TUPLE_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    if (tuple->isAddUpd) {
        tuple->isAddUpd = VOS_FALSE;
        tableType->tupleCntAddUpd--;
        valDest = BKF_DC_TUPLE_GET_VAL(tuple, tableType);
        if (valDest != VOS_NULL) {
            (void)memset_s(valDest, tableType->vTbl.tupleValLen, 0, tableType->vTbl.tupleValLen);
        }
    }
    return;
}

STATIC BOOL BkfDcIsTupleProcedByAllSeqItor(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple)
{
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    void *itor = VOS_NULL;

    /* 当前简单采用遍历方式。后续可以优化，基于数据结构O(1)查找 */
    for (seqItor = BkfDcGetFirstTupleSeqItor(dc, table, &itor); seqItor != VOS_NULL;
         seqItor = BkfDcGetNextTupleSeqItor(dc, table, &itor)) {
        if (tuple->seq >= seqItor->nextProcTupleSeq) {
            return VOS_FALSE;
        }
    }
    return VOS_TRUE;
}
BOOL BkfDcTry2FreeTuple(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple)
{
    uint32_t hashIdx = BKF_DC_GET_TUPLE_HASH_IDX(tuple->keyVal);
    uint8_t sliceBuf[BKF_1K / 8];
    uint8_t tupleKeyBuf[BKF_1K / 8];

    if (!BKF_SIGN_IS_VALID(tuple->sign, BKF_DC_TUPLE_SIGN)) {
        BKF_ASSERT(0);
        return VOS_FALSE;
    }
    BkfDcSetTupleDelState(dc, table, tuple);
    if (!BkfDcIsTupleProcedByAllSeqItor(dc, table, tuple)) {
        return VOS_FALSE;
    }

    /* 满足条件，free */
    BKF_LOG_DEBUG(dc->log, "dc(%#x), sliceKey(%s)/type(%u, %s)/tupleKey(%s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, table->slice->key, sliceBuf, sizeof(sliceBuf)), table->tableTypeId,
        BkfDcGetTableTypeIdStr(dc, table->tableTypeId),
        BkfDcGetTupleKeyStr(dc, table->tableTypeId, tuple->keyVal, tupleKeyBuf, sizeof(tupleKeyBuf)));
    if (table->tupleCache[hashIdx] == tuple) {
        table->tupleCache[hashIdx] = VOS_NULL;
    }
    BKF_SIGN_CLR(tuple->sign);
    if (BKF_DL_NODE_IS_IN(&tuple->dlNodeBySeq)) {
        BKF_DL_REMOVE(&tuple->dlNodeBySeq);
    } else {
        BKF_ASSERT(0);
    }
    VOS_AVLL_DELETE(table->tupleSet, tuple->avlNode);
    BKF_FREE(dc->argInit.memMng, tuple);
    return VOS_TRUE;
}

BOOL BkfDcTry2FreeAllTuple(BkfDc *dc, BkfDcTable *table)
{
    BkfDcTuple *tuple = VOS_NULL;
    void *itor = VOS_NULL;
    BOOL succ = VOS_TRUE;

    for (tuple = BkfDcGetFirstTuple(dc, table, &itor); tuple != VOS_NULL; tuple = BkfDcGetNextTuple(dc, table, &itor)) {
        succ = succ && BkfDcTry2FreeTuple(dc, table, tuple);
    }
    return succ;
}

BkfDcTuple *BkfDcFindTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey)
{
    (void)dc;
    uint32_t hashIdx = BKF_DC_GET_TUPLE_HASH_IDX(tupleKey);
    F_BKF_CMP cmp = table->tableType->vTbl.tupleKeyCmp;
    BOOL hit = VOS_FALSE;
    BkfDcTuple *tuple = VOS_NULL;

    tuple = table->tupleCache[hashIdx];
    hit = (tuple != VOS_NULL) && (cmp(tupleKey, &tuple->keyVal[0]) == 0);
    if (hit) {
        return tuple;
    }
    tuple = VOS_AVLL_FIND(table->tupleSet, tupleKey);
    if (tuple != VOS_NULL) {
        table->tupleCache[hashIdx] = tuple;
    }
    return tuple;
}

BkfDcTuple *BkfDcFindNextTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey)
{
    (void)dc;
    BkfDcTuple *tuple = VOS_NULL;

    tuple = VOS_AVLL_FIND_NEXT(table->tupleSet, tupleKey);
    return tuple;
}

BkfDcTuple *BkfDcGetFirstTuple(BkfDc *dc, BkfDcTable *table, void **itorOutOrNull)
{
    (void)dc;
    BkfDcTuple *tuple = VOS_NULL;

    tuple = VOS_AVLL_FIRST(table->tupleSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (tuple != VOS_NULL) ? VOS_AVLL_NEXT(table->tupleSet, tuple->avlNode) : VOS_NULL;
    }
    return tuple;
}

BkfDcTuple *BkfDcGetNextTuple(BkfDc *dc, BkfDcTable *table, void **itorInOut)
{
    (void)dc;
    BkfDcTuple *tuple = VOS_NULL;

    tuple = (*itorInOut);
    *itorInOut = (tuple != VOS_NULL) ? VOS_AVLL_NEXT(table->tupleSet, tuple->avlNode) : VOS_NULL;
    return tuple;
}

#if BKF_BLOCK("tuple limit")
STATIC uint32_t BkfDcTupleLimitOnSysLogCode(BkfDc *dc, uint16_t *tableTypeId, int32_t *logTimeTupleCntAddUpd,
    F_BKF_SYS_LOG_PRINTF logPrintf, uint32_t logPrintfParam1)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = BkfDcFindTableType(dc, *tableTypeId);
    if (tableType == VOS_NULL) {
        return BKF_ERR;
    }

    (void)logPrintf(logPrintfParam1, DIAG_ID_FMT_DC_TUPLE_EXCEED, tableType->vTbl.tableTypeId, tableType->name,
        *logTimeTupleCntAddUpd, tableType->vTbl.tupleCntMax);
    return BKF_OK;
}
uint32_t BkfDcTupleLimitInit(BkfDc *dc)
{
    BkfSysLogTypeVTbl vTbl = {0};

    vTbl.name = dc->name;
    vTbl.cookie = dc;
    vTbl.typeId = LOG_ID_DC_TUPLE_EXCEED;
    vTbl.needRestrain = VOS_TRUE;
    vTbl.keyLen = BKF_MBR_SIZE(BkfDcTableTypeVTbl, tableTypeId);
    vTbl.valLen = BKF_MBR_SIZE(BkfDcTableType, tupleCntAddUpd);
    vTbl.keyCmp = (F_BKF_CMP)Bkfuint16_tCmp;
    vTbl.keyGetStrOrNull = (F_BKF_GET_STR)Bkfuint16_tGetStr;
    vTbl.valGetStrOrNull = (F_BKF_GET_STR)BkfInt32GetStr;
    vTbl.out = (F_BKF_SYS_LOG_OUT)BkfDcTupleLimitOnSysLogCode;
    return BkfSysLogReg(dc->argInit.sysLogMng, &vTbl);
}

BOOL BkfDcTupleLimitMayAddTuple(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        return VOS_FALSE;
    }

    return (tableType->tupleCntAddUpd < tableType->vTbl.tupleCntMax);
}

uint32_t BkfDcTupleLimitSysLog(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSysLogFunc(dc->argInit.sysLogMng, LOG_ID_DC_TUPLE_EXCEED, &tableTypeId, &tableType->tupleCntAddUpd);
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
