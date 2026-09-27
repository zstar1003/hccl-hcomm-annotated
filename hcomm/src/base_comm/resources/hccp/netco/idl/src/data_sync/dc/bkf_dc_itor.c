/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((dc)->log)
#define BKF_MOD_NAME ((dc)->name)

#include "bkf_dc_itor_pri.h"
#include "bkf_str.h"
#include "securec.h"
#include "vos_base.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define BKF_DC_ITOR_PROC_TUPLE_CNT_MAX 1024

STATIC uint32_t BkfDcOnTableJobDelTuple(BkfDcTable *table, uint32_t *runCost);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
/* func */
uint32_t BkfDcItorInit(BkfDc *dc)
{
    uint32_t ret;

    if (dc == VOS_NULL) {
        return BKF_ERR;
    }

    ret = BkfJobRegType(dc->argInit.jobMng, dc->argInit.jobTypeId, (F_BKF_JOB_PROC)BkfDcOnTableJobDelTuple,
        dc->argInit.jobPrio);
    BKF_LOG_DEBUG(BKF_LOG_HND, "ret(%u)\n", ret);
    return ret;
}

void BkfDcItorUninit(BkfDc *dc)
{
    if (dc == VOS_NULL) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc itor uninit done\n");

    return;
}

/* keyItor */
BkfDcTupleKeyItor *BkfDcNewTupleKeyItor(BkfDc *dc, void *sliceKey, uint16_t tableTypeId)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    BkfDcTupleKeyItor *keyItor = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        goto error;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "sliceKey(%s)/table(%u, %s)\n",
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId));

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        goto error;
    }
    table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        goto error;
    }
    keyItor = BkfDcAddTupleKeyItor(dc, table);
    return keyItor;

error:

    /* keyItor一定为空 */
    return VOS_NULL;
}

void BkfDcDeleteTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor)
{
    uint16_t tableTypeId;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (keyItor == VOS_NULL)) {
        return;
    }
    if (!BKF_SIGN_IS_VALID(keyItor->sign, BKF_DC_TUPLE_KEY_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    tableTypeId = keyItor->tableType->vTbl.tableTypeId;
    BKF_LOG_DEBUG(BKF_LOG_HND, "sliceKey(%s)/table(%u, %s), keyItor(%#x)\n",
        BkfDcGetSliceKeyStr(dc, keyItor->sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId), BKF_MASK_ADDR(keyItor));

    BkfDcDelTupleKeyItor(dc, keyItor);
    return;
}

static inline void BkfDcLogTupleInfo(BkfDc *dc, const char *outStr, void *sliceKey, uint16_t tableTypeId,
    void *itorAddr, BkfDcTupleInfo *info)
{
    uint8_t sliceBuf[BKF_1K / 8];
    uint8_t keyBuf[BKF_1K / 8];
    uint8_t valBuf[BKF_1K / 8];

    BKF_LOG_DEBUG(BKF_LOG_HND,
        "%s, sliceKey(%s)/table(%u, %s)/itor(%#x), getTupleKey(%s)/val(%s)/isAddUpd(%u)/"
        "seq(%" VOS_PRIu64 ")\n",
        outStr, BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId), BKF_MASK_ADDR(itorAddr),
        BkfDcGetTupleKeyStr(dc, tableTypeId, info->key, keyBuf, sizeof(keyBuf)),
        BkfDcGetTupleValStr(dc, tableTypeId, info->valOrNull, valBuf, sizeof(valBuf)), info->isAddUpd, info->seq);
    return;
}

BOOL BkfDcGetTupleByKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor, BkfDcTupleInfo *info)
{
    BkfDcTableType *tableType = VOS_NULL;
    BOOL paramIsInvalid = VOS_FALSE;
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;

    paramIsInvalid = (dc == VOS_NULL) || (keyItor == VOS_NULL) || (info == VOS_NULL);
    if (paramIsInvalid) {
        return VOS_FALSE;
    }
    info->key = VOS_NULL;
    info->isAddUpd = VOS_FALSE;
    info->valOrNull = VOS_NULL;
    info->seq = BKF_DC_TUPLE_SEQ_MAX;
    if (!BKF_SIGN_IS_VALID(keyItor->sign, BKF_DC_TUPLE_KEY_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return VOS_FALSE;
    }
    tableType = keyItor->tableType;

    slice = BkfDcFindSlice(dc, keyItor->sliceKey);
    if (slice == VOS_NULL) {
        return VOS_FALSE;
    }
    table = BkfDcFindTable(dc, slice, tableType->vTbl.tableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }
    tuple = (keyItor->getFirstTuple) ? BkfDcGetFirstTuple(dc, table, VOS_NULL)
                                     : BkfDcFindNextTuple(dc, table, &keyItor->lastGetTupleKey[0]);
    if (tuple != VOS_NULL) {
        info->key = &tuple->keyVal[0];
        info->isAddUpd = tuple->isAddUpd;
        info->valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, tableType) : VOS_NULL;
        info->seq = tuple->seq;
    }

    BkfDcLogTupleInfo(dc, "GetTupleByKeyItor", keyItor->sliceKey, tableType->vTbl.tableTypeId, keyItor, info);
    return (tuple != VOS_NULL) ? VOS_TRUE : VOS_FALSE;
}

BOOL BkfDcGetTupleByKeyItorFromApp(BkfDc *dc, BkfDcTupleKeyItor *keyItor, BkfDcTableType *tableType,
    BkfDcTupleInfo *info)
{
    uint32_t ret = BKF_OK;
    if (keyItor->getFirstTuple) {
        ret = tableType->vTbl.getFirst(tableType->vTbl.cookie, tableType->vTbl.tableTypeId, keyItor->tupleKey,
            keyItor->tupleVal);
    } else {
        ret = tableType->vTbl.getNext(tableType->vTbl.cookie, tableType->vTbl.tableTypeId, keyItor->lastGetTupleKey,
            keyItor->tupleKey, keyItor->tupleVal);
    }
    if (ret != BKF_OK) {
        return VOS_FALSE;
    }

    info->isAddUpd = VOS_TRUE;
    info->key = keyItor->tupleKey;
    info->valOrNull = keyItor->tupleVal;
    BkfDcLogTupleInfo(dc, "GetTupleByKeyItorFromApp", keyItor->sliceKey, tableType->vTbl.tableTypeId, keyItor, info);
    return VOS_TRUE;
}

BOOL BkfDcGetTupleByKeyItorFromDcOrApp(BkfDc *dc, BkfDcTupleKeyItor *keyItor, BkfDcTupleInfo *info)
{
    BOOL paramIsInvalid = (dc == VOS_NULL) || (keyItor == VOS_NULL) || (info == VOS_NULL);
    if (paramIsInvalid) {
        return VOS_FALSE;
    }
    info->seq = BKF_DC_TUPLE_SEQ_MAX;
    if (!BKF_SIGN_IS_VALID(keyItor->sign, BKF_DC_TUPLE_KEY_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return VOS_FALSE;
    }
    BkfDcTableType *tableType = keyItor->tableType;

    BkfDcSlice *slice = BkfDcFindSlice(dc, keyItor->sliceKey);
    if (slice == VOS_NULL) {
        return VOS_FALSE;
    }
    BkfDcTable *table = BkfDcFindTable(dc, slice, tableType->vTbl.tableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }

    if (tableType->vTbl.noDcTuple) {
        return BkfDcGetTupleByKeyItorFromApp(dc, keyItor, tableType, info);
    }
    BkfDcTuple *tuple = (keyItor->getFirstTuple) ? BkfDcGetFirstTuple(dc, table, VOS_NULL)
                                                 : BkfDcFindNextTuple(dc, table, &keyItor->lastGetTupleKey[0]);
    if (tuple != VOS_NULL) {
        info->key = &tuple->keyVal[0];
        info->isAddUpd = tuple->isAddUpd;
        info->valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, tableType) : VOS_NULL;
        info->seq = tuple->seq;
    }

    BkfDcLogTupleInfo(dc, "GetTupleByKeyItorFromDcOrApp", keyItor->sliceKey, tableType->vTbl.tableTypeId, keyItor,
        info);
    return (tuple != VOS_NULL) ? VOS_TRUE : VOS_FALSE;
}

STATIC uint32_t BkfDcForwordTupleKeyItorChkParam(BkfDc *dc, BkfDcTupleKeyItor *keyItor)
{
    uint8_t sliceBuf[BKF_1K / 8];
    uint8_t keyBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (keyItor == VOS_NULL)) {
        return BKF_ERR;
    }

    if (!BKF_SIGN_IS_VALID(keyItor->sign, BKF_DC_TUPLE_KEY_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    uint16_t tableTypeId = keyItor->tableType->vTbl.tableTypeId;
    /* 此时itor指向节点不一定在tuple中，所以只输出itor内缓冲的数据 */
    BKF_LOG_DEBUG(BKF_LOG_HND, "sliceKey(%s)/table(%u, %s), keyItor(%#x), getFirstTuple(%u)/lastGetTupleKey(%s)\n",
        BkfDcGetSliceKeyStr(dc, keyItor->sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId), BKF_MASK_ADDR(keyItor), keyItor->getFirstTuple,
        BkfDcGetTupleKeyStr(dc, tableTypeId, keyItor->lastGetTupleKey, keyBuf, sizeof(keyBuf)));

    return BKF_OK;
}

void BkfDcForwordTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor)
{
    uint32_t ret;
    BkfDcTableTypeVTbl *vTbl = VOS_NULL;
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;

    ret = BkfDcForwordTupleKeyItorChkParam(dc, keyItor);
    if (ret != BKF_OK) {
        return;
    }

    vTbl = &keyItor->tableType->vTbl;
    slice = BkfDcFindSlice(dc, keyItor->sliceKey);
    if (slice == VOS_NULL) {
        return;
    }
    table = BkfDcFindTable(dc, slice, vTbl->tableTypeId);
    if (table == VOS_NULL) {
        return;
    }

    if (vTbl->noDcTuple) {
        if (keyItor->getFirstTuple) {
            keyItor->getFirstTuple = VOS_FALSE; /* 无论树上是否有tuple，都无条件置false */
        }
        (void)memcpy_s(&keyItor->lastGetTupleKey[0], vTbl->tupleKeyLen, keyItor->tupleKey, vTbl->tupleKeyLen);
        return;
    }

    if (keyItor->getFirstTuple) {
        keyItor->getFirstTuple = VOS_FALSE; /* 无论树上是否有tuple，都无条件置false */
        tuple = BkfDcGetFirstTuple(dc, table, VOS_NULL);
    } else {
        tuple = BkfDcFindNextTuple(dc, table, &keyItor->lastGetTupleKey[0]);
    }

    if (tuple != VOS_NULL) {
        (void)memcpy_s(&keyItor->lastGetTupleKey[0], vTbl->tupleKeyLen, &tuple->keyVal[0], vTbl->tupleKeyLen);
    }
    return;
}

void BkfDcForwordDelTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *delKeyItor)
{
    BkfDcTuple *tuple = VOS_NULL;
    uint32_t ret = BkfDcForwordTupleKeyItorChkParam(dc, delKeyItor);
    if (ret != BKF_OK) {
        return;
    }

    BkfDcTableTypeVTbl *vTbl = &delKeyItor->tableType->vTbl;
    BkfDcSlice *slice = BkfDcFindSlice(dc, delKeyItor->sliceKey);
    if (slice == VOS_NULL) {
        return;
    }
    BkfDcTable *table = BkfDcFindTable(dc, slice, vTbl->tableTypeId);
    if (table == VOS_NULL) {
        return;
    }

    if (delKeyItor->getFirstTuple) {
        delKeyItor->getFirstTuple = VOS_FALSE; /* 无论树上是否有tuple，都无条件置false */
        tuple = BkfDcGetFirstTuple(dc, table, VOS_NULL);
    } else {
        tuple = BkfDcFindNextTuple(dc, table, &delKeyItor->lastGetTupleKey[0]);
    }

    if (tuple != VOS_NULL) {
        (void)memcpy_s(&delKeyItor->lastGetTupleKey[0], vTbl->tupleKeyLen, &tuple->keyVal[0], vTbl->tupleKeyLen);
    }
    return;
}

/* seqItor */
BkfDcTupleSeqItor *BkfDcNewTupleSeqItor(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, BkfDcTupleItorVTbl *vTbl)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (vTbl == VOS_NULL)) {
        goto error;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "sliceKey(%s)/table(%u, %s), vTbl(%#x)\n",
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId), BKF_MASK_ADDR(vTbl));

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        goto error;
    }
    table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        goto error;
    }
    seqItor = BkfDcAddTupleSeqItor(dc, table, vTbl);
    return seqItor;

error:

    /* seqItor一定为空 */
    return VOS_NULL;
}

void BkfDcDeleteTupleSeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor)
{
    BkfDcTable *table = VOS_NULL;
    uint16_t tableTypeId = 0;
    BkfDcSlice *slice = VOS_NULL;
    char *resultStr = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];
    BkfDcTupleSeqItor *tmpSeqItor = VOS_NULL;
    BkfDcTuple *tmpDcTuple = VOS_NULL;
    uint64_t tmpSeq;

    BKF_ASSERT(dc != VOS_NULL);
    BKF_ASSERT(seqItor != VOS_NULL);
    if ((dc == VOS_NULL) || (seqItor == VOS_NULL)) {
        return;
    }
    if (!BKF_SIGN_IS_VALID(seqItor->sign, BKF_DC_TUPLE_SEQ_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    table = seqItor->table;
    tableTypeId = table->tableTypeId;
    slice = table->slice;
    tmpSeqItor = seqItor;
    tmpDcTuple = seqItor->nextProcTuple;
    tmpSeq = seqItor->nextProcTupleSeq;

    /*
    1. 删除迭代器
    2. 开启遍历tuple删数据流程
    */
    (void)BkfDcDelTupleSeqItor(dc, seqItor);

    if (table->jobIdDelTuple != VOS_NULL) {
        BkfJobDelete(dc->argInit.jobMng, table->jobIdDelTuple);
        table->jobIdDelTuple = VOS_NULL;
    }
    if (table->keyItorDelTuple != VOS_NULL) {
        BkfDcDeleteTupleKeyItor(dc, table->keyItorDelTuple);
        table->keyItorDelTuple = VOS_NULL;
    }
    table->keyItorDelTuple = BkfDcNewTupleKeyItor(dc, &slice->key[0], tableTypeId);
    if (table->keyItorDelTuple == VOS_NULL) {
        goto error;
    }
    table->jobIdDelTuple = BkfJobCreate(dc->argInit.jobMng, dc->argInit.jobTypeId, "jobIdDelTuple", table);

error:

    resultStr = ((table->jobIdDelTuple != VOS_NULL) && (table->keyItorDelTuple != VOS_NULL)) ? "ok" : "ng";
    BKF_LOG_DEBUG(BKF_LOG_HND,
        "sliceKey(%s)/table(%u, %s), del seqItor(%#x)/nextProcTuple(%#x)/"
        "nextProcTupleSeq(%" VOS_PRIx64 "), table keyItorDelTuple(%#x)/jobIdDelTuple(%#x), %s\n",
        BkfDcGetSliceKeyStr(dc, &slice->key[0], sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId), BKF_MASK_ADDR(tmpSeqItor), BKF_MASK_ADDR(tmpDcTuple), tmpSeq,
        BKF_MASK_ADDR(table->keyItorDelTuple), BKF_MASK_ADDR(table->jobIdDelTuple), resultStr);
    return;
}

BOOL BkfDcGetTupleBySeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor, BkfDcTupleInfo *info)
{
    BkfDcTable *table = VOS_NULL;
    BOOL paramIsInvalid = VOS_FALSE;
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTableType *tableType = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;
    BOOL isValid = VOS_FALSE;

    paramIsInvalid = (dc == VOS_NULL) || (seqItor == VOS_NULL) || (info == VOS_NULL);
    if (paramIsInvalid) {
        return VOS_FALSE;
    }
    if (!BKF_SIGN_IS_VALID(seqItor->sign, BKF_DC_TUPLE_SEQ_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return VOS_FALSE;
    }
    table = seqItor->table;
    slice = table->slice;
    tableType = table->tableType;

    info->key = VOS_NULL;
    info->isAddUpd = VOS_FALSE;
    info->valOrNull = VOS_NULL;
    info->seq = BKF_DC_TUPLE_SEQ_MAX;
    tuple = VOS_NULL;
    if (seqItor->nextProcTuple != VOS_NULL) {
        /* nextProcTuple 和 nextProcTupleSeq 是同步设置的。理论上不可能不匹配 */
        isValid = BKF_SIGN_IS_VALID(seqItor->nextProcTuple->sign, BKF_DC_TUPLE_SIGN) &&
                  (seqItor->nextProcTuple->seq == seqItor->nextProcTupleSeq);
        if (isValid) {
            tuple = seqItor->nextProcTuple;
        } else {
            BKF_ASSERT(0);
        }
    }
    if (tuple != VOS_NULL) {
        info->key = &tuple->keyVal[0];
        info->isAddUpd = tuple->isAddUpd;
        info->valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, tableType) : VOS_NULL;
        info->seq = tuple->seq;
    }

    BkfDcLogTupleInfo(dc, "GetTupleBySeqItor", slice->key, tableType->vTbl.tableTypeId, seqItor, info);
    return (tuple != VOS_NULL) ? VOS_TRUE : VOS_FALSE;
}

STATIC uint32_t BkfDcForwordTupleSeqItorChkParam(BkfDc *dc, BkfDcTupleSeqItor *seqItor)
{
    BkfDcTable *table = VOS_NULL;
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTableType *tableType = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;
    BkfDcTupleInfo info;

    if ((dc == VOS_NULL) || (seqItor == VOS_NULL)) {
        return BKF_ERR;
    }
    if (!BKF_SIGN_IS_VALID(seqItor->sign, BKF_DC_TUPLE_SEQ_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    table = seqItor->table;
    slice = table->slice;
    tableType = table->tableType;

    tuple = seqItor->nextProcTuple;
    if (tuple != VOS_NULL) {
        info.key = &tuple->keyVal[0];
        info.isAddUpd = tuple->isAddUpd;
        info.valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, tableType) : VOS_NULL;
        info.seq = tuple->seq;
    } else {
        info.key = VOS_NULL;
        info.isAddUpd = VOS_FALSE;
        info.valOrNull = VOS_NULL;
        info.seq = BKF_DC_TUPLE_SEQ_MAX;
    }
    BkfDcLogTupleInfo(dc, "ForwordTupleSeqItor", slice->key, tableType->vTbl.tableTypeId, seqItor, &info);
    return BKF_OK;
}
void BkfDcForwordTupleSeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor)
{
    uint32_t ret;
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *procedTuple = VOS_NULL;
    BkfDlNode *tempNode = VOS_NULL;

    ret = BkfDcForwordTupleSeqItorChkParam(dc, seqItor);
    if (ret != BKF_OK) {
        return;
    }
    table = seqItor->table;

    if (seqItor->nextProcTuple != VOS_NULL) {
        procedTuple = seqItor->nextProcTuple;
        if (!BKF_SIGN_IS_VALID(procedTuple->sign, BKF_DC_TUPLE_SIGN)) {
            BKF_ASSERT(0);
            return;
        }

        tempNode = BKF_DL_GET_NEXT(&table->tupleSetBySeq, &seqItor->nextProcTuple->dlNodeBySeq);
        if (tempNode != VOS_NULL) {
            seqItor->nextProcTuple = BKF_DL_GET_ENTRY(tempNode, BkfDcTuple, dlNodeBySeq);
            seqItor->nextProcTupleSeq = seqItor->nextProcTuple->seq;
        } else {
            seqItor->nextProcTuple = VOS_NULL;
            seqItor->nextProcTupleSeq = BKF_DC_TUPLE_SEQ_MAX;
        }

        if (!procedTuple->isAddUpd || table->tableType->vTbl.noDcTuple) {
            (void)BkfDcTry2FreeTuple(dc, table, procedTuple);
        }
    }
    return;
}

#endif

#if BKF_BLOCK("数据操作函数定义")
BkfDcTupleKeyItor *BkfDcAddTupleKeyItor(BkfDc *dc, BkfDcTable *table)
{
    uint16_t sliceKeyLen = dc->argInit.sliceVTbl.keyLen;
    BkfDcTableType *tableType = table->tableType;
    BkfDcTupleKeyItor *keyItor = VOS_NULL;
    uint32_t len;

    len = sizeof(BkfDcTupleKeyItor) + sliceKeyLen + tableType->vTbl.tupleKeyLen;
    keyItor = BKF_MALLOC(dc->argInit.memMng, len);
    if (keyItor == VOS_NULL) {
        goto error;
    }
    (void)memset_s(keyItor, len, 0, len);
    keyItor->getFirstTuple = VOS_TRUE;
    keyItor->tableType = tableType;
    keyItor->sliceKey = &keyItor->buf[0];
    if (tableType->vTbl.noDcTuple) {
        keyItor->tupleKey = BKF_MALLOC(dc->argInit.memMng, tableType->vTbl.tupleKeyLen);
        if (keyItor->tupleKey == VOS_NULL) {
            BKF_ASSERT(0);
            BKF_FREE(dc->argInit.memMng, keyItor);
            goto error;
        }
        (void)memset_s(keyItor->tupleKey, tableType->vTbl.tupleKeyLen, 0, tableType->vTbl.tupleKeyLen);
        if (tableType->vTbl.tupleValLen > 0) {
            keyItor->tupleVal = BKF_MALLOC(dc->argInit.memMng, tableType->vTbl.tupleValLen);
            if (keyItor->tupleVal == VOS_NULL) {
                BKF_ASSERT(0);
                BKF_FREE(dc->argInit.memMng, keyItor->tupleKey);
                BKF_FREE(dc->argInit.memMng, keyItor);
                goto error;
            }
            (void)memset_s(keyItor->tupleVal, tableType->vTbl.tupleValLen, 0, tableType->vTbl.tupleValLen);
        }
    }
    (void)memcpy_s(keyItor->sliceKey, sliceKeyLen, table->slice->key, sliceKeyLen);
    keyItor->lastGetTupleKey = keyItor->sliceKey + sliceKeyLen;
    BKF_SIGN_SET(keyItor->sign, BKF_DC_TUPLE_KEY_ITOR_SIGN);

    return keyItor;
error:

    /* keyItor一定为空 */
    return VOS_NULL;
}

void BkfDcDelTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor)
{
    if (!BKF_SIGN_IS_VALID(keyItor->sign, BKF_DC_TUPLE_KEY_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    BKF_SIGN_CLR(keyItor->sign);
    if (keyItor->tupleKey) {
        BKF_FREE(dc->argInit.memMng, keyItor->tupleKey);
    }
    if (keyItor->tupleVal) {
        BKF_FREE(dc->argInit.memMng, keyItor->tupleVal);
    }
    BKF_FREE(dc->argInit.memMng, keyItor);
    return;
}

BkfDcTupleSeqItor *BkfDcAddTupleSeqItor(BkfDc *dc, BkfDcTable *table, BkfDcTupleItorVTbl *vTbl)
{
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    uint32_t len;

    len = sizeof(BkfDcTupleSeqItor);
    seqItor = BKF_MALLOC(dc->argInit.memMng, len);
    if (seqItor == VOS_NULL) {
        goto error;
    }
    (void)memset_s(seqItor, len, 0, len);
    seqItor->table = table;
    BKF_DL_NODE_INIT(&seqItor->dlNode);
    seqItor->vTbl = *vTbl;
    seqItor->name = BkfStrNew(dc->argInit.memMng, "%s", vTbl->name);
    seqItor->vTbl.name = seqItor->name;
    seqItor->nextProcTuple = VOS_NULL;
    seqItor->nextProcTupleSeq = BKF_DC_TUPLE_SEQ_MAX;

    BKF_DL_ADD_LAST(&table->tupleSeqItorSet, &seqItor->dlNode);
    BKF_SIGN_SET(seqItor->sign, BKF_DC_TUPLE_SEQ_ITOR_SIGN);
    return seqItor;

error:

    /* seqItor 一定为空 */
    return VOS_NULL;
}

BOOL BkfDcDelTupleSeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor)
{
    if (!BKF_SIGN_IS_VALID(seqItor->sign, BKF_DC_TUPLE_SEQ_ITOR_SIGN)) {
        BKF_ASSERT(0);
        return VOS_FALSE;
    }
    if (seqItor->lockDel) {
        seqItor->softDel = VOS_TRUE;
        return VOS_FALSE;
    }

    BKF_DL_REMOVE(&seqItor->dlNode);
    BkfStrDel(seqItor->name);
    BKF_SIGN_CLR(seqItor->sign);
    BKF_FREE(dc->argInit.memMng, seqItor);
    return VOS_TRUE;
}

BkfDcTupleSeqItor *BkfDcGetFirstTupleSeqItor(BkfDc *dc, BkfDcTable *table, void **itorOutOrNull)
{
    (void)dc;
    BkfDlNode *tempNode = VOS_NULL;
    BkfDcTupleSeqItor *seqItor = VOS_NULL;

    tempNode = BKF_DL_GET_FIRST(&table->tupleSeqItorSet);
    if (tempNode != VOS_NULL) {
        seqItor = BKF_DL_GET_ENTRY(tempNode, BkfDcTupleSeqItor, dlNode);
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = BKF_DL_GET_NEXT(&table->tupleSeqItorSet, &seqItor->dlNode);
        }
    } else {
        seqItor = VOS_NULL;
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = VOS_NULL;
        }
    }
    return seqItor;
}

BkfDcTupleSeqItor *BkfDcGetNextTupleSeqItor(BkfDc *dc, BkfDcTable *table, void **itorInOut)
{
    (void)dc;
    BkfDlNode *tempNode = VOS_NULL;
    BkfDcTupleSeqItor *seqItor = VOS_NULL;

    tempNode = *itorInOut;
    if (tempNode != VOS_NULL) {
        seqItor = BKF_DL_GET_ENTRY(tempNode, BkfDcTupleSeqItor, dlNode);
        *itorInOut = BKF_DL_GET_NEXT(&table->tupleSeqItorSet, &seqItor->dlNode);
    } else {
        seqItor = VOS_NULL;
        *itorInOut = VOS_NULL;
    }
    return seqItor;
}

BkfDcTupleSeqItor *BkfDcGetNextTupleSeqItorBySeqItor(BkfDc *dc, BkfDcTable *table, BkfDcTupleSeqItor *seqItor)
{
    (void)dc;
    BkfDlNode *tempNode = VOS_NULL;
    BkfDcTupleSeqItor *nextSeqItor = VOS_NULL;

    tempNode = BKF_DL_GET_NEXT(&table->tupleSeqItorSet, &seqItor->dlNode);
    nextSeqItor = (tempNode != VOS_NULL) ? BKF_DL_GET_ENTRY(tempNode, BkfDcTupleSeqItor, dlNode) : VOS_NULL;
    return nextSeqItor;
}

void BkfDcUpdTableAllTupleSeqItorForTuple(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple)
{
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    void *itor = VOS_NULL;
    BkfDlNode *tempNode = VOS_NULL;

    /*
    1. 如果dataItor指向当前节点，指向当前节点的下一个
    2. 如果dataItor指向空，更新到当前节点
    3. 将当前节点挪到最后
    */
    tuple->seq = BKF_GET_NEXT_VAL(dc->seq64Seed);
    for (seqItor = BkfDcGetFirstTupleSeqItor(dc, table, &itor); seqItor != VOS_NULL;
         seqItor = BkfDcGetNextTupleSeqItor(dc, table, &itor)) {
        if (seqItor->nextProcTuple == tuple) {
            tempNode = BKF_DL_GET_NEXT(&table->tupleSetBySeq, &seqItor->nextProcTuple->dlNodeBySeq);
            if (tempNode != VOS_NULL) {
                seqItor->nextProcTuple = BKF_DL_GET_ENTRY(tempNode, BkfDcTuple, dlNodeBySeq);
                seqItor->nextProcTupleSeq = seqItor->nextProcTuple->seq;
            } else {
                seqItor->nextProcTuple = VOS_NULL;
                seqItor->nextProcTupleSeq = BKF_DC_TUPLE_SEQ_MAX;
            }
        }

        if (seqItor->nextProcTuple == VOS_NULL) {
            seqItor->nextProcTuple = tuple;
            seqItor->nextProcTupleSeq = tuple->seq;
        }
    }
    if (BKF_DL_NODE_IS_IN(&tuple->dlNodeBySeq)) {
        BKF_DL_REMOVE(&tuple->dlNodeBySeq);
    }
    BKF_DL_ADD_LAST(&table->tupleSetBySeq, &tuple->dlNodeBySeq);

    return;
}

void BkfDcNtfTableAllSeqItor(BkfDc *dc, BkfDcTable *table, F_DC_TUPLE_SEQ_ITOR_PROC proc)
{
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    BkfDcTupleSeqItor *seqItorNext = VOS_NULL;

    seqItor = BkfDcGetFirstTupleSeqItor(dc, table, VOS_NULL);
    while (seqItor != VOS_NULL) {
        seqItor->lockDel = VOS_TRUE;
        proc(seqItor);
        seqItor->lockDel = VOS_FALSE;

        seqItorNext = BkfDcGetNextTupleSeqItorBySeqItor(dc, table, seqItor);
        if (seqItor->softDel) {
            (void)BkfDcDelTupleSeqItor(dc, seqItor);
        }
        seqItor = seqItorNext;
    }
    return;
}

#endif

#if BKF_BLOCK("私有函数定义")
STATIC uint32_t BkfDcOnTableJobDelTuple(BkfDcTable *table, uint32_t *runCost)
{
    BkfDcSlice *slice = table->slice;
    BkfDc *dc = slice->dc;
    BkfDcTuple *tuple = VOS_NULL;
    uint32_t procedTupleCnt = 0;
    BkfDcTupleInfo info;

    *runCost = BkfJobGetRunCostMax(dc->argInit.jobMng);
    while (BkfDcGetTupleByKeyItor(dc, table->keyItorDelTuple, &info)) {
        BkfDcForwordDelTupleKeyItor(dc, table->keyItorDelTuple);
        if (!info.isAddUpd || table->tableType->vTbl.noDcTuple) {
            /* no数据模式所有tuple都要删除 */
            tuple = BkfDcFindTuple(dc, table, info.key);
            BKF_ASSERT((tuple != VOS_NULL) && !tuple->isAddUpd);
            if (tuple != VOS_NULL) {
                (void)BkfDcTry2FreeTuple(dc, table, tuple);
            }
        }

        procedTupleCnt++;
        if (procedTupleCnt >= BKF_DC_ITOR_PROC_TUPLE_CNT_MAX) {
            return BKF_JOB_CONTINUE;
        }
    }

    table->jobIdDelTuple = VOS_NULL;
    BkfDcDeleteTupleKeyItor(dc, table->keyItorDelTuple);
    table->keyItorDelTuple = VOS_NULL;
    return BKF_JOB_FINSIH;
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
