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
BkfDcTable *BkfDcAddTable(BkfDc *dc, BkfDcSlice *slice, uint16_t tableTypeId)
{
    uint8_t hashIdx = BKF_DC_GET_TABLE_HASH_IDX(tableTypeId);
    BkfDcTableType *tableType = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    uint32_t len;
    BOOL insOk = VOS_FALSE;

    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        goto error;
    }

    len = sizeof(BkfDcTable);
    table = BKF_MALLOC(dc->argInit.memMng, len);
    if (table == VOS_NULL) {
        goto error;
    }
    (void)memset_s(table, len, 0, len);
    table->tableType = tableType;
    table->slice = slice;
    table->tableTypeId = tableTypeId;
    VOS_AVLL_INIT_TREE(table->tupleSet, (AVLL_COMPARE)tableType->vTbl.tupleKeyCmp, BKF_OFFSET(BkfDcTuple, keyVal[0]),
        BKF_OFFSET(BkfDcTuple, avlNode));
    BKF_DL_INIT(&table->tupleSetBySeq);
    BKF_DL_INIT(&table->tupleSeqItorSet);
    VOS_AVLL_INIT_NODE(table->avlNode);
    insOk = VOS_AVLL_INSERT(slice->tableSet, table->avlNode);
    if (!insOk) {
        goto error;
    }

    slice->tableCache[hashIdx] = table;
    return table;

error:

    if (table != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(dc->argInit.memMng, table);
    }
    return VOS_NULL;
}

void BkfDcDelTable(BkfDc *dc, BkfDcTable *table)
{
    uint8_t hashIdx = BKF_DC_GET_TABLE_HASH_IDX(table->tableTypeId);
    BkfDcSlice *slice = table->slice;
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;
    void *itor = VOS_NULL;
    BOOL succ = VOS_FALSE;
    BOOL mayDelTable = VOS_TRUE;

    if (slice->tableCache[hashIdx] == table) {
        slice->tableCache[hashIdx] = VOS_NULL;
    }

    if (table->jobIdDelTuple != VOS_NULL) {
        BkfJobDelete(dc->argInit.jobMng, table->jobIdDelTuple);
    }
    if (table->keyItorDelTuple != VOS_NULL) {
        BkfDcDelTupleKeyItor(dc, table->keyItorDelTuple);
    }
    /* 上层创建seqItor，并保留其引用。正常情况上层主动删除，此处按道理应该为空。先尽力删除 */
    for (seqItor = BkfDcGetFirstTupleSeqItor(dc, table, &itor); seqItor != VOS_NULL;
         seqItor = BkfDcGetNextTupleSeqItor(dc, table, &itor)) {
        succ = BkfDcDelTupleSeqItor(dc, seqItor);
        if (!succ) {
            BKF_ASSERT(0);
            mayDelTable = VOS_FALSE;
        }
    }
    for (tuple = BkfDcGetFirstTuple(dc, table, &itor); tuple != VOS_NULL; tuple = BkfDcGetNextTuple(dc, table, &itor)) {
        succ = BkfDcTry2FreeTuple(dc, table, tuple);
        if (!succ) {
            BKF_ASSERT(0);
            mayDelTable = VOS_FALSE;
        }
    }
    if (mayDelTable) {
        VOS_AVLL_DELETE(slice->tableSet, table->avlNode);
        BKF_FREE(dc->argInit.memMng, table);
    }
    return;
}

void BkfDcDelAllTable(BkfDc *dc, BkfDcSlice *slice)
{
    BkfDcTable *table = VOS_NULL;
    void *itor = VOS_NULL;

    for (table = BkfDcGetFirstTable(dc, slice, &itor); table != VOS_NULL; table = BkfDcGetNextTable(dc, slice, &itor)) {
        BkfDcDelTable(dc, table);
    }
    return;
}

BkfDcTable *BkfDcFindTable(BkfDc *dc, BkfDcSlice *slice, uint16_t tableTypeId)
{
    (void)dc;
    uint8_t hashIdx = BKF_DC_GET_TABLE_HASH_IDX(tableTypeId);
    BkfDcTable *table = VOS_NULL;
    BOOL hit = VOS_FALSE;

    table = slice->tableCache[hashIdx];
    hit = (table != VOS_NULL) && (table->tableTypeId == tableTypeId);
    if (hit) {
        return table;
    }

    table = VOS_AVLL_FIND(slice->tableSet, &tableTypeId);
    if (table != VOS_NULL) {
        slice->tableCache[hashIdx] = table;
    }
    return table;
}

BkfDcTable *BkfDcFindNextTable(BkfDc *dc, BkfDcSlice *slice, uint16_t tableTypeId)
{
    (void)dc;
    BkfDcTable *table = VOS_NULL;

    table = VOS_AVLL_FIND_NEXT(slice->tableSet, &tableTypeId);
    return table;
}

BkfDcTable *BkfDcGetFirstTable(BkfDc *dc, BkfDcSlice *slice, void **itorOutOrNull)
{
    (void)dc;
    BkfDcTable *table = VOS_NULL;

    table = VOS_AVLL_FIRST(slice->tableSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (table != VOS_NULL) ? VOS_AVLL_NEXT(slice->tableSet, table->avlNode) : VOS_NULL;
    }
    return table;
}

BkfDcTable *BkfDcGetNextTable(BkfDc *dc, BkfDcSlice *slice, void **itorInOut)
{
    (void)dc;
    BkfDcTable *table = VOS_NULL;

    table = (*itorInOut);
    *itorInOut = (table != VOS_NULL) ? VOS_AVLL_NEXT(slice->tableSet, table->avlNode) : VOS_NULL;
    return table;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
