/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vos_base.h"
#include "bkf_dc_pri.h"
#include "bkf_dc_itor_pri.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

STATIC int32_t BkfDcDispGetTableTypeCnt(BkfDc *dc)
{
    BkfDcTableType *tableType = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (tableType = BkfDcGetFirstTableType(dc, &itor); tableType != VOS_NULL;
         tableType = BkfDcGetNextTableType(dc, &itor)) {
        cnt++;
    }
    return cnt;
}
STATIC int32_t BkfDcDispGetSliceCnt(BkfDc *dc)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        cnt++;
    }
    return cnt;
}
STATIC int32_t BkfDcDispGetSliceTableCnt(BkfDc *dc, BkfDcSlice *slice)
{
    BkfDcTable *table = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (table = BkfDcGetFirstTable(dc, slice, &itor); table != VOS_NULL; table = BkfDcGetNextTable(dc, slice, &itor)) {
        cnt++;
    }
    return cnt;
}
STATIC int32_t BkfDcDispGetAllTableCnt(BkfDc *dc)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        cnt += BkfDcDispGetSliceTableCnt(dc, slice);
    }
    return cnt;
}
STATIC void BkfDcDispGetTableTupleCnt(BkfDc *dc, BkfDcTable *table, int32_t *tupleAddUpdCnt, int32_t *tupleDelCnt)
{
    BkfDcTuple *tuple = VOS_NULL;
    void *itor = VOS_NULL;

    for (tuple = BkfDcGetFirstTuple(dc, table, &itor); tuple != VOS_NULL; tuple = BkfDcGetNextTuple(dc, table, &itor)) {
        if (tuple->isAddUpd) {
            (*tupleAddUpdCnt)++;
        } else {
            (*tupleDelCnt)++;
        }
    }
}
STATIC void BkfDcDispGetSliceTupleCnt(BkfDc *dc, BkfDcSlice *slice, int32_t *tupleAddUpdCnt, int32_t *tupleDelCnt)
{
    BkfDcTable *table = VOS_NULL;
    void *itor = VOS_NULL;

    for (table = BkfDcGetFirstTable(dc, slice, &itor); table != VOS_NULL; table = BkfDcGetNextTable(dc, slice, &itor)) {
        BkfDcDispGetTableTupleCnt(dc, table, tupleAddUpdCnt, tupleDelCnt);
    }
}
STATIC void BkfDcDispGetAllTupleCnt(BkfDc *dc, int32_t *tupleAddUpdCnt, int32_t *tupleDelCnt)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        BkfDcDispGetSliceTupleCnt(dc, slice, tupleAddUpdCnt, tupleDelCnt);
    }
}
STATIC int32_t BkfDcDispGetTableTupleMem(BkfDc *dc, BkfDcTable *table, int64_t *addUpdTupleMem, int64_t *delTupleMem)
{
    (void)delTupleMem;
    int32_t tupleAddUpdCnt = 0;
    int32_t tupleDelCnt = 0;
    BkfDcTableType *tableType = table->tableType;
    int32_t tupleLen = (int32_t)(sizeof(BkfDcTuple) + tableType->vTbl.tupleKeyLen + tableType->vTbl.tupleValLen);

    BkfDcDispGetTableTupleCnt(dc, table, &tupleAddUpdCnt, &tupleDelCnt);
    *addUpdTupleMem += (int64_t)tupleLen * tupleAddUpdCnt;
    *addUpdTupleMem += (int64_t)tupleLen * tupleDelCnt;
    return tupleLen;
}
STATIC void BkfDcDispGetSliceTupleMem(BkfDc *dc, BkfDcSlice *slice, int64_t *addUpdTupleMem, int64_t *delTupleMem)
{
    BkfDcTable *table = VOS_NULL;
    void *itor = VOS_NULL;

    for (table = BkfDcGetFirstTable(dc, slice, &itor); table != VOS_NULL; table = BkfDcGetNextTable(dc, slice, &itor)) {
        (void)BkfDcDispGetTableTupleMem(dc, table, addUpdTupleMem, delTupleMem);
    }
}
STATIC void BkfDcDispGetAllTupleMem(BkfDc *dc, int64_t *addUpdTupleMem, int64_t *delTupleMem)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        BkfDcDispGetSliceTupleMem(dc, slice, addUpdTupleMem, delTupleMem);
    }
}
STATIC int32_t BkfDcDispGetTableTupleSeqItorCnt(BkfDc *dc, BkfDcTable *table)
{
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (seqItor = BkfDcGetFirstTupleSeqItor(dc, table, &itor); seqItor != VOS_NULL;
         seqItor = BkfDcGetNextTupleSeqItor(dc, table, &itor)) {
        cnt++;
    }
    return cnt;
}
STATIC int32_t BkfDcDispGetSliceTupleSeqItorCnt(BkfDc *dc, BkfDcSlice *slice)
{
    BkfDcTable *table = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (table = BkfDcGetFirstTable(dc, slice, &itor); table != VOS_NULL; table = BkfDcGetNextTable(dc, slice, &itor)) {
        cnt += BkfDcDispGetTableTupleSeqItorCnt(dc, table);
    }
    return cnt;
}
STATIC int32_t BkfDcDispGetAllTupleSeqItorCnt(BkfDc *dc)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t cnt = 0;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        cnt += BkfDcDispGetSliceTupleSeqItorCnt(dc, slice);
    }
    return cnt;
}

STATIC void BkfDcDispMem(BkfDc *dc)
{
    BkfDisp *disp = dc->argInit.disp;
    int32_t tableTypeCnt = BkfDcDispGetTableTypeCnt(dc);
    int32_t sliceCnt = BkfDcDispGetSliceCnt(dc);
    int32_t tableCnt = BkfDcDispGetAllTableCnt(dc);
    int32_t tupleAddUpdCnt = 0;
    int32_t tupleDelCnt = 0;
    int32_t seqItorCnt = BkfDcDispGetAllTupleSeqItorCnt(dc);
    int64_t memHnd = sizeof(BkfDc);
    int64_t memTableType = (int64_t)sizeof(BkfDcTableType) * tableTypeCnt;
    int64_t memSlice = (int64_t)sizeof(BkfDcSlice) * sliceCnt;
    int64_t memTable = (int64_t)sizeof(BkfDcTable) * tableCnt;
    int64_t memAddUpdTuple = 0;
    int64_t memDelTuple = 0;
    int64_t memSeqItor = (int64_t)sizeof(BkfDcTupleSeqItor) * seqItorCnt;
    int64_t memTotal;

    BkfDcDispGetAllTupleCnt(dc, &tupleAddUpdCnt, &tupleDelCnt);
    BkfDcDispGetAllTupleMem(dc, &memAddUpdTuple, &memDelTuple);
    memTotal = memHnd + memTableType + memSlice + memTable + memAddUpdTuple + memDelTuple + memSeqItor;

    BKF_DISP_PRINTF(disp, "===cnt===\n");
    BKF_DISP_PRINTF(disp, "tableTypeCnt     : %-10d\n", tableTypeCnt);
    BKF_DISP_PRINTF(disp, "sliceCnt         : %-10d\n", sliceCnt);
    BKF_DISP_PRINTF(disp, "tableCnt         : %-10d\n", tableCnt);
    BKF_DISP_PRINTF(disp, "tupleAddUpdCnt   : %-10d\n", tupleAddUpdCnt);
    BKF_DISP_PRINTF(disp, "tupleDelCnt      : %-10d\n", tupleDelCnt);
    BKF_DISP_PRINTF(disp, "seqItorCnt       : %-10d\n", seqItorCnt);
    BKF_DISP_PRINTF(disp, "===mem===\n");
    BKF_DISP_PRINTF(disp, "memTotal         : %-10" VOS_PRId64 "\n", memTotal);
    BKF_DISP_PRINTF(disp, "memHnd           : %-10" VOS_PRId64 "\n", memHnd);
    BKF_DISP_PRINTF(disp, "memTableType     : %-10" VOS_PRId64 " = %d * %d\n", memTableType, sizeof(BkfDcTableType),
        tableTypeCnt);
    BKF_DISP_PRINTF(disp, "memSlice         : %-10" VOS_PRId64 " = %d * %d\n", memSlice, sizeof(BkfDcSlice), sliceCnt);
    BKF_DISP_PRINTF(disp, "memTable         : %-10" VOS_PRId64 " = %d * %d\n", memTable, sizeof(BkfDcTable), tableCnt);
    BKF_DISP_PRINTF(disp, "memAddUpdTuple   : %-10" VOS_PRId64 "\n", memAddUpdTuple);
    BKF_DISP_PRINTF(disp, "memDelTuple      : %-10" VOS_PRId64 "\n", memDelTuple);
    BKF_DISP_PRINTF(disp, "memSeqItor       : %-10" VOS_PRId64 " = %d * %d\n", memSeqItor, sizeof(BkfDcTupleSeqItor),
        seqItorCnt);
}
void BkfDcDisp(BkfDc *dc)
{
    BkfDisp *disp = dc->argInit.disp;

    BKF_DISP_PRINTF(disp, "%s\n", dc->name);
    BKF_DISP_PRINTF(disp, "dbgOn(%u)\n", dc->argInit.dbgOn);
    BKF_DISP_PRINTF(disp, "memMng(%#x)\n", BKF_MASK_ADDR(dc->argInit.memMng));
    BKF_DISP_PRINTF(disp, "disp(%#x)\n", BKF_MASK_ADDR(dc->argInit.disp));
    BKF_DISP_PRINTF(disp, "log(%#x/%#x)\n", BKF_MASK_ADDR(dc->argInit.log), BKF_MASK_ADDR(dc->log));
    BKF_DISP_PRINTF(disp, "jobMng(%#x)/jobTypeId(%u)/jobPrio(%u)\n", BKF_MASK_ADDR(dc->argInit.jobMng),
        dc->argInit.jobTypeId, dc->argInit.jobPrio);
    BKF_DISP_PRINTF(disp, "slice_keyLen(%u)/keyCmp(%#x)/keyGetStrOrNull(%#x)/keyCodec(%#x)\n",
        dc->argInit.sliceVTbl.keyLen, BKF_MASK_ADDR(dc->argInit.sliceVTbl.keyCmp),
        BKF_MASK_ADDR(dc->argInit.sliceVTbl.keyGetStrOrNull), BKF_MASK_ADDR(dc->argInit.sliceVTbl.keyCodec));
    BKF_DISP_PRINTF(disp, "sysLogMng(%#x)\n", BKF_MASK_ADDR(dc->argInit.sysLogMng));
    BKF_DISP_PRINTF(disp, "seq64Seed(%" VOS_PRIu64 ")\n", dc->seq64Seed);

    BkfDcDispMem(dc);
}

void BkfDcDispTableType(BkfDc *dc)
{
    BkfDisp *disp = dc->argInit.disp;
    BkfDcTableType *tableType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t memTupleTotal;

    int32_t idx = 0;
    for (tableType = BkfDcGetFirstTableType(dc, &itor); tableType != VOS_NULL;
         tableType = BkfDcGetNextTableType(dc, &itor)) {
        memTupleTotal = sizeof(BkfDcTuple) + tableType->vTbl.tupleKeyLen + tableType->vTbl.tupleValLen;
        BKF_DISP_PRINTF(disp,
            "tableType[%d]: id(%u)/name(%s)/cookie(%#x), memTupleTotal/key/val(%u, %u, %u), "
            "tupleKeyCmp(%#x)/tupleKeyGetStr(%#x)/tupleValGetStr(%#x)\n",
            idx, tableType->vTbl.tableTypeId, tableType->name, BKF_MASK_ADDR(tableType->vTbl.cookie), memTupleTotal,
            tableType->vTbl.tupleKeyLen, tableType->vTbl.tupleValLen, BKF_MASK_ADDR(tableType->vTbl.tupleKeyCmp),
            BKF_MASK_ADDR(tableType->vTbl.tupleKeyGetStrOrNull), BKF_MASK_ADDR(tableType->vTbl.tupleValGetStrOrNull));
        idx++;
    }
}

STATIC void BkfDcDispTableSummary(BkfDc *dc, BkfDcSlice *slice, uintptr_t dispTable)
{
    (void)dispTable;
    BkfDisp *disp = dc->argInit.disp;
    BkfDcTable *table = VOS_NULL;
    void *itor1 = VOS_NULL;
    int32_t tupleAddUpdCnt;
    int32_t tupleDelCnt;
    int32_t seqItorCnt;
    BkfDcTupleSeqItor *seqItor = VOS_NULL;
    void *itor2 = VOS_NULL;
    int32_t seqItorIdx;

    int32_t tableIdx = 0;
    for (table = BkfDcGetFirstTable(dc, slice, &itor1); table != VOS_NULL;
         table = BkfDcGetNextTable(dc, slice, &itor1)) {
        tupleAddUpdCnt = 0;
        tupleDelCnt = 0;
        BkfDcDispGetTableTupleCnt(dc, table, &tupleAddUpdCnt, &tupleDelCnt);
        seqItorCnt = BkfDcDispGetTableTupleSeqItorCnt(dc, table);
        BKF_DISP_PRINTF(disp, "++table[%d]: (%u, %s)/isRelease(%u)/jobIdDelTuple(%#x)/keyItorDelTuple(%#x)\n", tableIdx,
            table->tableTypeId, BkfDcGetTableTypeIdStr(dc, table->tableTypeId), table->isRelease,
            BKF_MASK_ADDR(table->jobIdDelTuple), BKF_MASK_ADDR(table->keyItorDelTuple));
        tableIdx++;

        BKF_DISP_PRINTF(disp, "++++tupleAddUpdCnt/tupleDelCnt/seqItorCnt: (%d/%d/%d)\n", tupleAddUpdCnt, tupleDelCnt,
            seqItorCnt);
        seqItorIdx = 0;
        for (seqItor = BkfDcGetFirstTupleSeqItor(dc, table, &itor2); seqItor != VOS_NULL;
             seqItor = BkfDcGetNextTupleSeqItor(dc, table, &itor2)) {
            BKF_DISP_PRINTF(disp,
                "++++seqItor[%d]: nextProcTuple(%#x)/nextProcTupleSeq(%" VOS_PRIu64 ")/"
                "name(%s)/cookie(%#x)/afterTupleChange(%#x)/afterTableRelease(%#x)\n",
                seqItorIdx, BKF_MASK_ADDR(seqItor->nextProcTuple), seqItor->nextProcTupleSeq, seqItor->name,
                BKF_MASK_ADDR(seqItor->vTbl.cookie), BKF_MASK_ADDR(seqItor->vTbl.afterTupleChangeOrNull),
                BKF_MASK_ADDR(seqItor->vTbl.afterTableReleaseOrNull));
            seqItorIdx++;
        }
    }
}
STATIC void BkfDcDispOneSliceSummary(BkfDc *dc, BkfDcSlice *slice, int32_t idx, uintptr_t dispTable)
{
    BkfDisp *disp = dc->argInit.disp;
    int32_t tableCnt = BkfDcDispGetSliceTableCnt(dc, slice);
    int32_t tupleAddUpdCnt = 0;
    int32_t tupleDelCnt = 0;
    int32_t seqItorCnt = BkfDcDispGetSliceTupleSeqItorCnt(dc, slice);
    int64_t memSlice = (int64_t)sizeof(BkfDcSlice);
    int64_t memTable = (int64_t)sizeof(BkfDcTable) * tableCnt;
    int64_t memAddUpdTuple = 0;
    int64_t memDelTuple = 0;
    int64_t memSeqItor = (int64_t)sizeof(BkfDcTupleSeqItor) * seqItorCnt;
    int64_t memTotal;
    uint8_t buf[BKF_1K];

    BkfDcDispGetSliceTupleCnt(dc, slice, &tupleAddUpdCnt, &tupleDelCnt);
    BkfDcDispGetSliceTupleMem(dc, slice, &memAddUpdTuple, &memDelTuple);
    memTotal = memSlice + memTable + memAddUpdTuple + memDelTuple + memSeqItor;

    BKF_DISP_PRINTF(disp, "===slice[%d]: [%s]============\n", idx,
        BkfDcGetSliceKeyStr(dc, slice->key, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp,
        "memTotal/slice/table/tupleAddUpd/tupleDel/seqItor_"
        "(%" VOS_PRId64 "/%" VOS_PRId64 "/%" VOS_PRId64 "/%" VOS_PRId64 "/%" VOS_PRId64 "/%" VOS_PRId64 ")\n",
        memTotal, memSlice, memTable, memAddUpdTuple, memDelTuple, memSeqItor);
    BKF_DISP_PRINTF(disp, "tableCnt/tupleAddUpdCnt/tupleDelCnt/seqItorCnt: (%d/%d/%d/%d)\n", tableCnt, tupleAddUpdCnt,
        tupleDelCnt, seqItorCnt);
    if (dispTable) {
        BkfDcDispTableSummary(dc, slice, dispTable);
    }
    BKF_DISP_PRINTF(disp, "\n");
}
void BkfDcDispSliceSummary(BkfDc *dc, uintptr_t dispTable)
{
    BkfDisp *disp = dc->argInit.disp;
    uint16_t sliceKeyLen = dc->argInit.sliceVTbl.keyLen;
    void *sliceKey = VOS_NULL;
    BkfDcSlice *slice = VOS_NULL;
    int32_t sliceCnt = 0;

    sliceKey = BKF_DISP_GET_LAST_CTX(disp, &sliceCnt);
    if (sliceKey == VOS_NULL) {
        slice = BkfDcGetFirstSlice(dc, VOS_NULL);
        sliceCnt = 0;
    } else {
        slice = BkfDcFindNextSlice(dc, sliceKey);
    }

    if (slice != VOS_NULL) {
        BkfDcDispOneSliceSummary(dc, slice, sliceCnt, dispTable);
        sliceCnt++;
        BKF_DISP_SAVE_CTX(disp, slice->key, sliceKeyLen, &sliceCnt, sizeof(sliceCnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d slice(s), ***\n", sliceCnt);
    }
}

typedef struct tagBkfDcDispTupleCtx2 {
    int32_t tupleAddUpdCnt;
    int32_t tupleDelCnt;

    BOOL sliceNoMoreTable;
    uint16_t tableTypeId;
    uint8_t pad1[2];
    BOOL tableNoMoreTuple;
    uint8_t tupleKey[BKF_DC_TUPLE_KEY_LEN_MAX];
} BkfDcDispTupleCtx2;
typedef struct tagBkfDcDispTupleTemp {
    BOOL isNewSlice;
    BkfDcSlice *slice;
    BOOL isNewTable;
    BkfDcTable *table;
    BkfDcTuple *tuple;
    uint16_t tupleKeyLen;
    uint16_t pad1[2];
} BkfDcDispTupleTemp;
STATIC void BkfDcDispTupleGetFirstTuple(BkfDc *dc, BkfDcDispTupleCtx2 *ctx2, BkfDcDispTupleTemp *temp)
{
    (void)ctx2;
    temp->isNewSlice = VOS_TRUE;
    temp->slice = BkfDcGetFirstSlice(dc, VOS_NULL);
    if (temp->slice != VOS_NULL) {
        temp->table = BkfDcGetFirstTable(dc, temp->slice, VOS_NULL);
    }
    temp->isNewTable = VOS_TRUE;
    if (temp->table != VOS_NULL) {
        temp->tuple = BkfDcGetFirstTuple(dc, temp->table, VOS_NULL);
    }
}
STATIC void BkfDcDispTupleGetIfLastSliceNoMoreTable(BkfDc *dc, void *sliceKey, BkfDcDispTupleCtx2 *ctx2,
    BkfDcDispTupleTemp *temp)
{
    (void)ctx2;
    temp->isNewSlice = VOS_TRUE;
    temp->slice = BkfDcFindNextSlice(dc, sliceKey);
    temp->isNewTable = VOS_TRUE;
    if (temp->slice != VOS_NULL) {
        temp->table = BkfDcGetFirstTable(dc, temp->slice, VOS_NULL);
    }
    if (temp->table != VOS_NULL) {
        temp->tuple = BkfDcGetFirstTuple(dc, temp->table, VOS_NULL);
    }
}
STATIC void BkfDcDispTupleGetIfLastTableNoMoreTuple(BkfDc *dc, void *sliceKey, BkfDcDispTupleCtx2 *ctx2,
    BkfDcDispTupleTemp *temp)
{
    temp->slice = BkfDcFindSlice(dc, sliceKey);
    if (temp->slice == VOS_NULL) {
        BkfDcDispTupleGetIfLastSliceNoMoreTable(dc, sliceKey, ctx2, temp);
    } else {
        temp->isNewSlice = VOS_FALSE;
        temp->isNewTable = VOS_TRUE;
        temp->table = BkfDcFindNextTable(dc, temp->slice, ctx2->tableTypeId);
        if (temp->table != VOS_NULL) {
            temp->tuple = BkfDcGetFirstTuple(dc, temp->table, VOS_NULL);
        }
    }
}
STATIC void BkfDcDispTupleGetNextTuple(BkfDc *dc, void *sliceKey, BkfDcDispTupleCtx2 *ctx2, BkfDcDispTupleTemp *temp)
{
    temp->slice = BkfDcFindSlice(dc, sliceKey);
    if (temp->slice == VOS_NULL) {
        BkfDcDispTupleGetIfLastSliceNoMoreTable(dc, sliceKey, ctx2, temp);
    } else {
        temp->table = BkfDcFindTable(dc, temp->slice, ctx2->tableTypeId);
        if (temp->table == VOS_NULL) {
            BkfDcDispTupleGetIfLastTableNoMoreTuple(dc, sliceKey, ctx2, temp);
        } else {
            temp->isNewSlice = VOS_FALSE;
            temp->isNewTable = VOS_FALSE;
            temp->tuple = BkfDcFindNextTuple(dc, temp->table, ctx2->tupleKey);
        }
    }
}
STATIC void BkfDcDispOneTuple(BkfDc *dc, BkfDcDispTupleCtx2 *ctx2, BkfDcSlice *slice, BkfDcTable *table,
    BkfDcTuple *tuple)
{
    BkfDisp *disp = dc->argInit.disp;
    uint8_t sliceBuf[BKF_1K / 8];
    uint8_t tupleKeyBuf[BKF_1K / 8];
    uint8_t tupleValBuf[BKF_1K / 8];
    int32_t idx;
    void *tupleValOrNull = VOS_NULL;

    idx = ctx2->tupleAddUpdCnt + ctx2->tupleDelCnt;
    tupleValOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, table->tableType) : VOS_NULL;
    BKF_DISP_PRINTF(disp, "slice(%s)/table(%u, %s), tuple[%d]_addUpd?(%u)/key(%s)/val(%s)\n",
        BkfDcGetSliceKeyStr(dc, slice->key, sliceBuf, sizeof(sliceBuf)), table->tableTypeId,
        BkfDcGetTableTypeIdStr(dc, table->tableTypeId), idx, tuple->isAddUpd,
        BkfDcGetTupleKeyStr(dc, table->tableTypeId, tuple->keyVal, tupleKeyBuf, sizeof(tupleKeyBuf)),
        BkfDcGetTupleValStr(dc, table->tableTypeId, tupleValOrNull, tupleValBuf, sizeof(tupleValBuf)));
}
STATIC void BkfDcDispOneDataAndUpdCtx2(BkfDc *dc, BkfDcDispTupleCtx2 *ctx2, BkfDcDispTupleTemp *temp)
{
    BkfDisp *disp = dc->argInit.disp;
    BkfDcSlice *slice = temp->slice;
    BkfDcTable *table = temp->table;
    BkfDcTuple *tuple = temp->tuple;
    uint8_t sliceBuf[BKF_1K / 8];
    errno_t err;

    temp->tupleKeyLen = 0;
    if (tuple != VOS_NULL) {
        BkfDcDispOneTuple(dc, ctx2, slice, table, tuple);
        if (tuple->isAddUpd) {
            ctx2->tupleAddUpdCnt++;
        } else {
            ctx2->tupleDelCnt++;
        }
        ctx2->sliceNoMoreTable = VOS_FALSE;
        ctx2->tableTypeId = table->tableTypeId;
        ctx2->tableNoMoreTuple = VOS_FALSE;
        temp->tupleKeyLen = table->tableType->vTbl.tupleKeyLen;
        err = memcpy_s(ctx2->tupleKey, sizeof(ctx2->tupleKey), tuple->keyVal, temp->tupleKeyLen);
        BKF_ASSERT(err == EOK);
    } else if (table != VOS_NULL) {
        if (temp->isNewTable) {
            BKF_DISP_PRINTF(disp, "slice(%s)/table(%u, %s), no_tuple\n",
                BkfDcGetSliceKeyStr(dc, slice->key, sliceBuf, sizeof(sliceBuf)), table->tableTypeId,
                BkfDcGetTableTypeIdStr(dc, table->tableTypeId));
        }
        ctx2->sliceNoMoreTable = VOS_FALSE;
        ctx2->tableTypeId = table->tableTypeId;
        ctx2->tableNoMoreTuple = VOS_TRUE;
    } else if (slice != VOS_NULL) {
        if (temp->isNewSlice) {
            BKF_DISP_PRINTF(disp, "slice(%s)/no_table\n",
                BkfDcGetSliceKeyStr(dc, slice->key, sliceBuf, sizeof(sliceBuf)));
        }
        ctx2->sliceNoMoreTable = VOS_TRUE;
    }
}
void BkfDcDispData(BkfDc *dc)
{
    BkfDisp *disp = dc->argInit.disp;
    void *sliceKey = VOS_NULL;
    BkfDcDispTupleCtx2 ctx2 = {0};
    BkfDcDispTupleTemp temp = {0};

    sliceKey = BKF_DISP_GET_LAST_CTX(disp, &ctx2);
    if (sliceKey == VOS_NULL) {
        ctx2.tupleAddUpdCnt = 0;
        ctx2.tupleDelCnt = 0;

        BkfDcDispTupleGetFirstTuple(dc, &ctx2, &temp);
    } else {
        if (ctx2.sliceNoMoreTable) {
            BkfDcDispTupleGetIfLastSliceNoMoreTable(dc, sliceKey, &ctx2, &temp);
        } else if (ctx2.tableNoMoreTuple) {
            BkfDcDispTupleGetIfLastTableNoMoreTuple(dc, sliceKey, &ctx2, &temp);
        } else {
            BkfDcDispTupleGetNextTuple(dc, sliceKey, &ctx2, &temp);
        }
    }

    if (temp.slice != VOS_NULL) {
        BkfDcDispOneDataAndUpdCtx2(dc, &ctx2, &temp);
        BKF_DISP_SAVE_CTX(disp, temp.slice->key, dc->argInit.sliceVTbl.keyLen, &ctx2,
            sizeof(ctx2) - sizeof(ctx2.tupleKey) + temp.tupleKeyLen);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d addUpdTuple(s), %d delTuple(s) ***\n", ctx2.tupleAddUpdCnt,
            ctx2.tupleDelCnt);
    }
}

void BkfDcDispInit(BkfDc *dc)
{
    BkfDisp *disp = dc->argInit.disp;
    char *objName = dc->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, dc);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfDcDisp, "disp dc", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfDcDispTableType, "disp dc table type", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfDcDispSliceSummary, "disp dc slice summary", objName, 1, "uniDispTable");
    (void)BKF_DISP_REG_FUNC(disp, BkfDcDispData, "disp dc data", objName, 0);
}

void BkfDcDispUninit(BkfDc *dc)
{
    BkfDispUnregObj(dc->argInit.disp, dc->name);
}
