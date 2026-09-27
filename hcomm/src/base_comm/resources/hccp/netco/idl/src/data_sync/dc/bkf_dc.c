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

#include "bkf_dc_pri.h"
#include "bkf_dc_itor_pri.h"
#include "bkf_dc_disp.h"
#include "bkf_str.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
/* func */
BkfDc *BkfDcInit(BkfDcInitArg *arg)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfDc *dc = VOS_NULL;
    uint32_t len;
    uint32_t ret;

    paramIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                     (arg->disp == VOS_NULL) || (arg->jobMng == VOS_NULL) || (arg->sysLogMng == VOS_NULL) ||
                     (arg->sliceVTbl.keyLen == 0) || (arg->sliceVTbl.keyLen > BKF_DC_SLICE_KEY_LEN_MAX) ||
                     !BKF_LEN_IS_ALIGN4(arg->sliceVTbl.keyLen) || (arg->pfm == VOS_NULL) ||
                     (arg->sliceVTbl.keyCmp == VOS_NULL) || (arg->sliceVTbl.keyCodec == VOS_NULL);
    if (paramIsInvalid) {
        goto error;
    }

    len = sizeof(BkfDc);
    dc = BKF_MALLOC(arg->memMng, len);
    if (dc == VOS_NULL) {
        goto error;
    }
    (void)memset_s(dc, len, 0, len);
    dc->argInit = *arg;
    dc->name = BkfStrNew(arg->memMng, "%s_dc", arg->name);
    dc->argInit.name = dc->name;
    dc->log = arg->log;

    VOS_AVLL_INIT_TREE(dc->tableTypeSet, (AVLL_COMPARE)Bkfuint16_tCmp,
        BKF_OFFSET(BkfDcTableType, vTbl) + BKF_OFFSET(BkfDcTableTypeVTbl, tableTypeId),
        BKF_OFFSET(BkfDcTableType, avlNode));
    VOS_AVLL_INIT_TREE(dc->sliceSet, (AVLL_COMPARE)dc->argInit.sliceVTbl.keyCmp, BKF_OFFSET(BkfDcSlice, key[0]),
        BKF_OFFSET(BkfDcSlice, avlNode));
    dc->seq64Seed = BKF_DC_SEQ_INVALID;

    ret = BkfDcItorInit(dc);
    if (ret != BKF_OK) {
        goto error;
    }
    BkfDcDispInit(dc);
    (void)BkfDcTupleLimitInit(dc);
    BKF_LOG_INFO(BKF_LOG_HND, "dc(%#x)/ret(%u) init123 ok\n", BKF_MASK_ADDR(dc), ret);
    return dc;

error:

    BkfDcUninit(dc);
    return VOS_NULL;
}

void BkfDcUninit(BkfDc *dc)
{
    if (dc == VOS_NULL) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "dc(%#x), 2unit\n", BKF_MASK_ADDR(dc));

    BkfDcDispUninit(dc);
    BkfDcItorUninit(dc);
    BkfDcDelAllSlice(dc);
    BkfDcDelAllTableType(dc);
    BkfStrDel(dc->name);
    BKF_FREE(dc->argInit.memMng, dc);
    return;
}

/* table type */
uint32_t BkfDcRegTableType(BkfDc *dc, BkfDcTableTypeVTbl *vTbl)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfDcTableType *tableType = VOS_NULL;

    paramIsInvalid = (dc == VOS_NULL) || (vTbl == VOS_NULL) || (vTbl->name == VOS_NULL) || (vTbl->tupleCntMax <= 0) ||
                     (vTbl->tupleKeyLen == 0) || !BKF_LEN_IS_ALIGN4(vTbl->tupleKeyLen) ||
                     (vTbl->tupleKeyLen > BKF_DC_TUPLE_KEY_LEN_MAX) || (vTbl->tupleKeyCmp == VOS_NULL);
    if (paramIsInvalid) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x)/table(%u, %s), vTbl(%#x)\n", BKF_MASK_ADDR(dc), vTbl->tableTypeId, vTbl->name,
        BKF_MASK_ADDR(vTbl));

    tableType = BkfDcAddTableType(dc, vTbl);
    if (tableType == VOS_NULL) {
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfDcUnregTableType(BkfDc *dc, uint16_t tabTypeId)
{
    if (dc == VOS_NULL) {
        return;
    }
    BkfDcTableType *tableType = BkfDcFindTableType(dc, tabTypeId);
    if (tableType == VOS_NULL) {
        return;
    }
    BkfDcDelTableType(dc, tableType);
    return;
}

STATIC void BkfDcNtfAfterTableCompleteBySeqItor(BkfDcTupleSeqItor *seqItor)
{
    if (seqItor->vTbl.afterTableCompleteOrNull != VOS_NULL) {
        seqItor->vTbl.afterTableCompleteOrNull(seqItor->vTbl.cookie);
    }
}
STATIC void BkfDcNtfAllMatchTableComplete(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;
    BkfDcTable *table = VOS_NULL;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        table = BkfDcFindTable(dc, slice, tableTypeId);
        if (table != VOS_NULL) {
            BkfDcNtfTableAllSeqItor(dc, table, BkfDcNtfAfterTableCompleteBySeqItor);
        }
    }
    return;
}
uint32_t BkfDcNotifyTableTypeComplete(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcTableType *tableType = VOS_NULL;

    if (dc == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x)/table(%u)\n", BKF_MASK_ADDR(dc), tableTypeId);

    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "table(%u)/tableType(%#x)\n", tableTypeId, BKF_MASK_ADDR(tableType));
        return BKF_ERR;
    }
    if (tableType->complete) {
        return BKF_OK;
    }

    tableType->complete = VOS_TRUE;
    BkfDcNtfAllMatchTableComplete(dc, tableTypeId);
    return BKF_OK;
}

void BkfDcResetTableTypeComplete(BkfDc *dc, uint16_t tableTypeId)
{
    if (dc == VOS_NULL) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x)/table(%u)\n", BKF_MASK_ADDR(dc), tableTypeId);

    BkfDcTableType *tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "table(%u)/tableType(%#x)\n", tableTypeId, BKF_MASK_ADDR(tableType));
        return;
    }
    tableType->complete = VOS_FALSE;
}

BOOL BkfDcIsTableTypeComplete(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcTableType *tableType = VOS_NULL;

    if (dc == VOS_NULL) {
        return VOS_FALSE;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x)/table(%u)\n", BKF_MASK_ADDR(dc), tableTypeId);

    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "table(%u)/tableType(%#x)\n", tableTypeId, BKF_MASK_ADDR(tableType));
        return VOS_FALSE;
    }

    return tableType->complete;
}

const char *BkfDcGetTableTypeIdStr(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType != VOS_NULL) {
        return tableType->name;
    } else {
        return "typeUnknown_notReg";
    }
}

/* slice */
uint32_t BkfDcCreateSlice(BkfDc *dc, void *sliceKey)
{
    BkfDcSlice *slice = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)));

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        slice = BkfDcAddSlice(dc, sliceKey);
    }
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "create fail dc(%#x), sliceKey(%s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)));
    }
    return ((slice != VOS_NULL) ? BKF_OK : BKF_ERR);
}

void BkfDcDeleteSlice(BkfDc *dc, void *sliceKey)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    void *itor = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)));

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_WARN(BKF_LOG_HND, "slice not exist, dc(%#x), sliceKey(%s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)));
        return;
    }

    /* 直接删切片之前，一定要有其下表 release相关的通知。否则上层的seqItor没有释放，
    table会在下面流程中释放，导致野指针 */
    for (table = BkfDcGetFirstTable(dc, slice, &itor); table != VOS_NULL; table = BkfDcGetNextTable(dc, slice, &itor)) {
        if (!table->isRelease) {
            BkfDcReleaseTable(dc, sliceKey, table->tableTypeId);
        }
    }
    BkfDcDelSlice(dc, slice);
    return;
}

BOOL BkfDcIsSliceExist(BkfDc *dc, void *sliceKey)
{
    BkfDcSlice *slice = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return VOS_FALSE;
    }

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        return VOS_FALSE;
    }

    return VOS_TRUE;
}

void *BkfDcGetFirstSliceInfo(BkfDc *dc)
{
    BkfDcSlice *slice = VOS_NULL;

    if (dc == VOS_NULL) {
        return VOS_NULL;
    }

    slice = BkfDcGetFirstSlice(dc, VOS_NULL);
    if (slice == VOS_NULL) {
        return VOS_NULL;
    }

    return slice->key;
}

void *BkfDcGetNextSliceInfo(BkfDc *dc, void *prevSliceKey)
{
    BkfDcSlice *slice = VOS_NULL;

    if ((dc == VOS_NULL) || (prevSliceKey == VOS_NULL)) {
        return VOS_NULL;
    }

    slice = BkfDcFindNextSlice(dc, prevSliceKey);
    if (slice == VOS_NULL) {
        return VOS_NULL;
    }

    return slice->key;
}

char *BkfDcGetSliceKeyStr(BkfDc *dc, void *sliceKey, uint8_t *buf, int32_t bufLen)
{
    if ((dc == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "sliceKeyNg";
    }
    if (sliceKey == VOS_NULL) {
        return "sliceKeyNg_Null";
    }

    if (dc->argInit.sliceVTbl.keyGetStrOrNull != VOS_NULL) {
        return dc->argInit.sliceVTbl.keyGetStrOrNull(sliceKey, buf, bufLen);
    } else {
        return BKF_GET_MEM_STD_STR(sliceKey, dc->argInit.sliceVTbl.keyLen, buf, bufLen);
    }
}

/* table */
uint32_t BkfDcCreateTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId));

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "slice not exist, dc(%#x), sliceKey(%s)/type(%u, %s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId));
        return BKF_ERR;
    }

    table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        table = BkfDcAddTable(dc, slice, tableTypeId);
    } else {
        table->isRelease = VOS_FALSE;
    }
    if (table == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "add fail, dc(%#x), sliceKey(%s)/type(%u, %s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId));
    }
    return ((table != VOS_NULL) ? BKF_OK : BKF_ERR);
}

STATIC void BkfDcNtfAfterTableReleaseBySeqItor(BkfDcTupleSeqItor *seqItor)
{
    if (seqItor->vTbl.afterTableReleaseOrNull != VOS_NULL) {
        seqItor->vTbl.afterTableReleaseOrNull(seqItor->vTbl.cookie);
    }
}
void BkfDcReleaseTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u,  %s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId));

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "slice not exist dc(%#x), sliceKey(%s)/type(%u,  %s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId));
        return;
    }
    table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "table not exist dc(%#x), sliceKey(%s)/type(%u,  %s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId));
        return;
    }

    table->isRelease = VOS_TRUE;
    BkfDcNtfTableAllSeqItor(dc, table, BkfDcNtfAfterTableReleaseBySeqItor);
    return;
}

void BkfDcDeleteTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId)
{
    uint8_t sliceBuf[BKF_1K / 8];
    BOOL paramIsInvalid = VOS_FALSE;

    paramIsInvalid = (dc == VOS_NULL) || (sliceKey == VOS_NULL);
    if (paramIsInvalid) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId));

    BkfDcSlice *slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "slice not exist dc(%#x), sliceKey(%s)/type(%u,  %s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId));
        return;
    }
    BkfDcTable *table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "table not exist dc(%#x), sliceKey(%s)/type(%u,  %s)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId));
        return;
    }
    /* 直接删表之前，一定要有release相关的通知。否则上层的seqItor没有释放，table会在下面流程中释放，导致野指针 */
    if (!table->isRelease) {
        BkfDcReleaseTable(dc, sliceKey, tableTypeId);
    }
    BkfDcDelTable(dc, table);
    return;
}

STATIC BkfDcSlice *BkfDcChkAndFindSlice(BkfDc *dc, void *sliceKey)
{
    BkfDcSlice *slice = VOS_NULL;
    BOOL paramIsInvalid = VOS_FALSE;

    paramIsInvalid = (dc == VOS_NULL) || (sliceKey == VOS_NULL);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    slice = BkfDcFindSlice(dc, sliceKey);
    return slice;
}
BOOL BkfDcIsTableExist(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, BOOL *isTableReleaseOrNull)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return VOS_FALSE;
    }

    slice = BkfDcChkAndFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        return VOS_FALSE;
    }
    table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }

    if (isTableReleaseOrNull != VOS_NULL) {
        *isTableReleaseOrNull = table->isRelease;
    }
    return VOS_TRUE;
}

BOOL BkfDcGetFirstTableInfo(BkfDc *dc, void *sliceKey, uint16_t *tableTypeId, BOOL *isTableReleaseOrNull)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (tableTypeId == VOS_NULL)) {
        return VOS_FALSE;
    }

    slice = BkfDcChkAndFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        return VOS_FALSE;
    }
    table = BkfDcGetFirstTable(dc, slice, VOS_NULL);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }

    *tableTypeId = table->tableTypeId;
    if (isTableReleaseOrNull != VOS_NULL) {
        *isTableReleaseOrNull = table->isRelease;
    }
    return VOS_TRUE;
}

BOOL BkfDcGetNextTableInfo(BkfDc *dc, void *sliceKey, uint16_t prevTableTypeId, uint16_t *tableTypeId,
    BOOL *isTableReleaseOrNull)
{
    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (tableTypeId == VOS_NULL)) {
        return VOS_FALSE;
    }

    BkfDcSlice *slice = BkfDcChkAndFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        return VOS_FALSE;
    }
    BkfDcTable *table = BkfDcFindNextTable(dc, slice, prevTableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }

    *tableTypeId = table->tableTypeId;
    if (isTableReleaseOrNull != VOS_NULL) {
        *isTableReleaseOrNull = table->isRelease;
    }
    return VOS_TRUE;
}

/* tuple */
STATIC void BkfDcUpdateTupleDoBefore(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey,
    void *tupleValOrNull, BkfDcTableType **tableType, BkfDcSlice **slice, BkfDcTable **table)
{
    uint8_t sliceBuf[BKF_1K / 8];
    uint8_t tupleKeyBuf[BKF_1K / 8];
    uint8_t tupleValBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (tupleKey == VOS_NULL)) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s)/tupleKey(%s)/val(%s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId),
        BkfDcGetTupleKeyStr(dc, tableTypeId, tupleKey, tupleKeyBuf, sizeof(tupleKeyBuf)),
        BkfDcGetTupleValStr(dc, tableTypeId, tupleValOrNull, tupleValBuf, sizeof(tupleValBuf)));

    *tableType = BkfDcFindTableType(dc, tableTypeId);
    if (*tableType == VOS_NULL) {
        return;
    }
    *slice = BkfDcFindSlice(dc, sliceKey);
    if (*slice == VOS_NULL) {
        return;
    }
    *table = BkfDcFindTable(dc, *slice, tableTypeId);
    if (*table == VOS_NULL) {
        return;
    }

    return;
}

STATIC void BkfDcNtfAfterTupleChangeBySeqItor(BkfDcTupleSeqItor *seqItor)
{
    if (seqItor->vTbl.afterTupleChangeOrNull != VOS_NULL) {
        seqItor->vTbl.afterTupleChangeOrNull(seqItor->vTbl.cookie);
    }
}

STATIC void BkfDcFillTupleInfo(BkfDcTuple *tuple, BkfDcTable *table, BkfDcTupleInfo *infoOrNull)
{
    if (infoOrNull == VOS_NULL) {
        return;
    }

    infoOrNull->key = &tuple->keyVal[0];
    infoOrNull->isAddUpd = tuple->isAddUpd;
    infoOrNull->valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, table->tableType) : VOS_NULL;
    infoOrNull->seq = tuple->seq;
    return;
}

STATIC void BkfDcUpdateTupleDoAfter(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple, BkfDcTupleInfo *infoOrNull)
{
    BkfDcFillTupleInfo(tuple, table, infoOrNull);
    BkfDcUpdTableAllTupleSeqItorForTuple(dc, table, tuple);
    BkfDcNtfTableAllSeqItor(dc, table, BkfDcNtfAfterTupleChangeBySeqItor);
    return;
}

uint32_t BkfDcUpdateTupleDoUpdate(BkfDc *dc, void *tupleValOrNull, BkfDcTable *table, BkfDcTuple *tuple)
{
    BOOL mayAddTuple = VOS_FALSE;
    uint8_t sliceBuf[BKF_1K / 8];

    mayAddTuple = tuple->isAddUpd || BkfDcTupleLimitMayAddTuple(dc, table->tableTypeId);
    if (!mayAddTuple) {
        BKF_LOG_WARN(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s), mayAddTuple(%u)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, table->slice->key, sliceBuf, sizeof(sliceBuf)), table->tableTypeId,
            BkfDcGetTableTypeIdStr(dc, table->tableTypeId), mayAddTuple);
        (void)BkfDcTupleLimitSysLog(dc, table->tableTypeId);
        return BKF_ERR;
    }

    (void)BkfDcUpdTuple(dc, table, tuple, tupleValOrNull);
    return BKF_OK;
}

BkfDcTuple *BkfDcUpdateTupleDoAdd(BkfDc *dc, void *tupleKey, void *tupleValOrNull, BkfDcTable *table)
{
    BOOL mayAddTuple = VOS_FALSE;
    BkfDcTuple *tuple = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];

    mayAddTuple = BkfDcTupleLimitMayAddTuple(dc, table->tableTypeId);
    if (!mayAddTuple) {
        BKF_LOG_WARN(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s), mayAddTuple(%u)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, table->slice->key, sliceBuf, sizeof(sliceBuf)), table->tableTypeId,
            BkfDcGetTableTypeIdStr(dc, table->tableTypeId), mayAddTuple);
        (void)BkfDcTupleLimitSysLog(dc, table->tableTypeId);
        return VOS_NULL;
    }

    tuple = BkfDcAddTuple(dc, table, tupleKey, tupleValOrNull);
    if (tuple == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s), add fail\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, table->slice->key, sliceBuf, sizeof(sliceBuf)), table->tableTypeId,
            BkfDcGetTableTypeIdStr(dc, table->tableTypeId));
        return VOS_NULL;
    }

    return tuple;
}

#define BKF_DC_DOBEFORE(dc)                                                                                            \
    do {                                                                                                               \
        if ((dc)->argInit.beforeProc) {                                                                                \
            (dc)->argInit.beforeProc((dc)->argInit.cookie);                                                            \
        }                                                                                                              \
    } while (0)

#define BKF_DC_DOAFTER(dc)                                                                                             \
    do {                                                                                                               \
        if ((dc)->argInit.afterProc) {                                                                                 \
            (dc)->argInit.afterProc((dc)->argInit.cookie);                                                             \
        }                                                                                                              \
    } while (0)

uint32_t BkfDcUpdateTuple(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey, void *tupleValOrNull,
    BkfDcTupleInfo *infoOrNull)
{
    BOOL isInvalid = VOS_FALSE;
    BkfDcTableType *tableType = VOS_NULL;
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];
    if (dc == VOS_NULL) {
        return BKF_ERR;
    }

    BKF_DC_DOBEFORE(dc);
    BkfDcUpdateTupleDoBefore(dc, sliceKey, tableTypeId, tupleKey, tupleValOrNull, &tableType, &slice, &table);
    isInvalid = (tableType == VOS_NULL) || (slice == VOS_NULL) || (table == VOS_NULL);
    if (isInvalid) {
        BKF_LOG_ERROR(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s), tableType(%#x), slice(%#x), table(%#x)\n",
            BKF_MASK_ADDR(dc), BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId), BKF_MASK_ADDR(tableType), BKF_MASK_ADDR(slice),
            BKF_MASK_ADDR(table));
        BKF_DC_DOAFTER(dc);
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId));

    if (table->isRelease) {
        BKF_LOG_INFO(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s)/isRelease(%u)\n", BKF_MASK_ADDR(dc),
            BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId), table->isRelease);
        BKF_DC_DOAFTER(dc);
        return BKF_ERR;
    }

    tuple = BkfDcFindTuple(dc, table, tupleKey);
    if (tuple == VOS_NULL) {
        tuple = BkfDcUpdateTupleDoAdd(dc, tupleKey, tupleValOrNull, table);
        if (tuple == VOS_NULL) {
            BKF_DC_DOAFTER(dc);
            return BKF_ERR;
        }
    } else {
        if (BkfDcUpdateTupleDoUpdate(dc, tupleValOrNull, table, tuple) != BKF_OK) {
            BKF_DC_DOAFTER(dc);
            return BKF_ERR;
        }
    }

    BkfDcUpdateTupleDoAfter(dc, table, tuple, infoOrNull);
    BkfDcSpyUpdateTuple(dc->cookieOfSpy, sliceKey, tableTypeId, tupleKey, tupleValOrNull);
    BKF_DC_DOAFTER(dc);
    return BKF_OK;
}

BkfDcTuple *BkfDcFindOrAddTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey)
{
    BkfDcTuple *tuple = BkfDcFindTuple(dc, table, tupleKey);
    if (tuple) {
        return tuple;
    }
    tuple = BkfDcAddTuple(dc, table, tupleKey, VOS_NULL);
    BKF_LOG_DEBUG(BKF_LOG_HND, "DcFindOrAddTuple: %#x\n", BKF_MASK_ADDR(tuple));
    return tuple;
}

void BkfDcDeleteTuple(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey)
{
    BkfDcTuple *tuple = VOS_NULL;
    uint8_t sliceBuf[BKF_1K / 8];
    uint8_t tupleKeyBuf[BKF_1K / 8];

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (tupleKey == VOS_NULL)) {
        return;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "dc(%#x), sliceKey(%s)/type(%u, %s)/tupleKey(%s)\n", BKF_MASK_ADDR(dc),
        BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
        BkfDcGetTableTypeIdStr(dc, tableTypeId),
        BkfDcGetTupleKeyStr(dc, tableTypeId, tupleKey, tupleKeyBuf, sizeof(tupleKeyBuf)));

    BkfDcSlice *slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "slice not exist, dc(%#x), sliceKey(%s)/type(%u, %s)/tupleKey(%s)\n",
            BKF_MASK_ADDR(dc), BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId),
            BkfDcGetTupleKeyStr(dc, tableTypeId, tupleKey, tupleKeyBuf, sizeof(tupleKeyBuf)));
        return;
    }
    BkfDcTable *table = BkfDcFindTable(dc, slice, tableTypeId);
    if (table == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "table not exist, dc(%#x), sliceKey(%s)/type(%u, %s)/tupleKey(%s)\n",
            BKF_MASK_ADDR(dc), BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId),
            BkfDcGetTupleKeyStr(dc, tableTypeId, tupleKey, tupleKeyBuf, sizeof(tupleKeyBuf)));
        return;
    }
    BKF_DC_DOBEFORE(dc);
    if (table->tableType->vTbl.noDcTuple) {
        tuple = BkfDcFindOrAddTuple(dc, table, tupleKey);
    } else {
        tuple = BkfDcFindTuple(dc, table, tupleKey);
    }
    if (tuple == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tuple not exist, dc(%#x), sliceKey(%s)/type(%u, %s)/tupleKey(%s)\n",
            BKF_MASK_ADDR(dc), BkfDcGetSliceKeyStr(dc, sliceKey, sliceBuf, sizeof(sliceBuf)), tableTypeId,
            BkfDcGetTableTypeIdStr(dc, tableTypeId),
            BkfDcGetTupleKeyStr(dc, tableTypeId, tupleKey, tupleKeyBuf, sizeof(tupleKeyBuf)));
        BKF_DC_DOAFTER(dc);
        return;
    }

    BkfDcSetTupleDelState(dc, table, tuple);
    BkfDcUpdTableAllTupleSeqItorForTuple(dc, table, tuple);
    BkfDcNtfTableAllSeqItor(dc, table, BkfDcNtfAfterTupleChangeBySeqItor);
    (void)BkfDcTry2FreeTuple(dc, table, tuple);
    BkfDcSpyDeleteTuple(dc->cookieOfSpy, sliceKey, tableTypeId, tupleKey);
    BKF_DC_DOAFTER(dc);
    return;
}

STATIC BkfDcTable *BkfDcChkAndFindTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId)
{
    BkfDcSlice *slice = VOS_NULL;
    BkfDcTable *table = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL)) {
        return VOS_NULL;
    }

    slice = BkfDcFindSlice(dc, sliceKey);
    if (slice == VOS_NULL) {
        return VOS_NULL;
    }
    table = BkfDcFindTable(dc, slice, tableTypeId);
    return table;
}
BOOL BkfDcGetTupleInfo(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey, BkfDcTupleInfo *info)
{
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (tupleKey == VOS_NULL) || (info == VOS_NULL)) {
        return VOS_FALSE;
    }

    table = BkfDcChkAndFindTable(dc, sliceKey, tableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }
    tuple = BkfDcFindTuple(dc, table, tupleKey);
    if (tuple == VOS_NULL) {
        return VOS_FALSE;
    }

    BkfDcFillTupleInfo(tuple, table, info);
    return VOS_TRUE;
}

BOOL BkfDcGetFirstTupleInfo(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, BkfDcTupleInfo *info)
{
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (info == VOS_NULL)) {
        return VOS_FALSE;
    }

    table = BkfDcChkAndFindTable(dc, sliceKey, tableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }
    tuple = BkfDcGetFirstTuple(dc, table, VOS_NULL);
    if (tuple == VOS_NULL) {
        return VOS_FALSE;
    }

    info->key = &tuple->keyVal[0];
    info->isAddUpd = tuple->isAddUpd;
    info->valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, table->tableType) : VOS_NULL;
    info->seq = tuple->seq;
    return VOS_TRUE;
}

BOOL BkfDcGetNextTupleInfo(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *prevTupleKey, BkfDcTupleInfo *info)
{
    BkfDcTable *table = VOS_NULL;
    BkfDcTuple *tuple = VOS_NULL;

    if ((dc == VOS_NULL) || (sliceKey == VOS_NULL) || (prevTupleKey == VOS_NULL) || (info == VOS_NULL)) {
        return VOS_FALSE;
    }

    table = BkfDcChkAndFindTable(dc, sliceKey, tableTypeId);
    if (table == VOS_NULL) {
        return VOS_FALSE;
    }
    tuple = BkfDcFindNextTuple(dc, table, prevTupleKey);
    if (tuple == VOS_NULL) {
        return VOS_FALSE;
    }

    info->key = &tuple->keyVal[0];
    info->isAddUpd = tuple->isAddUpd;
    info->valOrNull = tuple->isAddUpd ? BKF_DC_TUPLE_GET_VAL(tuple, table->tableType) : VOS_NULL;
    info->seq = tuple->seq;
    return VOS_TRUE;
}

char *BkfDcGetTupleKeyStr(BkfDc *dc, uint16_t tableTypeId, void *tupleKey, uint8_t *buf, int32_t bufLen)
{
    BkfDcTableType *tableType = VOS_NULL;

    if ((dc == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "paramNg";
    }
    if (tupleKey == VOS_NULL) {
        return "-";
    }
    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        return "**";
    }

    if (tableType->vTbl.tupleKeyGetStrOrNull != VOS_NULL) {
        return tableType->vTbl.tupleKeyGetStrOrNull(tupleKey, buf, bufLen);
    } else {
        return BKF_GET_MEM_STD_STR(tupleKey, tableType->vTbl.tupleKeyLen, buf, bufLen);
    }
}

char *BkfDcGetTupleValStr(BkfDc *dc, uint16_t tableTypeId, void *tupleVal, uint8_t *buf, int32_t bufLen)
{
    BkfDcTableType *tableType = VOS_NULL;

    if ((dc == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "paramNg";
    }
    if (tupleVal == VOS_NULL) {
        return "-";
    }
    tableType = BkfDcFindTableType(dc, tableTypeId);
    if (tableType == VOS_NULL) {
        return "**";
    }

    if (tableType->vTbl.tupleValGetStrOrNull != VOS_NULL) {
        return tableType->vTbl.tupleValGetStrOrNull(tupleVal, buf, bufLen);
    } else {
        return BKF_GET_MEM_STD_STR(tupleVal, tableType->vTbl.tupleValLen, buf, bufLen);
    }
}

void BkfDcEnableSpy(BkfDc *dc, void *cookie)
{
    if (dc == VOS_NULL) {
        return;
    }

    dc->cookieOfSpy = cookie;
}

void BkfDcSpyUpdateTuple(void *cookieOfSpy, void *sliceKey, uint16_t tableTypeId, void *tupleKey, void *tupleValOrNull)
{
    (void)cookieOfSpy;
    (void)sliceKey;
    (void)tableTypeId;
    (void)tupleKey;
    (void)tupleValOrNull;
}

void BkfDcSpyDeleteTuple(void *cookieOfSpy, void *sliceKey, uint16_t tableTypeId, void *tupleKey)
{
    (void)cookieOfSpy;
    (void)sliceKey;
    (void)tableTypeId;
    (void)tupleKey;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
