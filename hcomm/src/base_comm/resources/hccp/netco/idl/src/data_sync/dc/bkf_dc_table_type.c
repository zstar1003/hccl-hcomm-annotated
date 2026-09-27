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
#include "bkf_str.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
BkfDcTableType *BkfDcAddTableType(BkfDc *dc, BkfDcTableTypeVTbl *vTbl)
{
    uint8_t hashIdx = BKF_DC_GET_TABLE_TYPE_HASH_IDX(vTbl->tableTypeId);
    BkfDcTableType *tableType = VOS_NULL;
    uint32_t len;
    BOOL insOk = VOS_FALSE;

    len = sizeof(BkfDcTableType);
    tableType = BKF_MALLOC(dc->argInit.memMng, len);
    if (tableType == VOS_NULL) {
        goto error;
    }
    (void)memset_s(tableType, len, 0, len);
    tableType->vTbl = *vTbl;
    tableType->name = BkfStrNew(dc->argInit.memMng, "%s", vTbl->name);
    tableType->vTbl.name = tableType->name;
    VOS_AVLL_INIT_NODE(tableType->avlNode);
    insOk = VOS_AVLL_INSERT(dc->tableTypeSet, tableType->avlNode);
    if (!insOk) {
        goto error;
    }

    dc->tableTypeCache[hashIdx] = tableType;
    return tableType;

error:

    if (tableType != VOS_NULL) {
        /* insOk一定为false */
        BkfStrDel(tableType->name);
        BKF_FREE(dc->argInit.memMng, tableType);
    }
    return VOS_NULL;
}

void BkfDcDelTableType(BkfDc *dc, BkfDcTableType *tableType)
{
    uint8_t hashIdx = BKF_DC_GET_TABLE_TYPE_HASH_IDX(tableType->vTbl.tableTypeId);
    if (dc->tableTypeCache[hashIdx] == tableType) {
        dc->tableTypeCache[hashIdx] = VOS_NULL;
    }
    VOS_AVLL_DELETE(dc->tableTypeSet, tableType->avlNode);
    BkfStrDel(tableType->name);
    BKF_FREE(dc->argInit.memMng, tableType);
    return;
}

void BkfDcDelAllTableType(BkfDc *dc)
{
    BkfDcTableType *tableType = VOS_NULL;
    void *itor = VOS_NULL;

    for (tableType = BkfDcGetFirstTableType(dc, &itor); tableType != VOS_NULL;
         tableType = BkfDcGetNextTableType(dc, &itor)) {
        BkfDcDelTableType(dc, tableType);
    }
    return;
}

BkfDcTableType *BkfDcFindTableType(BkfDc *dc, uint16_t tableTypeId)
{
    uint8_t hashIdx = BKF_DC_GET_TABLE_TYPE_HASH_IDX(tableTypeId);
    BkfDcTableType *tableType = VOS_NULL;
    BOOL hit = VOS_FALSE;

    tableType = dc->tableTypeCache[hashIdx];
    hit = (tableType != VOS_NULL) && (tableType->vTbl.tableTypeId == tableTypeId);
    if (hit) {
        return tableType;
    }

    tableType = VOS_AVLL_FIND(dc->tableTypeSet, &tableTypeId);
    if (tableType != VOS_NULL) {
        dc->tableTypeCache[hashIdx] = tableType;
    }
    return tableType;
}

BkfDcTableType *BkfDcFindNextTableType(BkfDc *dc, uint16_t tableTypeId)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = VOS_AVLL_FIND_NEXT(dc->tableTypeSet, &tableTypeId);
    return tableType;
}

BkfDcTableType *BkfDcGetFirstTableType(BkfDc *dc, void **itorOutOrNull)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = VOS_AVLL_FIRST(dc->tableTypeSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (tableType != VOS_NULL) ? VOS_AVLL_NEXT(dc->tableTypeSet, tableType->avlNode) : VOS_NULL;
    }
    return tableType;
}

BkfDcTableType *BkfDcGetNextTableType(BkfDc *dc, void **itorInOut)
{
    BkfDcTableType *tableType = VOS_NULL;

    tableType = (*itorInOut);
    *itorInOut = (tableType != VOS_NULL) ? VOS_AVLL_NEXT(dc->tableTypeSet, tableType->avlNode) : VOS_NULL;
    return tableType;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
