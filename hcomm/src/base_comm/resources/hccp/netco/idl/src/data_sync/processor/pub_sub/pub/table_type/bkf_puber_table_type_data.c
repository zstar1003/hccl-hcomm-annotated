/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_table_type_data.h"
#include "bkf_str.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

BkfPuberTableTypeMng *BkfPuberTableTypeDataInit(BkfPuberInitArg *arg)
{
    uint32_t len = sizeof(BkfPuberTableTypeMng);
    BkfPuberTableTypeMng *tableTypeMng = BKF_MALLOC(arg->memMng, len);
    if (tableTypeMng == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(tableTypeMng, len, 0, len);
    tableTypeMng->argInit = arg;
    tableTypeMng->name = BkfStrNew(arg->memMng, "%s_puberTableType", arg->name);
    if (tableTypeMng->name == VOS_NULL) {
        BKF_FREE(tableTypeMng->argInit->memMng, tableTypeMng);
        return VOS_NULL;
    }
    tableTypeMng->log = arg->log;

    VOS_AVLL_INIT_TREE(tableTypeMng->tableTypeSet, (AVLL_COMPARE)Bkfuint16_tCmp,
        BKF_OFFSET(BkfPuberTableType, vTbl) + BKF_OFFSET(BkfPuberTableTypeVTbl, tableTypeId),
        BKF_OFFSET(BkfPuberTableType, avlNode));

    return tableTypeMng;
}

void BkfPuberTableTypeDataUninit(BkfPuberTableTypeMng *tableTypeMng)
{
    BkfPuberTableTypeDeleteAll(tableTypeMng);
    BkfStrDel(tableTypeMng->name);
    BKF_FREE(tableTypeMng->argInit->memMng, tableTypeMng);
}

BkfPuberTableType *BkfPuberTableTypeAdd(BkfPuberTableTypeMng *tableTypeMng, BkfPuberTableTypeVTbl *vTbl, void *userData,
    uint16_t dataLen)
{
    uint32_t len = sizeof(BkfPuberTableType) + dataLen;
    BkfPuberTableType *tableType = BKF_MALLOC(tableTypeMng->argInit->memMng, len);
    if (tableType == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(tableType, len, 0, len);

    tableType->tableTypeMng = tableTypeMng;
    tableType->vTbl = *vTbl;
    if (dataLen > 0) {
        (void)memcpy_s(tableType->userData, dataLen, userData, dataLen);
    }

    VOS_AVLL_INIT_NODE(tableType->avlNode);
    if (!VOS_AVLL_INSERT(tableTypeMng->tableTypeSet, tableType->avlNode)) {
        BKF_FREE(tableTypeMng->argInit->memMng, tableType);
        return VOS_NULL;
    }

    uint8_t hashIdx = BkfPuberTableTypeGetHashIdx(vTbl->tableTypeId);
    tableTypeMng->tableTypeCache[hashIdx] = tableType;
    return tableType;
}

void BkfPuberTableTypeDelete(BkfPuberTableType *tableType)
{
    BkfPuberTableTypeMng *tableTypeMng = tableType->tableTypeMng;
    uint8_t hashIdx = BkfPuberTableTypeGetHashIdx(tableType->vTbl.tableTypeId);
    if (tableTypeMng->tableTypeCache[hashIdx] == tableType) {
        tableTypeMng->tableTypeCache[hashIdx] = VOS_NULL;
    }

    VOS_AVLL_DELETE(tableTypeMng->tableTypeSet, tableType->avlNode);
    BKF_FREE(tableTypeMng->argInit->memMng, tableType);
}

void BkfPuberTableTypeDeleteAll(BkfPuberTableTypeMng *tableTypeMng)
{
    BkfPuberTableType *tableType = VOS_NULL;
    while ((tableType = VOS_AVLL_FIRST(tableTypeMng->tableTypeSet)) != VOS_NULL) {
        BkfPuberTableTypeDelete(tableType);
    }
}

BkfPuberTableType *BkfPuberTableTypeFind(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId)
{
    uint8_t hashIdx = BkfPuberTableTypeGetHashIdx(tableTypeId);
    BkfPuberTableType *tableType = tableTypeMng->tableTypeCache[hashIdx];
    BOOL hit = (tableType != VOS_NULL) && (tableType->vTbl.tableTypeId == tableTypeId);
    if (hit) {
        return tableType;
    }

    tableType = VOS_AVLL_FIND(tableTypeMng->tableTypeSet, &tableTypeId);
    if (tableType != VOS_NULL) {
        tableTypeMng->tableTypeCache[hashIdx] = tableType;
    }
    return tableType;
}

BkfPuberTableType *BkfPuberTableTypeFindNext(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId)
{
    return VOS_AVLL_FIND_NEXT(tableTypeMng->tableTypeSet, &tableTypeId);
}

BkfPuberTableType *BkfPuberTableTypeGetFirst(BkfPuberTableTypeMng *tableTypeMng, void **itorOutOrNull)
{
    BkfPuberTableType *tableType = VOS_AVLL_FIRST(tableTypeMng->tableTypeSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (tableType != VOS_NULL) ? VOS_AVLL_NEXT(tableTypeMng->tableTypeSet, tableType->avlNode)
                                                 : VOS_NULL;
    }
    return tableType;
}

BkfPuberTableType *BkfPuberTableTypeGetNext(BkfPuberTableTypeMng *tableTypeMng, void **itorInOut)
{
    BkfPuberTableType *tableType = (*itorInOut);
    *itorInOut = (tableType != VOS_NULL) ? VOS_AVLL_NEXT(tableTypeMng->tableTypeSet, tableType->avlNode) : VOS_NULL;
    return tableType;
}

#ifdef __cplusplus
}
#endif
