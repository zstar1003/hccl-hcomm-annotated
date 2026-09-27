/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_table.h"
#include "v_avll.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

struct tagBkfTableMng {
    BkfTableInitArg initArg;
    AVLL_TREE entrySet;
    uint32_t entryCnt;
};

typedef struct tagBkfTableEntry {
    AVLL_NODE avlNode;
    uint8_t keyValue[0];
} BkfTableEntry;

BkfTableMng *BkfTableInit(BkfTableInitArg *arg)
{
    if (unlikely(arg == VOS_NULL || arg->memMng == VOS_NULL || arg->keyCmp == VOS_NULL || arg->keyLen == 0)) {
        return VOS_NULL;
    }

    uint32_t len = sizeof(BkfTableMng);
    BkfTableMng *tableMng = BKF_MALLOC(arg->memMng, len);
    if (tableMng == VOS_NULL) {
        return VOS_NULL;
    }
    tableMng->initArg = *arg;
    VOS_AVLL_INIT_TREE(tableMng->entrySet, (AVLL_COMPARE)(arg->keyCmp), BKF_OFFSET(BkfTableEntry, keyValue),
        BKF_OFFSET(BkfTableEntry, avlNode));
    tableMng->entryCnt = 0;
    return tableMng;
}

void BkfTableDelAll(BkfTableMng *tableMng)
{
    if (unlikely(tableMng == VOS_NULL)) {
        return;
    }

    BkfTableEntry *entry = VOS_NULL;
    BkfTableEntry *entryNext = VOS_NULL;

    for (entry = VOS_AVLL_FIRST(tableMng->entrySet); entry != NULL; entry = entryNext) {
        entryNext = VOS_AVLL_NEXT(tableMng->entrySet, entry->avlNode);
        VOS_AVLL_DELETE(tableMng->entrySet, entry->avlNode);
        BKF_FREE(tableMng->initArg.memMng, entry);
    }
    tableMng->entryCnt = 0;
}

void BkfTableUnInit(BkfTableMng *tableMng)
{
    if (unlikely(tableMng == VOS_NULL)) {
        return;
    }

    BkfTableDelAll(tableMng);
    BKF_FREE(tableMng->initArg.memMng, tableMng);
}

uint32_t BkfTableGetCnt(BkfTableMng *tableMng)
{
    if (unlikely(tableMng == VOS_NULL)) {
        return 0;
    }

    return tableMng->entryCnt;
}

STATIC BkfTableEntry *BkfTableNewEntry(BkfTableMng *tableMng, void *key)
{
    uint32_t len = sizeof(BkfTableEntry) + tableMng->initArg.keyLen + tableMng->initArg.valLen;
    BkfTableEntry *entry = BKF_MALLOC(tableMng->initArg.memMng, len);
    if (entry == VOS_NULL) {
        return VOS_NULL;
    }

    VOS_AVLL_INIT_NODE(entry->avlNode);
    if (memcpy_s(entry->keyValue, len - BKF_MBR_SIZE(BkfTableEntry, avlNode), key, tableMng->initArg.keyLen) != 0) {
        goto error;
    }

    BOOL insOK = VOS_AVLL_INSERT(tableMng->entrySet, entry->avlNode);
    if (!insOK) {
        goto error;
    }

    if (tableMng->entryCnt < 0xFFFFFFFF) {
        tableMng->entryCnt++;
    }

    return entry;
error:
    BKF_FREE(tableMng->initArg.memMng, entry);
    return VOS_NULL;
}

uint32_t BkfTableAdd(BkfTableMng *tableMng, void *key, void *val, BkfTableInfo *infoOutOrNull)
{
    if (unlikely(tableMng == VOS_NULL || key == VOS_NULL || val == VOS_NULL)) {
        return BKF_ERR;
    }

    BkfTableEntry *entry = VOS_AVLL_FIND(tableMng->entrySet, key);
    if (entry == VOS_NULL) {
        entry = BkfTableNewEntry(tableMng, key);
        if (entry == VOS_NULL) {
            return BKF_ERR;
        }
    }

    if (memcpy_s(entry->keyValue + tableMng->initArg.keyLen, tableMng->initArg.valLen, val, tableMng->initArg.valLen) !=
        0) {
        goto error;
    }

    if (infoOutOrNull != VOS_NULL) {
        infoOutOrNull->key = entry->keyValue;
        infoOutOrNull->val = entry->keyValue + tableMng->initArg.keyLen;
    }
    return BKF_OK;

error:
    VOS_AVLL_DELETE(tableMng->entrySet, entry->avlNode);
    BKF_FREE(tableMng->initArg.memMng, entry);
    if (tableMng->entryCnt > 0) {
        tableMng->entryCnt--;
    }
    return BKF_ERR;
}

void BkfTableDel(BkfTableMng *tableMng, void *key)
{
    if (unlikely(tableMng == VOS_NULL || key == VOS_NULL)) {
        return;
    }

    BkfTableEntry *entry = VOS_AVLL_FIND(tableMng->entrySet, key);
    if (entry == VOS_NULL) {
        return;
    }
    VOS_AVLL_DELETE(tableMng->entrySet, entry->avlNode);
    BKF_FREE(tableMng->initArg.memMng, entry);
    if (tableMng->entryCnt > 0) {
        tableMng->entryCnt--;
    }
}

uint32_t BkfTableFind(BkfTableMng *tableMng, void *key, BkfTableInfo *infoOut)
{
    if (unlikely(tableMng == VOS_NULL || infoOut == VOS_NULL)) {
        return BKF_ERR;
    }

    BkfTableEntry *entry = VOS_AVLL_FIND(tableMng->entrySet, key);
    if (entry == VOS_NULL) {
        return BKF_ERR;
    }
    infoOut->key = entry->keyValue;
    infoOut->val = entry->keyValue + tableMng->initArg.keyLen;
    return BKF_OK;
}

uint32_t BkfTableFindNext(BkfTableMng *tableMng, void *key, BkfTableInfo *infoOut)
{
    if (unlikely(tableMng == VOS_NULL || infoOut == VOS_NULL)) {
        return BKF_ERR;
    }

    BkfTableEntry *entry = VOS_AVLL_FIND_NEXT(tableMng->entrySet, key);
    if (entry == VOS_NULL) {
        return BKF_ERR;
    }
    infoOut->key = entry->keyValue;
    infoOut->val = entry->keyValue + tableMng->initArg.keyLen;
    return BKF_OK;
}

uint32_t BkfTableGetFirst(BkfTableMng *tableMng, BkfTableInfo *infoOut, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }

    if (unlikely(tableMng == VOS_NULL || infoOut == VOS_NULL)) {
        return BKF_ERR;
    }

    BkfTableEntry *entry = VOS_AVLL_FIRST(tableMng->entrySet);
    if (entry == VOS_NULL) {
        return BKF_ERR;
    }

    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(tableMng->entrySet, entry->avlNode);
    }
    infoOut->key = entry->keyValue;
    infoOut->val = entry->keyValue + tableMng->initArg.keyLen;
    return BKF_OK;
}

uint32_t BkfTableGetNext(BkfTableMng *tableMng, BkfTableInfo *infoOut, void **itorInOut)
{
    if (unlikely(tableMng == VOS_NULL || itorInOut == VOS_NULL || infoOut == VOS_NULL)) {
        return BKF_ERR;
    }

    BkfTableEntry *entry = *(BkfTableEntry **)itorInOut;
    if (entry != VOS_NULL) {
        infoOut->key = entry->keyValue;
        infoOut->val = entry->keyValue + tableMng->initArg.keyLen;
        *itorInOut = VOS_AVLL_NEXT(tableMng->entrySet, entry->avlNode);
        return BKF_OK;
    } else {
        return BKF_ERR;
    }
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif