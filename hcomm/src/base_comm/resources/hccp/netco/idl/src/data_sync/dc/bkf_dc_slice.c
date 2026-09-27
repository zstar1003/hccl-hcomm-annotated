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
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
BkfDcSlice *BkfDcAddSlice(BkfDc *dc, void *sliceKey)
{
    uint32_t hashIdx = BKF_DC_GET_SLICE_HASH_IDX(sliceKey);
    BkfDcSlice *slice = VOS_NULL;
    uint32_t len;
    BOOL insOk = VOS_FALSE;

    len = sizeof(BkfDcSlice) + dc->argInit.sliceVTbl.keyLen;
    slice = BKF_MALLOC(dc->argInit.memMng, len);
    if (slice == VOS_NULL) {
        goto error;
    }
    (void)memset_s(slice, len, 0, len);
    slice->dc = dc;
    VOS_AVLL_INIT_TREE(slice->tableSet, (AVLL_COMPARE)Bkfuint16_tCmp, BKF_OFFSET(BkfDcTable, tableTypeId),
        BKF_OFFSET(BkfDcTable, avlNode));
    (void)memcpy_s(slice->key, dc->argInit.sliceVTbl.keyLen, sliceKey, dc->argInit.sliceVTbl.keyLen);
    VOS_AVLL_INIT_NODE(slice->avlNode);
    insOk = VOS_AVLL_INSERT(dc->sliceSet, slice->avlNode);
    if (!insOk) {
        goto error;
    }

    dc->sliceCache[hashIdx] = slice;
    return slice;

error:

    if (slice != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(dc->argInit.memMng, slice);
    }
    return VOS_NULL;
}

void BkfDcDelSlice(BkfDc *dc, BkfDcSlice *slice)
{
    uint32_t hashIdx = BKF_DC_GET_SLICE_HASH_IDX(slice->key);
    if (dc->sliceCache[hashIdx] == slice) {
        dc->sliceCache[hashIdx] = VOS_NULL;
    }
    BkfDcDelAllTable(dc, slice);
    VOS_AVLL_DELETE(dc->sliceSet, slice->avlNode);
    BKF_FREE(dc->argInit.memMng, slice);
    return;
}

void BkfDcDelAllSlice(BkfDc *dc)
{
    BkfDcSlice *slice = VOS_NULL;
    void *itor = VOS_NULL;

    for (slice = BkfDcGetFirstSlice(dc, &itor); slice != VOS_NULL; slice = BkfDcGetNextSlice(dc, &itor)) {
        BkfDcDelSlice(dc, slice);
    }
    return;
}

BkfDcSlice *BkfDcFindSlice(BkfDc *dc, void *sliceKey)
{
    uint32_t hashIdx = BKF_DC_GET_SLICE_HASH_IDX(sliceKey);
    BkfDcSlice *slice = VOS_NULL;
    BOOL hit = VOS_FALSE;

    slice = dc->sliceCache[hashIdx];
    hit = (slice != VOS_NULL) && (dc->argInit.sliceVTbl.keyCmp(sliceKey, slice->key) == 0);
    if (hit) {
        return slice;
    }

    slice = VOS_AVLL_FIND(dc->sliceSet, sliceKey);
    if (slice != VOS_NULL) {
        dc->sliceCache[hashIdx] = slice;
    }
    return slice;
}

BkfDcSlice *BkfDcFindNextSlice(BkfDc *dc, void *sliceKey)
{
    BkfDcSlice *slice = VOS_NULL;

    slice = VOS_AVLL_FIND_NEXT(dc->sliceSet, sliceKey);
    return slice;
}

BkfDcSlice *BkfDcGetFirstSlice(BkfDc *dc, void **itorOutOrNull)
{
    BkfDcSlice *slice = VOS_NULL;

    slice = VOS_AVLL_FIRST(dc->sliceSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (slice != VOS_NULL) ? VOS_AVLL_NEXT(dc->sliceSet, slice->avlNode) : VOS_NULL;
    }
    return slice;
}

BkfDcSlice *BkfDcGetNextSlice(BkfDc *dc, void **itorInOut)
{
    BkfDcSlice *slice = VOS_NULL;

    slice = (*itorInOut);
    *itorInOut = (slice != VOS_NULL) ? VOS_AVLL_NEXT(dc->sliceSet, slice->avlNode) : VOS_NULL;
    return slice;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
