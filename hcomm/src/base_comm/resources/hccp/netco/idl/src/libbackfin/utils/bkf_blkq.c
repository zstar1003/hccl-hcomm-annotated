/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_blkq.h"
#include "bkf_assert.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
typedef struct tagBkfBlk {
    int32_t dataTotalLen;
    int32_t dataUsedLen;
    uint8_t *data;
} BkfBlk;

struct tagBkfBlkq {
    BkfMemMng *memMng;
    int32_t blkMaxCnt;
    int32_t blkUsedCnt;
    int32_t blkFrontIdx;       /* blkq可能为空，需要加上 usedCnt 一起判断此值是否真正有效 */
    BkfBlk blkLastNotRealFree; /* 重用内存 */
    BkfBlk blk[0];
};
#define BKF_BLKQ_CNT_AND_IDX_IS_INVALID(blkq)                                                                          \
    (((blkq)->blkMaxCnt <= 0) || ((blkq)->blkUsedCnt > (blkq)->blkMaxCnt) || ((blkq)->blkFrontIdx < 0) ||              \
        ((blkq)->blkFrontIdx >= (blkq)->blkMaxCnt))

#pragma pack()

/* func */
BkfBlkq *BkfBlkqInit(BkfMemMng *memMng, int32_t blkMaxCnt)
{
    BkfBlkq *blkq = VOS_NULL;
    uint32_t len;

    if ((memMng == VOS_NULL) || (blkMaxCnt <= 0)) {
        return VOS_NULL;
    }

    len = sizeof(BkfBlkq) + sizeof(BkfBlk) * ((uint32_t)blkMaxCnt);
    blkq = BKF_MALLOC(memMng, len);
    if (blkq == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(blkq, len, 0, len);
    blkq->memMng = memMng;
    blkq->blkMaxCnt = blkMaxCnt;
    return blkq;
}

void BkfBlkqReset(BkfBlkq *blkq)
{
    BkfBlk *blk = VOS_NULL;
    int32_t i;

    blkq->blkUsedCnt = 0;
    blkq->blkFrontIdx = 0;

    blk = &blkq->blkLastNotRealFree;
    if (blk->data != VOS_NULL) {
        BKF_FREE(blkq->memMng, blk->data);
        blk->data = VOS_NULL;
    }
    (void)memset_s(blk, sizeof(BkfBlk), 0, sizeof(BkfBlk));

    for (i = 0; i < blkq->blkMaxCnt; i++) {
        blk = &blkq->blk[i];
        if (blk->data != VOS_NULL) {
            BKF_FREE(blkq->memMng, blk->data);
            blk->data = VOS_NULL;
        }
        (void)memset_s(blk, sizeof(BkfBlk), 0, sizeof(BkfBlk));
    }
}

void BkfBlkqUninit(BkfBlkq *blkq)
{
    if (blkq == VOS_NULL) {
        return;
    }

    BkfBlkqReset(blkq);
    BKF_FREE(blkq->memMng, blkq);
    return;
}

STATIC BkfBlk *BkfBlkqEnFreeBlk(BkfBlkq *blkq, uint8_t *blkData, int32_t blkDataLen, BOOL *hasError)
{
    int32_t nextFreeIdx;
    BkfBlk *nextFreeBlk = VOS_NULL;

    *hasError = VOS_TRUE;
    if ((blkq == VOS_NULL) || (blkData == VOS_NULL) || (blkDataLen < 0)) {
        goto error;
    }
    if (BKF_BLKQ_CNT_AND_IDX_IS_INVALID(blkq)) {
        BKF_ASSERT(0);
        goto error;
    }

    if (blkDataLen == 0) {
        *hasError = VOS_FALSE;
        goto error;
    }
    if (blkq->blkUsedCnt >= blkq->blkMaxCnt) {
        goto error;
    }
    nextFreeIdx = (blkq->blkFrontIdx + blkq->blkUsedCnt) % blkq->blkMaxCnt;
    nextFreeBlk = &blkq->blk[nextFreeIdx];
    if (nextFreeBlk->data != VOS_NULL) {
        BKF_ASSERT(0);
        goto error;
    }

    return nextFreeBlk;

error:

    return VOS_NULL;
}
int32_t BkfBlkqEn(BkfBlkq *blkq, uint8_t *blkData, int32_t blkDataLen)
{
    BkfBlk *nextFreeBlk = VOS_NULL;
    BOOL hasError = VOS_FALSE;
    BOOL mayReuse = VOS_FALSE;

    nextFreeBlk = BkfBlkqEnFreeBlk(blkq, blkData, blkDataLen, &hasError);
    if (nextFreeBlk == VOS_NULL) {
        return !hasError ? 0 : -1;
    }

    if (blkq->blkLastNotRealFree.data != VOS_NULL) {
        mayReuse = (blkq->blkLastNotRealFree.dataTotalLen >= blkDataLen);
        if (mayReuse) {
            *nextFreeBlk = blkq->blkLastNotRealFree;
        } else {
            BKF_FREE(blkq->memMng, blkq->blkLastNotRealFree.data);
        }
        (void)memset_s(&blkq->blkLastNotRealFree, sizeof(BkfBlk), 0, sizeof(BkfBlk));
    }
    if (nextFreeBlk->data == VOS_NULL) {
        nextFreeBlk->data = BKF_MALLOC(blkq->memMng, blkDataLen);
        if (nextFreeBlk->data == VOS_NULL) {
            return -1;
        }

        nextFreeBlk->dataTotalLen = blkDataLen;
    }

    (void)memcpy_s(nextFreeBlk->data, nextFreeBlk->dataTotalLen, blkData, blkDataLen);
    nextFreeBlk->dataUsedLen = blkDataLen;
    blkq->blkUsedCnt++;
    return blkDataLen;
}

STATIC BkfBlk *BkfBlkqGetAtBlk(BkfBlkq *blkq, int32_t pos, BOOL *hasError)
{
    int32_t posIdx;

    *hasError = VOS_TRUE;
    if (blkq == VOS_NULL) {
        goto error;
    }
    if (BKF_BLKQ_CNT_AND_IDX_IS_INVALID(blkq)) {
        BKF_ASSERT(0);
        goto error;
    }

    if (pos >= blkq->blkMaxCnt) {
        goto error;
    } else {
        *hasError = VOS_FALSE;
        if (pos >= blkq->blkUsedCnt) {
            goto error;
        }

        posIdx = (blkq->blkFrontIdx + pos) % blkq->blkMaxCnt;
        return &blkq->blk[posIdx];
    }

error:

    return VOS_NULL;
}
int32_t BkfBlkqDe(BkfBlkq *blkq)
{
    BkfBlk *frontBlk = VOS_NULL;
    int32_t frontBlkDataUsedLen;
    BOOL hasError = VOS_FALSE;

    frontBlk = BkfBlkqGetAtBlk(blkq, 0, &hasError);
    if (frontBlk == VOS_NULL) {
        return !hasError ? 0 : -1;
    }

    if (blkq->blkLastNotRealFree.data != VOS_NULL) {
        BKF_FREE(blkq->memMng, blkq->blkLastNotRealFree.data);
    }
    blkq->blkLastNotRealFree = *frontBlk;
    blkq->blkLastNotRealFree.dataUsedLen = 0;
    frontBlkDataUsedLen = frontBlk->dataUsedLen;
    (void)memset_s(frontBlk, sizeof(BkfBlk), 0, sizeof(BkfBlk));
    blkq->blkFrontIdx = (blkq->blkFrontIdx + 1) % blkq->blkMaxCnt;
    blkq->blkUsedCnt--;
    return frontBlkDataUsedLen;
}

uint8_t *BkfBlkqGetAt(BkfBlkq *blkq, int32_t pos, int32_t *posDataLenOrNull)
{
    BkfBlk *posBlk = VOS_NULL;
    BOOL hasError = VOS_FALSE;

    posBlk = BkfBlkqGetAtBlk(blkq, pos, &hasError);
    if (posBlk == VOS_NULL) {
        if (posDataLenOrNull != VOS_NULL) {
            *posDataLenOrNull = !hasError ? 0 : -1;
        }
        return VOS_NULL;
    }

    if (posDataLenOrNull != VOS_NULL) {
        *posDataLenOrNull = posBlk->dataUsedLen;
    }
    return posBlk->data;
}

uint8_t *BkfBlkqGetFront(BkfBlkq *blkq, int32_t *frontDataLenOrNull)
{
    return BkfBlkqGetAt(blkq, 0, frontDataLenOrNull);
}

int32_t BkfBlkqGetUsedCnt(BkfBlkq *blkq)
{
    if (blkq == VOS_NULL) {
        return -1;
    }
    if (BKF_BLKQ_CNT_AND_IDX_IS_INVALID(blkq)) {
        BKF_ASSERT(0);
        return -1;
    }

    return blkq->blkUsedCnt;
}

int32_t BkfBlkqGetFreeCnt(BkfBlkq *blkq)
{
    if (blkq == VOS_NULL) {
        return -1;
    }
    if (BKF_BLKQ_CNT_AND_IDX_IS_INVALID(blkq)) {
        BKF_ASSERT(0);
        return -1;
    }

    return (blkq->blkMaxCnt - blkq->blkUsedCnt);
}

BOOL BkfBlkqIsFull(BkfBlkq *blkq)
{
    return (BkfBlkqGetFreeCnt(blkq) <= 0) ? VOS_TRUE : VOS_FALSE;
}

BOOL BkfBlkqIsEmpty(BkfBlkq *blkq)
{
    return (BkfBlkqGetUsedCnt(blkq) <= 0) ? VOS_TRUE : VOS_FALSE;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
