/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_bufq.h"
#include "bkf_assert.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
#define BKF_BUFQ_GUARD 0xAA
struct tagBkfBufq {
    BkfMemMng *memMng;
    int32_t bufLen;
    int32_t usedLen;
    uint8_t *guard;
    uint8_t buf[0];
};

#pragma pack()

/* func */
BkfBufq *BkfBufqInit(BkfMemMng *memMng, int32_t bufLen)
{
    BkfBufq *bufq = VOS_NULL;
    uint32_t len;

    if ((memMng == VOS_NULL) || (bufLen <= 0)) {
        return VOS_NULL;
    }

    len = sizeof(BkfBufq) + (uint32_t)bufLen + sizeof(*bufq->guard);
    bufq = BKF_MALLOC(memMng, len);
    if (bufq == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(bufq, len, 0, len);
    bufq->memMng = memMng;
    bufq->bufLen = bufLen;
    bufq->usedLen = 0;
    bufq->guard = (uint8_t *)(bufq + 1) + bufLen;
    BKF_SIGN_SET(*bufq->guard, BKF_BUFQ_GUARD);
    return bufq;
}

void BkfBufqUninit(BkfBufq *bufq)
{
    if (bufq == VOS_NULL) {
        return;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        /* 继续释放 */
    }

    BKF_SIGN_CLR(*bufq->guard);
    BKF_FREE(bufq->memMng, bufq);
    return;
}

void BkfBufqReset(BkfBufq *bufq)
{
    if (bufq == VOS_NULL) {
        return;
    }
    bufq->usedLen = 0;
    return;
}

int32_t BkfBufqEn(BkfBufq *bufq, uint8_t *data, int32_t dataLen)
{
    int32_t freeLen;
    int32_t cpLen;
    errno_t err;

    if ((bufq == VOS_NULL) || (data == VOS_NULL) || (dataLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return -1;
    }

    if (dataLen == 0) {
        return 0;
    }
    freeLen = bufq->bufLen - bufq->usedLen;
    if (freeLen <= 0) {
        return freeLen;
    }
    cpLen = BKF_GET_MIN(dataLen, freeLen);
    err = memcpy_s(bufq->buf + bufq->usedLen, freeLen, data, cpLen);
    if (err != EOK) {
        BKF_ASSERT(0);
        return -1;
    }
    bufq->usedLen += cpLen;
    return cpLen;
}

int32_t BkfBufqEnDataLen(BkfBufq *bufq, int32_t dataLen)
{
    int32_t freeLen;

    if ((bufq == VOS_NULL) || (dataLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return -1;
    }
    freeLen = bufq->bufLen - bufq->usedLen;
    if (dataLen > freeLen) {
        return -1;
    }

    bufq->usedLen += dataLen;
    return dataLen;
}

int32_t BkfBufqDe(BkfBufq *bufq, int32_t deLen)
{
    uint8_t *leftBegin = VOS_NULL;
    int32_t leftLen;
    errno_t err;

    if ((bufq == VOS_NULL) || (deLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return -1;
    }
    if (deLen > bufq->usedLen) {
        return -1;
    }

    leftBegin = bufq->buf + deLen;
    leftLen = bufq->usedLen - deLen;
    err = memmove_s(bufq->buf, bufq->bufLen, leftBegin, leftLen);
    if (err != EOK) {
        BKF_ASSERT(0);
        return -1;
    }
    bufq->usedLen = leftLen;
    return deLen;
}

int32_t BkfBufqDe2Left(BkfBufq *bufq, int32_t leftLen)
{
    int32_t deLen;

    if ((bufq == VOS_NULL) || (leftLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return -1;
    }
    if (leftLen > bufq->usedLen) {
        return -1;
    }

    deLen = bufq->usedLen - leftLen;
    return (BkfBufqDe(bufq, deLen) == deLen) ? leftLen : -1;
}

uint8_t *BkfBufqGetUsedBegin(BkfBufq *bufq)
{
    if (bufq == VOS_NULL) {
        return VOS_NULL;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    return bufq->buf;
}

int32_t BkfBufqGetUsedLen(BkfBufq *bufq)
{
    if (bufq == VOS_NULL) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return -1;
    }

    return bufq->usedLen;
}

uint8_t *BkfBufqGetFreeBegin(BkfBufq *bufq)
{
    if (bufq == VOS_NULL) {
        return VOS_NULL;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    return bufq->buf + bufq->usedLen;
}

int32_t BkfBufqGetFreeLen(BkfBufq *bufq)
{
    if (bufq == VOS_NULL) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufq->guard, BKF_BUFQ_GUARD)) {
        BKF_ASSERT(0);
        return -1;
    }

    return (bufq->bufLen - bufq->usedLen);
}

BOOL BkfBufqIsFull(BkfBufq *bufq)
{
    return (BkfBufqGetFreeLen(bufq) <= 0) ? VOS_TRUE : VOS_FALSE;
}

BOOL BkfBufqIsEmpty(BkfBufq *bufq)
{
    return (BkfBufqGetUsedLen(bufq) <= 0) ? VOS_TRUE : VOS_FALSE;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
