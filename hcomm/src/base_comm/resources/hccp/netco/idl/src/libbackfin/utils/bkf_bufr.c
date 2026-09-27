/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_bufr.h"
#include "bkf_assert.h"
#include "securec.h"
#include "v_stringlib.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
/* API ok，实现后续可以优化 */
#define BKF_BUFR_GUARD ((uint8_t)0xA9)
struct tagBkfBufr {
    BkfMemMng *memMng;
    int32_t bufLen;
    int32_t startIdx;
    int32_t usedLen;
    uint64_t validPutTotalLen;
    uint64_t overwriteLen;
    uint64_t getTotalLen;
    uint8_t *buf;
    uint8_t *guard;
};

#define BKF_BUFR_LEN_IS_VALID(bufr)                                                                                    \
    (((bufr)->bufLen > 0) && ((bufr)->startIdx >= 0) && ((bufr)->startIdx <= (bufr)->bufLen) &&                        \
        ((bufr)->usedLen >= 0) && ((bufr)->usedLen <= (bufr)->bufLen))

#pragma pack()

/* func */
STATIC uint32_t BkfBufrInitVal(BkfBufr *bufr, BkfMemMng *memMng, int32_t bufLen)
{
    bufr->memMng = memMng;
    bufr->bufLen = bufLen;
    bufr->startIdx = 0;
    bufr->usedLen = 0;
    bufr->validPutTotalLen = 0;
    bufr->overwriteLen = 0;
    bufr->getTotalLen = 0;

    bufr->buf = BKF_MALLOC(memMng, bufLen + sizeof(*bufr->guard));
    if (bufr->buf == VOS_NULL) {
        return BKF_ERR;
    }
    bufr->guard = bufr->buf + bufr->bufLen;
    BKF_SIGN_SET(*bufr->guard, BKF_BUFR_GUARD);
    return BKF_OK;
}

BkfBufr *BkfBufrInit(BkfMemMng *memMng, int32_t bufLen)
{
    if ((memMng == VOS_NULL) || (bufLen <= 0)) {
        return VOS_NULL;
    }

    uint32_t len = sizeof(BkfBufr);
    BkfBufr *bufr = BKF_MALLOC(memMng, len);
    if (bufr == VOS_NULL) {
        return VOS_NULL;
    }

    uint32_t ret = BkfBufrInitVal(bufr, memMng, bufLen);
    if (ret != BKF_OK) {
        BkfBufrUninit(bufr);
        return VOS_NULL;
    }

    return bufr;
}

void BkfBufrUninit(BkfBufr *bufr)
{
    if (bufr == VOS_NULL) {
        return;
    }

    if (bufr->buf != VOS_NULL) {
        if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD)) {
            BKF_ASSERT(0);
            /* 继续释放 */
        }
        BKF_SIGN_CLR(*bufr->guard);
        BKF_FREE(bufr->memMng, bufr->buf);
    }

    BKF_FREE(bufr->memMng, bufr);
    return;
}

uint32_t BkfBufrReset(BkfBufr *bufr)
{
    if (bufr == VOS_NULL) {
        return BKF_ERR;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    bufr->startIdx = 0;
    bufr->usedLen = 0;
    bufr->validPutTotalLen = 0;
    bufr->overwriteLen = 0;
    bufr->getTotalLen = 0;
    return BKF_OK;
}

STATIC int32_t BkfBufrPut2Free(BkfBufr *bufr, uint8_t *data, int32_t dataLen)
{
    int32_t freeLen = bufr->bufLen - bufr->usedLen;
    uint8_t *leftData = VOS_NULL;
    int32_t leftLen;
    int32_t putLen;
    int32_t putTotalLen;
    errno_t err;

    if ((dataLen <= 0) || (freeLen <= 0)) {
        return 0;
    }

    leftData = data;
    leftLen = BKF_GET_MIN(dataLen, freeLen);
    putTotalLen = 0;
    if ((bufr->startIdx + bufr->usedLen) < bufr->bufLen) {
        putLen = bufr->bufLen - (bufr->startIdx + bufr->usedLen);
        putLen = BKF_GET_MIN(leftLen, putLen);
        err = memcpy_s(bufr->buf + bufr->startIdx + bufr->usedLen, putLen, leftData, putLen);
        if (err != EOK) {
            BKF_ASSERT(0);
            return -1;
        }
        bufr->usedLen += putLen;
        putTotalLen += putLen;
        leftLen -= putLen;
        if (leftLen <= 0) {
            return putTotalLen;
        }
        leftData += putLen;
        freeLen = bufr->bufLen - bufr->usedLen;
    }

    if (freeLen > 0) {
        putLen = BKF_GET_MIN(leftLen, freeLen);
        err = memcpy_s(bufr->buf + (bufr->startIdx + bufr->usedLen) % bufr->bufLen, putLen, leftData, putLen);
        if (err != EOK) {
            BKF_ASSERT(0);
            return -1;
        }
        bufr->usedLen += putLen;
        putTotalLen += putLen;
    }

    return putTotalLen;
}
int32_t BkfBufrPut(BkfBufr *bufr, uint8_t *data, int32_t dataLen)
{
    int32_t putLen;

    if ((bufr == VOS_NULL) || (data == VOS_NULL) || (dataLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    putLen = BkfBufrPut2Free(bufr, data, dataLen);
    if (putLen >= 0) {
        bufr->validPutTotalLen += (uint32_t)putLen;
    }
    return putLen;
}

int32_t BkfBufrPutForce(BkfBufr *bufr, uint8_t *data, int32_t dataLen)
{
    uint8_t *leftData = VOS_NULL;
    int32_t leftLen;
    int32_t freeLen;
    int32_t dropLen;
    int32_t putLen;

    if ((bufr == VOS_NULL) || (data == VOS_NULL) || (dataLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    if (dataLen > bufr->bufLen) {
        dropLen = dataLen - bufr->bufLen;
        leftData = &data[dropLen];
        leftLen = bufr->bufLen;
        bufr->overwriteLen += (uint32_t)dropLen;
    } else {
        leftData = data;
        leftLen = dataLen;
    }
    freeLen = bufr->bufLen - bufr->usedLen;
    if (leftLen > freeLen) {
        dropLen = leftLen - freeLen;
        bufr->startIdx = (bufr->startIdx + dropLen) % bufr->bufLen;
        bufr->usedLen -= dropLen;
        bufr->overwriteLen += (uint32_t)dropLen;
    }
    putLen = BkfBufrPut2Free(bufr, leftData, leftLen);
    BKF_ASSERT(putLen == leftLen);
    bufr->validPutTotalLen += (uint32_t)dataLen;
    return dataLen;
}

int32_t BkfBufrGetCycle(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen)
{
    int32_t getLen;
    int32_t getTotalLen;
    errno_t err;

    getTotalLen = 0;
    getLen = BKF_GET_MIN(bufr->usedLen, bufr->bufLen - bufr->startIdx);
    getLen = BKF_GET_MIN(getLen, outBufLen);
    if (getLen <= 0) {
        return 0;
    }
    err = memcpy_s(outBuf, getLen, bufr->buf + bufr->startIdx, getLen);
    if (err != EOK) {
        BKF_ASSERT(0);
        return -1;
    }
    bufr->startIdx = (bufr->startIdx + getLen) % bufr->bufLen;
    bufr->usedLen -= getLen;
    getTotalLen += getLen;

    if ((bufr->usedLen > 0) && (getTotalLen < outBufLen)) {
        getLen = BKF_GET_MIN(bufr->usedLen, outBufLen - getTotalLen);
        err = memcpy_s(outBuf + getTotalLen, getLen, bufr->buf + bufr->startIdx, getLen);
        if (err != EOK) {
            BKF_ASSERT(0);
            return -1;
        }
        bufr->startIdx = (bufr->startIdx + getLen) % bufr->bufLen;
        bufr->usedLen -= getLen;
        getTotalLen += getLen;
    }

    return getTotalLen;
}
int32_t BkfBufrGet(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen)
{
    int32_t getLen;

    if ((bufr == VOS_NULL) || (outBuf == VOS_NULL) || (outBufLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    getLen = BkfBufrGetCycle(bufr, outBuf, outBufLen);
    if (getLen >= 0) {
        bufr->getTotalLen += (uint32_t)getLen;
    }
    return getLen;
}

int32_t BkfBufrPeek(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen)
{
    BkfBufr temp;

    if ((bufr == VOS_NULL) || (outBuf == VOS_NULL) || (outBufLen < 0)) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    temp = *bufr;
    return BkfBufrGetCycle(&temp, outBuf, outBufLen);
}

int32_t BkfBufrGetUsedLen(BkfBufr *bufr)
{
    if (bufr == VOS_NULL) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    return bufr->usedLen;
}

int32_t BkfBufrGetFreeLen(BkfBufr *bufr)
{
    if (bufr == VOS_NULL) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    return (bufr->bufLen - bufr->usedLen);
}

BOOL BkfBufrIsFull(BkfBufr *bufr)
{
    return (BkfBufrGetFreeLen(bufr) <= 0) ? VOS_TRUE : VOS_FALSE;
}

BOOL BkfBufrIsEmpty(BkfBufr *bufr)
{
    return (BkfBufrGetUsedLen(bufr) <= 0) ? VOS_TRUE : VOS_FALSE;
}

char *BkfBufrGetStr(BkfBufr *bufr, uint8_t *buf, int32_t bufLen)
{
    if (bufr == VOS_NULL) {
        return "bufr_null";
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD)) {
        return "bufr_sign_invalid";
    }
    if (buf == VOS_NULL) {
        return "info_buf_null";
    }
    if (bufLen <= 0) {
        return "info_buf_len_ng";
    }

    (void)snprintf_truncated_s((char *)buf, bufLen,
        "bufr(%#x), memMng(%#x), bufLen(%d)/startIdx(%d)/usedLen(%d), "
        "validPutTotalLen(%" VOS_PRIu64 ")/overwriteLen(%" VOS_PRIu64 ")/"
        "getTotalLen(%" VOS_PRIu64 "), buf(%#x)/guard(%#x, %#x)",
        BKF_MASK_ADDR(bufr), BKF_MASK_ADDR(bufr->memMng), bufr->bufLen, bufr->startIdx, bufr->usedLen,
        bufr->validPutTotalLen, bufr->overwriteLen, bufr->getTotalLen, BKF_MASK_ADDR(bufr->buf),
        BKF_MASK_ADDR(bufr->guard), *bufr->guard);
    return (char *)buf;
}

STATIC int32_t BkfBufrDbgPeekLast(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen, BkfBufrDbgPeekItor *itor)
{
    BkfBufr temp;
    uint8_t peekData[BKF_MBR_SIZE(BkfBufrDbgPeekItor, prePeekData)];
    int32_t peekLen;
    int32_t getLen;
    BOOL hasChg = VOS_FALSE;

    /*
    1. 初始化
    2. 通过 prePeek ，推断数据有无覆盖
    3. 获取数据 & 更新itor
    4. 更新 prePeek 数据
    */
    /* 1 */
    temp = *bufr;
    if (itor->peekStartIdx >= temp.bufLen) {
        return -1;
    }
    temp.startIdx = itor->peekStartIdx;
    /* 2 */
    if (itor->prePeekDataLen > 0) {
        peekLen = BkfBufrPeek(&temp, peekData, itor->prePeekDataLen);
        hasChg = (peekLen != itor->prePeekDataLen) ||
                 (VOS_MemCmp(peekData, itor->prePeekData, itor->prePeekDataLen) != 0);
        if (hasChg) {
            return -1;
        }
    }
    /* 3 */
    getLen = BkfBufrGetCycle(&temp, outBuf, BKF_GET_MIN(itor->peekLeftLen, outBufLen));
    if ((getLen < 0) || (getLen > itor->peekLeftLen)) {
        return -1;
    }
    itor->peekStartIdx = temp.startIdx;
    itor->peekLeftLen -= getLen;
    /* 4 */
    itor->prePeekDataLen = BkfBufrGetCycle(&temp, itor->prePeekData, sizeof(itor->prePeekData));
    if (itor->prePeekDataLen < 0) {
        return -1;
    }

    return getLen;
}
int32_t BkfBufrDbgPeekLastFirst(BkfBufr *bufr, int32_t lastLen, uint8_t *outBuf, int32_t outBufLen,
    BkfBufrDbgPeekItor *itor)
{
    BOOL argIsInvalid = VOS_FALSE;

    argIsInvalid = (bufr == VOS_NULL) || (lastLen < 0) || (outBuf == VOS_NULL) || (outBufLen < 0) || (itor == VOS_NULL);
    if (argIsInvalid) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    if (bufr->usedLen > lastLen) {
        itor->peekStartIdx = (bufr->startIdx + (bufr->usedLen - lastLen)) % bufr->bufLen;
        itor->peekLeftLen = lastLen;
    } else {
        itor->peekStartIdx = bufr->startIdx;
        itor->peekLeftLen = bufr->usedLen;
    }
    itor->prePeekDataLen = 0;
    return BkfBufrDbgPeekLast(bufr, outBuf, outBufLen, itor);
}

int32_t BkfBufrDbgPeekLastNext(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen, BkfBufrDbgPeekItor *itor)
{
    BOOL argIsInvalid = VOS_FALSE;

    argIsInvalid = (bufr == VOS_NULL) || (outBuf == VOS_NULL) || (outBufLen < 0) || (itor == VOS_NULL) ||
                   (itor->peekLeftLen < 0);
    if (argIsInvalid) {
        return -1;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return -1;
    }

    if (itor->prePeekDataLen <= 0) {
        return 0;
    }
    return BkfBufrDbgPeekLast(bufr, outBuf, outBufLen, itor);
}

uint32_t BkfBufrResize(BkfBufr *bufr, int32_t newBufLen)
{
    if (bufr == VOS_NULL) {
        return BKF_ERR;
    }
    if (!BKF_SIGN_IS_VALID(*bufr->guard, BKF_BUFR_GUARD) || !BKF_BUFR_LEN_IS_VALID(bufr)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    uint8_t *temp = BKF_MALLOC(bufr->memMng, newBufLen + sizeof(*bufr->guard));
    if (temp == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    int32_t usedLen = BkfBufrGetCycle(bufr, temp, newBufLen);
    if (usedLen < 0) {
        BKF_ASSERT(0);
        BKF_FREE(bufr->memMng, temp);
        return BKF_ERR;
    }

    BKF_SIGN_CLR(*bufr->guard);
    BKF_FREE(bufr->memMng, bufr->buf);

    bufr->usedLen = usedLen;
    bufr->startIdx = 0;
    bufr->buf = temp;
    bufr->bufLen = newBufLen;
    bufr->guard = bufr->buf + bufr->bufLen;
    BKF_SIGN_SET(*bufr->guard, BKF_BUFR_GUARD);
    return BKF_OK;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
