/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_BUFR_H
#define BKF_BUFR_H

#include "bkf_comm.h"
#include "bkf_mem.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
typedef struct tagBkfBufr BkfBufr;

typedef struct tagBkfBufrDbgPeekItor {
    int32_t peekStartIdx;
    int32_t peekLeftLen;
    int32_t prePeekDataLen;
    uint8_t prePeekData[0x20];
} BkfBufrDbgPeekItor;

/* func */
BkfBufr *BkfBufrInit(BkfMemMng *memMng, int32_t bufLen);
void BkfBufrUninit(BkfBufr *bufr);

uint32_t BkfBufrReset(BkfBufr *bufr);
uint32_t BkfBufrResize(BkfBufr *bufr, int32_t newBufLen);
/* 如果buf空闲长度不够，put将尽力写入，直至buf满，返回实际写入的长度；putForce会覆盖最旧已经写入的数据 */
int32_t BkfBufrPut(BkfBufr *bufr, uint8_t *data, int32_t dataLen);
int32_t BkfBufrPutForce(BkfBufr *bufr, uint8_t *data, int32_t dataLen);
int32_t BkfBufrGet(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen);
int32_t BkfBufrPeek(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen);
int32_t BkfBufrGetUsedLen(BkfBufr *bufr);
int32_t BkfBufrGetFreeLen(BkfBufr *bufr);
BOOL BkfBufrIsFull(BkfBufr *bufr);
BOOL BkfBufrIsEmpty(BkfBufr *bufr);

char *BkfBufrGetStr(BkfBufr *bufr, uint8_t *buf, int32_t bufLen);
/* 如果lastLen接近bufr的缓冲长度，而bufr不停往里填数据，很容易导致数据被覆写而获取失败 */
int32_t BkfBufrDbgPeekLastFirst(BkfBufr *bufr, int32_t lastLen, uint8_t *outBuf, int32_t outBufLen,
    BkfBufrDbgPeekItor *itor);
int32_t BkfBufrDbgPeekLastNext(BkfBufr *bufr, uint8_t *outBuf, int32_t outBufLen, BkfBufrDbgPeekItor *itor);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
