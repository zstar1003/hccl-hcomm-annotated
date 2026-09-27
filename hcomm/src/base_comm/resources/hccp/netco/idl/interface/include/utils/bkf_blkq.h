/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_BLKQ_H
#define BKF_BLKQ_H

#include "bkf_comm.h"
#include "bkf_mem.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/*
de时对占用的内存没有立即释放。如果紧跟的en的数据长度不大于上次de的，复用。
那好处是减少内存申请释放的次数，坏处是占用内存可能稍多。
*/

/* common */
typedef struct tagBkfBlkq BkfBlkq;

/* func */
BkfBlkq *BkfBlkqInit(BkfMemMng *memMng, int32_t blkMaxCnt);
void BkfBlkqUninit(BkfBlkq *blkq);

int32_t BkfBlkqEn(BkfBlkq *blkq, uint8_t *blkData, int32_t blkDataLen);
int32_t BkfBlkqDe(BkfBlkq *blkq);
void BkfBlkqReset(BkfBlkq *blkq);

/* front 的 pos = 0 */
uint8_t *BkfBlkqGetAt(BkfBlkq *blkq, int32_t pos, int32_t *posDataLenOrNull);
uint8_t *BkfBlkqGetFront(BkfBlkq *blkq, int32_t *frontDataLenOrNull);
int32_t BkfBlkqGetUsedCnt(BkfBlkq *blkq);
int32_t BkfBlkqGetFreeCnt(BkfBlkq *blkq);
BOOL BkfBlkqIsFull(BkfBlkq *blkq);
BOOL BkfBlkqIsEmpty(BkfBlkq *blkq);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
