/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_BUFQ_H
#define BKF_BUFQ_H

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
typedef struct tagBkfBufq BkfBufq;

/* func */
BkfBufq *BkfBufqInit(BkfMemMng *memMng, int32_t bufLen);
void BkfBufqUninit(BkfBufq *bufq);

/*
如果buf空闲长度不够：
1. en将尽力写入，直至buf满，返回实际写入的长度
2. EnDataLen是为了减少数据拷贝次数，外部填充，此处只是更新长度。如果长度不够，返回负数
*/
int32_t BkfBufqEn(BkfBufq *bufq, uint8_t *data, int32_t dataLen);
int32_t BkfBufqEnDataLen(BkfBufq *bufq, int32_t dataLen);

int32_t BkfBufqDe(BkfBufq *bufq, int32_t deLen);
int32_t BkfBufqDe2Left(BkfBufq *bufq, int32_t leftLen);

uint8_t *BkfBufqGetUsedBegin(BkfBufq *bufq);
int32_t BkfBufqGetUsedLen(BkfBufq *bufq);
uint8_t *BkfBufqGetFreeBegin(BkfBufq *bufq);
int32_t BkfBufqGetFreeLen(BkfBufq *bufq);
BOOL BkfBufqIsFull(BkfBufq *bufq);
BOOL BkfBufqIsEmpty(BkfBufq *bufq);
void BkfBufqReset(BkfBufq *bufq);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
