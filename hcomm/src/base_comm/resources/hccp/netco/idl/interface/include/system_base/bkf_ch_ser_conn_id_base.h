/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_SER_CONN_ID_BASE_H
#define BKF_CH_SER_CONN_ID_BASE_H

#include "bkf_comm.h"
#include "bkf_ch_ser_adef.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* comm */
/* 此结构必须为 conn id 结构的第一个成员 */
typedef struct tagBkfChSerConnIdBase {
    uint16_t sign;
    uint8_t urlType;
    uint8_t pad1[1];
} BkfChSerConnIdBase;

/* func */
uint32_t BkfChSerConnIdBaseInit(BkfChSerConnIdBase *base, uint8_t urlType);
void BkfChSerConnIdBaseUninit(BkfChSerConnIdBase *base);
BkfChSerConnIdBase *BkfChSerConnIdGetBase(BkfChSerConnId *connId);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
