/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_SUB_STATIC_H
#define BKF_SUBER_SUB_STATIC_H

#include "bkf_suber_env.h"
#include "bkf_suber_data.h"
#include "bkf_subscriber.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

typedef struct tagBkfSuberTableTypeMng BkfSuberTableTypeMng;
typedef struct tagBkfSuberTableTypeMngInitArg {
    BkfSuberEnv *env;
} BkfSuberTableTypeMngInitArg;

BkfSuberTableTypeMng *BkfSuberTableTypeMngInit(BkfSuberTableTypeMngInitArg *initArg);
void BkfSuberTableTypeMngUnInit(BkfSuberTableTypeMng *tableTypeMng);

uint32_t BkfSuberTableTypeAdd(BkfSuberTableTypeMng *tableTypeMng, uint8_t mode, void *vtbl, void *userData,
    uint16_t userDataLen);
uint32_t BkfSuberTableTypeDel(BkfSuberTableTypeMng *tableTypeMng, uint16_t type);
BkfSuberTableTypeVTbl *BkfSuberTableTypeGetVtbl(BkfSuberTableTypeMng *tableTypeMng, uint16_t tablTypeId);
void *BkfSuberTableTypeGetUserData(BkfSuberTableTypeMng *tableTypeMng, uint16_t tablTypeId);
#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif