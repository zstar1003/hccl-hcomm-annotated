/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_TABLE_TYPE_H
#define BKF_PUBER_TABLE_TYPE_H

#include "bkf_puber_adef.h"
#include "bkf_assert.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
typedef struct TagBkfPuberTableTypeMng BkfPuberTableTypeMng;
#pragma pack()

BkfPuberTableTypeMng *BkfPuberTableTypeInit(BkfPuberInitArg *arg);
void BkfPuberTableTypeUninit(BkfPuberTableTypeMng *tableTypeMng);
uint32_t BkfPuberTableTypeAttachEx(BkfPuberTableTypeMng *tableTypeMng, BkfPuberTableTypeVTbl *vTbl, void *userData,
    uint16_t userDataLen);
BkfPuberTableTypeVTbl *BkfPuberTableTypeGetVTbl(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId);
void *BkfPuberTableTypeGetUserData(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId);
void BkfPuberTableTypeDispSummary(BkfPuberTableTypeMng *tableTypeMng, BkfDisp *disp);
void BkfPuberTableTypeDeattach(BkfPuberTableTypeMng *tableTypeMng, uint16_t tabType);

#ifdef __cplusplus
}
#endif

#endif
