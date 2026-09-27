/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_TABLE_TYPE_DATA_H
#define BKF_PUBER_TABLE_TYPE_DATA_H

#include "bkf_puber_adef.h"
#include "bkf_puber_table_type.h"
#include "v_avll.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
#define BKF_PUBER_TABLE_TYPE_HASH_MOD 0x0f

typedef struct {
    BkfPuberTableTypeMng *tableTypeMng;
    AVLL_NODE avlNode;
    BkfPuberTableTypeVTbl vTbl;
    uint8_t userData[0];
} BkfPuberTableType;

struct TagBkfPuberTableTypeMng {
    BkfPuberInitArg *argInit;
    char *name;
    BkfLog *log;
    uint8_t dispInitOk : 1;
    uint8_t rsv : 7;
    uint8_t pad1[3];
    AVLL_TREE tableTypeSet;
    BkfPuberTableType *tableTypeCache[BKF_PUBER_TABLE_TYPE_HASH_MOD + 1];
};
#pragma pack()

BkfPuberTableTypeMng *BkfPuberTableTypeDataInit(BkfPuberInitArg *arg);
void BkfPuberTableTypeDataUninit(BkfPuberTableTypeMng *tableTypeMng);
BkfPuberTableType *BkfPuberTableTypeAdd(BkfPuberTableTypeMng *tableTypeMng, BkfPuberTableTypeVTbl *vTbl, void *userData,
    uint16_t dataLen);
void BkfPuberTableTypeDelete(BkfPuberTableType *tableType);
void BkfPuberTableTypeDeleteAll(BkfPuberTableTypeMng *tableTypeMng);
BkfPuberTableType *BkfPuberTableTypeFind(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId);
BkfPuberTableType *BkfPuberTableTypeFindNext(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId);
BkfPuberTableType *BkfPuberTableTypeGetFirst(BkfPuberTableTypeMng *tableTypeMng, void **itorOutOrNull);
BkfPuberTableType *BkfPuberTableTypeGetNext(BkfPuberTableTypeMng *tableTypeMng, void **itorInOut);

static inline uint8_t BkfPuberTableTypeGetHashIdx(uint16_t tableTypeId)
{
    return BKF_GET_U8_FOLD4_VAL((uint8_t)tableTypeId);
}

#ifdef __cplusplus
}
#endif

#endif
