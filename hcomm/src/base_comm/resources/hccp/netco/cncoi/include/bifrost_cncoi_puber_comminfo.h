/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIFROST_CNCOI_PUBER_COMMINFO_H

#define BIFROST_CNCOI_PUBER_COMMINFO_H

#include "bifrost_cncoi_slice.h"
#include "bifrost_cncoi_puber.h"
#include "bifrost_cncoi_reader.h"

#if __cplusplus
extern "C" {
#endif
#pragma pack(4)

typedef int32_t (*F_BIFROST_CNCOI_PUBER_ON_FILL_COMMINFO_UPDATE_DATA)(void *cookieOfReg, void *builder,
    BifrostCncoiSliceKeyT *sliceKey, BifrostCncoiComminfoKeyT *tupleKey, void *tupleVal, void *codeBuf, int32_t bufLen);
typedef void (*F_BIFROST_CNCOI_PUBER_ON_COMMINFO_TABLE)(void *cookieOfReg, BifrostCncoiSliceKeyT *sliceKey);
typedef char *(*F_BIFROST_CNCOI_PUBER_ON_GET_COMMINFO_TUPLE_VAL_STR)(void *cookieOfReg, void *tupleVal, uint8_t *buf,
    int32_t bufLen);

typedef uint32_t (*F_BIFROST_CNCOI_PUBER_COMMINFO_GETTUPLE_FIRST)(void *cookie, uint32_t type, void *outputKey,
    void *outputTupleVal);
typedef uint32_t (*F_BIFROST_CNCOI_PUBER_COMMINFO_GETTUPLE_NEXT)(void *cookie, uint32_t type, void *lastKey,
    void *outputKey, void *outputTupleVal);

typedef struct tagBifrostCncoiPuberComminfoVTbl {
    uint16_t tupleValLen;
    uint8_t noDcTuple;
    uint8_t pad001[1];
    int32_t tupleCntMax;
    void *cookie;
    F_BIFROST_CNCOI_PUBER_ON_FILL_COMMINFO_UPDATE_DATA onFillUpdateData;
    F_BIFROST_CNCOI_PUBER_ON_COMMINFO_TABLE onSub;
    F_BIFROST_CNCOI_PUBER_ON_COMMINFO_TABLE onUnSub;
    F_BIFROST_CNCOI_PUBER_ON_GET_COMMINFO_TUPLE_VAL_STR onGetTupleValStr;
    F_BIFROST_CNCOI_PUBER_COMMINFO_GETTUPLE_FIRST getFirstTuple;
    F_BIFROST_CNCOI_PUBER_COMMINFO_GETTUPLE_NEXT getNextTuple;
} BifrostCncoiPuberComminfoVTbl;

uint32_t BifrostCncoiPuberComminfoReg(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiPuberComminfoVTbl *appVTbl);
uint32_t BifrostCncoiPuberComminfoNotifyTableComplete(BifrostCncoiPuber *bifrostCncoiPuber);
uint32_t BifrostCncoiPuberComminfoCreateTable(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey);
void BifrostCncoiPuberComminfoDeleteTable(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey);
void BifrostCncoiPuberComminfoReleaseTable(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey);
uint32_t BifrostCncoiPuberComminfoUpdate(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiComminfoKeyT *tupleKey, void *val);
void BifrostCncoiPuberComminfoDelete(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiComminfoKeyT *tupleKey);

#pragma pack()

#if __cplusplus
}
#endif

#endif
