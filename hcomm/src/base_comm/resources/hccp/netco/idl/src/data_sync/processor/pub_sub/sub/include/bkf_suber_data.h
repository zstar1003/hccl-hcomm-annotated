/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_DATA_H
#define BKF_SUBER_DATA_H

#include "bkf_url.h"
#include "bkf_suber_env.h"
#include "bkf_subscriber.h"
#include "bkf_dl.h"
#include "v_avll.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

typedef struct tagBkfSuberDataMng BkfSuberDataMng;
typedef struct tagBkfSuberInst BkfSuberInst;
typedef struct tagBkfSuberSlice BkfSuberSlice;
typedef struct tagBkfSuberAppSub BkfSuberAppSub;

typedef void (*F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER)(void *cookieInit, BkfSuberInst *inst);
typedef void (*F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER)(void *cookieInit, BkfSuberSlice *slice, uint64_t instId);
typedef void (*F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER)(void *cookieInit, BkfSuberAppSub *appSub);

typedef struct tagBkfSuberSubDataMngInitArg {
    BkfSuberEnv *env;
    void *cookie;
    F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER ntfPuberUrlAdd;
    F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER ntfPuberUrlDel;
    F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER ntfSliceInstAdd;
    F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER ntfSliceInstDel;
    F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER ntfAppSubAdd;
    F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER ntfAppSubDel;
} BkfSuberSubDataMngInitArg;

struct tagBkfSuberInst {
    BkfSuberDataMng *dataMng;
    uint64_t instId;
    BkfUrl puberUrl;
    BkfUrl localUrl;
    AVLL_NODE avlNode;
    BkfDl obSlice; /* 管理属于本inst的切片 */
    uint32_t sliceCnt;
    BkfDlNode nodeObBySubConn; /* 插入subConn管理链 */
    uint8_t isTemp : 1;
    uint8_t pad : 7;
};

#define BKF_SUBER_INVALID_INST_ID 0xFFFFFFFFFFFFFFFF

struct tagBkfSuberSlice {
    BkfSuberDataMng *dataMng;
    uint64_t instId;
    AVLL_NODE avlNode;
    AVLL_TREE sliceInstSet; /* 管理BkfSuberSliceInst  */
    BkfDl obAppSub;
    uint32_t instCnt; /* 绑定本切片的实例数 */
    uint32_t appSubCnt;
    uint8_t isTemp : 1;
    uint8_t pad : 7;
    uint8_t sliceKey[0];
};

typedef struct tagBkfSuberSliceInst {
    AVLL_NODE avlNode; /* 插入  */
    BkfDlNode nodeObByInst;
    uint64_t instId; /* key */
    BkfSuberSlice *slice;
    BkfSuberInst *inst;
} BkfSuberSliceInst;

typedef struct tagBkfSuberAppSubKey {
    uint16_t tableTypeId;
    uint8_t pad1[2];
    uint64_t instId;
    F_BKF_CMP sliceKeyCmp;
    void *sliceKey;
} BkfSuberAppSubKey;

struct tagBkfSuberAppSub {
    BkfSuberDataMng *dataMng;
    BkfSuberAppSubKey key;
    AVLL_NODE avlNode;
    BkfDlNode nodeObBySlice;
    uint8_t isVerify : 1;
    uint8_t pad : 7;
    uint8_t sliceKey[0];
};

BkfSuberDataMng *BkfSuberDataMngInit(BkfSuberSubDataMngInitArg *initArg);
void BkfSuberDataMngUnInit(BkfSuberDataMng *dataMng);

BkfSuberInst *BkfSuberDataAddInst(BkfSuberDataMng *dataMng, uint64_t instId);
void BkfSuberDataDelInstByKey(BkfSuberDataMng *dataMng, uint64_t instId);
void BkfSuberDataDelInst(BkfSuberInst *inst);
void BkfSuberDataDelAllInst(BkfSuberDataMng *dataMng);
BkfSuberInst *BkfSuberDataGetFirstInst(BkfSuberDataMng *dataMng, void **itorOutOrNull);
BkfSuberInst *BkfSuberDataGetNextInst(BkfSuberDataMng *dataMng, void **itorInOut);
BkfSuberInst *BkfSuberDataFindInst(BkfSuberDataMng *dataMng, uint64_t instId);
BkfSuberInst *BkfSuberDataFindNextInst(BkfSuberDataMng *dataMng, uint64_t instId);
uint32_t BkfSuberDataInstAddPuberUrl(BkfSuberInst *inst, BkfUrl *url);
uint32_t BkfSuberDataInstAddLocalUrl(BkfSuberInst *inst, BkfUrl *url);
uint32_t BkfSuberDataInstAddPuberUrlEx(BkfSuberInst *inst, BkfUrl *url);
void BkfSuberDataInstDelPuberUrl(BkfSuberInst *inst);
void BkfSuberDataInstDelLocalUrl(BkfSuberInst *inst);

BkfSuberSlice *BkfSuberDataAddSlice(BkfSuberDataMng *dataMng, void *sliceKey);
void BkfSuberDataDelSliceByKey(BkfSuberDataMng *dataMng, void *sliceKey);
void BkfSuberDataDelSliceSpecInst(BkfSuberSlice *slice, uint64_t instId);
void BkfSuberDataDelSlice(BkfSuberSlice *slice);
void BkfSuberDataDelAllSlice(BkfSuberDataMng *dataMng);
BkfSuberSlice *BkfSuberDataGetFirstSlice(BkfSuberDataMng *dataMng, void **itorOutOrNull);
BkfSuberSlice *BkfSuberDataGetNextSlice(BkfSuberDataMng *dataMng, void **itorInOut);
BkfSuberSlice *BkfSuberDataFindSlice(BkfSuberDataMng *dataMng, void *sliceKey);
BkfSuberSlice *BkfSuberDataFindNextSlice(BkfSuberDataMng *dataMng, void *sliceKey);
uint32_t BkfSuberDataSliceAddInst(BkfSuberSlice *slice, uint64_t instId);

uint32_t BkfSuberDataAddAppSub(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId, uint64_t instId,
    BOOL isVerify);

void BkfSuberDataDelAppSubByKey(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId, uint64_t instId,
    BOOL isVerify);
void BkfSuberDataDelAppSub(BkfSuberAppSub *appSub);
void BkfSuberDataDelAllAppSub(BkfSuberDataMng *dataMng);
BkfSuberAppSub *BkfSuberDataGetFirstAppSub(BkfSuberDataMng *dataMng, void **itorOutOrNull);
BkfSuberAppSub *BkfSuberDataGetNextAppSub(BkfSuberDataMng *dataMng, void **itorInOut);
BkfSuberAppSub *BkfSuberDataFindAppSub(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId, uint64_t instId);
BkfSuberAppSub *BkfSuberDataFindNextAppSub(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId,
    uint64_t instId);

uint32_t BkfSuberDataTableTypeAdd(BkfSuberDataMng *dataMng, uint8_t mode, void *vtbl, void *userData,
    uint16_t userDataLen);
uint32_t BkfSuberDataTableTypeDel(BkfSuberDataMng *dataMng, uint16_t tabType);
BkfSuberTableTypeVTbl *BkfSuberDataTableTypeGetVtbl(BkfSuberDataMng *dataMng, uint16_t tablTypeId);
void *BkfSuberDataTableTypeGetUserData(BkfSuberDataMng *dataMng, uint16_t tablTypeId);
typedef void (*F_BKF_SUBER_DATA_SCAN_APPSUB_BY_SLICE_PROC)(BkfSuberAppSub *appSub, void *usrData, BOOL *needContinue);
void BkfSuberDataScanAppSubBySlice(BkfSuberSlice *slice, uint64_t instId,
    F_BKF_SUBER_DATA_SCAN_APPSUB_BY_SLICE_PROC proc, void *usrData);
uint8_t BkfSuberDataTableTypeGetMode(BkfSuberTableTypeVTbl *vTbl);
BkfSuberSliceInst *BkfSuberDataGetFirstSliceInst(BkfSuberSlice *slice, void **itorOutOrNull);
BkfSuberSliceInst *BkfSuberDataGetNextSliceInst(BkfSuberSlice *slice, void **itorInOut);
#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif