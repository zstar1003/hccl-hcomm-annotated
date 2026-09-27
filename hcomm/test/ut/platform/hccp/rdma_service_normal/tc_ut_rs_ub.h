/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TC_UT_RS_UB_H
#define _TC_UT_RS_UB_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void TcRsUbGetRdevCb();
void TcRsUrmaApiInitAbnormal();
void TcRsUbV2();
void TcRsUbGetDevEidInfoNum();
void TcRsUbGetDevEidInfoList();
struct rs_cb* TcRsUbV2Init(int mode, unsigned int* devIndex);
void TcRsUbV2Deinit(struct rs_cb* rsCb, int mode, unsigned int devIndex);
void TcRsUbCtxTokenIdAlloc();
void TcRsUbCtxTokenIdAlloc1();
void TcRsUbCtxTokenIdAlloc2();
void TcRsUbCtxTokenIdAlloc3();
void TcRsUbCtxJfceCreate();
void TcRsUbCtxJfcCreate();
void TcRsUbCtxJfcCreateNormal();
void TcRsUbCtxJettyCreate();
void TcRsUbCtxJettyImport();
void TcRsUbCtxJettyBind();
void TcRsUbCtxBatchSendWr();
void TcRsUbFreeCbList();
void TcRsUbCtxExtJettyCreate();
void TcRsUbCtxRmemImport();
void TcRsGetTpInfoList();
void TcRsUbCtxDrvJettyImport();
void TcRsUbDevCbInit();
void TcRsUbCtxInit();
void TcRsUbCtxJfcDestroy();
void TcRsUbCtxExtJettyDelete();
void TcRsUbCtxChanCreate();
void TcRsUbCtxDeinit();
void TcRsUbInitSegCb();
void TcRsUbCtxLmemReg();
void TcRsUbCtxJfcCreateFail();
void TcRsUbCtxInitJettyCb();
void TcRsUbCtxJettyCreateFail();
void TcRsUbCtxJettyImportFail();
void TcRsUbCtxBatchSendWrFail();
void TcRsUbCtxJettyDestroyBatch();
void TcRsUbCtxQueryJettyBatch();
void TcRsGetEidByIp();
void TcRsUbGetEidByIp();
void TcRsGetIpByEid();
void TcRsUbGetIpByEid();
void TcRsUbCtxGetAuxInfo();
void TcRsUbGetTpAttr();
void TcRsUbSetTpAttr();
void TcRsEpollEventJfcInHandle();
void TcRsHandleEpollPollJfc();
void TcRsJfcCallbackProcess();
#ifdef __cplusplus
}
#endif
#endif
