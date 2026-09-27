/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_RA_CTX_H
#define TC_RA_CTX_H
#ifdef __cplusplus
extern "C" {
#endif
void TcRaGetDevEidInfoNum();
void TcRaGetDevEidInfoList();
void TcRaCtxInit();
void TcRaGetDevBaseAttr();
void TcRaCtxDeinit();
void TcRaCtxLmemRegister();
void TcRaCtxRmemImport();
void TcRaCtxChanCreate();
void TcRaCtxTokenIdAlloc();
void TcRaCtxTokenIdAlloc1();
void TcRaCtxTokenIdAlloc2();
void TcRaCtxTokenIdAlloc3();
void TcRaCtxCqCreate();
void TcRaCtxQpCreate();
void TcRaCtxQpImport();
void TcRaCtxQpBind();
void TcRaBatchSendWr();
void TcRaCtxUpdateCi();
void TcRaGetEidByIp();
void TcRaHdcGetEidByIp();
void TcRaRsGetEidByIp();
void TcRaPeerGetEidByIp();
void TcRaGetIpByEid();
void TcRaHdcGetIpByEid();
void TcRaRsGetIpByEid();
void TcRaPeerGetIpByEid();
void TcRaCtxGetAuxInfo();
void TcRaHdcCtxGetAuxInfo();
void TcRaRsCtxGetAuxInfo();
void TcRaCtxGetCrErrInfoList();
void TcRaHdcCtxGetCrErrInfoList();
void TcRaRsCtxGetCrErrInfoList();
void TcRaCtxGetJettyContext();
void TcRaHdcCtxGetJettyContext();
void TcRaPeerCtxGetJettyContext();
void TcRaCtxNotifyEvent();
void TcRaHdcCtxNotifyEvent();

void TcRaGetTpInfoListAsync();
void TcRaHdcGetTpInfoListAsync();
void TcRaRsGetTpInfoList();
void TcRaRsAsyncHdcSessionConnect();
void TcRaHdcAsyncSendPkt();
void TcRaHdcPoolAddTask();
void TcRaHdcAsyncRecvPkt();
void TcHdcAsyncRecvPkt();
void TcRaHdcPoolCreate();
void TcRaAsyncHandlePkt();
void TcRaHdcAsyncHandleSocketListenStart();
void TcRaHdcAsyncHandleQpImport();
void TcRaPeerCtxInit();
void TcRaPeerCtxDeinit();
void TcRaPeerGetDevEidInfoNum();
void TcRaPeerGetDevEidInfoList();
void TcRaPeerCtxTokenIdAlloc();
void TcRaPeerCtxTokenIdFree();
void TcRaPeerCtxLmemRegister();
void TcRaPeerCtxLmemUnregister();
void TcRaPeerCtxRmemImport();
void TcRaPeerCtxRmemUnimport();
void TcRaPeerCtxChanCreate();
void TcRaPeerCtxChanDestroy();
void TcRaPeerCtxCqCreate();
void TcRaPeerCtxCqDestroy();
void TcRaPeerCtxQpCreate();
void TcRaCtxPrepareQpCreate();
void TcRaPeerCtxQpDestroy();
void TcRaPeerCtxQpImport();
void TcRaPeerCtxQpUnimport();
void TcRaPeerCtxQpBind();
void TcRaPeerCtxQpUnbind();
void TcRaCtxQpDestroyBatchAsync();
void TcQpDestroyBatchParamCheck();
void TcRaHdcCtxQpDestroyBatchAsync();
void TcRaRsCtxQpDestroyBatch();
void TcRaCtxQpQueryBatch();
void TcQpQueryBatchParamCheck();
void TcRaHdcCtxQpQueryBatch();
void TcRaRsCtxQpQueryBatch();
void TcRaGetTpAttrAsync();
void TcRaHdcGetTpAttrAsync();
void TcRaRsGetTpAttr();
void TcRaSetTpAttrAsync();
void TcRaHdcSetTpAttrAsync();
void TcRaRsSetTpAttr();
void TcRaCtxGetTpInfoList();
void TcRaPeerCtxGetTpInfoList();
void TcRaCtxGetTpAttr();
void TcRaPeerCtxGetTpAttr();
void TcRaCtxSetTpAttr();
void TcRaPeerCtxSetTpAttr();
#ifdef __cplusplus
}
#endif
#endif
