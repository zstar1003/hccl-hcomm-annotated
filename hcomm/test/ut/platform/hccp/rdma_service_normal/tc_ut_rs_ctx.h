/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TC_UT_RS_CTX_H
#define _TC_UT_RS_CTX_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void TcRsGetDevEidInfoNum();
void TcRsGetDevEidInfoList();
void TcRsCtxInit();
void TcRsCtxDeinit();
void TcRsCtxTokenIdAlloc();
void TcRsCtxTokenIdFree();
void TcRsCtxLmemReg();
void TcRsCtxLmemUnreg();
void TcRsCtxRmemImport();
void TcRsCtxRmemUnimport();
void TcRsCtxChanCreate();
void TcRsCtxChanDestroy();
void TcRsCtxCqCreate();
void TcRsCtxCqDestroy();
void TcRsCtxQpCreate();
void TcRsCtxQpDestroy();
void TcRsCtxQpImport();
void TcRsCtxQpUnimport();
void TcRsCtxQpBind();
void TcRsCtxQpUnbind();
void TcRsCtxBatchSendWr();
void TcRsCtxUpdateCi();
void TcRsCtxEsched();
void TcDlCcuApiInit();
void TcRsGetTpInfoList();
void TcRsCtxQpDestroyBatch();
void TcRsCtxQpQueryBatch();
void TcRsNetApiInitDeinit();
void TcRsNetAllocJfcId();
void TcRsNetFreeJfcId();
void TcRsNetAllocJettyId();
void TcRsNetFreeJettyId();
void TcRsNetGetCqeBaseAddr();
void TcRsCcuGetCqeBaseAddr();
void TcRsCtxGetAuxInfo();
void TcRsGetTpAttr();
void TcRsSetTpAttr();
void TcRsCtxGetCrErrInfoList();
void TcRsCtxNotifyEvent();
void TcRsCtxNotifyEventAubdfxFail();
void TcRsAubdfxApiInitRollback();
#ifdef __cplusplus
}
#endif
#endif
