/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_RA_HDC_H
#define TC_RA_HDC_H
#ifdef __cplusplus
extern "C" {
#endif
void TcHdc();
void TcHdcInit();
void TcHdcInitFail();
void TcHdcDeinitFail();
void TcHdcSocketBatchConnect();
void TcHdcSocketBatchClose();
void TcHdcSocketListenStart();
void TcHdcSocketBatchAbort();
void TcHdcSocketListenStop();
void TcHdcGetSockets();
void TcHdcSocketSend();
void TcHdcSocketRecv();
void TcHdcQpCreateDestroy();
void TcHdcGetQpStatus();
void TcHdcQpConnectAsync();
void TcHdcMrReg();
void TcHdcMrDereg();
void TcHdcSendWr();
void TcHdcSendWrlist();
void TcHdcGetNotifyBaseAddr();
void TcHdcSocketInit();
void TcHdcSocketDeinit();
void TcHdcRdevInit();
void TcHdcRdevDeinit();
void TcHdcSocketWhiteListAdd();
void TcHdcSocketWhiteListDel();
void TcHdcGetIfaddrs();
void TcHdcGetIfaddrsV2();
void TcHdcGetIfnum();
void TcHdcMessageProcessFail();
void TcHdcSocketRecvFail();
void TcRaHdcSendWrlistExtInit();
void TcRaHdcSendWrlistExt();
void TcRaHdcSendNormalWrlist();
void TcRaHdcSetQpAttrQos();
void TcRaHdcSetQpAttrTimeout();
void TcRaHdcSetQpAttrRetryCnt();
void TcRaHdcGetCqeErrInfo();
void TcRaHdcGetCqeErrInfoList();
void TcRaHdcQpCreateOp();
void TcRaHdcGetQpStatusOp();
void TcHdcSendWrOp();
void TcHdcLiteSendWrOp();
void TcHdcRecvWrlist();
void TcHdcPollCq();
void TcHdcGetLiteSupport();
void TcRaRdevGetSupportLite();
void TcRaRdevGetHandle();
void TcRaIsFirstOrLastUsed();
void TcRaHdcLiteCtxInit();
void RcRaHdcLiteQpCreate();
void TcRaHdcTlvRequest();
void TcRaHdcGetTlvRecvMsg();
void TcRaHdcQpCreateWithAttrs();
#ifdef __cplusplus
}
#endif
#endif
