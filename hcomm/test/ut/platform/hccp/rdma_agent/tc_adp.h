/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_RA_ADP_H
#define TC_RA_ADP_H

#define MAX_TEST_MESSAGE 10
#ifdef __cplusplus
extern "C" {
#endif
void TcAdapter();
void TcHccpInit();
void TcHccpInitFail();
void TcHccpDeinitFail();
void TcSocketConnect();
void TcSocketClose();
void TcSocketAbort();
void TcSocketListenStart();
void TcSocketListenStop();
void TcSocketInfo();
void TcSocketSend();
void TcSocketRecv();
void TcSocketInit();
void TcSocketDeinit();
void TcSetTsqpDepth();
void TcGetTsqpDepth();
void TcQpCreate();
void TcQpDestroy();
void TcQpStatus();
void TcQpInfo();
void TcQpConnect();
void TcMrReg();
void TcMrDreg();
void TcSendWr();
void TcSendWrlist();
void TcRdevInit();
void TcRdevDeinit();
void TcGetNotifyBa();
void TcSetPid();
void TcGetVnicIp();
void TcSocketWhiteListAdd();
void TcSocketWhiteListDel();
void TcGetIfaddrs();
void TcGetIfaddrsV2();
void TcGetIfnum();
void TcGetInterfaceVersion();
void TcMessageProcessFail();
void TcSetNotifyCfg();
void TcGetNotifyCfg();
void TcTlvInit();
void TcTlvDeinit();
void TcTlvRequest();
void TcRaRsSendWrList();
void TcRaRsSendWrListExt();
void TcRaRsSendNormalWrlist();
void TcRaRsSetQpAttrQos();
void TcRaRsSetQpAttrTimeout();
void TcRaRsSetQpAttrRetryCnt();
void TcRaRsGetCqeErrInfo();
void TcRaRsGetCqeErrInfoNum();
void TcRaRsGetCqeErrInfoList();
void TcRaRsGetLiteSupport();
void tc_ra_RsGetLiteRdevCap();
void TcRaRsGetLiteQpCqAttr();
void TcRaRsGetLiteConnectedInfo();
void TcRaRsSocketWhiteListV2();
void TcRaRsSocketCreditAdd();
void TcRaRsGetLiteMemAttr();
void TcRaRsPingInit();
void TcRaRsPingTargetAdd();
void TcRaRsPingTaskStart();
void TcRaRsPingGetResults();
void TcRaRsPingTaskStop();
void TcRaRsPingTargetDel();
void TcRaRsPingDeinit();
void TcRaRsRemapMr();
void TcRaRsTestCtxOps();
void TcRaRsGetTlsEnable0();

#ifdef __cplusplus
}
#endif
#endif
