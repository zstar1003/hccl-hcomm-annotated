/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_RA_HOST_H
#define TC_RA_HOST_H
extern "C" void TcHost();
extern "C" void TcIfaddr();
extern "C" void TcRaRecvWrlist(void);
extern "C" void TcHostRaSendWrlistExt();
extern "C" void TcHostRaSendNormalWrlist();
extern "C" void TcRaSetQpAttrQos(void);
extern "C" void TcRaSetQpAttrTimeout(void);
extern "C" void TcRaSetQpAttrRetryCnt(void);
extern "C" void TcRaCreateEventHandle(void);
extern "C" void TcRaCtlEventHandle(void);
extern "C" void TcRaWaitEventHandle(void);
extern "C" void TcRaDestroyEventHandle(void);
extern "C" void TcRaPollCq(void);
extern "C" void TcGetVnicIpInfos(void);
extern "C" void TcRaSocketBatchAbort(void);
extern "C" void TcRaGetClientSocketErrInfo(void);
extern "C" void TcRaGetServerSocketErrInfo(void);
extern "C" void TcRaSocketAcceptCreditAdd(void);
extern "C" void TcRaRemapMr(void);
extern "C" void TcRaRegisterMr(void);
extern "C" void TcRaGetLbMax(void);
extern "C" void TcRaSetQpLbValue(void);
extern "C" void TcRaGetQpLbValue(void);
extern "C" void TcRaGetQpHyperFeature(void);
extern "C" void TcRaSetDebugConfig(void);
#endif
