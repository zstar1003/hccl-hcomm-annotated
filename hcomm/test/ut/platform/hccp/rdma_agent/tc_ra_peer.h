/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_RA_PEER_H
#define TC_RA_PEER_H
void TcPeer();
void TcPeerFail();
void TcRaPeerEpollCtlAdd();
void TcRaPeerSetTcpRecvCallback();
void TcRaPeerEpollCtlMod();
void TcRaPeerEpollCtlDel();
void TcRaPeerCqCreate();
void TcRaPeerNormalQpCreate();
void TcRaPeerCreateEventHandle();
void TcRaPeerCtlEventHandle();
void TcRaPeerWaitEventHandle();
void TcRaPeerDestroyEventHandle();
void TcRaPeerSocketBatchAbort();
void TcRaLoopbackQpCreate();
void TcRaPeerLoopbackQpCreate();
void TcRaPeerLoopbackSingleQpCreate();
void TcRaPeerSetQpLbValue();
void TcRaPeerGetQpLbValue();
void TcRaPeerGetLbMax();
void TcRaPeerGetHccnCfg();
#endif
