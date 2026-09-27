/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TC_RDMA_AGENT_H
#define __TC_RDMA_AGENT_H

#include <stdio.h>

void TcHostAbnormalQpModeTest(); /* SOP mode test */

void TcHdcSendRecvPktRecvCheck();

void TcRaPeerSocketWhiteListAdd01();
void TcRaPeerSocketWhiteListAdd02();
void TcRaPeerSocketWhiteListDel();

void TcRaPeerRdevInit01();
void TcRaPeerRdevInit02();
void TcRaPeerRdevInit03();
void TcRaPeerRdevInit04();
void TcRaPeerRdevDeinit01();
void TcRaPeerRdevDeinit02();
void TcRaPeerRdevDeinit03();
void TcRaPeerRdevGetPortStatusSucc();
void TcRaPeerRdevGetPortStatusFail();
void TcRaPeerSocketBatchConnect();
void TcRaPeerSocketListenStart01();
void TcRaPeerSocketListenStart02();
void TcRaPeerSocketListenStop();
void TcRaPeerSetRsConnParam();
void TcRaInetPton01();
void TcRaInetPton02();
void TcRaSocketInit();
void TcRaSocketInitV1();
void TcRaSendWrlist();
void TcRaRdevInit();
void TcRaRdevGetPortStatus();
void TcRaHdcSocketBatchClose();
void TcRaHdcRdevDeinit();
void TcRaHdcSocketWhiteListAdd();
void TcRaHdcSocketWhiteListDel();
void TcRaHdcSocketAcceptCreditAdd();
void TcRaHdcRdevInit();
void TcRaHdcInitApart();
void TcRaHdcQpDestroy();
void TcRaHdcQpDestroy01();
void TcMsgSendHeadCheck();
void TcMsgRecvHeadCheck();
void TcRaGetSocketConnectInfo();
void TcRaGetSocketListenInfo();
void TcRaGetSocketListenResult();
void TcRaHwHdcInit();
void TcHccpInitDeinit();
void TcRaPeerInitFail001();
void TcRaPeerSocketDeinit001();
void TcHostNotifyBaseAddrInit();
void TcHostNotifyBaseAddrInit001();
void TcHostNotifyBaseAddrInit002();
void TcHostNotifyBaseAddrInit003();
void TcHostNotifyBaseAddrInit005();
void TcHostNotifyBaseAddrInit006();
void TcHostNotifyBaseAddrInit007();

void TcHostNotifyBaseAddrUninit();
void TcHostNotifyBaseAddrUninit001();
void TcHostNotifyBaseAddrUninit002();
void TcHostNotifyBaseAddrUninit003();
void TcHostNotifyBaseAddrUninit004();
void TcHostNotifyBaseAddrUninit005();
void TcRaPeerSendWrlist();
void TcRaPeerSendWrlist001();
void TcRaPeerGetAllSockets();
void TcRaPeerGetSocketNum();
void TcRaGetQpContext();
void TcRaCreateCq();
void TcRaCreateNotmalQp();
void TcRaCreateCompChannel();
void TcRaGetCqeErrInfo();
void TcRaRdevGetCqeErrInfoList();
void TcRaRsGetIfnum();
void TcRaCreateSrq();
void TcRaRsSocketPortIsUse();
void TcRaRsGetVnicIpInfosV1();
void TcRaRsGetVnicIpInfos();
void TcRaRsTypicalMrReg();
void TcRaRsTypicalQpCreate();
void TcRaHdcRecvHandleSendPktUnsuccess();
void TcRaGetTlsEnable();
void TcRaGetSecRandom();
void TcRaRsGetSecRandom();
void TcRaRsGetTlsEnable();
void TcRaGetHccnCfg();
void TcRaRsGetHccnCfg();
void TcRaSaveSnapshotInput();
void TcRaSaveSnapshotPre();
void TcRaSaveSnapshotPost();
void TcHdcAsyncDelReqHandle();
void TcRaHdcDeinitAsyncAll();
#endif
