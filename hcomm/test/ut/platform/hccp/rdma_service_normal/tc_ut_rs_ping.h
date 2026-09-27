/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_UT_RS_PING_H
#define TC_UT_RS_PING_H

void TcRsPayloadHeaderResvCustomCheck();
void TcRsPingHandleInit();
void TcRsPingHandleDeinit();
void TcRsPingInit();
void TcRsPingTargetAdd();
void TcRsGetPingCb();
void TcRsPingClientPostSend();
void TcRsPingGetResults();
void TcRsPingTaskStop();
void TcRsPingTargetDel();
void TcRsPingDeinit();
void TcRsPingCompareRdmaInfo();
void TcRsPingRoceFindTargetNode();
void TcRsPongFindTargetNode();
void TcRsPongFindAllocTargetNode();
void TcRsPingPollSendCq();
void TcRsPingServerPostSend();
void TcRsPingPostRecv();
void TcRsPingClientPollCq();
void TcRsEpollEventPingHandle();
void TcRsPingGetTripTime();
void TcRsPingCbInitMutex();
void TcRsPingResolveResponsePacket();
void TcRsPingServerPollCq();
void TcRsPingCbGetDevRdevIndex();
void TcRsPingInitMrCb();
void TcRsPingCommonDeinitLocalBuffer();
void TcRsPingCommonDeinitLocalQp();
void TcRsPingRocePollScq();
void TcRsPingPongInitLocalInfo();
void TcRsPingHandle();
void TcRsPingRocePingCbDeinit();

#endif
