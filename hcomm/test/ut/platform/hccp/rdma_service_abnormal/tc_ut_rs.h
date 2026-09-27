/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TC_UT_RS_H
#define _TC_UT_RS_H

#include <stdio.h>

#define RS_TEST_MEM_SIZE 32
#define RS_TEST_MEM_PAGE_SIZE 4096

void TcRsInit2();
void TcRsDeinit2();

void TcRsSocketInit();
void TcRsSocketDeinit();

void TcRsRdevInit();
void TcRsRdevDeinit();

void TcRsGetTsqpDepthAbnormal();
void TcRsSetTsqpDepthAbnormal();

void TcRsSocketListenStart2();
void TcRsSocketBatchConnect2();
void TcRsQpCreate2();
void TcRsMrOps2();

void TcRsAbnormal2();
void TcRsEpollOps2();
void TcRsSocketOps2();
void TcRsSocketClose2();
void TcRsMrAbnormal2();
void TcRsGetGidIndex2();
void TcRsQpConnectAsync2();
void TcRsSendWr2();
void TcTlsAbnormal1();
void TcRsSocketNodeid2vnic();
void TcRsServerValidAsyncInit();
void TcRsConnectHandle();
void TcRsGetQpContext();
void TcRsSocketGetBindByChip();
void TcRsSocketBatchAbort();
void TcRsSocketSendAndRecvLogTest();
void TcRsTcpRecvTagInHandle();
void TcRsServerValidAsyncAbnormal();
void TcRsServerValidAsyncAbnormal01();
void TcRsNetApiInitFail();
#endif
