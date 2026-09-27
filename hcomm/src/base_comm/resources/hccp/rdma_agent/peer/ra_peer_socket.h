/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_PEER_SOCKET_H
#define RA_PEER_SOCKET_H

#include "hccp_common.h"

int RaPeerGetClientSocketErrInfo(unsigned int phyId, struct SocketConnectInfoT conn[], struct SocketErrInfo err[],
    unsigned int num);
int RaPeerGetServerSocketErrInfo(unsigned int phyId, struct SocketListenInfoT conn[], struct ServerSocketErrInfo err[],
    unsigned int num);
int RaPeerSocketAcceptCreditAdd(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num,
    unsigned int creditLimit);
#endif // RA_PEER_SOCKET_H
