/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <sys/time.h>
#include <sys/epoll.h>
#include <errno.h>
#include "hccp_tlv.h"
#include "ra_rs_comm.h"
#include "ra_comm.h"

int RsTlvInit(unsigned int moduleType, unsigned int phyId, unsigned int* bufferSize) { return 0; }

int RsTlvDeinit(unsigned int moduleType, unsigned int phyId) { return 0; }

int RsTlvRequest(struct TlvRequestMsgHead* head, char* data) { return 0; }

int RsCtxCustomChannel(const struct ChannelInfoIn* in, struct ChannelInfoOut* out) { return 0; }
