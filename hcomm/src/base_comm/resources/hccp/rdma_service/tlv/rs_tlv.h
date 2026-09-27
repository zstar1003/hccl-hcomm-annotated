/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_TLV_H
#define RS_TLV_H

#include "hccp_tlv.h"
#include "ra_rs_comm.h"
#include "rs.h"

#define RS_TLV_BUFFER_SIZE (64 * 1024) // 64KB

RS_ATTRI_VISI_DEF int RsTlvInit(unsigned int phyId, unsigned int *bufferSize);
RS_ATTRI_VISI_DEF int RsTlvDeinit(unsigned int phyId);
RS_ATTRI_VISI_DEF int RsTlvRequest(struct TlvRequestMsgHead *head, char *dataIn, char *dataOut,
    unsigned int *bufferSize, unsigned int dataMaxLength);
RS_ATTRI_VISI_DEF int RsCtxCustomChannel(const struct CustomChanInfoIn *in, struct CustomChanInfoOut *out);
#endif // RS_TLV_H
