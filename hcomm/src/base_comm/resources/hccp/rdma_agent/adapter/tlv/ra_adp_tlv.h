/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_TLV_H
#define RA_ADP_TLV_H

int RaRsTlvInitV1(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsTlvInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsTlvDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsTlvRequest(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsTlvRequestV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCustomChannel(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
#endif // RA_ADP_TLV_H
