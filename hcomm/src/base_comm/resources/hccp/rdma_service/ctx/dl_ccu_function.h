/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_CCU_FUNCTION_H
#define DL_CCU_FUNCTION_H

#include "ccu_u_api.h"

struct RsCcuOps {
    int (*rsCcuInit)(void);
    int (*rsCcuUninit)(void);
    int (*rsCcuCustomChannel)(const struct channel_info_in *in, struct channel_info_out *out);
    unsigned long long (*rsCcuGetCqeBaseAddr)(unsigned int dieId);
    int (*rsCcuGetMemInfo)(unsigned int dieId, unsigned long long memTypeBitmap, struct ccu_mem_rsp *rsp);
    int (*rsCcuTlvRequest)(struct ccu_tlv_request_info *tlvRequestInfo);
};

int RsCcuApiInit(void);
void RsCcuApiDeinit(void);

int RsCcuInit(void);
int RsCcuUninit(void);
int RsCcuCustomChannel(const struct channel_info_in *in, struct channel_info_out *out);
int RsCcuGetCqeBaseAddr(unsigned int dieId, unsigned long long *cqeBaseAddr);
int RsCcuGetMemInfo(char *dataIn, char *dataOut, unsigned int *bufferSize);
int RsCcuTlvRequest(unsigned int type, char *dataIn, char *dataOut, unsigned int dataInlength,
    unsigned int *dataOutLength, unsigned int dataOutMaxLength);
bool isCcuTlvReqExist(void);
#endif // DL_CCU_FUNCTION_H
