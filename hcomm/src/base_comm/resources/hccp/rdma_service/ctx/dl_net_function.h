/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_NET_FUNCTION_H
#define DL_NET_FUNCTION_H

struct RsNetOps {
    int (*rsNetAdaptInit)(void);
    void (*rsNetAdaptUninit)(void);
    int (*rsNetAllocJfcId)(const char *udevName, unsigned int jfcMode, unsigned int *jfcId);
    int (*rsNetFreeJfcId)(const char *udevName, unsigned int jfcMode, unsigned int jfcId);
    int (*rsNetAllocJettyId)(const char *udevName, unsigned int jettyMode, unsigned int *jettyId);
    int (*rsNetFreeJettyId)(const char *udevName, unsigned int jettyMode, unsigned int jettyId);
    unsigned long long (*rsNetGetCqeBaseAddr)(unsigned int dieId);
    unsigned int (*rsNetGetApiVersion)(void);
};

int RsNetApiInit(void);
void RsNetApiDeinit(void);

int RsNetAdaptInit(void);
void RsNetAdaptUninit(void);
int RsNetAllocJfcId(const char *udevName, unsigned int jfcMode, unsigned int *jfcId);
int RsNetFreeJfcId(const char *udevName, unsigned int jfcMode, unsigned int jfcId);
int RsNetAllocJettyId(const char *udevName, unsigned int jettyMode, unsigned int *jettyId);
int RsNetFreeJettyId(const char *udevName, unsigned int jettyMode, unsigned int jettyId);
int RsNetGetCqeBaseAddr(unsigned int dieId, unsigned long long *cqeBaseAddr);
unsigned int RsNetGetApiVersion(void);

#endif // DL_NET_FUNCTION_H
