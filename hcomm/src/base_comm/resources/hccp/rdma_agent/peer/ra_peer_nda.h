/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_PEER_NDA_H
#define RA_PEER_NDA_H

#include "hccp_nda.h"
#include "ra_comm.h"

int RaPeerNdaGetDirectFlag(struct RaRdmaHandle *rdmaHandle, int *directFlag);
int RaPeerNdaCqCreate(struct RaRdmaHandle *rdmaHandle, struct NdaCqInitAttr *attr, struct NdaCqInfo *info,
    void **cqHandle);
int RaPeerNdaCqDestroy(struct RaRdmaHandle *rdmaHandle, void *cqHandle);
int RaPeerNdaQpCreate(struct RaRdmaHandle *rdmaHandle, struct NdaQpInitAttr *attr, struct NdaQpInfo *info,
    void **qpHandle);
int RaPeerNdaQpDestroy(struct RaQpHandle *qpPeer);
int RaPeerGetQpHyperFeature(struct RaQpHandle *qpPeer, struct HyperFeature *hyperFeature);

#endif // RA_PEER_NDA_H
