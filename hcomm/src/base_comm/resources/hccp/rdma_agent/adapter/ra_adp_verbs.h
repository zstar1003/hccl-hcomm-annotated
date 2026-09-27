/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_VERBS_H
#define RA_ADP_VERBS_H
#include "ra_hdc_lite.h"

struct RaRsVerbsOps {
    int (*qpCreateWithCQWithAttrs)(unsigned int phyId, unsigned int rdevIndex, unsigned int sendCqn,
        unsigned int recvCqn, struct RsQpNormWithAttrs *qpNorm, struct RsQpRespWithAttrs *qpResp);
    int (*typicalCqCreate)(unsigned int phyId, unsigned int rdevIndex, unsigned int cqDepth, unsigned int *cqn);
    int (*typicalCqDestroy)(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn);
    int (*getLiteCqAttr)(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn,
        struct rdma_lite_device_cq_attr *deviceCqAttr);
    int (*qpDestroyWithoutCQ)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn);
    int (*getLiteQpAttr)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteQpAttrResp *resp);
};

int RaRsQpCreateWithCQWithAttrs(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsTypicalCqCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsTypicalCqDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetLiteCqAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsQpDestroyWithoutCQ(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetLiteQpAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);

#endif // RA_ADP_VERBS_H
