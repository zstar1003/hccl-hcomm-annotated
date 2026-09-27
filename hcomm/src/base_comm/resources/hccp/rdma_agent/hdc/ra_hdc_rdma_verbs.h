/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_RDMA_VERBS_H
#define RA_HDC_RDMA_VERBS_H
#include "ra_hdc_rdma.h"

union OpQpCreateWithCQWithAttrsData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int sendCqn;
        unsigned int recvCqn;
        struct QpExtAttrs extAttrs;
    } txData;

    struct {
        unsigned int qpn;
        unsigned int psn;
        unsigned int gidIdx;
        unsigned int rsvd;
    } rxData;
};

union OpTypicalCqCreateData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int cqDepth;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int cqn;
        unsigned int rsvd[RA_RSVD_NUM_6];
    } rxData;
};

union OpTypicalCqDestroyData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int cqn;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

int RaHdcTypicalCqCreate(struct RaRdmaHandle *rdmaHandle, unsigned int cqDepth, unsigned int *cqn, void **cqHandle);
int RaHdcTypicalCqDestroy(struct RaRdmaHandle *rdmaHandle, unsigned int cqn, void *cqHandle);
int RaHdcQpCreateWithCQWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, unsigned int sendCqn,
    unsigned int recvCqn, void **qpHandle);
int RaHdcPollTypicalCq(struct RaTypicalCqHandle *cqHdc, unsigned int numEntries, void *wc);
int RaHdcQpDestroyWithoutCQ(struct RaQpHandle *qpHdc);
int RaHdcSendWrVerbs(struct RaQpHandle *qpHdc, struct SendWrVerbs *wr, struct SendWrRsp *opRsp);
int RaHdcRecvWrVerbs(struct RaQpHandle *qpHdc, struct RecvWrVerbs *wr);

#endif // RA_HDC_RDMA_VERBS_H
