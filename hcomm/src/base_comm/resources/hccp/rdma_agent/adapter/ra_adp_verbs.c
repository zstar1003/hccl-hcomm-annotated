/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include "ra_adp.h"
#include "rs.h"
#include "ra_adp_verbs.h"

static struct RaRsVerbsOps gRaRsVerbsOps = {
    .qpCreateWithCQWithAttrs = RsQpCreateWithCQWithAttrs,
    .typicalCqCreate = RsTypicalCqCreate,
    .typicalCqDestroy = RsTypicalCqDestroy,
    .getLiteCqAttr = RsGetLiteCqAttr,
    .qpDestroyWithoutCQ = RsQpDestroyWithoutCQ,
    .getLiteQpAttr = RsGetLiteQpAttr,
};

int RaRsQpCreateWithCQWithAttrs(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpQpCreateWithCQWithAttrsData *createData = NULL;
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpResp = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpCreateWithCQWithAttrsData), sizeof(struct MsgHead), rcvBufLen,
        opResult);

    createData = (union OpQpCreateWithCQWithAttrsData *)(inBuf + sizeof(struct MsgHead));

    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    qpNorm.extAttrs = createData->txData.extAttrs;

    *opResult = gRaRsVerbsOps.qpCreateWithCQWithAttrs(createData->txData.phyId, createData->txData.rdevIndex,
        createData->txData.sendCqn, createData->txData.recvCqn, &qpNorm, &qpResp);
    if (*opResult != 0) {
        hccp_err("qp create with cq with attrs failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpQpCreateWithCQWithAttrsData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.qpn = qpResp.qpn;
    createData->rxData.psn = qpResp.psn;
    createData->rxData.gidIdx = qpResp.gidIdx;

    return 0;
}

int RaRsTypicalCqCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTypicalCqCreateData *createData = (union OpTypicalCqCreateData *)(inBuf + sizeof(struct MsgHead));
    unsigned int cqn = 0;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalCqCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsVerbsOps.typicalCqCreate(createData->txData.phyId, createData->txData.rdevIndex,
        createData->txData.cqDepth, &cqn);
    if (*opResult != 0) {
        hccp_err("typical cq create failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpTypicalCqCreateData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.cqn = cqn;

    return 0;
}

int RaRsTypicalCqDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpTypicalCqDestroyData *destroyData = (union OpTypicalCqDestroyData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalCqDestroyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsVerbsOps.typicalCqDestroy(destroyData->txData.phyId, destroyData->txData.rdevIndex,
        destroyData->txData.cqn);
    if (*opResult != 0) {
        hccp_err("typical cq destroy failed ret[%d].", *opResult);
        return 0;
    }

    return 0;
}

int RaRsGetLiteCqAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetLiteCqAttrData *getLiteCqAttrData = (union OpGetLiteCqAttrData *)(inBuf + sizeof(struct MsgHead));
    unsigned int cqn = getLiteCqAttrData->txData.cqn;
    struct rdma_lite_device_cq_attr deviceCqAttr = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetLiteCqAttrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsVerbsOps.getLiteCqAttr(getLiteCqAttrData->txData.phyId, getLiteCqAttrData->txData.rdevIndex, cqn,
        &deviceCqAttr);
    if (*opResult != 0) {
        hccp_err("get lite cq attr failed ret[%d], cqn[%u].", *opResult, cqn);
        return 0;
    }

    getLiteCqAttrData = (union OpGetLiteCqAttrData *)(outBuf + sizeof(struct MsgHead));
    getLiteCqAttrData->rxData.deviceCqAttr = deviceCqAttr;

    return 0;
}

int RaRsQpDestroyWithoutCQ(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpQpDestroyData *qpDestroyData = (union OpQpDestroyData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpDestroyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsVerbsOps.qpDestroyWithoutCQ(qpDestroyData->txData.phyId, qpDestroyData->txData.rdevIndex,
        qpDestroyData->txData.qpn);
    if (*opResult != 0) {
        hccp_err("qp destroy without cq failed ret[%d].", *opResult);
    }

    return 0;
}

int RaRsGetLiteQpAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLiteQpAttrData *liteQpAttrData = (union OpLiteQpAttrData *)(inBuf + sizeof(struct MsgHead));
    union OpLiteQpAttrData *liteQpAttrOut = (union OpLiteQpAttrData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLiteQpAttrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsVerbsOps.getLiteQpAttr(liteQpAttrData->txData.phyId, liteQpAttrData->txData.rdevIndex,
        liteQpAttrData->txData.qpn, (void *)&liteQpAttrOut->rxData.resp);
    if (*opResult != 0) {
        hccp_err("get_lite_qp_attr failed ret[%d].", *opResult);
    }

    return 0;
}
