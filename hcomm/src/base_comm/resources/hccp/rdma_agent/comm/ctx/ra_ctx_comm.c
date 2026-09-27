/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include "securec.h"
#include "user_log.h"
#include "ra_ctx_comm.h"

int RaCtxPrepareLmemRegister(struct MrRegInfoT *lmemInfo, struct MemRegAttrT *memAttr)
{
    struct RaTokenIdHandle *tokenIdHandle = NULL;
    bool isTokenIdValid = false;

    memAttr->mem = lmemInfo->in.mem;
    isTokenIdValid = lmemInfo->in.ub.flags.bs.tokenIdValid == 1;
    CHK_PRT_RETURN(isTokenIdValid && lmemInfo->in.ub.tokenIdHandle == NULL,
        hccp_err("[init][ra_ctx_lmem]lmem_info specify token id, but tokenIdHandle is NULL"), -EINVAL);
    memAttr->ub.flags = lmemInfo->in.ub.flags;
    memAttr->ub.tokenValue = lmemInfo->in.ub.tokenValue;
    tokenIdHandle = (struct RaTokenIdHandle *)(lmemInfo->in.ub.tokenIdHandle);
    memAttr->ub.tokenIdAddr = isTokenIdValid ? tokenIdHandle->addr : 0;

    return 0;
}

void RaCtxGetLmemInfo(struct MemRegInfoT *memInfo, struct MrRegInfoT *lmemInfo, struct RaLmemHandle *lmemHandle)
{
    lmemInfo->out.key = memInfo->key;
    lmemInfo->out.ub.tokenId = memInfo->ub.tokenId;
    lmemInfo->out.ub.targetSegHandle = memInfo->ub.targetSegHandle;
    lmemHandle->addr = lmemInfo->out.ub.targetSegHandle;
}

void RaCtxPrepareRmemImport(struct MrImportInfoT *rmemInfo, struct MemImportAttrT *memAttr)
{
    memAttr->key = rmemInfo->in.key;
    memAttr->ub.flags = rmemInfo->in.ub.flags;
    memAttr->ub.mappingAddr = rmemInfo->in.ub.mappingAddr;
    memAttr->ub.tokenValue = rmemInfo->in.ub.tokenValue;
}

void RaCtxPrepareCqCreate(struct CqInfoT *info, struct CtxCqAttr *cqAttr)
{
    struct RaChanHandle *chanHandle = NULL;

    cqAttr->depth = info->in.depth;
    cqAttr->ub.userCtx = info->in.ub.userCtx;
    cqAttr->ub.mode = info->in.ub.mode;
    cqAttr->ub.ceqn = info->in.ub.ceqn;
    cqAttr->ub.flag = info->in.ub.flag;
    if (info->in.chanHandle != NULL) {
        chanHandle = (struct RaChanHandle *)info->in.chanHandle;
        cqAttr->chanAddr = chanHandle->addr;
    }
}

void RaCtxGetCqCreateInfo(struct CtxCqInfo *cqInfo, struct CqInfoT *info)
{
    info->out.va = cqInfo->addr;
    info->out.id = cqInfo->ub.id;
    info->out.cqeSize = cqInfo->ub.cqeSize;
    info->out.bufAddr = cqInfo->ub.bufAddr;
    info->out.swdbAddr = cqInfo->ub.swdbAddr;
}

int RaCtxPrepareQpCreate(struct QpCreateAttr *qpAttr, struct CtxQpAttr *ctxQpAttr)
{
    struct RaTokenIdHandle *tokenIdHandle = NULL;

    CHK_PRT_RETURN(qpAttr->scqHandle == NULL, hccp_err("[init][ra_ctx_qp]scq_handle is NULL"), -EINVAL);
    CHK_PRT_RETURN(qpAttr->rcqHandle == NULL, hccp_err("[init][ra_ctx_qp]rcq_handle is NULL"), -EINVAL);

    ctxQpAttr->scqIndex = ((struct RaCqHandle *)qpAttr->scqHandle)->addr;
    ctxQpAttr->rcqIndex = ((struct RaCqHandle *)qpAttr->rcqHandle)->addr;
    ctxQpAttr->srqIndex = 0;
    ctxQpAttr->sqDepth = qpAttr->sqDepth;
    ctxQpAttr->rqDepth = qpAttr->rqDepth;
    ctxQpAttr->transportMode = qpAttr->transportMode;

    ctxQpAttr->ub.mode = qpAttr->ub.mode;
    ctxQpAttr->ub.jettyId = qpAttr->ub.jettyId;
    ctxQpAttr->ub.flag = qpAttr->ub.flag;
    ctxQpAttr->ub.jfsFlag = qpAttr->ub.jfsFlag;
    ctxQpAttr->ub.tokenValue = qpAttr->ub.tokenValue;
    ctxQpAttr->ub.priority = qpAttr->ub.priority;
    ctxQpAttr->ub.rnrRetry = qpAttr->ub.rnrRetry;
    ctxQpAttr->ub.errTimeout = qpAttr->ub.errTimeout;

    if (qpAttr->ub.tokenIdHandle != NULL) {
        tokenIdHandle = (struct RaTokenIdHandle *)(qpAttr->ub.tokenIdHandle);
        ctxQpAttr->ub.tokenIdAddr = tokenIdHandle->addr;
    }

    return 0;
}

void RaCtxGetQpCreateInfo(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
    struct RaCtxQpHandle *qpHandle)
{
    qpHandle->devIndex = ctxHandle->devIndex;
    qpHandle->phyId = ctxHandle->attr.phyId;
    qpHandle->ctxHandle = ctxHandle;
    qpHandle->protocol = ctxHandle->protocol;
    qpHandle->id = qpInfo->ub.id;
    (void)memcpy_s(&qpHandle->qpAttr, sizeof(struct QpCreateAttr), qpAttr, sizeof(struct QpCreateAttr));
    (void)memcpy_s(&qpHandle->qpInfo, sizeof(struct QpCreateInfo), qpInfo, sizeof(struct QpCreateInfo));
}

void RaCtxPrepareQpImport(struct QpImportInfoT *qpInfo, struct RaRsJettyImportAttr *importAttr)
{
    importAttr->mode = qpInfo->in.ub.mode;
    importAttr->tokenValue = qpInfo->in.ub.tokenValue;
    importAttr->policy = qpInfo->in.ub.policy;
    importAttr->type = qpInfo->in.ub.type;
    importAttr->flag = qpInfo->in.ub.flag;
    importAttr->expImportCfg = qpInfo->in.ub.expImportCfg;
    importAttr->tpType = qpInfo->in.ub.tpType;
}

void RaCtxGetQpImportInfo(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo,
    struct RaRsJettyImportInfo *importInfo, struct RaCtxRemQpHandle *qpHandle)
{
    qpInfo->out.ub.tjettyHandle = importInfo->tjettyHandle;
    qpInfo->out.ub.tpn = importInfo->tpn;
    qpHandle->devIndex = ctxHandle->devIndex;
    qpHandle->phyId = ctxHandle->attr.phyId;
    qpHandle->protocol = ctxHandle->protocol;
    qpHandle->qpKey = qpInfo->in.key;
}
