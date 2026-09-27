/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <errno.h>
#include "securec.h"
#include "config_log.h"
#include "ra_comm.h"
#include "rs_nda.h"
#include "ra_peer.h"
#include "ra_peer_nda.h"

int RaPeerNdaGetDirectFlag(struct RaRdmaHandle *rdmaHandle, int *directFlag)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret = 0;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsNdaGetDirectFlag(phyId, rdmaHandle->rdevIndex, directFlag);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[get][directFlag]RsNdaGetDirectFlag failed ret:%d phyId:%u", ret, phyId);
    }
    return ret;
}

int RaPeerNdaCqCreate(struct RaRdmaHandle *rdmaHandle, struct NdaCqInitAttr *attr, struct NdaCqInfo *info,
    void **cqHandle)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaCqHandleExt *cqPeer = NULL;
    void *ibvCqExt = NULL;
    int ret = 0;

    cqPeer = (struct RaCqHandleExt *)calloc(1, sizeof(struct RaCqHandleExt));
    CHK_PRT_RETURN(cqPeer == NULL, hccp_err("[create][RaNdaCq]cqPeer calloc failed phyId:%u", phyId), -ENOMEM);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsNdaCqCreate(phyId, rdmaHandle->rdevIndex, attr, info, &ibvCqExt);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[create][RaNdaCq]RsNdaCqCreate failed ret:%d phyId:%u", ret, phyId);
        goto free_cq_handle;
    }
    cqPeer->addr = (unsigned long long)(uintptr_t)ibvCqExt;

    *cqHandle = cqPeer;
    return ret;

free_cq_handle:
    free(cqPeer);
    cqPeer = NULL;
    return ret;
}

int RaPeerNdaCqDestroy(struct RaRdmaHandle *rdmaHandle, void *cqHandle)
{
    struct RaCqHandleExt *cqPeer = (struct RaCqHandleExt *)cqHandle;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    void *ibvCqExt;
    int ret = 0;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ibvCqExt = (void *)(uintptr_t)cqPeer->addr;
    ret = RsNdaCqDestroy(phyId, rdmaHandle->rdevIndex, ibvCqExt);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[destroy][RaNdaCq]RsNdaCqDestroy failed ret:%d phyId:%u", ret, phyId);
    }
    free(cqPeer);
    cqPeer = NULL;
    return ret;
}

int RaPeerNdaQpCreate(struct RaRdmaHandle *rdmaHandle, struct NdaQpInitAttr *attr, struct NdaQpInfo *info,
    void **qpHandle)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpPeer = NULL;
    struct RsQpResp qpResp = {0};
    int ret = 0;

    qpPeer = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpPeer == NULL, hccp_err("[create][RaNdaQp]qpPeer calloc failed phyId:%u", phyId), -ENOMEM);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsNdaQpCreate(phyId, rdmaHandle->rdevIndex, attr, info, &qpResp);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[create][RaNdaQp]RsNdaQpCreate failed ret:%d phyId:%u", ret, phyId);
        goto free_qp_handle;
    }
    qpPeer->phyId = phyId;
    qpPeer->qpn = qpResp.qpn;
    qpPeer->psn = qpResp.psn;
    qpPeer->gidIdx = qpResp.gidIdx;
    qpPeer->directFlag = qpResp.directFlag;
    qpPeer->rdevIndex = rdmaHandle->rdevIndex;
    qpPeer->rdmaHandle = rdmaHandle;
    qpPeer->rdmaOps = rdmaHandle->rdmaOps;

    *qpHandle = qpPeer;
    return ret;

free_qp_handle:
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerNdaQpDestroy(struct RaQpHandle *qpPeer)
{
    int ret = 0;

    RaPeerMutexLock(qpPeer->phyId);
    RsSetCtx(qpPeer->phyId);
    ret = RsNdaQpDestroy(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn);
    RaPeerMutexUnlock(qpPeer->phyId);
    if (ret != 0) {
        hccp_err("[destroy][RaNdaQp]RsNdaQpDestroy failed ret:%d phyId:%u", ret, qpPeer->phyId);
    }
    free(qpPeer);
    qpPeer = NULL;
    return ret;
}

int RaPeerGetQpHyperFeature(struct RaQpHandle *qpPeer, struct HyperFeature *hyperFeature)
{
    int ret = 0;

    RaPeerMutexLock(qpPeer->phyId);
    RsSetCtx(qpPeer->phyId);
    ret = RsGetQpHyperFeature(qpPeer->phyId, qpPeer->rdevIndex, qpPeer->qpn, hyperFeature);
    RaPeerMutexUnlock(qpPeer->phyId);
    CHK_PRT_RETURN(ret, hccp_warn_rma("RsGetQpHyperFeature unsuccessful ret(%d) phyId(%u)", ret, qpPeer->phyId), ret);
    return ret;
}
