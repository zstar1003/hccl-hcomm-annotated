/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ra_rs_ctx.h"
#include "ra_peer.h"
#include "rs_ctx.h"
#include "ra_ctx_comm.h"
#include "ra_peer_ctx.h"

int RaPeerGetDevEidInfoNum(struct RaInfo info, unsigned int *num)
{
    unsigned int phyId = info.phyId;
    int ret = 0;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetDevEidInfoNum(phyId, num);
    if (ret != 0) {
        hccp_err("[get][eid]rs_get_dev_eid_info_num failed ret[%d], phyId[%u]", ret, phyId);
    }
    RaPeerMutexUnlock(phyId);
    return ret;
}

int RaPeerGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int *num)
{
    unsigned int startIndex = 0;
    unsigned int count = *num;
    int ret = 0;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetDevEidInfoList(phyId, infoList, startIndex, count);
    if (ret != 0) {
        *num = 0;
        hccp_err("[get][eid]rs_get_dev_eid_info_list failed ret[%d], phyId[%u]", ret, phyId);
    } else {
        *num = count;
    }
    RaPeerMutexUnlock(phyId);
    return ret;
}

int RaPeerCtxInit(struct RaCtxHandle *ctxHandle, struct CtxInitAttr *attr, unsigned int *devIndex,
    struct DevBaseAttr *devBaseAttr)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    int ret = 0;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxInit(attr, devIndex, devBaseAttr);
    if (ret != 0) {
        hccp_err("[init][ra_peer_ctx]ctx init failed, ret[%d] phy_id[%u]", ret, phyId);
    }

    RaPeerMutexUnlock(phyId);
    return ret;
}

int RaPeerCtxGetAsyncEvents(struct RaCtxHandle *ctxHandle, struct AsyncEvent events[], unsigned int *num)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxGetAsyncEvents(&devInfo, events, num);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[get][async_events]RsCtxGetAsyncEvents failed ret:%d phyId:%u devIndex:0x%x", ret, phyId,
            ctxHandle->devIndex);
    }

    return ret;
}

int RaPeerCtxDeinit(struct RaCtxHandle *ctxHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxDeinit(&devInfo);
    if (ret != 0) {
        hccp_err("[deinit][ra_peer_ctx]ctx deinit failed[%d] phy_id[%u]", ret, phyId);
    }

    RaPeerMutexUnlock(phyId);
    return ret;
}

int RaPeerGetEidByIp(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetEidByIp(&devInfo, ip, eid, num);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[get][eid_by_ip]rs_get_eid_by_ip failed ret[%d] phy_id[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerGetIpByEid(struct RaCtxHandle *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetIpByEid(&devInfo, eid, ip, num);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[get][IpByEid]RsGetIpByEid failed ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxTokenIdAlloc(struct RaCtxHandle *ctxHandle, struct HccpTokenId *info,
    struct RaTokenIdHandle *tokenIdHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxTokenIdAlloc(&devInfo, &tokenIdHandle->addr, &info->tokenId);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[init][ra_token_id]rs_ctx_token_id_alloc failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxTokenIdFree(struct RaCtxHandle *ctxHandle, struct RaTokenIdHandle *tokenIdHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxTokenIdFree(&devInfo, tokenIdHandle->addr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[deinit][ra_token_id]rs_ctx_token_id_free failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxLmemRegister(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo, struct RaLmemHandle *lmemHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    struct MemRegAttrT memAttr = {0};
    struct MemRegInfoT memInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);
    ret = RaCtxPrepareLmemRegister(lmemInfo, &memAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer_lmem]ra_ctx_prepare_lmem_register failed, ret[%d]", ret), ret);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxLmemReg(&devInfo, &memAttr, &memInfo);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer_lmem]rs_ctx_lmem_reg failed, ret[%d] phyId[%u]", ret, phyId),
        ret);

    RaCtxGetLmemInfo(&memInfo, lmemInfo, lmemHandle);

    return ret;
}

int RaPeerCtxLmemUnregister(struct RaCtxHandle *ctxHandle, struct RaLmemHandle *lmemHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxLmemUnreg(&devInfo, lmemHandle->addr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[init][ra_peer_lmem]rs_ctx_lmem_unreg failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxRmemImport(struct RaCtxHandle *ctxHandle, struct MrImportInfoT *rmemInfo)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct MemImportAttrT memAttr = {0};
    struct MemImportInfoT memInfo = {0};
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);
    RaCtxPrepareRmemImport(rmemInfo, &memAttr);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxRmemImport(&devInfo, &memAttr, &memInfo);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer_rmem]rs_ctx_rmem_import failed, ret[%d] phyId[%u]", ret, phyId),
        ret);

    rmemInfo->out.ub.targetSegHandle = memInfo.ub.targetSegHandle;

    return ret;
}

int RaPeerCtxRmemUnimport(struct RaCtxHandle *ctxHandle, struct RaRmemHandle *rmemHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxRmemUnimport(&devInfo, rmemHandle->addr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[deinit][ra_peer_rmem]rs_ctx_rmem_unimport failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxChanCreate(struct RaCtxHandle *ctxHandle, struct ChanInfoT *chanInfo, struct RaChanHandle *chanHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxChanCreate(&devInfo, chanInfo->in.dataPlaneFlag, &chanHandle->addr, &chanInfo->out.fd);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[init][ctx_chan]rs_ctx_chan_create failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxChanDestroy(struct RaCtxHandle *ctxHandle, struct RaChanHandle *chanHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxChanDestroy(&devInfo, chanHandle->addr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[deinit][ctx_chan]rs_ctx_chan_destroy failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxCqCreate(struct RaCtxHandle *ctxHandle, struct CqInfoT *info, struct RaCqHandle *cqHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    int ret = 0;

    CHK_PRT_RETURN(info->in.ub.mode != JFC_MODE_NORMAL,
        hccp_err("[init][ctx_cq]jfc_mode[%d] not support, only JFC_MODE_NORMAL is supported, phyId[%u]",
            info->in.ub.mode, phyId),
        -EINVAL);

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);
    RaCtxPrepareCqCreate(info, &cqAttr);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxCqCreate(&devInfo, &cqAttr, &cqInfo);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ctx_cq]rs_ctx_cq_create failed, ret[%d] phyId[%u]", ret, phyId), ret);

    cqHandle->addr = cqInfo.addr;
    RaCtxGetCqCreateInfo(&cqInfo, info);
    return ret;
}

int RaPeerCtxCqDestroy(struct RaCtxHandle *ctxHandle, struct RaCqHandle *cqHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxCqDestroy(&devInfo, cqHandle->addr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[deinit][ctx_cq]rs_ctx_cq_destroy failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxQpCreate(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
    struct RaCtxQpHandle *qpHandle)
{
    unsigned int devIndex = ctxHandle->devIndex;
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    struct CtxQpAttr ctxQpAttr = {0};
    int ret = 0;

    CHK_PRT_RETURN(qpAttr->ub.mode != JETTY_MODE_URMA_NORMAL,
        hccp_err("[init][ctx_cq]jetty_mode[%d] not support, only JETTY_MODE_URMA_NORMAL is supported,"
                 " phyId[%u]",
            qpAttr->ub.mode, phyId),
        -EINVAL);

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);
    ret = RaCtxPrepareQpCreate(qpAttr, &ctxQpAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_peer_qp]ra_ctx_prepare_qp_create failed ret[%d] phyId[%u], devIndex[%u]", ret, phyId,
            devIndex),
        ret);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxQpCreate(&devInfo, &ctxQpAttr, qpInfo);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer_qp]rs_ctx_qp_create failed, ret[%d] phyId[%u]", ret, phyId), ret);

    RaCtxGetQpCreateInfo(ctxHandle, qpAttr, qpInfo, qpHandle);

    return ret;
}

int RaPeerCtxQpDestroy(struct RaCtxQpHandle *qpHandle)
{
    unsigned int phyId = qpHandle->phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, qpHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxQpDestroy(&devInfo, qpHandle->id);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[deinit][ra_peer_qp]rs_ctx_qp_destroy failed, ret[%d] phyId[%u]", ret, phyId);
    }

    return ret;
}

int RaPeerCtxGetTpInfoList(struct RaCtxHandle *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetTpInfoList(&devInfo, cfg, infoList, num);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[get][RaTpInfo]RsGetTpInfoList failed, ret[%d] phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
    }

    return ret;
}

int RaPeerCtxGetTpAttr(struct RaCtxHandle *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsGetTpAttr(&devInfo, attrBitmap, tpHandle, attr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[get][RaTpAttr]RsGetTpAttr failed, ret[%d] phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
    }

    return ret;
}

int RaPeerCtxSetTpAttr(struct RaCtxHandle *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsSetTpAttr(&devInfo, attrBitmap, tpHandle, attr);
    RaPeerMutexUnlock(phyId);
    if (ret != 0) {
        hccp_err("[set][RaTpAttr]RsSetTpAttr failed, ret[%d] phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
    }

    return ret;
}

STATIC void RaPeerPrepareQpImport(struct QpImportInfoT *qpInfo, struct RsJettyImportAttr *importAttr)
{
    struct RaRsJettyImportAttr *raRsImportAttr = NULL;

    raRsImportAttr = &(importAttr->attr);
    importAttr->key = qpInfo->in.key;
    RaCtxPrepareQpImport(qpInfo, raRsImportAttr);
}

STATIC void RaPeerGetQpImportInfo(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo,
    struct RsJettyImportInfo *importInfo, struct RaCtxRemQpHandle *qpHandle)
{
    struct RaRsJettyImportInfo *raRsImportInfo = NULL;

    raRsImportInfo = &(importInfo->info);
    RaCtxGetQpImportInfo(ctxHandle, qpInfo, raRsImportInfo, qpHandle);
    qpHandle->id = importInfo->remJettyId;
}

int RaPeerCtxQpImport(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo, struct RaCtxRemQpHandle *remQpHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importInfo = {0};
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, ctxHandle->devIndex);
    RaPeerPrepareQpImport(qpInfo, &importAttr);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxQpImport(&devInfo, &importAttr, &importInfo);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer_qp]rs_ctx_qp_import failed, ret[%d] phyId[%u]", ret, phyId), ret);

    RaPeerGetQpImportInfo(ctxHandle, qpInfo, &importInfo, remQpHandle);
    return ret;
}

int RaPeerCtxQpUnimport(struct RaCtxRemQpHandle *remQpHandle)
{
    unsigned int phyId = remQpHandle->phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, remQpHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxQpUnimport(&devInfo, remQpHandle->qpKey.value, REM_JETTY_ID_SIZE);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra_peer_qp]rs_ctx_qp_unimport failed, ret[%d] phyId[%u]", ret, phyId),
        ret);

    return ret;
}

int RaPeerCtxQpBind(struct RaCtxQpHandle *qpHandle, struct RaCtxRemQpHandle *remQpHandle)
{
    struct RsCtxQpInfo remoteQpInfo = {0};
    struct RsCtxQpInfo localQpInfo = {0};
    unsigned int phyId = qpHandle->phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, remQpHandle->devIndex);
    localQpInfo.id = qpHandle->id;
    remoteQpInfo.id = remQpHandle->id;

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxQpBind(&devInfo, &localQpInfo, &remoteQpInfo);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_peer_qp]rs_ctx_qp_bind failed, ret[%d] phyId[%u]", ret, phyId), ret);

    return ret;
}

int RaPeerCtxQpUnbind(struct RaCtxQpHandle *qpHandle)
{
    unsigned int phyId = qpHandle->phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, qpHandle->devIndex);

    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxQpUnbind(&devInfo, qpHandle->id);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra_peer_qp]rs_ctx_qp_unbind failed, ret[%d] phyId[%u]", ret, phyId),
        ret);

    return ret;
}

int RaPeerCtxGetJettyContext(struct RaCtxQpHandle *qpHandle, uint8_t context[], unsigned int *len)
{
    unsigned int phyId = qpHandle->phyId;
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    RaRsSetDevInfo(&devInfo, phyId, qpHandle->devIndex);
    RaPeerMutexLock(phyId);
    RsSetCtx(phyId);
    ret = RsCtxGetUbContext(&devInfo, qpHandle->id, CONTEXT_TYPE_JETTY, context, len);
    RaPeerMutexUnlock(phyId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][jettyContext]RsCtxGetJettyContext failed, ret[%d] phyId[%u], devIndex[%u]", ret, phyId,
            qpHandle->devIndex),
        ret);

    return ret;
}