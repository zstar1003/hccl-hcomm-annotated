/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "hccp_common.h"
#include "hccp_ctx.h"
#include "ra.h"
#include "ra_rs_err.h"
#include "ra_hdc_ctx.h"
#include "ra_peer_ctx.h"
#include "ra_ctx.h"

struct RaCtxOps gRaHdcCtxOps = {
    .raCtxInit = RaHdcCtxInit,
    .raCtxGetAsyncEvents = RaHdcCtxGetAsyncEvents,
    .raCtxDeinit = RaHdcCtxDeinit,
    .raCtxGetEidByIp = RaHdcGetEidByIp,
    .raCtxGetIpByEid = RaHdcGetIpByEid,
    .raCtxTokenIdAlloc = RaHdcCtxTokenIdAlloc,
    .raCtxTokenIdFree = RaHdcCtxTokenIdFree,
    .raCtxLmemRegister = RaHdcCtxLmemRegister,
    .raCtxLmemUnregister = RaHdcCtxLmemUnregister,
    .raCtxRmemImport = RaHdcCtxRmemImport,
    .raCtxRmemUnimport = RaHdcCtxRmemUnimport,
    .raCtxChanCreate = RaHdcCtxChanCreate,
    .raCtxChanDestroy = RaHdcCtxChanDestroy,
    .raCtxCqCreate = RaHdcCtxCqCreate,
    .raCtxCqDestroy = RaHdcCtxCqDestroy,
    .raCtxQpCreate = RaHdcCtxQpCreate,
    .raCtxQpDestroy = RaHdcCtxQpDestroy,
    .raCtxQpImport = RaHdcCtxQpImport,
    .raCtxQpUnimport = RaHdcCtxQpUnimport,
    .raCtxQpBind = RaHdcCtxQpBind,
    .raCtxQpUnbind = RaHdcCtxQpUnbind,
    .raCtxBatchSendWr = RaHdcCtxBatchSendWr,
    .raCtxUpdateCi = RaHdcCtxUpdateCi,
    .raCtxQueryQpBatch = RaHdcCtxQpQueryBatch,
    .raCtxGetAuxInfo = RaHdcCtxGetAuxInfo,
    .raCtxNotifyEvent = RaHdcCtxNotifyEvent,
    .raCtxGetJettyContext = RaHdcCtxGetJettyContext,
};

struct RaCtxOps gRaPeerCtxOps = {
    .raCtxInit = RaPeerCtxInit,
    .raCtxGetAsyncEvents = RaPeerCtxGetAsyncEvents,
    .raCtxDeinit = RaPeerCtxDeinit,
    .raCtxGetEidByIp = RaPeerGetEidByIp,
    .raCtxGetIpByEid = RaPeerGetIpByEid,
    .raCtxTokenIdAlloc = RaPeerCtxTokenIdAlloc,
    .raCtxTokenIdFree = RaPeerCtxTokenIdFree,
    .raCtxLmemRegister = RaPeerCtxLmemRegister,
    .raCtxLmemUnregister = RaPeerCtxLmemUnregister,
    .raCtxRmemImport = RaPeerCtxRmemImport,
    .raCtxRmemUnimport = RaPeerCtxRmemUnimport,
    .raCtxChanCreate = RaPeerCtxChanCreate,
    .raCtxChanDestroy = RaPeerCtxChanDestroy,
    .raCtxCqCreate = RaPeerCtxCqCreate,
    .raCtxCqDestroy = RaPeerCtxCqDestroy,
    .raCtxQpCreate = RaPeerCtxQpCreate,
    .raCtxQpDestroy = RaPeerCtxQpDestroy,
    .raCtxQpImport = RaPeerCtxQpImport,
    .raCtxQpUnimport = RaPeerCtxQpUnimport,
    .raCtxQpBind = RaPeerCtxQpBind,
    .raCtxQpUnbind = RaPeerCtxQpUnbind,
    .raCtxBatchSendWr = NULL,
    .raCtxUpdateCi = NULL,
    .raCtxQueryQpBatch = NULL,
    .raCtxGetAuxInfo = NULL,
    .raCtxNotifyEvent = NULL,
    .raCtxGetJettyContext = RaPeerCtxGetJettyContext,
};

HCCP_ATTRI_VISI_DEF int RaGetDevEidInfoNum(struct RaInfo info, unsigned int *num)
{
    int ret;

    CHK_PRT_RETURN(num == NULL, hccp_err("[get][eid]num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(info.phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][eid]phy_id(%u) must be smaller than %u", info.phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phy_id[%u], nic_position:[%d]", info.phyId, info.mode);
    if (info.mode == NETWORK_OFFLINE) {
        ret = RaHdcGetDevEidInfoNum(info, num);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][eid]ra_hdc_get_dev_eid_info_num failed, ret(%d) phyId(%u)", ret, info.phyId),
            ConverReturnCode(RDMA_OP, ret));
    } else if (info.mode == NETWORK_PEER_ONLINE) {
        ret = RaPeerGetDevEidInfoNum(info, num);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][eid]ra_peer_get_dev_eid_info_num failed, ret(%d) phyId(%u)", ret, info.phyId),
            ConverReturnCode(RDMA_OP, ret));
    } else {
        hccp_err("[get][eid]mode(%d) do not support, phyId(%u)", info.mode, info.phyId);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo infoList[], unsigned int *num)
{
    int ret;

    CHK_PRT_RETURN(infoList == NULL || num == NULL, hccp_err("[get][eid]info_list or num is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(info.phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][eid]phy_id(%u) must be smaller than %u", info.phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phy_id[%u], nic_position:[%d]", info.phyId, info.mode);
    if (info.mode == NETWORK_OFFLINE) {
        ret = RaHdcGetDevEidInfoList(info.phyId, infoList, num);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][eid]ra_hdc_get_dev_eid_info_list failed, ret(%d) phyId(%u)", ret, info.phyId),
            ConverReturnCode(RDMA_OP, ret));
    } else if (info.mode == NETWORK_PEER_ONLINE) {
        ret = RaPeerGetDevEidInfoList(info.phyId, infoList, num);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][eid]ra_peer_get_dev_eid_info_list failed, ret(%d) phyId(%u)", ret, info.phyId),
            ConverReturnCode(RDMA_OP, ret));
    } else {
        hccp_err("[get][eid]mode(%d) do not support, phyId(%u)", info.mode, info.phyId);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    return 0;
}

STATIC int RaGetInitCtxHandle(struct CtxInitCfg *cfg, struct CtxInitAttr *attr, struct RaCtxHandle *ctxHandle)
{
    ctxHandle->protocol = PROTOCOL_UDMA;

    if (cfg->mode == NETWORK_OFFLINE) {
        ctxHandle->ctxOps = &gRaHdcCtxOps;
    } else if (cfg->mode == NETWORK_PEER_ONLINE) {
        ctxHandle->ctxOps = &gRaPeerCtxOps;
    } else {
        hccp_err("[init][ra_ctx]mode(%d) do not support, phyId(%u)", cfg->mode, attr->phyId);
        return -EINVAL;
    }

    CHK_PRT_RETURN(ctxHandle->ctxOps->raCtxInit == NULL,
        hccp_err("[init][ra_ctx]ra_ctx_init is NULL, phyId(%u)", attr->phyId), -EINVAL);

    (void)memcpy_s(&(ctxHandle->attr), sizeof(struct CtxInitAttr), attr, sizeof(struct CtxInitAttr));
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCtxInit(struct CtxInitCfg *cfg, struct CtxInitAttr *attr, void **ctxHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(cfg == NULL || attr == NULL || ctxHandle == NULL,
        hccp_err("[init][ra_ctx]cfg or attr or ctx_handle is NULL"), ConverReturnCode(HCCP_INIT, -EINVAL));
    CHK_PRT_RETURN(attr->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[init][ra_ctx]phy_id(%u) must be smaller than %u", attr->phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    ctxHandleTmp = calloc(1, sizeof(struct RaCtxHandle));
    CHK_PRT_RETURN(ctxHandleTmp == NULL,
        hccp_err("[init][ra_ctx]calloc ctx_handle failed, errno(%d) phyId(%u)", errno, attr->phyId),
        ConverReturnCode(HCCP_INIT, -ENOMEM));

    ret = RaGetInitCtxHandle(cfg, attr, ctxHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_ctx]ra_get_init_ctx_handle failed, ret(%d) phyId(%u)", ret, attr->phyId);
        goto err;
    }

    hccp_run_info("Input parameters: phy_id[%u], nic_position:[%d]", attr->phyId, cfg->mode);
    ret = ctxHandleTmp->ctxOps->raCtxInit(ctxHandleTmp, attr, &(ctxHandleTmp->devIndex), &(ctxHandleTmp->devAttr));
    if (ret != 0) {
        hccp_err("[init][ra_ctx]ctx init failed, ret(%d) phyId(%u)", ret, attr->phyId);
        goto err;
    }

    *ctxHandle = (void *)ctxHandleTmp;
    return 0;

err:
    free(ctxHandleTmp);
    ctxHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetDevBaseAttr(void *ctxHandle, struct DevBaseAttr *attr)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;

    CHK_PRT_RETURN(ctxHandle == NULL || attr == NULL, hccp_err("[get][dev_attr]ctx_handle or attr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    (void)memcpy_s(attr, sizeof(struct DevBaseAttr), &(ctxHandleTmp->devAttr), sizeof(struct DevBaseAttr));

    hccp_info_rma("[get][dev_attr]phy_id(%u), devIndex(%u)", ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex);
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCtxGetAsyncEvents(void *ctxHandle, struct AsyncEvent events[], unsigned int *num)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || events == NULL || num == NULL,
        hccp_err("[get][async_events]ctx_handle or events or num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*num == 0 || *num > ASYNC_EVENT_MAX_NUM,
        hccp_err("[get][async_events]num:%u must be greater than 0 "
                 "and less than or equal to %d",
            *num, ASYNC_EVENT_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxGetAsyncEvents == NULL,
        hccp_err("[get][async_events]ctx_ops or ra_ctx_get_async_events is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ret = ctxHandleTmp->ctxOps->raCtxGetAsyncEvents(ctxHandleTmp, events, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][async_events]get async events failed, ret:%d phyId(%u) devIndex:0x%x", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxDeinit(void *ctxHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL, hccp_err("[deinit][ra_ctx]ctx_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxDeinit == NULL,
        hccp_err("[deinit][ra_ctx]ctx_ops or ra_ctx_deinit is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phy_id[%u], devIndex[%u], protocol[%d]", ctxHandleTmp->attr.phyId,
        ctxHandleTmp->devIndex, ctxHandleTmp->protocol);
    ret = ctxHandleTmp->ctxOps->raCtxDeinit(ctxHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_ctx]ctx deinit failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
    }

    ctxHandleTmp->ctxOps = NULL;
    free(ctxHandleTmp);
    ctxHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetEidByIp(void *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || ip == NULL || eid == NULL || num == NULL,
        hccp_err("[get][eid_by_ip]ctx_handle or ip or eid or num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM,
        hccp_err("[get][eid_by_ip]num(%u) must be greater than 0 "
                 "and less than or equal to %d",
            *num, HCCP_EID_IP_QUERY_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;

    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxGetEidByIp == NULL,
        hccp_err("[get][eid_by_ip]ctx_ops or ra_ctx_get_eid_by_ip is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId:%u, devIndex:0x%x num:%u", ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex,
        *num);

    ret = ctxHandleTmp->ctxOps->raCtxGetEidByIp(ctxHandle, ip, eid, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][eid_by_ip]ra_ctx_get_eid_by_ip failed, ret(%d) phyId(%u) devIndex(0x%x)", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetIpByEid(void *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || eid == NULL || ip == NULL || num == NULL,
        hccp_err("[get][IpByEid]ctxHandle or eid or ip or num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM,
        hccp_err("[get][IpByEid]num(%u) must be greater than 0 "
                 "and less than or equal to %d",
            *num, HCCP_EID_IP_QUERY_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;

    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxGetIpByEid == NULL,
        hccp_err("[get][IpByEid]ctxOps or raCtxGetIpByEid is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phy_id:%u, devIndex:0x%x num:%u", ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex,
        *num);

    ret = ctxHandleTmp->ctxOps->raCtxGetIpByEid(ctxHandle, eid, ip, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][IpByEid]raCtxGetIpByEid failed, ret(%d) phyId(%u) devIndex(0x%x)", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxTokenIdAlloc(void *ctxHandle, struct HccpTokenId *info, void **tokenIdHandle)
{
    struct RaTokenIdHandle *tokenIdHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || info == NULL || tokenIdHandle == NULL,
        hccp_err("[init][ra_token_id]ctx_handle or info or token_id_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxTokenIdAlloc == NULL,
        hccp_err("[init][ra_token_id]ctx_ops or ra_ctx_token_id_alloc is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    tokenIdHandleTmp = (struct RaTokenIdHandle *)calloc(1, sizeof(struct RaTokenIdHandle));
    CHK_PRT_RETURN(tokenIdHandleTmp == NULL,
        hccp_err("[init][ra_token_id]calloc token_id_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxTokenIdAlloc(ctxHandleTmp, info, tokenIdHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_token_id]alloc failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
        goto err;
    }

    *tokenIdHandle = (void *)tokenIdHandleTmp;
    return 0;

err:
    free(tokenIdHandleTmp);
    tokenIdHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxTokenIdFree(void *ctxHandle, void *tokenIdHandle)
{
    struct RaTokenIdHandle *tokenIdHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || tokenIdHandle == NULL,
        hccp_err("[deinit][ra_token_id]ctx_handle or token_id_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxTokenIdFree == NULL,
        hccp_err("[deinit][ra_token_id]ctx_ops or ra_ctx_token_id_free is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    tokenIdHandleTmp = (struct RaTokenIdHandle *)tokenIdHandle;
    ret = ctxHandleTmp->ctxOps->raCtxTokenIdFree(ctxHandleTmp, tokenIdHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_token_id]free failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
    }

    free(tokenIdHandleTmp);
    tokenIdHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxLmemRegister(void *ctxHandle, struct MrRegInfoT *lmemInfo, void **lmemHandle)
{
    struct RaLmemHandle *lmemHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || lmemInfo == NULL || lmemHandle == NULL,
        hccp_err("[init][ra_lmem]ctx_handle or lmem_info or lmem_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxLmemRegister == NULL,
        hccp_err("[init][ra_lmem]ctx_ops or ra_ctx_lmem_register is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    lmemHandleTmp = calloc(1, sizeof(struct RaLmemHandle));
    CHK_PRT_RETURN(lmemHandleTmp == NULL,
        hccp_err("[init][ra_lmem]calloc lmem_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxLmemRegister(ctxHandleTmp, lmemInfo, lmemHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_lmem]register failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
        goto err;
    }

    *lmemHandle = (void *)lmemHandleTmp;
    return 0;

err:
    free(lmemHandleTmp);
    lmemHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxLmemUnregister(void *ctxHandle, void *lmemHandle)
{
    struct RaLmemHandle *lmemHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || lmemHandle == NULL,
        hccp_err("[deinit][ra_lmem]ctx_handle or lmem_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxLmemUnregister == NULL,
        hccp_err("[deinit][ra_lmem]ctx_ops or ra_ctx_lmem_unregister is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    lmemHandleTmp = (struct RaLmemHandle *)lmemHandle;
    ret = ctxHandleTmp->ctxOps->raCtxLmemUnregister(ctxHandleTmp, lmemHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_lmem]unregister failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
    }

    free(lmemHandleTmp);
    lmemHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxRmemImport(void *ctxHandle, struct MrImportInfoT *rmemInfo, void **rmemHandle)
{
    struct RaRmemHandle *rmemHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || rmemInfo == NULL || rmemHandle == NULL,
        hccp_err("[init][ra_rmem]ctx_handle or rmem_info or rmem_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxRmemImport == NULL,
        hccp_err("[init][ra_rmem]ctx_ops is NULL or ra_ctx_rmem_import ops is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    rmemHandleTmp = calloc(1, sizeof(struct RaRmemHandle));
    CHK_PRT_RETURN(rmemHandleTmp == NULL,
        hccp_err("[init][ra_rmem]calloc rmem_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxRmemImport(ctxHandleTmp, rmemInfo);
    if (ret != 0) {
        hccp_err("[init][ra_lmem]import failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
        goto err;
    }

    rmemHandleTmp->key = rmemInfo->in.key;
    rmemHandleTmp->addr = rmemInfo->out.ub.targetSegHandle;
    *rmemHandle = (void *)rmemHandleTmp;
    return 0;

err:
    free(rmemHandleTmp);
    rmemHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxRmemUnimport(void *ctxHandle, void *rmemHandle)
{
    struct RaRmemHandle *rmemHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || rmemHandle == NULL,
        hccp_err("[deinit][ra_rmem]ctx_handle or rmem_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxRmemUnimport == NULL,
        hccp_err("[deinit][ra_rmem]ctx_ops or ra_ctx_rmem_unimport is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    rmemHandleTmp = (struct RaRmemHandle *)rmemHandle;
    ret = ctxHandleTmp->ctxOps->raCtxRmemUnimport(ctxHandleTmp, rmemHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_rmem]unimport failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
    }

    free(rmemHandleTmp);
    rmemHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxChanCreate(void *ctxHandle, struct ChanInfoT *chanInfo, void **chanHandle)
{
    struct RaChanHandle *chanHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || chanInfo == NULL || chanHandle == NULL,
        hccp_err("[init][ra_chan]ctx_handle or chan_info or chan_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxChanCreate == NULL,
        hccp_err("[init][ra_chan]ctx_ops or ra_ctx_chan_create ops is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    chanHandleTmp = (struct RaChanHandle *)calloc(1, sizeof(struct RaChanHandle));
    CHK_PRT_RETURN(chanHandleTmp == NULL,
        hccp_err("[init][ra_chan]calloc chan_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxChanCreate(ctxHandleTmp, chanInfo, chanHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_chan]create failed, ret(%d)", ret);
        goto err;
    }

    *chanHandle = (void *)chanHandleTmp;
    return 0;

err:
    free(chanHandleTmp);
    chanHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxChanDestroy(void *ctxHandle, void *chanHandle)
{
    struct RaChanHandle *chanHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || chanHandle == NULL,
        hccp_err("[deinit][ra_chan]ctx_handle or chan_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxChanDestroy == NULL,
        hccp_err("[deinit][ra_chan]ctx_ops or ra_ctx_chan_destroy is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    chanHandleTmp = (struct RaChanHandle *)chanHandle;
    ret = ctxHandleTmp->ctxOps->raCtxChanDestroy(ctxHandleTmp, chanHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_chan]destroy failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
    }

    free(chanHandleTmp);
    chanHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxCqCreate(void *ctxHandle, struct CqInfoT *info, void **cqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    struct RaCqHandle *cqHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || info == NULL || cqHandle == NULL,
        hccp_err("[init][ra_cq]ctx_handle or info or cq_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxCqCreate == NULL,
        hccp_err("[init][ra_cq]ctx_ops or ra_ctx_cq_create ops is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    cqHandleTmp = (struct RaCqHandle *)calloc(1, sizeof(struct RaCqHandle));
    CHK_PRT_RETURN(cqHandleTmp == NULL,
        hccp_err("[init][ra_cq]calloc cq_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxCqCreate(ctxHandleTmp, info, cqHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_cq]create failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
        goto err;
    }

    *cqHandle = (void *)cqHandleTmp;
    return 0;

err:
    free(cqHandleTmp);
    cqHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxCqDestroy(void *ctxHandle, void *cqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    struct RaCqHandle *cqHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || cqHandle == NULL, hccp_err("[deinit][ra_cq]ctx_handle or cq_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxCqDestroy == NULL,
        hccp_err("[deinit][ra_cq]ctx_ops or ra_ctx_cq_destroy is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    cqHandleTmp = (struct RaCqHandle *)cqHandle;
    ret = ctxHandleTmp->ctxOps->raCtxCqDestroy(ctxHandleTmp, cqHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_cq]destroy failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
    }

    free(cqHandleTmp);
    cqHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpCreate(void *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info,
    void **qpHandle)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || attr == NULL || info == NULL || qpHandle == NULL,
        hccp_err("[init][ra_qp]ctx_handle or attr or info or qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxQpCreate == NULL,
        hccp_err("[init][ra_qp]ctx_ops or ra_ctx_qp_create is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = calloc(1, sizeof(struct RaCtxQpHandle));
    CHK_PRT_RETURN(qpHandleTmp == NULL,
        hccp_err("[init][ra_qp]calloc qp_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxQpCreate(ctxHandleTmp, attr, info, qpHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_qp]create failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
        goto err;
    }

    *qpHandle = (void *)qpHandleTmp;
    return 0;

err:
    free(qpHandleTmp);
    qpHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

STATIC int QpQueryBatchParamCheck(void *qpHandle[], unsigned int *num, unsigned int phyId, unsigned int ids[])
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    unsigned int i;

    for (i = 0; i < *num; i++) {
        qpHandleTmp = (struct RaCtxQpHandle *)qpHandle[i];
        CHK_PRT_RETURN(qpHandleTmp == NULL, hccp_err("[query][ra_qp]qp_handle[%u] is NULL", i), -EINVAL);
        CHK_PRT_RETURN(qpHandleTmp->phyId != phyId,
            hccp_err("[query][ra_qp]qp_handle[%u] comes from different devices, phyId[%u] != qpHandle[0]->phyId(%u)", i,
                qpHandleTmp->phyId, phyId),
            -EINVAL);
        CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                           qpHandleTmp->ctxHandle->ctxOps->raCtxQueryQpBatch == NULL,
            hccp_err("[query][ra_qp]ctx_handle or ctx_ops or ra_ctx_query_qp_batch is NULL"), -EINVAL);

        ids[i] = qpHandleTmp->id;
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCtxQpQueryBatch(void *qpHandle[], struct JettyAttr attr[], unsigned int *num)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    unsigned int ids[HCCP_MAX_QP_QUERY_NUM] = {0};
    unsigned int phyId, devIndex;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || attr == NULL, hccp_err("[query][ra_qp]qp_handle or attr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(num == NULL, hccp_err("num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(*num == 0 || *num > HCCP_MAX_QP_QUERY_NUM,
        hccp_err("[query][ra_qp]num(%u) is out of range(0, %u]", *num, HCCP_MAX_QP_QUERY_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle[0];
    CHK_PRT_RETURN(qpHandleTmp == NULL, hccp_err("[query][ra_qp]qp_handle[0] is NULL"), -EINVAL);
    phyId = qpHandleTmp->phyId;
    devIndex = qpHandleTmp->devIndex;

    ret = QpQueryBatchParamCheck(qpHandle, num, phyId, ids);
    CHK_PRT_RETURN(ret != 0, hccp_err("[query][ra_qp]qp_query_batch_param_check failed, ret(%d)", ret),
        ConverReturnCode(RDMA_OP, ret));

    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxQueryQpBatch(phyId, devIndex, ids, attr, num);
    CHK_PRT_RETURN(ret != 0, hccp_err("[query][ra_qp]query_qp_batch failed, ret(%d)", ret),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpDestroy(void *qpHandle)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[deinit][ra_qp]qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                       qpHandleTmp->ctxHandle->ctxOps->raCtxQpDestroy == NULL,
        hccp_err("[deinit][ra_qp]ctx_handle or ctx_ops "
                 "or ra_ctx_qp_destroy is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxQpDestroy(qpHandleTmp);
    if (ret == -ENODEV) {
        hccp_warn_rma("[deinit][ra_qp]destroy unsuccessful, ret(%d) phyId(%u) devIndex(%u) qp_id(%u)", ret,
            qpHandleTmp->phyId, qpHandleTmp->devIndex, qpHandleTmp->id);
        goto out;
    }
    if (ret != 0) {
        hccp_err("[deinit][ra_qp]destroy failed, ret(%d) phyId(%u) devIndex(%u) qp_id(%u)", ret, qpHandleTmp->phyId,
            qpHandleTmp->devIndex, qpHandleTmp->id);
    }

out:
    free(qpHandleTmp);
    qpHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxGetTpInfoList(void *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || cfg == NULL, hccp_err("[get][RaTpInfo]ctxHandle or cfg is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(infoList == NULL || num == NULL, hccp_err("[get][RaTpInfo]infoList or num is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(*num == 0 || *num > HCCP_MAX_TPID_INFO_NUM,
        hccp_err("[get][RaTpInfo]*num(%u) is out of range[0, %d]", *num, HCCP_MAX_TPID_INFO_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaPeerCtxGetTpInfoList(ctxHandleTmp, cfg, infoList, num);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxGetTpAttr(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || attrBitmap == NULL || attr == NULL,
        hccp_err("[get][RaTpAttr]ctxHandle or attrBitmap or attr is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaPeerCtxGetTpAttr(ctxHandleTmp, tpHandle, attrBitmap, attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][RaTpAttr]RaPeerCtxGetTpAttr failed, ret[%d] phyId[%u] devIndex[0x%x]", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxSetTpAttr(void *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || attr == NULL, hccp_err("[set][RaTpAttr]ctxHandle or attr or attr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaPeerCtxSetTpAttr(ctxHandleTmp, tpHandle, attrBitmap, attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][RaTpAttr]RaPeerCtxSetTpAttr failed, ret[%d] phyId[%u] devIndex[0x%x]", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpImport(void *ctxHandle, struct QpImportInfoT *qpInfo, void **remQpHandle)
{
    struct RaCtxRemQpHandle *remQpHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || qpInfo == NULL || remQpHandle == NULL,
        hccp_err("[init][ra_qp]ctx_handle or qp_info or rem_qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxQpImport == NULL,
        hccp_err("[init][ra_qp]ctx_ops or ra_ctx_qp_import is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    remQpHandleTmp = calloc(1, sizeof(struct RaCtxRemQpHandle));
    CHK_PRT_RETURN(remQpHandleTmp == NULL,
        hccp_err("[init][ra_qp]calloc rem_qp_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = ctxHandleTmp->ctxOps->raCtxQpImport(ctxHandleTmp, qpInfo, remQpHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_qp]import failed, ret(%d) phyId(%u) devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex);
        goto err;
    }

    *remQpHandle = (void *)remQpHandleTmp;
    return 0;

err:
    free(remQpHandleTmp);
    remQpHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpUnimport(void *ctxHandle, void *remQpHandle)
{
    struct RaCtxRemQpHandle *remQpHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || remQpHandle == NULL,
        hccp_err("[deinit][ra_qp]ctx_handle or rem_qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxQpUnimport == NULL,
        hccp_err("[deinit][ra_qp]ctx_ops or ra_ctx_qp_unimport is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    remQpHandleTmp = (struct RaCtxRemQpHandle *)remQpHandle;
    ret = ctxHandleTmp->ctxOps->raCtxQpUnimport(remQpHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_qp]unimport failed, ret(%d) phyId(%u) devIndex(%u) qp_id(%u)", ret, remQpHandleTmp->phyId,
            remQpHandleTmp->devIndex, remQpHandleTmp->id);
    }

    free(remQpHandleTmp);
    remQpHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpBind(void *qpHandle, void *remQpHandle)
{
    struct RaCtxRemQpHandle *remQpHandleTmp = NULL;
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || remQpHandle == NULL, hccp_err("[init][ra_qp]qp_handle or rem_qp_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                       qpHandleTmp->ctxHandle->ctxOps->raCtxQpBind == NULL,
        hccp_err("[init][ra_qp]ctx_handle or ctx_ops "
                 "or ra_ctx_qp_bind is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    remQpHandleTmp = (struct RaCtxRemQpHandle *)remQpHandle;
    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxQpBind(qpHandleTmp, remQpHandleTmp);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_qp]bind failed, ret(%d) phyId(%u) devIndex(%u) "
                 "local_id(%u) remote_id(%u)",
            ret, qpHandleTmp->ctxHandle->attr.phyId, qpHandleTmp->ctxHandle->devIndex, qpHandleTmp->id,
            remQpHandleTmp->id),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCtxQpUnbind(void *qpHandle)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[deinit][ra_qp]qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                       qpHandleTmp->ctxHandle->ctxOps->raCtxQpUnbind == NULL,
        hccp_err("[deinit][ra_qp]ctx_handle or ctx_ops "
                 "or ra_ctx_qp_unbind is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxQpUnbind(qpHandleTmp);
    CHK_PRT_RETURN(ret == -ENODEV,
        hccp_warn_rma("[deinit][ra_qp]unbind unsuccessful, ret(%d) phyId(%u) devIndex(%u)", ret,
            qpHandleTmp->ctxHandle->attr.phyId, qpHandleTmp->ctxHandle->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_qp]unbind failed, ret(%d) phyId(%u) devIndex(%u)", ret,
            qpHandleTmp->ctxHandle->attr.phyId, qpHandleTmp->ctxHandle->devIndex),
        ConverReturnCode(RDMA_OP, ret));
    return ConverReturnCode(RDMA_OP, ret);
}

STATIC int RaCtxBatchSendWrCheck(struct RaCtxQpHandle *qpHandle, struct SendWrData wrList[], unsigned int num)
{
    enum ProtocolTypeT protocol = qpHandle->protocol;
    bool isInline = false;
    unsigned int i, j;

    for (i = 0; i < num; i++) {
        // NOP opcode no need to check
        if (protocol == PROTOCOL_UDMA && wrList[i].ub.opcode == RA_UB_OPC_NOP) {
            continue;
        }

        CHK_PRT_RETURN(wrList[i].rmemHandle == NULL, hccp_err("[send][ra_qp]wr[%u] rmem_handle is NULL", i), -EINVAL);

        isInline = ((protocol == PROTOCOL_RDMA && (wrList[i].rdma.flags & RA_SEND_INLINE) != 0) ||
                    (protocol == PROTOCOL_UDMA && wrList[i].ub.flags.bs.inlineFlag != 0));
        if (!isInline) {
            CHK_PRT_RETURN((wrList[i].sges == NULL), hccp_err("[send][ra_qp]wr[%u] sges is NULL", i), -EINVAL);

            for (j = 0; j < wrList[i].numSge; j++) {
                CHK_PRT_RETURN((wrList[i].sges[j].lmemHandle == NULL),
                    hccp_err("[send][ra_qp]wr[%u] sges[%u] lmem_handle is NULL", i, j), -EINVAL);
            }
        } else if (wrList[i].inlineData == NULL) {
            hccp_err("[send][ra_qp]wr[%u] inline_data is NULL", i);
            return -EINVAL;
        }

        if (protocol == PROTOCOL_UDMA) {
            if (wrList[i].ub.remQpHandle == NULL ||
                (wrList[i].ub.opcode == RA_UB_OPC_WRITE_NOTIFY && wrList[i].ub.notifyInfo.notifyHandle == NULL)) {
                hccp_err("[send][ra_qp]wr[%u] opcode[%d] rem_qp_handle or notify_handle is NULL", i,
                    wrList[i].ub.opcode);
                return -EINVAL;
            }

            // check reduce, only write & write with notify & read op support reduce
            if ((wrList[i].ub.opcode != RA_UB_OPC_WRITE && wrList[i].ub.opcode != RA_UB_OPC_WRITE_NOTIFY &&
                    wrList[i].ub.opcode != RA_UB_OPC_READ) &&
                wrList[i].ub.reduceInfo.reduceEn) {
                hccp_err("[send][ra_qp]wr[%u] opcode[%d] does not support reduce", i, wrList[i].ub.opcode);
                return -EINVAL;
            }
        }
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaBatchSendWr(void *qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[],
    unsigned int num, unsigned int *completeNum)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wrList == NULL || opResp == NULL || num == 0 || completeNum == NULL,
        hccp_err("[send][ra_qp]qp_handle or wr_list or op_resp or complete_num is NULL, or num[%u] is 0", num),
        ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                       qpHandleTmp->ctxHandle->ctxOps->raCtxBatchSendWr == NULL,
        hccp_err("[send][ra_qp]ctx_handle or ctx_ops or ra_ctx_batch_send_wr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = RaCtxBatchSendWrCheck(qpHandleTmp, wrList, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[send][ra_qp]check failed, ret(%d), protocol(%d) phyId(%u), devIndex(%u)", ret, qpHandleTmp->protocol,
            qpHandleTmp->ctxHandle->attr.phyId, qpHandleTmp->ctxHandle->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxBatchSendWr(qpHandleTmp, wrList, opResp, num, completeNum);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxUpdateCi(void *qpHandle, uint16_t ci)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[update][ra_qp]qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                       qpHandleTmp->ctxHandle->ctxOps->raCtxUpdateCi == NULL,
        hccp_err("[update][ra_qp]ctx_handle or ctx_ops "
                 "or ra_ctx_update_ci is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxUpdateCi(qpHandleTmp, ci);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxGetAuxInfo(void *ctxHandle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || in == NULL || out == NULL,
        hccp_err("[get][aux_info]ctx_handle or in or out is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(in->type < AUX_INFO_IN_TYPE_CQE || in->type >= AUX_INFO_IN_TYPE_MAX,
        hccp_err("[get][aux_info]in->type(%d) must be greater than or equal to %d and less than %d", in->type,
            AUX_INFO_IN_TYPE_CQE, AUX_INFO_IN_TYPE_MAX),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxGetAuxInfo == NULL,
        hccp_err("[get][aux_info]ctx_ops or ra_ctx_get_aux_info is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ret = ctxHandleTmp->ctxOps->raCtxGetAuxInfo(ctxHandle, in, out);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][aux_info]ra_ctx_get_aux_info failed, ret(%d) phyId(%u) devIndex(0x%x)", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxGetCrErrInfoList(void *ctxHandle, struct CrErrInfo *infoList, unsigned int *num)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || infoList == NULL || num == NULL,
        hccp_err("[get][cr_err_info_list]ctx_handle or info_list or num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*num == 0 || *num > CR_ERR_INFO_MAX_NUM,
        hccp_err("[get][cr_err_info_list]num:%u must be greater than 0 "
                 "and less than or equal to %d",
            *num, CR_ERR_INFO_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;

    hccp_run_info("Input parameters: phy_id(%u), devIndex(0x%x) num(%u)", ctxHandleTmp->attr.phyId,
        ctxHandleTmp->devIndex, *num);

    ret = RaHdcCtxGetCrErrInfoList(ctxHandleTmp, infoList, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][cr_err_info_list]ra_hdc_ctx_get_cr_err_info_list failed, ret:%d "
                 "phyId:%u devIndex:0x%x",
            ret, ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxGetJettyContext(void *qpHandle, uint8_t context[], unsigned int *len)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(qpHandle == NULL || context == NULL || len == NULL,
        hccp_err("[get][jettyContext]qpHandle or context or len is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*len < CONTEXT_MAX_LEN, hccp_err("[get][jettyContext]len:%u less than %d", *len, CONTEXT_MAX_LEN),
        ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL || qpHandleTmp->ctxHandle->ctxOps == NULL ||
                       qpHandleTmp->ctxHandle->ctxOps->raCtxGetJettyContext == NULL,
        hccp_err("[get][jettyContext]ctxHandle or"
                 " ctxOps or raCtxGetJettyContext is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId:%u, devIndex:0x%x jettyId:%u len:%u", qpHandleTmp->phyId,
        qpHandleTmp->devIndex, qpHandleTmp->qpInfo.ub.id, *len);

    ret = qpHandleTmp->ctxHandle->ctxOps->raCtxGetJettyContext(qpHandleTmp, context, len);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][jettyContext]raCtxGetJettyContext failed, ret:%d phyId:%u devIndex"
                 ":0x%x",
            ret, qpHandleTmp->phyId, qpHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxNotifyEvent(void *ctxHandle, struct CtxNotifyEvent *event)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || event == NULL, hccp_err("[notify][event]ctxHandle or event is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    CHK_PRT_RETURN(ctxHandleTmp->ctxOps == NULL || ctxHandleTmp->ctxOps->raCtxNotifyEvent == NULL,
        hccp_err("[notify][event]ctxOps or raCtxNotifyEvent is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    if (!RaHdcHasCapability(ctxHandleTmp->attr.phyId, RA_CAP_UDMA_NOTIFY_EVENT)) {
        hccp_run_warn("[notify][event]capability not support, phyId:%u", ctxHandleTmp->attr.phyId);
        return ConverReturnCode(RDMA_OP, -ENOTSUPP);
    }

    hccp_run_info("Input parameters: phyId:%u, devIndex:0x%x serviceType:%u errorType:%u", ctxHandleTmp->attr.phyId,
        ctxHandleTmp->devIndex, event->serviceType, event->errorType);

    ret = ctxHandleTmp->ctxOps->raCtxNotifyEvent(ctxHandleTmp, event);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[notify][event]raCtxNotifyEvent failed, ret:%d phyId:%u devIndex:0x%x", ret, ctxHandleTmp->attr.phyId,
            ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}
