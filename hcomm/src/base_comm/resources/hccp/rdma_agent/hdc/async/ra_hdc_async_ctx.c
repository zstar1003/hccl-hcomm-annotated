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
#include "ra_async.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "ra_rs_ctx.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async.h"
#include "ra_hdc_async_ctx.h"

int RaHdcGetEidByIpAsync(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num,
    void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaResponseEidList *asyncRsp = NULL;
    union OpGetEidByIpData asyncData = {0};
    int ret = 0;

    asyncRsp = (struct RaResponseEidList *)calloc(1, sizeof(struct RaResponseEidList));
    CHK_PRT_RETURN(asyncRsp == NULL,
        hccp_err("[get][eid_by_ip]calloc async_rsp failed, phyId[%u] devIndex[0x%x]", phyId, ctxHandle->devIndex),
        -ENOMEM);
    asyncRsp->eidList = eid;
    asyncRsp->num = num;

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[get][eid_by_ip]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId, ctxHandle->devIndex);
        ret = -ENOMEM;
        goto out;
    }

    RaHdcPrepareGetEidByIp(ctxHandle, ip, *num, &asyncData);
    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)asyncRsp;
    ret = RaHdcSendMsgAsync(RA_RS_GET_EID_BY_IP, phyId, (char *)&asyncData, sizeof(union OpGetEidByIpData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[get][eid_by_ip]hdc async send message failed ret[%d], phyId[%u], devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    *reqHandle = (void *)reqHandleTmp;
    return ret;

out:
    free(asyncRsp);
    asyncRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleGetEidByIp(struct RaRequestHandle *reqHandle)
{
    struct RaResponseEidList *asyncRsp = NULL;
    union OpGetEidByIpData *asyncData = NULL;
    unsigned int ipNum = 0;
    int ret = 0;

    asyncData = (union OpGetEidByIpData *)reqHandle->recvBuf;
    asyncRsp = (struct RaResponseEidList *)reqHandle->privData;
    ipNum = *asyncRsp->num;

    ret = RaHdcGetEidResults(asyncData, ipNum, asyncRsp->eidList, asyncRsp->num);
    if (ret != 0) {
        hccp_err("[get][eid_by_ip]ra_hdc_get_eid_results failed ret[%d], phyId[%u] devIndex[0x%x]", ret,
            reqHandle->phyId, reqHandle->devIndex);
        reqHandle->opRet = ret;
    }

    free(reqHandle->privData);
    reqHandle->privData = NULL;
    return;
}

int RaHdcGetIpByEidAsync(struct RaCtxHandle *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num,
    void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    struct RaResponseIpList *asyncRsp = NULL;
    union OpGetIpByEidData asyncData = {0};
    int ret = 0;

    asyncRsp = (struct RaResponseIpList *)calloc(1, sizeof(struct RaResponseIpList));
    CHK_PRT_RETURN(asyncRsp == NULL,
        hccp_err("[get][IpByEid]calloc asyncRsp failed, phyId[%u] devIndex[0x%x]", phyId, ctxHandle->devIndex),
        -ENOMEM);
    asyncRsp->ipList = ip;
    asyncRsp->num = num;

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[get][IpByEid]calloc reqHandleTmp failed, phyId[%u], devIndex[0x%x]", phyId, ctxHandle->devIndex);
        ret = -ENOMEM;
        goto free_async;
    }

    RaHdcPrepareGetIpByEid(ctxHandle, eid, *num, &asyncData);
    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)asyncRsp;
    ret = RaHdcSendMsgAsync(RA_RS_GET_IP_BY_EID, phyId, (char *)&asyncData, sizeof(union OpGetIpByEidData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[get][IpByEid]hdc async send message failed ret[%d], phyId[%u], devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto free_async;
    }

    *reqHandle = (void *)reqHandleTmp;
    return ret;

free_async:
    free(asyncRsp);
    asyncRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleGetIpByEid(struct RaRequestHandle *reqHandle)
{
    union OpGetIpByEidData *asyncData = NULL;
    struct RaResponseIpList *asyncRsp = NULL;
    unsigned int eidNum = 0;
    int ret = 0;

    asyncData = (union OpGetIpByEidData *)reqHandle->recvBuf;
    asyncRsp = (struct RaResponseIpList *)reqHandle->privData;
    eidNum = *asyncRsp->num;

    ret = RaHdcGetIpResults(asyncData, eidNum, asyncRsp->ipList, asyncRsp->num);
    if (ret != 0) {
        hccp_err("[get][IpByEid]RaHdcGetIpResults failed ret[%d], phyId[%u] devIndex[0x%x]", ret, reqHandle->phyId,
            reqHandle->devIndex);
        reqHandle->opRet = ret;
    }

    free(reqHandle->privData);
    reqHandle->privData = NULL;
    return;
}

int RaHdcCtxLmemRegisterAsync(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo,
    struct RaLmemHandle *lmemHandle, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpLmemRegInfoData asyncData = {0};
    int ret;

    ret = RaHdcCtxPrepareLmemRegister(ctxHandle, lmemInfo, &asyncData);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_lmem]prepare register failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[init][ra_hdc_lmem]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        -ENOMEM);

    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)&lmemInfo->out;
    reqHandleTmp->privHandle = (void *)lmemHandle;
    ret = RaHdcSendMsgAsync(RA_RS_LMEM_REG, phyId, (char *)&asyncData, sizeof(union OpLmemRegInfoData), reqHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_lmem]hdc async send message failed ret[%d], phyId[%u], devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

void RaHdcAsyncHandleLmemRegister(struct RaRequestHandle *reqHandle)
{
    union OpLmemRegInfoData *asyncData = NULL;
    struct RaLmemHandle *lmemHandle = NULL;
    struct MemRegInfo *info = NULL;
    int ret;

    asyncData = (union OpLmemRegInfoData *)reqHandle->recvBuf;
    lmemHandle = (struct RaLmemHandle *)reqHandle->privHandle;
    info = (struct MemRegInfo *)reqHandle->privData;
    ret = memcpy_s(info, sizeof(struct MemRegInfo), &asyncData->rxData.memInfo, sizeof(struct MemRegInfoT));
    if (ret != 0) {
        hccp_err("[init][ra_hdc_lmem]memcpy_s mem_info failed ret[%d], phyId[%u] devIndex[0x%x]", ret, reqHandle->phyId,
            reqHandle->devIndex);
        reqHandle->opRet = -ESAFEFUNC;
        return;
    }

    lmemHandle->addr = info->ub.targetSegHandle;
    return;
}

int RaHdcCtxLmemUnregisterAsync(struct RaCtxHandle *ctxHandle, struct RaLmemHandle *lmemHandle, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpLmemUnregInfoData asyncData = {0};
    unsigned int phyId = ctxHandle->attr.phyId;
    int ret = 0;

    asyncData.txData.phyId = phyId;
    asyncData.txData.devIndex = ctxHandle->devIndex;
    asyncData.txData.addr = lmemHandle->addr;
    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[deinit][ra_hdc_lmem]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        -ENOMEM);

    ret = RaHdcSendMsgAsync(RA_RS_LMEM_UNREG, phyId, (char *)&asyncData, sizeof(union OpLmemUnregInfoData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_hdc_lmem]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

int RaHdcCtxQpCreateAsync(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info,
    struct RaCtxQpHandle *qpHandle, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxQpCreateData asyncData = {0};
    int ret;

    ret = RaHdcCtxPrepareQpCreate(ctxHandle, attr, &asyncData);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_qp]prepare qp_create failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[init][ra_hdc_qp]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        -ENOMEM);
    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)info;
    qpHandle->ctxHandle = ctxHandle;
    qpHandle->devIndex = ctxHandle->devIndex;
    qpHandle->phyId = ctxHandle->attr.phyId;
    qpHandle->protocol = ctxHandle->protocol;
    (void)memcpy_s(&qpHandle->qpAttr, sizeof(struct QpCreateAttr), attr, sizeof(struct QpCreateAttr));
    reqHandleTmp->privHandle = (void *)qpHandle;

    ret = RaHdcSendMsgAsync(RA_RS_CTX_QP_CREATE, phyId, (char *)&asyncData, sizeof(union OpCtxQpCreateData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_qp]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

void RaHdcAsyncHandleQpCreate(struct RaRequestHandle *reqHandle)
{
    union OpCtxQpCreateData *asyncData = NULL;
    struct RaCtxQpHandle *qpHandle = NULL;
    struct QpCreateInfo *info = NULL;

    asyncData = (union OpCtxQpCreateData *)reqHandle->recvBuf;
    info = (struct QpCreateInfo *)reqHandle->privData;
    qpHandle = (struct RaCtxQpHandle *)reqHandle->privHandle;
    (void)memcpy_s(info, sizeof(struct QpCreateInfo), &asyncData->rxData.qpInfo, sizeof(struct QpCreateInfo));
    qpHandle->id = info->ub.id;
    (void)memcpy_s(&qpHandle->qpInfo, sizeof(struct QpCreateInfo), info, sizeof(struct QpCreateInfo));

    return;
}

int RaHdcCtxQpDestroyAsync(struct RaCtxQpHandle *qpHandle, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpCtxQpDestroyData asyncData = {0};
    unsigned int phyId = qpHandle->phyId;
    int ret;

    asyncData.txData.phyId = phyId;
    asyncData.txData.devIndex = qpHandle->devIndex;
    asyncData.txData.id = qpHandle->id;

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[deinit][ra_hdc_qp]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId,
            qpHandle->devIndex),
        -ENOMEM);

    ret = RaHdcSendMsgAsync(RA_RS_CTX_QP_DESTROY, phyId, (char *)&asyncData, sizeof(union OpCtxQpDestroyData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_hdc_qp]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            qpHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

int RaHdcCtxQpImportAsync(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *info,
    struct RaCtxRemQpHandle *remQpHandle, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxQpImportData asyncData = {0};
    int ret = 0;

    ret = RaHdcCtxPrepareQpImport(ctxHandle, info, &asyncData);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_qp]prepare qp_import failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[init][ra_hdc_qp]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        -ENOMEM);

    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)&info->out;
    remQpHandle->devIndex = ctxHandle->devIndex;
    remQpHandle->phyId = ctxHandle->attr.phyId;
    remQpHandle->protocol = ctxHandle->protocol;
    remQpHandle->qpKey = info->in.key;
    reqHandleTmp->privHandle = (void *)remQpHandle;

    ret = RaHdcSendMsgAsync(RA_RS_CTX_QP_IMPORT, phyId, (char *)&asyncData, sizeof(union OpCtxQpImportData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_qp]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

void RaHdcAsyncHandleQpImport(struct RaRequestHandle *reqHandle)
{
    struct RaCtxRemQpHandle *remQpHandle = NULL;
    union OpCtxQpImportData *asyncData = NULL;
    struct QpImportInfo *info = NULL;

    asyncData = (union OpCtxQpImportData *)reqHandle->recvBuf;
    info = (struct QpImportInfo *)reqHandle->privData;
    remQpHandle = (struct RaCtxRemQpHandle *)reqHandle->privHandle;

    info->ub.tjettyHandle = asyncData->rxData.info.tjettyHandle;
    info->ub.tpn = asyncData->rxData.info.tpn;
    remQpHandle->id = asyncData->rxData.remJettyId;

    return;
}

int RaHdcCtxQpUnimportAsync(struct RaCtxRemQpHandle *remQpHandle, void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpCtxQpUnimportData asyncData = {0};
    unsigned int phyId = remQpHandle->phyId;
    int ret;

    asyncData.txData.phyId = phyId;
    asyncData.txData.devIndex = remQpHandle->devIndex;
    (void)memcpy_s(asyncData.txData.rawRemJettyId, REM_JETTY_ID_SIZE, remQpHandle->qpKey.value, REM_JETTY_ID_SIZE);

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[deinit][ra_hdc_qp]calloc req_handle_tmp failed, phyId[%u], devIndex[0x%x]", phyId,
            remQpHandle->devIndex),
        -ENOMEM);

    ret = RaHdcSendMsgAsync(RA_RS_CTX_QP_UNIMPORT, phyId, (char *)&asyncData, sizeof(union OpCtxQpUnimportData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_hdc_qp]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            remQpHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

int RaHdcGetTpInfoListAsync(struct RaCtxHandle *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num, void **reqHandle)
{
    struct RaResponseTpInfoList *asyncRsp = NULL;
    struct RaRequestHandle *reqHandleTmp = NULL;
    union OpGetTpInfoListData asyncData = {0};
    unsigned int phyId = ctxHandle->attr.phyId;
    int ret = 0;

    asyncRsp = (struct RaResponseTpInfoList *)calloc(1, sizeof(struct RaResponseTpInfoList));
    CHK_PRT_RETURN(asyncRsp == NULL,
        hccp_err("[get][ra_hdc_tp_info]calloc async_rsp failed, phyId[%u] devIndex[0x%x]", phyId, ctxHandle->devIndex),
        -ENOMEM);
    asyncRsp->infoList = infoList;
    asyncRsp->num = num;

    asyncData.txData.phyId = phyId;
    asyncData.txData.devIndex = ctxHandle->devIndex;
    asyncData.txData.num = *num;
    (void)memcpy_s(&asyncData.txData.cfg, sizeof(struct GetTpCfg), cfg, sizeof(struct GetTpCfg));

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[get][ra_hdc_tp_info]calloc RaRequestHandle failed, phyId[%u], devIndex[0x%x]", phyId,
            ctxHandle->devIndex);
        ret = -ENOMEM;
        goto out;
    }

    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)asyncRsp;
    ret = RaHdcSendMsgAsync(RA_RS_GET_TP_INFO_LIST, phyId, (char *)&asyncData, sizeof(union OpGetTpInfoListData),
        reqHandleTmp);
    if (ret != 0) {
        hccp_err("[get][ra_hdc_tp_info]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
out:
    free(asyncRsp);
    asyncRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleTpInfoList(struct RaRequestHandle *reqHandle)
{
    struct RaResponseTpInfoList *asyncRsp = NULL;
    union OpGetTpInfoListData *asyncData = NULL;
    int ret;

    if (reqHandle->opRet != 0) {
        goto out;
    }
    asyncData = (union OpGetTpInfoListData *)reqHandle->recvBuf;
    asyncRsp = (struct RaResponseTpInfoList *)reqHandle->privData;
    if (asyncData->rxData.num == 0) {
        *asyncRsp->num = asyncData->rxData.num;
        goto out;
    }

    ret = memcpy_s(asyncRsp->infoList, (*asyncRsp->num) * sizeof(struct HccpTpInfo), asyncData->rxData.infoList,
        asyncData->rxData.num * sizeof(struct HccpTpInfo));
    if (ret != 0) {
        hccp_err("[get][ra_hdc_tp_info]memcpy_s tp_info failed ret[%d] *async_rsp->num[%u] rx_data.num[%u], "
                 "phyId[%u] devIndex[0x%x]",
            ret, *asyncRsp->num, asyncData->rxData.num, reqHandle->phyId, reqHandle->devIndex);
        reqHandle->opRet = -ESAFEFUNC;
        goto out;
    }
    *asyncRsp->num = asyncData->rxData.num;
out:
    free(reqHandle->privData);
    reqHandle->privData = NULL;
    return;
}

int RaHdcGetTpAttrAsync(struct RaCtxHandle *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr,
    void **reqHandle)
{
    struct RaResponseGetTpAttr *asyncRsp = NULL;
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpGetTpAttrData asyncData = {0};
    int ret = 0;

    asyncRsp = (struct RaResponseGetTpAttr *)calloc(1, sizeof(struct RaResponseGetTpAttr));
    CHK_PRT_RETURN(asyncRsp == NULL,
        hccp_err("[get][ra_hdc_tp_attr]calloc ra_response_get_tp_attr failed, phyId[%u] devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        -ENOMEM);
    asyncRsp->attr = attr;
    asyncRsp->attrBitmap = attrBitmap;

    asyncData.txData.devIndex = ctxHandle->devIndex;
    asyncData.txData.phyId = phyId;
    asyncData.txData.tpHandle = tpHandle;
    asyncData.txData.attrBitmap = *attrBitmap;
    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    if (reqHandleTmp == NULL) {
        hccp_err("[get][ra_hdc_tp_attr]calloc RaRequestHandle failed, phyId[%u] devIndex[0x%x]", phyId,
            ctxHandle->devIndex);
        ret = -ENOMEM;
        goto out;
    }

    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)asyncRsp;
    ret = RaHdcSendMsgAsync(RA_RS_GET_TP_ATTR, phyId, (char *)&asyncData, sizeof(union OpGetTpAttrData), reqHandleTmp);
    if (ret != 0) {
        hccp_err("[get][ra_hdc_tp_attr]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        goto out;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
out:
    free(asyncRsp);
    asyncRsp = NULL;
    return ret;
}

void RaHdcAsyncHandleGetTpAttr(struct RaRequestHandle *reqHandle)
{
    struct RaResponseGetTpAttr *asyncRsp = NULL;
    union OpGetTpAttrData *asyncData = NULL;

    if (reqHandle->opRet != 0) {
        goto out;
    }
    asyncData = (union OpGetTpAttrData *)reqHandle->recvBuf;
    asyncRsp = (struct RaResponseGetTpAttr *)reqHandle->privData;
    *asyncRsp->attrBitmap = asyncData->rxData.attrBitmap;
    (void)memcpy_s(asyncRsp->attr, sizeof(struct TpAttr), &asyncData->rxData.attr, sizeof(struct TpAttr));

out:
    free(reqHandle->privData);
    reqHandle->privData = NULL;
    return;
}

int RaHdcSetTpAttrAsync(struct RaCtxHandle *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr,
    void **reqHandle)
{
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpSetTpAttrData asyncData = {0};
    int ret = 0;

    asyncData.txData.devIndex = ctxHandle->devIndex;
    asyncData.txData.phyId = phyId;
    asyncData.txData.tpHandle = tpHandle;
    asyncData.txData.attrBitmap = attrBitmap;
    (void)memcpy_s(&asyncData.txData.attr, sizeof(struct TpAttr), attr, sizeof(struct TpAttr));
    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[set][ra_hdc_tp_attr]calloc RaRequestHandle failed, phyId[%u] devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        -ENOMEM);

    reqHandleTmp->devIndex = ctxHandle->devIndex;
    ret = RaHdcSendMsgAsync(RA_RS_SET_TP_ATTR, phyId, (char *)&asyncData, sizeof(union OpSetTpAttrData), reqHandleTmp);
    if (ret != 0) {
        hccp_err("[set][ra_hdc_tp_attr]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

STATIC int QpDestroyBatchParamCheck(struct RaCtxHandle *ctxHandle, void *qpHandle[], unsigned int ids[],
    unsigned int *num)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    unsigned int i;

    for (i = 0; i < *num; ++i) {
        qpHandleTmp = (struct RaCtxQpHandle *)qpHandle[i];
        CHK_PRT_RETURN(qpHandleTmp == NULL, hccp_err("[destroy_batch][ra_hdc_ctx_qp]qp_handle[%u] is NULL", i),
            -EINVAL);
        CHK_PRT_RETURN(qpHandleTmp->ctxHandle == NULL,
            hccp_err("[destroy_batch][ra_hdc_ctx_qp]ctx_handle[%u] is NULL", i), -EINVAL);
        CHK_PRT_RETURN(qpHandleTmp->ctxHandle != ctxHandle,
            hccp_err("[destroy_batch][ra_hdc_ctx_qp]qp_handle[%u]->ctx_handle is different from others", i), -EINVAL);

        ids[i] = qpHandleTmp->id;
    }

    return 0;
}

int RaHdcCtxQpDestroyBatchAsync(struct RaCtxHandle *ctxHandle, void *qpHandle[], unsigned int *num, void **reqHandle)
{
    union OpCtxQpDestroyBatchData asyncData = {0};
    struct RaRequestHandle *reqHandleTmp = NULL;
    unsigned int phyId = ctxHandle->attr.phyId;
    int ret;

    ret = QpDestroyBatchParamCheck(ctxHandle, qpHandle, asyncData.txData.ids, num);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[destroy_batch][ra_hdc_ctx_qp]param check failed, phyId[%u] devIndex[0x%x]", phyId,
            ctxHandle->devIndex),
        ret);

    asyncData.txData.phyId = phyId;
    asyncData.txData.devIndex = ctxHandle->devIndex;
    asyncData.txData.num = *num;

    reqHandleTmp = (struct RaRequestHandle *)calloc(1, sizeof(struct RaRequestHandle));
    CHK_PRT_RETURN(reqHandleTmp == NULL,
        hccp_err("[destroy_batch][ra_hdc_ctx_qp]calloc RaRequestHandle failed, "
                 "phyId[%u] devIndex[0x%x]",
            phyId, ctxHandle->devIndex),
        -ENOMEM);

    reqHandleTmp->devIndex = ctxHandle->devIndex;
    reqHandleTmp->privData = (void *)num;
    ret = RaHdcSendMsgAsync(RA_RS_CTX_QP_DESTROY_BATCH, phyId, (char *)&asyncData,
        sizeof(union OpCtxQpDestroyBatchData), reqHandleTmp);
    if (ret != 0) {
        hccp_err("[destroy_batch][ra_hdc_ctx_qp]hdc async send message failed ret[%d], phyId[%u] devIndex[0x%x]", ret,
            phyId, ctxHandle->devIndex);
        free(reqHandleTmp);
        reqHandleTmp = NULL;
        return ret;
    }

    *reqHandle = (void *)reqHandleTmp;
    return 0;
}

void RaHdcAsyncHandleQpDestroyBatch(struct RaRequestHandle *reqHandle)
{
    union OpCtxQpDestroyBatchData *asyncData = NULL;
    unsigned int *num = NULL;

    asyncData = (union OpCtxQpDestroyBatchData *)reqHandle->recvBuf;
    num = (unsigned int *)reqHandle->privData;
    *num = asyncData->rxData.num;

    return;
}
