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
#include "hccp_ctx.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async_ctx.h"
#include "ra_rs_ctx.h"
#include "ra_rs_comm.h"
#include "rs_ctx.h"
#include "ra_adp.h"
#include "ra_adp_ctx.h"

struct RsCtxOps gRaRsCtxOps = {
    .getDevEidInfoNum = RsGetDevEidInfoNum,
    .getDevEidInfoList = RsGetDevEidInfoList,
    .ctxInit = RsCtxInit,
    .ctxGetAsyncEvents = RsCtxGetAsyncEvents,
    .ctxDeinit = RsCtxDeinit,
    .getEidByIp = RsGetEidByIp,
    .getIpByEid = RsGetIpByEid,
    .getTpInfoList = RsGetTpInfoList,
    .getTpAttr = RsGetTpAttr,
    .setTpAttr = RsSetTpAttr,
    .ctxTokenIdAlloc = RsCtxTokenIdAlloc,
    .ctxTokenIdFree = RsCtxTokenIdFree,
    .ctxLmemReg = RsCtxLmemReg,
    .ctxLmemUnreg = RsCtxLmemUnreg,
    .ctxRmemImport = RsCtxRmemImport,
    .ctxRmemUnimport = RsCtxRmemUnimport,
    .ctxChanCreate = RsCtxChanCreate,
    .ctxChanDestroy = RsCtxChanDestroy,
    .ctxCqCreate = RsCtxCqCreate,
    .ctxCqDestroy = RsCtxCqDestroy,
    .ctxQpCreate = RsCtxQpCreate,
    .ctxQpQueryBatch = RsCtxQpQueryBatch,
    .ctxQpDestroy = RsCtxQpDestroy,
    .ctxQpDestroyBatch = RsCtxQpDestroyBatch,
    .ctxQpImport = RsCtxQpImport,
    .ctxQpUnimport = RsCtxQpUnimport,
    .ctxQpBind = RsCtxQpBind,
    .ctxQpUnbind = RsCtxQpUnbind,
    .ctxBatchSendWr = RsCtxBatchSendWr,
    .ctxUpdateCi = RsCtxUpdateCi,
    .ctxGetAuxInfo = RsCtxGetAuxInfo,
    .ctxGetCrErrInfoList = RsCtxGetCrErrInfoList,
    .ctxNotifyEvent = RsCtxNotifyEvent,
    .ctxGetUbContext = RsCtxGetUbContext,
};

int RaRsGetDevEidInfoNum(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetDevEidInfoNumData *opData = (union OpGetDevEidInfoNumData *)(inBuf + sizeof(struct MsgHead));
    unsigned int num = 0;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetDevEidInfoNumData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsCtxOps.getDevEidInfoNum(opData->txData.phyId, &num);
    CHK_PRT_RETURN(*opResult != 0, hccp_err("[get][eid]get_dev_eid_info_num failed, ret[%d].", *opResult), 0);

    opData = (union OpGetDevEidInfoNumData *)(outBuf + sizeof(struct MsgHead));
    opData->rxData.num = num;
    return 0;
}

int RaRsGetDevEidInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetDevEidInfoListData *opData = (union OpGetDevEidInfoListData *)(inBuf + sizeof(struct MsgHead));
    struct HccpDevEidInfo infoList[MAX_DEV_INFO_TRANS_NUM] = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetDevEidInfoListData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.count, 0, MAX_DEV_INFO_TRANS_NUM, opResult);

    *opResult = gRaRsCtxOps.getDevEidInfoList(opData->txData.phyId, infoList, opData->txData.startIndex,
        opData->txData.count);
    CHK_PRT_RETURN(*opResult != 0, hccp_err("[get][eid]get_dev_eid_info_list failed, ret[%d].", *opResult), 0);

    opData = (union OpGetDevEidInfoListData *)(outBuf + sizeof(struct MsgHead));
    (void)memcpy_s(opData->rxData.infoList, sizeof(struct HccpDevEidInfo) * MAX_DEV_INFO_TRANS_NUM, infoList,
        sizeof(struct HccpDevEidInfo) * MAX_DEV_INFO_TRANS_NUM);
    return 0;
}

int RaRsCtxInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxInitData *opDataOut = (union OpCtxInitData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxInitData *opData = (union OpCtxInitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxInitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsCtxOps.ctxInit(&opData->txData.attr, &opDataOut->rxData.devIndex, &opDataOut->rxData.devAttr);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_ctx]init failed, ret[%d]", *opResult);
    }

    return 0;
}

int RaRsCtxGetAsyncEvents(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxGetAsyncEventsData *opDataOut = (union OpCtxGetAsyncEventsData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxGetAsyncEventsData *opData = (union OpCtxGetAsyncEventsData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxGetAsyncEventsData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, ASYNC_EVENT_MAX_NUM, opResult);

    opDataOut->rxData.num = opData->txData.num;
    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxGetAsyncEvents(&devInfo, opDataOut->rxData.events, &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[get][async_events]ctx_get_async_events failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult,
            devInfo.phyId, devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxDeinitData *opData = (union OpCtxDeinitData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxDeinitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxDeinit(&devInfo);
    if (*opResult != 0) {
        hccp_err("[deinit][ra_rs_ctx]deinit failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsGetEidByIp(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetEidByIpData *opDataOut = (union OpGetEidByIpData *)(outBuf + sizeof(struct MsgHead));
    union OpGetEidByIpData *opData = (union OpGetEidByIpData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetEidByIpData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, HCCP_EID_IP_QUERY_MAX_NUM, opResult);

    opDataOut->rxData.num = opData->txData.num;
    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.getEidByIp(&devInfo, opData->txData.ip, opDataOut->rxData.eid, &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[get][eid_by_ip]get_eid_by_ip failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsGetIpByEid(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetIpByEidData *opDataOut = (union OpGetIpByEidData *)(outBuf + sizeof(struct MsgHead));
    union OpGetIpByEidData *opData = (union OpGetIpByEidData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetIpByEidData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, HCCP_EID_IP_QUERY_MAX_NUM, opResult);

    opDataOut->rxData.num = opData->txData.num;
    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.getIpByEid(&devInfo, opData->txData.eid, opDataOut->rxData.ip, &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[get][IpByEid]getIpByEid failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsGetTpInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetTpInfoListData *opDataOut = (union OpGetTpInfoListData *)(outBuf + sizeof(struct MsgHead));
    union OpGetTpInfoListData *opData = (union OpGetTpInfoListData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetTpInfoListData), sizeof(struct MsgHead), rcvBufLen, opResult);

    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, HCCP_MAX_TPID_INFO_NUM, opResult);
    opDataOut->rxData.num = opData->txData.num;
    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.getTpInfoList(&devInfo, &opData->txData.cfg, opDataOut->rxData.infoList,
        &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[get][ra_rs_ctx]get_tp_info_list failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsGetTpAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetTpAttrData *opDataOut = (union OpGetTpAttrData *)(outBuf + sizeof(struct MsgHead));
    union OpGetTpAttrData *opData = (union OpGetTpAttrData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetTpAttrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    opDataOut->rxData.attrBitmap = opData->txData.attrBitmap;
    *opResult = gRaRsCtxOps.getTpAttr(&devInfo, &opDataOut->rxData.attrBitmap, opData->txData.tpHandle,
        &opDataOut->rxData.attr);
    if (*opResult != 0) {
        hccp_err("[get_tp_attr][ra_rs_ctx]get attr failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsSetTpAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSetTpAttrData *opData = (union OpSetTpAttrData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetTpAttrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.setTpAttr(&devInfo, opData->txData.attrBitmap, opData->txData.tpHandle,
        &opData->txData.attr);
    if (*opResult != 0) {
        hccp_err("[set_tp_attr][ra_rs_ctx]set attr failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsCtxTokenIdAlloc(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTokenIdAllocData *opDataOut = (union OpTokenIdAllocData *)(outBuf + sizeof(struct MsgHead));
    union OpTokenIdAllocData *opData = (union OpTokenIdAllocData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTokenIdAllocData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxTokenIdAlloc(&devInfo, &opDataOut->rxData.addr, &opDataOut->rxData.tokenId);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_token_id]alloc failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }
    return 0;
}

int RaRsCtxTokenIdFree(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpTokenIdFreeData *opData = (union OpTokenIdFreeData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTokenIdFreeData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxTokenIdFree(&devInfo, opData->txData.addr);
    if (*opResult != 0) {
        hccp_err("[deinit][ra_rs_token_id]free failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }
    return 0;
}

int RaRsLmemReg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLmemRegInfoData *opDataOut = (union OpLmemRegInfoData *)(outBuf + sizeof(struct MsgHead));
    union OpLmemRegInfoData *opData = (union OpLmemRegInfoData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLmemRegInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxLmemReg(&devInfo, &opData->txData.memAttr, &opDataOut->rxData.memInfo);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_lmem]reg failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsLmemUnreg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpLmemUnregInfoData *opData = (union OpLmemUnregInfoData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLmemUnregInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxLmemUnreg(&devInfo, opData->txData.addr);
    if (*opResult != 0) {
        hccp_err("[deinit][ra_rs_lmem]unreg failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsRmemImport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpRmemImportInfoData *opDataOut = (union OpRmemImportInfoData *)(outBuf + sizeof(struct MsgHead));
    union OpRmemImportInfoData *opData = (union OpRmemImportInfoData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRmemImportInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxRmemImport(&devInfo, &opData->txData.memAttr, &opDataOut->rxData.memInfo);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_rmem]import failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsRmemUnimport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpRmemUnimportInfoData *opData = (union OpRmemUnimportInfoData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRmemUnimportInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxRmemUnimport(&devInfo, opData->txData.addr);
    if (*opResult != 0) {
        hccp_err("[deinit][ra_rs_rmem]unimport failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxChanCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxChanCreateData *opDataOut = (union OpCtxChanCreateData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxChanCreateData *opData = (union OpCtxChanCreateData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxChanCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxChanCreate(&devInfo, opData->txData.dataPlaneFlag, &opDataOut->rxData.addr,
        &opDataOut->rxData.fd);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_chan]create failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxChanDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxChanDestroyData *opData = (union OpCtxChanDestroyData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxChanDestroyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxChanDestroy(&devInfo, opData->txData.addr);
    if (*opResult != 0) {
        hccp_err("[deinit][ra_rs_chan]destroy failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxCqCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxCqCreateData *opDataOut = (union OpCtxCqCreateData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxCqCreateData *opData = (union OpCtxCqCreateData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxCqCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxCqCreate(&devInfo, &opData->txData.attr, &opDataOut->rxData.info);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_cq]create failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxCqDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxCqDestroyData *opData = (union OpCtxCqDestroyData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxCqDestroyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxCqDestroy(&devInfo, opData->txData.addr);
    if (*opResult != 0) {
        hccp_err("[deinit][ra_rs_cq]destroy failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxQpCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxQpCreateData *opDataOut = (union OpCtxQpCreateData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxQpCreateData *opData = (union OpCtxQpCreateData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxQpCreate(&devInfo, &opData->txData.qpAttr, &opDataOut->rxData.qpInfo);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_qp]create failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxQpQueryBatch(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxQpQueryBatchData *opData = (union OpCtxQpQueryBatchData *)(inBuf + sizeof(struct MsgHead));
    union OpCtxQpQueryBatchData *opDataOut = (union OpCtxQpQueryBatchData *)(outBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpQueryBatchData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, HCCP_MAX_QP_QUERY_NUM, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    opDataOut->rxData.num = opData->txData.num;
    *opResult = gRaRsCtxOps.ctxQpQueryBatch(&devInfo, opData->txData.ids, opDataOut->rxData.attr,
        &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[qp_batch][ra_rs_ctx]query failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsCtxQpDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxQpDestroyData *opData = (union OpCtxQpDestroyData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpDestroyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);

    *opResult = gRaRsCtxOps.ctxQpDestroy(&devInfo, opData->txData.id);
    CHK_PRT_RETURN(*opResult == -ENODEV,
        hccp_warn_rma("[deinit][ra_rs_qp]jetty not found, ret[%d] phyId[%u] "
                      "devIndex[0x%x] qpId[%u]",
            *opResult, devInfo.phyId, devInfo.devIndex, opData->txData.id),
        0);
    CHK_PRT_RETURN(*opResult != 0,
        hccp_err("[deinit][ra_rs_qp]destroy failed, ret[%d] phyId[%u] devIndex[0x%x] "
                 "qpId[%u]",
            *opResult, devInfo.phyId, devInfo.devIndex, opData->txData.id),
        0);
    return 0;
}

int RaRsCtxQpDestroyBatch(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxQpDestroyBatchData *opDataOut = (union OpCtxQpDestroyBatchData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxQpDestroyBatchData *opData = (union OpCtxQpDestroyBatchData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpDestroyBatchData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, HCCP_MAX_QP_DESTROY_BATCH_NUM, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    opDataOut->rxData.num = opData->txData.num;
    *opResult = gRaRsCtxOps.ctxQpDestroyBatch(&devInfo, opData->txData.ids, &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[qp_batch][ra_rs_ctx]destroy failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsCtxQpImport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxQpImportData *opDataOut = (union OpCtxQpImportData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxQpImportData *opData = (union OpCtxQpImportData *)(inBuf + sizeof(struct MsgHead));
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importInfo = {0};
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpImportData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    importAttr.key = opData->txData.key;
    importAttr.attr = opData->txData.attr;
    *opResult = gRaRsCtxOps.ctxQpImport(&devInfo, &importAttr, &importInfo);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_qp]import failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    opDataOut->rxData.remJettyId = importInfo.remJettyId;
    opDataOut->rxData.info = importInfo.info;

    return 0;
}

int RaRsCtxQpUnimport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxQpUnimportData *opData = (union OpCtxQpUnimportData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpUnimportData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxQpUnimport(&devInfo, opData->txData.rawRemJettyId, REM_JETTY_ID_SIZE);
    if (*opResult != 0) {
        hccp_err("unimport failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId, devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxQpUnimportDeprecated(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxQpUnimportData *opData = (union OpCtxQpUnimportData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpUnimportData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxQpUnimport(&devInfo, opData->txData.rawRemJettyId, REM_JETTY_ID_DEPRECATED_SIZE);
    if (*opResult != 0) {
        hccp_err("unimport failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId, devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxQpBind(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxQpBindData *opData = (union OpCtxQpBindData *)(inBuf + sizeof(struct MsgHead));
    struct RsCtxQpInfo remoteQpInfo = {0};
    struct RsCtxQpInfo localQpInfo = {0};
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpBindData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    localQpInfo.id = opData->txData.id;
    localQpInfo.key = opData->txData.localQpKey;
    remoteQpInfo.id = opData->txData.remId;
    remoteQpInfo.key = opData->txData.remoteQpKey;
    *opResult = gRaRsCtxOps.ctxQpBind(&devInfo, &localQpInfo, &remoteQpInfo);
    if (*opResult != 0) {
        hccp_err("[init][ra_rs_qp]bind failed, ret[%d] phyId[%u] devIndex[0x%x] qpId[%u] rem_qp_id[%u]", *opResult,
            devInfo.phyId, devInfo.devIndex, opData->txData.id, opData->txData.remId);
    }

    return 0;
}

int RaRsCtxQpUnbind(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxQpUnbindData *opData = (union OpCtxQpUnbindData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxQpUnbindData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);

    *opResult = gRaRsCtxOps.ctxQpUnbind(&devInfo, opData->txData.id);
    CHK_PRT_RETURN(*opResult == -ENODEV,
        hccp_warn_rma("[deinit][ra_rs_qp]jetty not found, ret[%d] phyId[%u] "
                      "devIndex[0x%x] qpId[%u]",
            *opResult, devInfo.phyId, devInfo.devIndex, opData->txData.id),
        0);
    CHK_PRT_RETURN(*opResult != 0,
        hccp_err("[deinit][ra_rs_qp]unbind failed, ret[%d] phyId[%u] devIndex[0x%x] "
                 "qpId[%u]",
            *opResult, devInfo.phyId, devInfo.devIndex, opData->txData.id),
        0);
    return 0;
}

int RaRsCtxBatchSendWr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxBatchSendWrData *opData = (union OpCtxBatchSendWrData *)(inBuf + sizeof(struct MsgHead));
    union OpCtxBatchSendWrData *opDataOut = (union OpCtxBatchSendWrData *)(outBuf + sizeof(struct MsgHead));
    struct WrlistSendCompleteNum wrlistNum = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxBatchSendWrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    wrlistNum.sendNum = opData->txData.sendNum;
    wrlistNum.completeNum = &opDataOut->rxData.completeNum;
    *opResult = gRaRsCtxOps.ctxBatchSendWr(&opData->txData.baseInfo, opData->txData.wrData, opDataOut->rxData.wrResp,
        &wrlistNum);
    if (*opResult != 0) {
        hccp_err("[send][ra_rs_ctx]batch send wr failed, ret[%d] qpId[%u] sendNum[%d] completeNum[%d]", *opResult,
            opData->txData.baseInfo.qpn, wrlistNum.sendNum, *wrlistNum.completeNum);
    }

    return 0;
}

int RaRsCtxUpdateCi(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxUpdateCiData *opData = (union OpCtxUpdateCiData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxUpdateCiData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxUpdateCi(&devInfo, opData->txData.jettyId, opData->txData.ci);
    if (*opResult != 0) {
        hccp_err("[update_ci][ra_rs_ctx]update ci failed, ret[%d] phyId[%u] devIndex[0x%x] qpId[%u]", *opResult,
            devInfo.phyId, devInfo.devIndex, opData->txData.jettyId);
    }

    return 0;
}

int RaRsCtxGetAuxInfo(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxGetAuxInfoData *opDataOut = (union OpCtxGetAuxInfoData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxGetAuxInfoData *opData = (union OpCtxGetAuxInfoData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxGetAuxInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxGetAuxInfo(&devInfo, &opData->txData.info, &opDataOut->rxData.info);
    if (*opResult != 0) {
        hccp_err("[get_aux_info][ra_rs_ctx]get aux info failed, ret[%d] phyId[%u] devIndex[0x%x]", *opResult,
            devInfo.phyId, devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxGetCrErrInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxGetCrErrInfoListData *opDataOut = (union OpCtxGetCrErrInfoListData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxGetCrErrInfoListData *opData = (union OpCtxGetCrErrInfoListData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxGetCrErrInfoListData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.num, 0, CR_ERR_INFO_MAX_NUM, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    opDataOut->rxData.num = opData->txData.num;
    *opResult = gRaRsCtxOps.ctxGetCrErrInfoList(&devInfo, opDataOut->rxData.infoList, &opDataOut->rxData.num);
    if (*opResult != 0) {
        hccp_err("[get][cr_err]ctx_get_cr_err_info_list failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}

int RaRsCtxGetUbContext(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCtxGetContextData *opDataOut = (union OpCtxGetContextData *)(outBuf + sizeof(struct MsgHead));
    union OpCtxGetContextData *opData = (union OpCtxGetContextData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxGetContextData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    opDataOut->rxData.len = opData->txData.len;
    *opResult = gRaRsCtxOps.ctxGetUbContext(&devInfo, opData->txData.id, opData->txData.contextType,
        opDataOut->rxData.context, &opDataOut->rxData.len);
    if (*opResult != 0) {
        hccp_err("[get][jettyContext]ctxGetJettyContext failed, ret:%d, phyId:%u devIndex:0x%x", *opResult,
            devInfo.phyId, devInfo.devIndex);
    }

    return 0;
}

int RaRsCtxNotifyEvent(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpCtxNotifyEventData *opData = (union OpCtxNotifyEventData *)(inBuf + sizeof(struct MsgHead));
    struct RaRsDevInfo devInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCtxNotifyEventData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RaRsSetDevInfo(&devInfo, opData->txData.phyId, opData->txData.devIndex);
    *opResult = gRaRsCtxOps.ctxNotifyEvent(&devInfo, &opData->txData.event);
    if (*opResult != 0) {
        hccp_err("[notify][event]ctxNotifyEvent failed, ret[%d], phyId[%u] devIndex[0x%x]", *opResult, devInfo.phyId,
            devInfo.devIndex);
    }

    return 0;
}
