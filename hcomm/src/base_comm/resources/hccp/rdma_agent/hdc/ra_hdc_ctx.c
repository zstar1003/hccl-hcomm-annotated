/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "hccp_ctx.h"
#include "ra.h"
#include "ra_ctx.h"
#include "ra_ctx_comm.h"
#include "ra_rs_err.h"
#include "ra_hdc.h"
#include "ra_hdc_ctx.h"

int RaHdcGetDevEidInfoNum(struct RaInfo info, unsigned int *num)
{
    union OpGetDevEidInfoNumData opData = {0};
    int ret;

    *num = 0;
    opData.txData.phyId = info.phyId;
    ret = RaHdcProcessMsg(RA_RS_GET_DEV_EID_INFO_NUM, info.phyId, (char *)&opData,
        sizeof(union OpGetDevEidInfoNumData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[get][eid]hdc message process failed ret[%d], phyId[%u]", ret, info.phyId), ret);

    *num = opData.rxData.num;
    return 0;
}

STATIC int RaHdcGetDevEidSubInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex,
    unsigned int count)
{
    union OpGetDevEidInfoListData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.startIndex = startIndex;
    opData.txData.count = count;
    ret = RaHdcProcessMsg(RA_RS_GET_DEV_EID_INFO_LIST, phyId, (char *)&opData, sizeof(union OpGetDevEidInfoListData));
    CHK_PRT_RETURN(ret, hccp_err("[get][eid]hdc message process failed ret[%d], phyId[%u]", ret, phyId), ret);

    (void)memcpy_s(infoList, sizeof(struct HccpDevEidInfo) * MAX_DEV_INFO_TRANS_NUM, opData.rxData.infoList,
        sizeof(struct HccpDevEidInfo) * MAX_DEV_INFO_TRANS_NUM);
    return 0;
}

int RaHdcGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int *num)
{
    struct HccpDevEidInfo subInfoList[MAX_DEV_INFO_TRANS_NUM] = {0};
    unsigned int infoSize = *num;
    unsigned int remainCount = 0;
    unsigned int startIndex = 0;
    int ret = 0;

    *num = 0;
    // get MAX_DEV_INFO_TRANS_NUM num of eid info every time: will fallthrough here
    for (startIndex = 0; startIndex + MAX_DEV_INFO_TRANS_NUM <= infoSize; startIndex += MAX_DEV_INFO_TRANS_NUM) {
        ret = RaHdcGetDevEidSubInfoList(phyId, subInfoList, startIndex, MAX_DEV_INFO_TRANS_NUM);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[get][eid]get sub_info_list failed, ret(%d) phyId(%u) "
                     "startIndex(%u) count(%d)",
                ret, phyId, startIndex, MAX_DEV_INFO_TRANS_NUM),
            ret);

        *num += MAX_DEV_INFO_TRANS_NUM;
        (void)memcpy_s(infoList + startIndex, sizeof(struct HccpDevEidInfo) * MAX_DEV_INFO_TRANS_NUM, subInfoList,
            sizeof(struct HccpDevEidInfo) * MAX_DEV_INFO_TRANS_NUM);
    }

    remainCount = infoSize % MAX_DEV_INFO_TRANS_NUM;
    if (remainCount == 0) {
        return ret;
    }

    // get remain count of eid info
    ret = RaHdcGetDevEidSubInfoList(phyId, subInfoList, startIndex, remainCount);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][eid]get sub_info_list failed, ret(%d) phyId(%u) "
                 "startIndex(%u) remainCount(%u)",
            ret, phyId, startIndex, remainCount),
        ret);

    *num += remainCount;
    (void)memcpy_s(infoList + startIndex, sizeof(struct HccpDevEidInfo) * remainCount, subInfoList,
        sizeof(struct HccpDevEidInfo) * remainCount);
    return ret;
}

int RaHdcCtxInit(struct RaCtxHandle *ctxHandle, struct CtxInitAttr *attr, unsigned int *devIndex,
    struct DevBaseAttr *devAttr)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxInitData opData = {0};
    int ret;

    (void)memcpy_s(&(opData.txData.attr), sizeof(struct CtxInitAttr), attr, sizeof(struct CtxInitAttr));
    ret = RaHdcProcessMsg(RA_RS_CTX_INIT, attr->phyId, (char *)&opData, sizeof(union OpCtxInitData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_hdc_ctx]hdc message process failed ret[%d], phyId[%u]", ret, phyId),
        ret);

    *devIndex = opData.rxData.devIndex;
    (void)memcpy_s(devAttr, sizeof(struct DevBaseAttr), &opData.rxData.devAttr, sizeof(struct DevBaseAttr));

    return 0;
}

int RaHdcCtxGetAsyncEvents(struct RaCtxHandle *ctxHandle, struct AsyncEvent events[], unsigned int *num)
{
    union OpCtxGetAsyncEventsData opData = {0};
    unsigned int phyId = ctxHandle->attr.phyId;
    unsigned int expectedNum = *num;
    unsigned int i;
    int ret = 0;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.num = *num;
    ret = RaHdcProcessMsg(RA_RS_CTX_GET_ASYNC_EVENTS, phyId, (char *)&opData, sizeof(union OpCtxGetAsyncEventsData));

    CHK_PRT_RETURN(opData.rxData.num > expectedNum,
        hccp_err("[get][async_events]rxData.num:%u > expectedNum:%u,"
                 " phyId:%u dev_index:0x%x",
            opData.rxData.num, expectedNum, phyId, ctxHandle->devIndex),
        -EINVAL);

    for (i = 0; i < opData.rxData.num; i++) {
        (void)memcpy_s(&events[i], sizeof(struct AsyncEvent), &opData.rxData.events[i], sizeof(struct AsyncEvent));
    }
    *num = opData.rxData.num;

    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][async_events]hdc message process failed ret:%d, phyId:%u"
                 " dev_index:0x%x",
            ret, phyId, ctxHandle->devIndex),
        ret);

    return ret;
}

int RaHdcCtxDeinit(struct RaCtxHandle *ctxHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxDeinitData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    ret = RaHdcProcessMsg(RA_RS_CTX_DEINIT, phyId, (char *)&opData, sizeof(union OpCtxDeinitData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ra_hdc_ctx]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    return 0;
}

void RaHdcPrepareGetEidByIp(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], unsigned int ipNum,
    union OpGetEidByIpData *opData)
{
    unsigned int i = 0;

    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.devIndex = ctxHandle->devIndex;
    opData->txData.num = ipNum;
    for (i = 0; i < ipNum; i++) {
        (void)memcpy_s(&opData->txData.ip[i], sizeof(struct IpInfo), &ip[i], sizeof(struct IpInfo));
    }
}

int RaHdcGetEidResults(union OpGetEidByIpData *opData, unsigned int ipNum, union HccpEid eid[], unsigned int *num)
{
    unsigned int i = 0;

    *num = 0;
    CHK_PRT_RETURN(opData->rxData.num > ipNum,
        hccp_err("[get][eid_by_ip]rx_data.num:%u > ip_num:%u", opData->rxData.num, ipNum), -EINVAL);

    *num = opData->rxData.num;
    for (i = 0; i < opData->rxData.num; i++) {
        (void)memcpy_s(&eid[i], sizeof(union HccpEid), &opData->rxData.eid[i], sizeof(union HccpEid));
    }

    return 0;
}

int RaHdcGetEidByIp(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpGetEidByIpData opData = {0};
    unsigned int ipNum = *num;
    int retTmp = 0;
    int ret = 0;

    RaHdcPrepareGetEidByIp(ctxHandle, ip, ipNum, &opData);
    ret = RaHdcProcessMsg(RA_RS_GET_EID_BY_IP, phyId, (char *)&opData, sizeof(union OpGetEidByIpData));

    retTmp = RaHdcGetEidResults(&opData, ipNum, eid, num);
    CHK_PRT_RETURN(retTmp != 0,
        hccp_err("[get][eid_by_ip]ra_hdc_get_eid_results failed ret[%d], phyId[%u]"
                 " devIndex[0x%x]",
            retTmp, phyId, ctxHandle->devIndex),
        retTmp);

    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][eid_by_ip]hdc message process failed ret[%d], phyId[%u]"
                 " devIndex[0x%x]",
            ret, phyId, ctxHandle->devIndex),
        ret);

    return ret;
}

void RaHdcPrepareGetIpByEid(struct RaCtxHandle *ctxHandle, union HccpEid eid[], unsigned int eidNum,
    union OpGetIpByEidData *opData)
{
    unsigned int i = 0;

    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.devIndex = ctxHandle->devIndex;
    opData->txData.num = eidNum;
    for (i = 0; i < eidNum; i++) {
        (void)memcpy_s(&opData->txData.eid[i], sizeof(union HccpEid), &eid[i], sizeof(union HccpEid));
    }
}

int RaHdcGetIpResults(union OpGetIpByEidData *opData, unsigned int eidNum, struct IpInfo ip[], unsigned int *num)
{
    unsigned int i = 0;

    *num = 0;
    CHK_PRT_RETURN(opData->rxData.num > eidNum,
        hccp_err("[get][IpByEid]rxData.num:%u > eidNum:%u", opData->rxData.num, eidNum), -EINVAL);

    *num = opData->rxData.num;
    for (i = 0; i < opData->rxData.num; i++) {
        (void)memcpy_s(&ip[i], sizeof(struct IpInfo), &opData->rxData.ip[i], sizeof(struct IpInfo));
    }

    return 0;
}

int RaHdcGetIpByEid(struct RaCtxHandle *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpGetIpByEidData opData = {0};
    unsigned int eidNum = *num;
    int retTmp = 0;
    int ret = 0;

    RaHdcPrepareGetIpByEid(ctxHandle, eid, eidNum, &opData);
    ret = RaHdcProcessMsg(RA_RS_GET_IP_BY_EID, phyId, (char *)&opData, sizeof(union OpGetIpByEidData));

    retTmp = RaHdcGetIpResults(&opData, eidNum, ip, num);
    CHK_PRT_RETURN(retTmp != 0,
        hccp_err("[get][IpByEid]RaHdcGetIpResults failed ret[%d], phyId[%u]"
                 " devIndex[0x%x]",
            retTmp, phyId, ctxHandle->devIndex),
        retTmp);

    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][IpByEid]hdc message process failed ret[%d], phyId[%u]"
                 " devIndex[0x%x]",
            ret, phyId, ctxHandle->devIndex),
        ret);

    return ret;
}

int RaHdcCtxTokenIdAlloc(struct RaCtxHandle *ctxHandle, struct HccpTokenId *info, struct RaTokenIdHandle *tokenIdHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpTokenIdAllocData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;

    ret = RaHdcProcessMsg(RA_RS_CTX_TOKEN_ID_ALLOC, phyId, (char *)&opData, sizeof(union OpTokenIdAllocData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_token_id]hdc message process failed ret[%d], phyId[%u] "
                 "devIndex[%u]",
            ret, phyId, ctxHandle->devIndex),
        ret);

    info->tokenId = opData.rxData.tokenId;
    tokenIdHandle->addr = opData.rxData.addr;
    return 0;
}

int RaHdcCtxTokenIdFree(struct RaCtxHandle *ctxHandle, struct RaTokenIdHandle *tokenIdHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpTokenIdFreeData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.addr = tokenIdHandle->addr;
    ret = RaHdcProcessMsg(RA_RS_CTX_TOKEN_ID_FREE, phyId, (char *)&opData, sizeof(union OpTokenIdFreeData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_token_id]hdc message process failed ret[%d], phyId[%u] "
                 "devIndex[%u]",
            ret, phyId, ctxHandle->devIndex),
        ret);

    return 0;
}

int RaHdcCtxPrepareLmemRegister(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo,
    union OpLmemRegInfoData *opData)
{
    struct MemRegAttrT *memAttr = NULL;
    int ret = 0;

    memAttr = &(opData->txData.memAttr);
    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.devIndex = ctxHandle->devIndex;
    ret = RaCtxPrepareLmemRegister(lmemInfo, memAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_hdc_lmem]ra_ctx_prepare_lmem_register failed, ret[%d]", ret), ret);

    return 0;
}

int RaHdcCtxLmemRegister(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo, struct RaLmemHandle *lmemHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpLmemRegInfoData opData = {0};
    int ret;

    ret = RaHdcCtxPrepareLmemRegister(ctxHandle, lmemInfo, &opData);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_lmem]prepare register failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    ret = RaHdcProcessMsg(RA_RS_LMEM_REG, phyId, (char *)&opData, sizeof(union OpLmemRegInfoData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_lmem]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    RaCtxGetLmemInfo(&(opData.rxData.memInfo), lmemInfo, lmemHandle);

    return 0;
}

int RaHdcCtxLmemUnregister(struct RaCtxHandle *ctxHandle, struct RaLmemHandle *lmemHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpLmemUnregInfoData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.addr = lmemHandle->addr;
    ret = RaHdcProcessMsg(RA_RS_LMEM_UNREG, phyId, (char *)&opData, sizeof(union OpLmemUnregInfoData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ra_hdc_lmem]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    return 0;
}

STATIC void RaHdcPrepareRmemImport(struct RaCtxHandle *ctxHandle, union OpRmemImportInfoData *opData,
    struct MrImportInfoT *rmemInfo)
{
    struct MemImportAttrT *memAttr = NULL;

    memAttr = &(opData->txData.memAttr);
    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.devIndex = ctxHandle->devIndex;
    RaCtxPrepareRmemImport(rmemInfo, memAttr);
}

int RaHdcCtxRmemImport(struct RaCtxHandle *ctxHandle, struct MrImportInfoT *rmemInfo)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpRmemImportInfoData opData = {0};
    int ret;

    RaHdcPrepareRmemImport(ctxHandle, &opData, rmemInfo);

    ret = RaHdcProcessMsg(RA_RS_RMEM_IMPORT, phyId, (char *)&opData, sizeof(union OpRmemImportInfoData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_rmem]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    rmemInfo->out.ub.targetSegHandle = opData.rxData.memInfo.ub.targetSegHandle;
    return 0;
}

int RaHdcCtxRmemUnimport(struct RaCtxHandle *ctxHandle, struct RaRmemHandle *rmemHandle)
{
    union OpRmemUnimportInfoData opData = {0};
    unsigned int phyId = ctxHandle->attr.phyId;
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.addr = rmemHandle->addr;
    ret = RaHdcProcessMsg(RA_RS_RMEM_UNIMPORT, phyId, (char *)&opData, sizeof(union OpRmemUnimportInfoData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ra_hdc_rmem]hdc message process failed ret[%d], phyId[%u], devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    return 0;
}

int RaHdcCtxChanCreate(struct RaCtxHandle *ctxHandle, struct ChanInfoT *chanInfo, struct RaChanHandle *chanHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxChanCreateData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.dataPlaneFlag = chanInfo->in.dataPlaneFlag;
    ret = RaHdcProcessMsg(RA_RS_CTX_CHAN_CREATE, phyId, (char *)&opData, sizeof(union OpCtxChanCreateData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ctx_chan]hdc message process failed ret[%d], phyId[%u], devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    chanHandle->addr = opData.rxData.addr;
    chanInfo->out.fd = opData.rxData.fd;
    return 0;
}

int RaHdcCtxChanDestroy(struct RaCtxHandle *ctxHandle, struct RaChanHandle *chanHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxChanDestroyData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.addr = chanHandle->addr;

    ret = RaHdcProcessMsg(RA_RS_CTX_CHAN_DESTROY, phyId, (char *)&opData, sizeof(union OpCtxChanDestroyData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ctx_chan]hdc message process failed ret[%d], phyId[%u], devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    return 0;
}

STATIC void RaHdcPrepareCqCreate(struct RaCtxHandle *ctxHandle, union OpCtxCqCreateData *opData, struct CqInfoT *info)
{
    struct CtxCqAttr *cqAttr = NULL;

    cqAttr = &(opData->txData.attr);
    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.devIndex = ctxHandle->devIndex;
    RaCtxPrepareCqCreate(info, cqAttr);
    if (opData->txData.attr.ub.mode == JFC_MODE_CCU_POLL && info->in.ub.ccuExCfg.valid) {
        opData->txData.attr.ub.ccuExCfg.valid = info->in.ub.ccuExCfg.valid;
        opData->txData.attr.ub.ccuExCfg.cqeFlag = info->in.ub.ccuExCfg.cqeFlag;
    }
}

int RaHdcCtxCqCreate(struct RaCtxHandle *ctxHandle, struct CqInfoT *info, struct RaCqHandle *cqHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxCqCreateData opData = {0};
    int ret;

    RaHdcPrepareCqCreate(ctxHandle, &opData, info);

    ret = RaHdcProcessMsg(RA_RS_CTX_CQ_CREATE, phyId, (char *)&opData, sizeof(union OpCtxCqCreateData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_cq]hdc message process failed ret[%d], phyId[%u], devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    cqHandle->addr = opData.rxData.info.addr;
    RaCtxGetCqCreateInfo(&opData.rxData.info, info);
    return 0;
}

int RaHdcCtxCqDestroy(struct RaCtxHandle *ctxHandle, struct RaCqHandle *cqHandle)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxCqDestroyData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.addr = cqHandle->addr;
    ret = RaHdcProcessMsg(RA_RS_CTX_CQ_DESTROY, phyId, (char *)&opData, sizeof(union OpCtxCqDestroyData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ra_cq]hdc message process failed ret[%d], phyId[%u], devIndex[%u]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    return 0;
}

int RaHdcCtxPrepareQpCreate(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, union OpCtxQpCreateData *opData)
{
    struct CtxQpAttr *ctxQpAttr = NULL;
    int ret = 0;

    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.devIndex = ctxHandle->devIndex;
    ctxQpAttr = &opData->txData.qpAttr;
    ret = RaCtxPrepareQpCreate(qpAttr, ctxQpAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_hdc_qp]ra_ctx_prepare_qp_create failed, ret[%d]", ret), ret);

    if (ctxQpAttr->ub.mode == JETTY_MODE_CCU_TA_CACHE) {
        ctxQpAttr->ub.taCacheMode.lockFlag = qpAttr->ub.taCacheMode.lockFlag;
        ctxQpAttr->ub.taCacheMode.sqeBufIdx = qpAttr->ub.taCacheMode.sqeBufIdx;
    } else {
        ctxQpAttr->ub.extMode.sq = qpAttr->ub.extMode.sq;
        ctxQpAttr->ub.extMode.piType = qpAttr->ub.extMode.piType;
        ctxQpAttr->ub.extMode.cstmFlag = qpAttr->ub.extMode.cstmFlag;
        ctxQpAttr->ub.extMode.sqebbNum = qpAttr->ub.extMode.sqebbNum;
    }

    return 0;
}

int RaHdcCtxQpCreate(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
    struct RaCtxQpHandle *qpHandle)
{
    unsigned int devIndex = ctxHandle->devIndex;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxQpCreateData opData = {0};
    int ret;

    ret = RaHdcCtxPrepareQpCreate(ctxHandle, qpAttr, &opData);
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_qp]prepare qp_create failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId, devIndex),
        ret);

    ret = RaHdcProcessMsg(RA_RS_CTX_QP_CREATE, phyId, (char *)&opData, sizeof(union OpCtxQpCreateData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId, devIndex),
        ret);

    RaCtxGetQpCreateInfo(ctxHandle, qpAttr, &(opData.rxData.qpInfo), qpHandle);
    (void)memcpy_s(qpInfo, sizeof(struct QpCreateInfo), &(opData.rxData.qpInfo), sizeof(struct QpCreateInfo));

    return 0;
}

int RaHdcCtxQpDestroy(struct RaCtxQpHandle *qpHandle)
{
    unsigned int phyId = qpHandle->phyId;
    union OpCtxQpDestroyData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = qpHandle->devIndex;
    opData.txData.id = qpHandle->id;

    ret = RaHdcProcessMsg(RA_RS_CTX_QP_DESTROY, phyId, (char *)&opData, sizeof(union OpCtxQpDestroyData));
    CHK_PRT_RETURN(ret == -ENODEV,
        hccp_warn_rma("[deinit][ra_hdc_qp]hdc message process ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            qpHandle->devIndex),
        ret);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_hdc_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            qpHandle->devIndex),
        ret);
    return 0;
}

int RaHdcCtxPrepareQpImport(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo,
    union OpCtxQpImportData *opData)
{
    struct RaRsJettyImportAttr *importAttr = NULL;

    importAttr = &(opData->txData.attr);
    opData->txData.devIndex = ctxHandle->devIndex;
    opData->txData.phyId = ctxHandle->attr.phyId;
    opData->txData.key = qpInfo->in.key;
    RaCtxPrepareQpImport(qpInfo, importAttr);

    return 0;
}

int RaHdcCtxQpImport(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo, struct RaCtxRemQpHandle *remQpHandle)
{
    unsigned int devIndex = ctxHandle->devIndex;
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxQpImportData opData = {0};
    int ret;

    ret = RaHdcCtxPrepareQpImport(ctxHandle, qpInfo, &opData);
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_qp]prepare qp_import failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId, devIndex),
        ret);

    ret = RaHdcProcessMsg(RA_RS_CTX_QP_IMPORT, phyId, (char *)&opData, sizeof(union OpCtxQpImportData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId, devIndex),
        ret);

    RaCtxGetQpImportInfo(ctxHandle, qpInfo, &(opData.rxData.info), remQpHandle);
    remQpHandle->id = opData.rxData.remJettyId;

    return 0;
}

int RaHdcCtxQpUnimport(struct RaCtxRemQpHandle *remQpHandle)
{
    unsigned int phyId = remQpHandle->phyId;
    union OpCtxQpUnimportData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = remQpHandle->devIndex;
    (void)memcpy_s(opData.txData.rawRemJettyId, REM_JETTY_ID_SIZE, remQpHandle->qpKey.value, REM_JETTY_ID_SIZE);
    ret = RaHdcProcessMsg(RA_RS_CTX_QP_UNIMPORT, phyId, (char *)&opData, sizeof(union OpCtxQpUnimportData));
    CHK_PRT_RETURN(ret,
        hccp_err("[deinit][ra_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            remQpHandle->devIndex),
        ret);

    return 0;
}

int RaHdcCtxQpBind(struct RaCtxQpHandle *qpHandle, struct RaCtxRemQpHandle *remQpHandle)
{
    unsigned int devIndex = qpHandle->devIndex;
    unsigned int phyId = qpHandle->phyId;
    union OpCtxQpBindData opData = {0};
    int ret;

    opData.txData.phyId = qpHandle->phyId;
    opData.txData.devIndex = qpHandle->devIndex;
    opData.txData.id = qpHandle->id;
    opData.txData.remId = remQpHandle->id;

    ret = RaHdcProcessMsg(RA_RS_CTX_QP_BIND, phyId, (char *)&opData, sizeof(union OpCtxQpBindData));
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId, devIndex),
        ret);

    return 0;
}

int RaHdcCtxQpUnbind(struct RaCtxQpHandle *qpHandle)
{
    union OpCtxQpUnbindData opData = {0};
    unsigned int phyId = qpHandle->phyId;
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = qpHandle->devIndex;
    opData.txData.id = qpHandle->id;

    ret = RaHdcProcessMsg(RA_RS_CTX_QP_UNBIND, phyId, (char *)&opData, sizeof(union OpCtxQpUnbindData));
    CHK_PRT_RETURN(ret == -ENODEV,
        hccp_warn_rma("[deinit][ra_qp]hdc message process ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            qpHandle->devIndex),
        ret);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            qpHandle->devIndex),
        ret);
    return 0;
}

STATIC int RaHdcSendWrDataProtocolInit(struct SendWrData *wr, struct BatchSendWrData *wrData,
    enum ProtocolTypeT protocol, bool *isInline)
{
    struct RaCtxRemQpHandle *remQpHandle = NULL;
    struct RaRmemHandle *rmemHandle = NULL;
    int ret;

    if (protocol == PROTOCOL_UDMA) {
        *isInline = (wr->ub.flags.bs.inlineFlag != 0);
        wrData->ub.userCtx = wr->ub.userCtx;
        wrData->ub.opcode = wr->ub.opcode;
        wrData->ub.flags = wr->ub.flags;
        /* nop no need to init other infos */
        if (wr->ub.opcode == RA_UB_OPC_NOP) {
            return 0;
        }
        remQpHandle = (struct RaCtxRemQpHandle *)wr->ub.remQpHandle;
        wrData->ub.remJetty = remQpHandle->id;
        /* notify */
        if (wr->ub.opcode == RA_UB_OPC_WRITE_NOTIFY) {
            wrData->ub.notifyInfo.notifyAddr = wr->ub.notifyInfo.notifyAddr;
            wrData->ub.notifyInfo.notifyData = wr->ub.notifyInfo.notifyData;
            rmemHandle = (struct RaRmemHandle *)(wr->ub.notifyInfo.notifyHandle);
            wrData->ub.notifyInfo.notifyHandle = rmemHandle->addr;
        }
    } else {
        *isInline = ((wr->rdma.flags & RA_SEND_INLINE) != 0);
        ret = memcpy_s(&wrData->rdma, sizeof(wrData->rdma), &(wr->rdma), sizeof(wr->rdma));
        CHK_PRT_RETURN(ret, hccp_err("[send][ra_hdc_ctx]memcpy_s protocol failed, ret[%d]", ret), -ESAFEFUNC);
    }

    return 0;
}

STATIC int RaHdcSendWrDataInit(struct RaCtxQpHandle *qpHandle, struct SendWrData wrList[],
    union OpCtxBatchSendWrData *opData, unsigned int completeCnt, unsigned int sendNum)
{
    unsigned int currBatchNum = (sendNum - completeCnt) >= MAX_CTX_WR_NUM ? MAX_CTX_WR_NUM : (sendNum - completeCnt);
    struct RaLmemHandle *lmemHandle = NULL;
    struct RaRmemHandle *rmemHandle = NULL;
    struct BatchSendWrData *hdcWr = NULL;
    struct SendWrData *wr = NULL;
    unsigned int i, j;
    bool isInline;
    int ret;

    (void)memset_s(opData, sizeof(union OpCtxBatchSendWrData), 0, sizeof(union OpCtxBatchSendWrData));
    opData->txData.baseInfo.phyId = qpHandle->phyId;
    opData->txData.baseInfo.devIndex = qpHandle->devIndex;
    opData->txData.baseInfo.qpn = qpHandle->id;
    opData->txData.sendNum = currBatchNum;

    for (i = 0; i < currBatchNum; i++) {
        wr = &wrList[completeCnt + i];
        hdcWr = &opData->txData.wrData[i];
        /* protocol cfg */
        ret = RaHdcSendWrDataProtocolInit(wr, hdcWr, qpHandle->protocol, &isInline);
        if (ret != 0) {
            hccp_err("[send][ra_hdc_ctx]init protocol cfg failed, ret[%d], wr[%u]", ret, (completeCnt + i));
            return ret;
        }

        /* nop no need init other infos */
        if (qpHandle->protocol == PROTOCOL_UDMA && wr->ub.opcode == RA_UB_OPC_NOP) {
            continue;
        }

        /* lmem */
        if (isInline) {
            ret = memcpy_s(hdcWr->inlineData, MAX_INLINE_SIZE, wr->inlineData, wr->inlineSize);
            CHK_PRT_RETURN(ret, hccp_err("[send][ra_hdc_ctx]memcpy_s inline data failed, ret[%d]", ret), -ESAFEFUNC);
            hdcWr->inlineSize = wr->inlineSize;
        } else {
            for (j = 0; j < wr->numSge; j++) {
                hdcWr->sges[j].addr = wr->sges[j].addr;
                hdcWr->sges[j].len = wr->sges[j].len;
                lmemHandle = (struct RaLmemHandle *)wr->sges[j].lmemHandle;
                hdcWr->sges[j].devLmemHandle = lmemHandle->addr;
            }
            hdcWr->numSge = wr->numSge;
        }

        /* rmem */
        hdcWr->remoteAddr = wr->remoteAddr;
        rmemHandle = (struct RaRmemHandle *)(wr->rmemHandle);
        hdcWr->devRmemHandle = rmemHandle->addr;

        /* inline reduce */
        hdcWr->ub.reduceInfo = wr->ub.reduceInfo;

        /* imm data */
        hdcWr->immData = wr->immData;
    }

    return 0;
}

int RaHdcCtxBatchSendWr(struct RaCtxQpHandle *qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[],
    unsigned int sendNum, unsigned int *completeNum)
{
    union OpCtxBatchSendWrData opData = {0};
    unsigned int completeCnt = 0;
    unsigned int currSendNum;
    bool isFinished = false;
    int ret = 0;

    while (completeCnt < sendNum) {
        ret = RaHdcSendWrDataInit(qpHandle, wrList, &opData, completeCnt, sendNum);
        if (ret != 0) {
            hccp_err("[send][ra_hdc_ctx]ra_hdc_send_wr_data_init failed, ret[%d] phyId[%u] devIndex[%u] qp_id[%u]", ret,
                qpHandle->phyId, qpHandle->devIndex, qpHandle->id);
            break;
        }

        currSendNum = opData.txData.sendNum;
        ret = RaHdcProcessMsg(RA_RS_CTX_BATCH_SEND_WR, qpHandle->phyId, (char *)&opData,
            sizeof(union OpCtxBatchSendWrData));

        if (opData.rxData.completeNum > currSendNum) {
            hccp_err("[send][ra_hdc_ctx]complete_num[%u] is larger than send_num[%u], ret[%d]",
                opData.rxData.completeNum, currSendNum, ret);
            ret = -EINVAL;
            break;
        }
        if (ret != 0 || opData.rxData.completeNum < currSendNum) {
            hccp_err("[send][ra_hdc_ctx]batch send wr failed, ret[%d], sendNum[%u], completeNum[%u]", ret, currSendNum,
                opData.rxData.completeNum);
            ret = -EOPENSRC;
            isFinished = true;
        }

        ret = memcpy_s(&opResp[completeCnt], (sizeof(struct SendWrResp) * (sendNum - completeCnt)),
            opData.rxData.wrResp, (sizeof(struct SendWrResp) * opData.rxData.completeNum));
        if (ret != 0) {
            hccp_err("[send][ra_hdc_ctx]memcpy_s wr_resp failed, ret[%d]", ret);
            break;
        }

        completeCnt = completeCnt + opData.rxData.completeNum;
        if (isFinished) {
            break;
        }
    }

    *completeNum = completeCnt;
    return ret;
}

int RaHdcCtxUpdateCi(struct RaCtxQpHandle *qpHandle, uint16_t ci)
{
    unsigned int phyId = qpHandle->phyId;
    union OpCtxUpdateCiData opData = {0};
    int ret;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = qpHandle->devIndex;
    opData.txData.jettyId = qpHandle->id;
    opData.txData.ci = ci;

    ret = RaHdcProcessMsg(RA_RS_CTX_UPDATE_CI, phyId, (char *)&opData, sizeof(union OpCtxUpdateCiData));
    CHK_PRT_RETURN(ret,
        hccp_err("[update][ra_qp]hdc message process failed ret[%d], phyId[%u] devIndex[%u]", ret, phyId,
            qpHandle->devIndex),
        ret);

    return 0;
}

int RaHdcCtxQpQueryBatch(unsigned int phyId, unsigned int devIndex, unsigned int ids[], struct JettyAttr attr[],
    unsigned int *num)
{
    union OpCtxQpQueryBatchData opData = {0};
    int ret, retTmp;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = devIndex;
    opData.txData.num = *num;
    (void)memcpy_s(opData.txData.ids, sizeof(unsigned int) * (*num), ids, sizeof(unsigned int) * (*num));
    ret = RaHdcProcessMsg(RA_RS_CTX_QUERY_QP_BATCH, phyId, (char *)&opData, sizeof(union OpCtxQpQueryBatchData));
    CHK_PRT_RETURN(opData.rxData.num > *num,
        hccp_err("[query][ra_qp]op_data.rx_data.num[%u] is larger than num[%u], "
                 "ret[%d]",
            opData.rxData.num, *num, ret),
        ret);

    if (ret != 0 || opData.rxData.num < *num) {
        hccp_err("[query][ra_qp]hdc message process failed ret[%d], phyId[%u], num[%u], opData.rxData.num[%u]", ret,
            phyId, *num, opData.rxData.num);
        ret = -EOPENSRC;
    }

    retTmp = memcpy_s(attr, sizeof(struct JettyAttr) * (*num), opData.rxData.attr,
        sizeof(struct JettyAttr) * opData.rxData.num);
    CHK_PRT_RETURN(retTmp != 0,
        hccp_err("[query][ra_qp]memcpy_s failed, ret[%d], phyId[%u], num[%u],"
                 "opData.rxData.num[%u]",
            ret, phyId, *num, opData.rxData.num),
        ret);

    *num = opData.rxData.num;
    return ret;
}

int RaHdcCtxGetAuxInfo(struct RaCtxHandle *ctxHandle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxGetAuxInfoData opData = {0};
    int ret = 0;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    (void)memcpy_s(&(opData.txData.info), sizeof(struct HccpAuxInfoIn), in, sizeof(struct HccpAuxInfoIn));
    ret = RaHdcProcessMsg(RA_RS_CTX_GET_AUX_INFO, phyId, (char *)&opData, sizeof(union OpCtxGetAuxInfoData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][aux_info]hdc message process failed ret[%d], phyId[%u] "
                 "devIndex[0x%x]",
            ret, phyId, ctxHandle->devIndex),
        ret);

    (void)memcpy_s(&(out->auxInfoType), sizeof(unsigned int) * AUX_INFO_NUM_MAX, &(opData.rxData.info.auxInfoType),
        sizeof(unsigned int) * AUX_INFO_NUM_MAX);
    (void)memcpy_s(&(out->auxInfoValue), sizeof(unsigned int) * AUX_INFO_NUM_MAX, &(opData.rxData.info.auxInfoValue),
        sizeof(unsigned int) * AUX_INFO_NUM_MAX);
    out->auxInfoNum = opData.rxData.info.auxInfoNum;
    return ret;
}

int RaHdcCtxGetCrErrInfoList(struct RaCtxHandle *ctxHandle, struct CrErrInfo *infoList, unsigned int *num)
{
    union OpCtxGetCrErrInfoListData opData = {0};
    unsigned int phyId = ctxHandle->attr.phyId;
    unsigned int expectedNum = *num;
    unsigned int i;
    int ret = 0;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    opData.txData.num = *num;
    ret = RaHdcProcessMsg(RA_RS_CTX_GET_CR_ERR_INFO_LIST, phyId, (char *)&opData,
        sizeof(union OpCtxGetCrErrInfoListData));

    if (opData.rxData.num > expectedNum) {
        hccp_err("[get][cr_err_info_list]rx_data.num(%u) > expected_num(%u), phyId(%u) devIndex(0x%x)",
            opData.rxData.num, expectedNum, phyId, ctxHandle->devIndex);
        return -EINVAL;
    }

    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][cr_err_info_list]hdc message process failed ret[%d], phyId[%u]"
                 " devIndex[0x%x]",
            ret, phyId, ctxHandle->devIndex),
        ret);

    for (i = 0; i < opData.rxData.num; i++) {
        (void)memcpy_s(&infoList[i], sizeof(struct CrErrInfo), &opData.rxData.infoList[i], sizeof(struct CrErrInfo));
    }
    *num = opData.rxData.num;

    return ret;
}

int RaHdcCtxNotifyEvent(struct RaCtxHandle *ctxHandle, struct CtxNotifyEvent *event)
{
    unsigned int phyId = ctxHandle->attr.phyId;
    union OpCtxNotifyEventData opData = {0};
    int ret = 0;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = ctxHandle->devIndex;
    (void)memcpy_s(&opData.txData.event, sizeof(struct CtxNotifyEvent), event, sizeof(struct CtxNotifyEvent));
    ret = RaHdcProcessMsg(RA_RS_CTX_NOTIFY_EVENT, phyId, (char *)&opData, sizeof(union OpCtxNotifyEventData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[notify][event]hdc message process failed ret[%d], phyId[%u] devIndex[0x%x]", ret, phyId,
            ctxHandle->devIndex),
        ret);

    return ret;
}

int RaHdcCtxGetJettyContext(struct RaCtxQpHandle *qpHandle, uint8_t context[], unsigned int *len)
{
    union OpCtxGetContextData opData = {0};
    unsigned int phyId = qpHandle->phyId;
    int ret = 0;

    opData.txData.phyId = phyId;
    opData.txData.devIndex = qpHandle->devIndex;
    opData.txData.id = qpHandle->id;
    opData.txData.contextType = CONTEXT_TYPE_JETTY;
    opData.txData.len = *len;
    ret = RaHdcProcessMsg(RA_RS_CTX_GET_UB_CONTEXT, phyId, (char *)&opData, sizeof(union OpCtxGetContextData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][jettyContext]hdc message process failed ret:%d, phyId:%u"
                 " devIndex:0x%x",
            ret, phyId, qpHandle->devIndex),
        ret);

    ret = memcpy_s(context, *len, opData.rxData.context, opData.rxData.len);
    CHK_PRT_RETURN(ret != 0, hccp_err("[get][jettyContext]memcpy_s for context failed, ret:%d", ret), -ESAFEFUNC);
    *len = opData.rxData.len;
    return ret;
}
