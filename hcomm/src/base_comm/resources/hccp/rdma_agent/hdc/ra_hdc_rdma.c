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
#include "user_log.h"
#include "config_log.h"
#include "ra_hdc.h"
#include "securec.h"
#include "ra.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "dl_hal_function.h"
#include "ra_rdma_lite.h"
#include "ra_hdc_lite.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma_verbs.h"
#include "ra_hdc_rdma.h"

STATIC int RaHdcNotifyBaseAddrInit(unsigned int notifyType, unsigned int phyId, unsigned long long **notifyVa)
{
    unsigned long long moudleId = HCCP;
    unsigned int notifySize = 0;
    unsigned int logicId;
    int ret, drvRet;

    CHK_PRT_RETURN(notifyType != NOTIFY, hccp_err("[init][base_addr]notify_type[%u] error", notifyType), -EINVAL);
    ret = DlDrvDeviceGetIndexByPhyId(phyId, &logicId);
    CHK_PRT_RETURN(ret, hccp_err("[init][base_addr]drvDeviceGetIndexByPhyId failed, ret(%d), phyId(%u)", ret, phyId),
        ret);

    ret = DlHalNotifyGetInfo(logicId, 0, RA_NOTIFY_TYPE_TOTAL_SIZE, &notifySize);
    CHK_PRT_RETURN(ret, hccp_err("[init][base_addr]halNotifyGetInfo failed, ret(%d), logicId(%u)", ret, logicId), ret);

    ret = DlHalMemAlloc((void *)notifyVa, (unsigned long long)notifySize,
        RA_MEM_TYPE_HBM | (moudleId << MEM_MODULE_ID_BIT));
    CHK_PRT_RETURN(ret, hccp_err("[init][base_addr]halMemAlloc failed, ret(%d), phyId(%u)", ret, phyId), ret);

    hccp_info_rma("notify info: size[%u]", notifySize);
    ret = RaHdcNotifyCfgSet(phyId, (uintptr_t)*notifyVa, notifySize);
    if (ret) {
        hccp_err("[init][base_addr]ra_hdc_notify_cfg_set failed, ret(%d), phyId(%u)", ret, phyId);
        goto free_mem;
    }
    return 0;

free_mem:
    drvRet = DlHalMemFree((void *)*notifyVa);
    if (drvRet) {
        hccp_err("[init][base_addr]halMemFree failed! ret(%d)", drvRet);
    }
    return ret;
}

static void RaHdcGetQpHdc(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, unsigned int qpn,
    struct RaQpHandle *qpHdc)
{
    qpHdc->phyId = rdmaHandle->rdevInfo.phyId;
    qpHdc->rdevIndex = rdmaHandle->rdevIndex;
    qpHdc->rdmaHandle = rdmaHandle;
    qpHdc->rdmaOps = rdmaHandle->rdmaOps;
    qpHdc->qpMode = qpMode;
    qpHdc->flag = flag;
    qpHdc->qpn = qpn;
}

STATIC int RaHdcCmdQpDestroy(struct RaQpHandle *qpHdc)
{
    int ret;
    union OpQpDestroyData qpDestroyData = {0};

    qpDestroyData.txData.qpn = qpHdc->qpn;
    qpDestroyData.txData.phyId = qpHdc->phyId;
    qpDestroyData.txData.rdevIndex = qpHdc->rdevIndex;
    ret = RaHdcProcessMsg(RA_RS_QP_DESTROY, qpHdc->phyId, (char *)&qpDestroyData, sizeof(union OpQpDestroyData));
    if (ret) {
        hccp_err("[destroy][ra_hdc_qp]hdc_send_recv_pkt failed ret(%d) phyId(%u)", ret, qpHdc->phyId);
    }

    return ret;
}

STATIC int RaHdcCmdQpDestroyWithoutCQ(struct RaQpHandle *qpHdc)
{
    int ret;
    union OpQpDestroyData qpDestroyData = {0};

    qpDestroyData.txData.qpn = qpHdc->qpn;
    qpDestroyData.txData.phyId = qpHdc->phyId;
    qpDestroyData.txData.rdevIndex = qpHdc->rdevIndex;
    ret = RaHdcProcessMsg(RA_RS_QP_DESTROY_WITHOUT_CQ, qpHdc->phyId, (char *)&qpDestroyData,
        sizeof(union OpQpDestroyData));
    if (ret) {
        hccp_err("[destroy][ra_hdc_qp]hdc_send_recv_pkt failed ret(%d) phyId(%u)", ret, qpHdc->phyId);
    }

    return ret;
}

int RaHdcQpCreate(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, void **qpHandle)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    union OpQpCreateData qpCreateData = {0};
    struct RaQpHandle *qpHdc = NULL;
    struct rdma_lite_qp_cap cap;
    int ret;

    qpHdc = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpHdc == NULL, hccp_err("[create][ra_hdc_qp]qp_hdc calloc failed phyId(%u)", phyId), -ENOMEM);

    qpCreateData.txData.phyId = phyId;
    qpCreateData.txData.rdevIndex = rdmaHandle->rdevIndex;
    qpCreateData.txData.flag = flag;
    qpCreateData.txData.qpMode = qpMode;
    qpCreateData.txData.memAlign = rdmaHandle->supportLite;

    ret = RaHdcProcessMsg(RA_RS_QP_CREATE, phyId, (char *)&qpCreateData, sizeof(union OpQpCreateData));
    if (ret) {
        hccp_err("[create][ra_hdc_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        free(qpHdc);
        qpHdc = NULL;
        return ret;
    }

    RaHdcGetQpHdc(rdmaHandle, flag, qpMode, qpCreateData.rxData.qpn, qpHdc);
    qpHdc->psn = qpCreateData.rxData.psn;
    qpHdc->gidIdx = qpCreateData.rxData.gidIdx;

    cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
    cap.max_send_sge = QP_DEFAULT_MIN_CAP_SEND_SGE;
    cap.max_recv_sge = QP_DEFAULT_MIN_CAP_RECV_SGE;
    cap.max_send_wr = RA_QP_32K_DEPTH;
    cap.max_recv_wr = RA_QP_128_DEPTH;
    ret = RaHdcLiteQpCreate(rdmaHandle, qpHdc, &cap);
    if (ret) {
        (void)RaHdcCmdQpDestroy(qpHdc);
        hccp_err("[create][ra_hdc_qp]ra_hdc_lite_qp_create failed ret(%d) phyId(%u)", ret, phyId);
        free(qpHdc);
        qpHdc = NULL;
        return ret;
    }

    qpHdc->sqDepth = cap.max_send_wr;
    *qpHandle = qpHdc;

    return 0;
}

int RaHdcQpCreateWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, void **qpHandle)
{
    int flag = (extAttrs->qpAttr.qp_type == IBV_QPT_RC) ? 0 : 1;
    union OpQpCreateWithAttrsData opData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpHdc = NULL;
    struct rdma_lite_qp_cap cap;
    int ret;

    qpHdc = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpHdc == NULL, hccp_err("[create][ra_hdc_qp_with_attrs]qp_hdc calloc failed phyId(%u)", phyId),
        -ENOMEM);

    opData.txData.phyId = phyId;
    opData.txData.rdevIndex = rdmaHandle->rdevIndex;
    ret = memcpy_s(&opData.txData.extAttrs, sizeof(struct QpExtAttrs), extAttrs, sizeof(struct QpExtAttrs));
    if (ret) {
        hccp_err("[create][ra_hdc_qp_with_attrs]memcpy_s for ext_attrs failed, ret:%d", ret);
        ret = -ESAFEFUNC;
        goto out;
    }

    opData.txData.extAttrs.memAlign = rdmaHandle->supportLite;
    ret = RaHdcProcessMsg(RA_RS_QP_CREATE_WITH_ATTRS, phyId, (char *)&opData, sizeof(union OpQpCreateWithAttrsData));
    if (ret) {
        hccp_err("[create][ra_hdc_qp_with_attrs]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    RaHdcGetQpHdc(rdmaHandle, flag, extAttrs->qpMode, opData.rxData.qpn, qpHdc);
    qpHdc->psn = opData.rxData.psn;
    qpHdc->gidIdx = opData.rxData.gidIdx;

    cap.max_inline_data = extAttrs->qpAttr.cap.max_inline_data;
    cap.max_send_sge = extAttrs->qpAttr.cap.max_send_sge;
    cap.max_recv_sge = extAttrs->qpAttr.cap.max_recv_sge;
    cap.max_send_wr = extAttrs->qpAttr.cap.max_send_wr;
    cap.max_recv_wr = extAttrs->qpAttr.cap.max_recv_wr;
    ret = RaHdcLiteQpCreate(rdmaHandle, qpHdc, &cap);
    if (ret) {
        (void)RaHdcCmdQpDestroy(qpHdc);
        hccp_err("[create][ra_hdc_qp_with_attrs]ra_hdc_lite_qp_create failed ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    qpHdc->sqSigAll = extAttrs->qpAttr.sq_sig_all;
    qpHdc->typicalQpAttr.udpSport = extAttrs->udpSport;
    qpHdc->sqDepth = cap.max_send_wr;
    *qpHandle = qpHdc;
    return 0;

out:
    free(qpHdc);
    qpHdc = NULL;
    return ret;
}

int RaHdcQpCreateWithCQWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, unsigned int sendCqn,
    unsigned int recvCqn, void **qpHandle)
{
    int flag = (extAttrs->qpAttr.qp_type == IBV_QPT_RC) ? 0 : 1;
    union OpQpCreateWithCQWithAttrsData opData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpHdc = NULL;
    struct rdma_lite_qp_cap cap;
    int ret;

    qpHdc = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpHdc == NULL, hccp_err("[create][ra_hdc_qp_with_cq_attrs]qp_hdc calloc failed phyId(%u)", phyId),
        -ENOMEM);

    opData.txData.phyId = phyId;
    opData.txData.rdevIndex = rdmaHandle->rdevIndex;
    opData.txData.sendCqn = sendCqn;
    opData.txData.recvCqn = recvCqn;
    ret = memcpy_s(&opData.txData.extAttrs, sizeof(struct QpExtAttrs), extAttrs, sizeof(struct QpExtAttrs));
    if (ret) {
        hccp_err("[create][ra_hdc_qp_with_cq_attrs]memcpy_s for ext_attrs failed, ret:%d", ret);
        ret = -ESAFEFUNC;
        goto out;
    }

    opData.txData.extAttrs.memAlign = rdmaHandle->supportLite;
    ret = RaHdcProcessMsg(RA_RS_QP_CREATE_WITH_CQ_WITH_ATTRS, phyId, (char *)&opData,
        sizeof(union OpQpCreateWithCQWithAttrsData));
    if (ret) {
        hccp_err("[create][ra_hdc_qp_with_cq_attrs]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    RaHdcGetQpHdc(rdmaHandle, flag, extAttrs->qpMode, opData.rxData.qpn, qpHdc);
    qpHdc->psn = opData.rxData.psn;
    qpHdc->gidIdx = opData.rxData.gidIdx;

    cap.max_inline_data = extAttrs->qpAttr.cap.max_inline_data;
    cap.max_send_wr = extAttrs->qpAttr.cap.max_send_wr;
    cap.max_recv_wr = extAttrs->qpAttr.cap.max_recv_wr;
    cap.max_send_sge = extAttrs->qpAttr.cap.max_send_sge;
    cap.max_recv_sge = extAttrs->qpAttr.cap.max_recv_sge;
    ret = RaHdcLiteQpCreateWithCQ(rdmaHandle, qpHdc, &cap, RaHdcLiteFindTypicalCq(rdmaHandle, sendCqn),
        RaHdcLiteFindTypicalCq(rdmaHandle, recvCqn), sendCqn, recvCqn);
    if (ret) {
        (void)RaHdcCmdQpDestroyWithoutCQ(qpHdc);
        hccp_err("[create][ra_hdc_qp_with_cq_attrs]ra_hdc_lite_qp_create failed ret(%d) phyId(%u)", ret, phyId);
        goto out;
    }

    qpHdc->sqDepth = cap.max_send_wr;
    qpHdc->typicalQpAttr.udpSport = extAttrs->udpSport;
    qpHdc->sqSigAll = extAttrs->qpAttr.sq_sig_all;
    *qpHandle = qpHdc;
    return 0;

out:
    free(qpHdc);
    qpHdc = NULL;
    return ret;
}

int RaHdcAiQpCreate(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, struct AiQpInfo *info,
    void **qpHandle)
{
#define AI_QP_DEFAULT_GID_IDX 3U
    int flag = extAttrs->qpAttr.qp_type == IBV_QPT_RC ? 0 : 1;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    union OpAiQpCreateData qpCreateData = {0};
    struct RaQpHandle *qpHdc = NULL;
    int qpMode = extAttrs->qpMode;
    int ret;

    qpHdc = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpHdc == NULL, hccp_err("[create][ra_hdc_ai_qp]qp_hdc calloc failed phyId(%u)", phyId), -ENOMEM);

    qpCreateData.txData.phyId = phyId;
    qpCreateData.txData.rdevIndex = rdmaHandle->rdevIndex;
    ret = memcpy_s(&qpCreateData.txData.extAttrs, sizeof(struct QpExtAttrs), extAttrs, sizeof(struct QpExtAttrs));
    if (ret) {
        hccp_err("[create][ra_hdc_ai_qp]memcpy_s for ext_attrs failed, ret:%d", ret);
        free(qpHdc);
        qpHdc = NULL;
        return -ESAFEFUNC;
    }

    ret = RaHdcProcessMsg(RA_RS_AI_QP_CREATE, phyId, (char *)&qpCreateData, sizeof(union OpAiQpCreateData));
    if (ret) {
        hccp_err("[create][ra_hdc_ai_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        free(qpHdc);
        qpHdc = NULL;
        return ret;
    }

    RaHdcGetQpHdc(rdmaHandle, flag, qpMode, qpCreateData.rxData.qpn, qpHdc);
    qpHdc->psn = qpCreateData.rxData.psn;
    // set a default gid_idx due to compatibility issue, rs will refresh it if it is different
    qpHdc->gidIdx = AI_QP_DEFAULT_GID_IDX;
    info->aiQpAddr = qpCreateData.rxData.aiQpAddr;
    info->sqIndex = qpCreateData.rxData.sqIndex;
    info->dbIndex = qpCreateData.rxData.dbIndex;
    qpHdc->typicalQpAttr.udpSport = extAttrs->udpSport;
    *qpHandle = qpHdc;

    return 0;
}

int RaHdcAiQpCreateWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, struct AiQpInfo *info,
    void **qpHandle)
{
    int flag = extAttrs->qpAttr.qp_type == IBV_QPT_RC ? 0 : 1;
    union OpAiQpCreateWithAttrsData qpCreateData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpHdc = NULL;
    int qpMode = extAttrs->qpMode;
    int ret;

    qpHdc = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpHdc == NULL, hccp_err("[create][ra_hdc_ai_qp]qp_hdc calloc failed phyId(%u)", phyId), -ENOMEM);

    qpCreateData.txData.phyId = phyId;
    qpCreateData.txData.rdevIndex = rdmaHandle->rdevIndex;
    ret = memcpy_s(&qpCreateData.txData.extAttrs, sizeof(struct QpExtAttrs), extAttrs, sizeof(struct QpExtAttrs));
    if (ret) {
        hccp_err("[create][ra_hdc_ai_qp]memcpy_s for ext_attrs failed, ret:%d", ret);
        free(qpHdc);
        qpHdc = NULL;
        return -ESAFEFUNC;
    }

    ret = RaHdcProcessMsg(RA_RS_AI_QP_CREATE_WITH_ATTRS, phyId, (char *)&qpCreateData,
        sizeof(union OpAiQpCreateWithAttrsData));
    if (ret) {
        hccp_err("[create][ra_hdc_ai_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        free(qpHdc);
        qpHdc = NULL;
        return ret;
    }

    qpHdc->typicalQpAttr.udpSport = extAttrs->udpSport;
    RaHdcGetQpHdc(rdmaHandle, flag, qpMode, qpCreateData.rxData.qpn, qpHdc);
    qpHdc->gidIdx = qpCreateData.rxData.gidIdx;
    qpHdc->psn = qpCreateData.rxData.psn;
    info->aiQpAddr = qpCreateData.rxData.aiQpAddr;
    info->sqIndex = qpCreateData.rxData.sqIndex;
    info->dbIndex = qpCreateData.rxData.dbIndex;
    info->aiScqAddr = qpCreateData.rxData.aiScqAddr;
    info->aiRcqAddr = qpCreateData.rxData.aiRcqAddr;
    (void)memcpy_s(&info->dataPlaneInfo, sizeof(struct AiDataPlaneInfo), &qpCreateData.rxData.dataPlaneInfo,
        sizeof(struct AiDataPlaneInfo));
    *qpHandle = qpHdc;

    return 0;
}

int RaHdcTypicalQpCreate(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, struct TypicalQp *qpInfo,
    void **qpHandle)
{
    union OpTypicalQpCreateData qpCreateData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpHdc = NULL;
    struct rdma_lite_qp_cap cap;
    int ret;

    qpHdc = (struct RaQpHandle *)calloc(1, sizeof(struct RaQpHandle));
    CHK_PRT_RETURN(qpHdc == NULL, hccp_err("[create][ra_hdc_typical_qp]qp_hdc calloc failed phyId(%u)", phyId),
        -ENOMEM);

    qpCreateData.txData.phyId = phyId;
    qpCreateData.txData.rdevIndex = rdmaHandle->rdevIndex;
    qpCreateData.txData.flag = flag;
    qpCreateData.txData.qpMode = qpMode;
    qpCreateData.txData.memAlign = rdmaHandle->supportLite;

    ret = RaHdcProcessMsg(RA_RS_TYPICAL_QP_CREATE, phyId, (char *)&qpCreateData, sizeof(union OpTypicalQpCreateData));
    if (ret) {
        hccp_err("[create][ra_hdc_typical_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        free(qpHdc);
        qpHdc = NULL;
        return ret;
    }

    qpInfo->gidIdx = qpCreateData.rxData.gidIdx;
    qpInfo->psn = qpCreateData.rxData.psn;
    qpInfo->qpn = qpCreateData.rxData.qpn;
    (void)memcpy_s(qpInfo->gid, HCCP_GID_RAW_LEN, qpCreateData.rxData.gid.raw, HCCP_GID_RAW_LEN);

    RaHdcGetQpHdc(rdmaHandle, flag, qpMode, qpInfo->qpn, qpHdc);
    qpHdc->psn = qpCreateData.rxData.psn;
    qpHdc->gidIdx = qpCreateData.rxData.gidIdx;

    cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
    cap.max_send_sge = QP_DEFAULT_MIN_CAP_SEND_SGE;
    cap.max_recv_sge = QP_DEFAULT_MIN_CAP_RECV_SGE;
    cap.max_send_wr = RA_QP_32K_DEPTH;
    cap.max_recv_wr = RA_QP_128_DEPTH;
    ret = RaHdcLiteQpCreate(rdmaHandle, qpHdc, &cap);
    if (ret) {
        (void)RaHdcCmdQpDestroy(qpHdc);
        hccp_err("[create][ra_hdc_typical_qp]ra_hdc_lite_qp_create failed ret(%d) phyId(%u)", ret, phyId);
        free(qpHdc);
        qpHdc = NULL;
        return ret;
    }

    qpHdc->sqDepth = cap.max_send_wr;
    *qpHandle = qpHdc;

    return 0;
}

int RaHdcTypicalCqCreate(struct RaRdmaHandle *rdmaHandle, unsigned int cqDepth, unsigned int *cqn, void **cqHandle)
{
    union OpTypicalCqCreateData cqCreateData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaTypicalCqHandle *cqHdc = NULL;
    int ret;

    cqHdc = (struct RaTypicalCqHandle *)calloc(1, sizeof(struct RaTypicalCqHandle));
    CHK_PRT_RETURN(cqHdc == NULL, hccp_err("[create][ra_hdc_typical_cq]cq_hdc calloc failed phyId(%u)", phyId),
        -ENOMEM);

    cqCreateData.txData.phyId = phyId;
    cqCreateData.txData.rdevIndex = rdmaHandle->rdevIndex;
    cqCreateData.txData.cqDepth = cqDepth;

    ret = RaHdcProcessMsg(RA_RS_TYPICAL_CQ_CREATE, phyId, (char *)&cqCreateData, sizeof(union OpTypicalCqCreateData));
    if (ret) {
        hccp_err("[create][ra_hdc_typical_cq]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        free(cqHdc);
        cqHdc = NULL;
        return ret;
    }
    hccp_info_rma("RaHdcProcessMsg RA_RS_TYPICAL_CQ_CREATE Done");

    *cqn = cqCreateData.rxData.cqn;

    cqHdc->cqn = *cqn;
    cqHdc->phyId = phyId;
    cqHdc->rdevIndex = rdmaHandle->rdevIndex;
    cqHdc->rdmaHandle = rdmaHandle;
    cqHdc->rdmaOps = rdmaHandle->rdmaOps;

    if (rdmaHandle->supportLite != 0) {
        struct rdma_lite_cq *liteCq = NULL;
        ret = RaHdcLiteCqCreate(rdmaHandle, cqDepth, &cqCreateData, &liteCq);
        if (ret) {
            union OpTypicalCqDestroyData destroyData = {0};
            hccp_err("[create][ra_hdc_typical_cq]ra_hdc_lite_cq_create failed ret(%d) phyId(%u)", ret, phyId);
            destroyData.txData.cqn = cqHdc->cqn;
            destroyData.txData.phyId = cqHdc->phyId;
            destroyData.txData.rdevIndex = cqHdc->rdevIndex;
            (void)RaHdcProcessMsg(RA_RS_TYPICAL_CQ_DESTROY, phyId, (char *)&destroyData,
                sizeof(union OpTypicalCqDestroyData));
            free(cqHdc);
            cqHdc = NULL;
            return ret;
        }
        cqHdc->liteCq = liteCq;
    }

    *cqHandle = cqHdc;

    return 0;
}

int RaHdcTypicalCqDestroy(struct RaRdmaHandle *rdmaHandle, unsigned int cqn, void *cqHandle)
{
    struct RaTypicalCqHandle *cqHdc = (struct RaTypicalCqHandle *)cqHandle;
    union OpTypicalCqDestroyData destroyData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret;

    if (cqHdc == NULL) {
        hccp_err("[destroy][ra_hdc_typical_cq]cq handle is NULL phyId(%u)", phyId);
        return -EINVAL;
    }

    if (cqHdc->liteCq != NULL) {
        RaHdcLiteRemoveTypicalCq(rdmaHandle, cqHdc->cqn);
        (void)RaRdmaLiteDestroyCq(cqHdc->liteCq);
        cqHdc->liteCq = NULL;
    }

    destroyData.txData.phyId = phyId;
    destroyData.txData.rdevIndex = rdmaHandle->rdevIndex;
    destroyData.txData.cqn = cqn;
    ret = RaHdcProcessMsg(RA_RS_TYPICAL_CQ_DESTROY, phyId, (char *)&destroyData, sizeof(union OpTypicalCqDestroyData));
    if (ret) {
        hccp_err("[destroy][ra_hdc_typical_cq]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
    }

    free(cqHdc);
    cqHdc = NULL;
    return ret;
}

int RaHdcQpDestroy(struct RaQpHandle *qpHdc)
{
    int ret;

    RaHdcLiteQpDestroy(qpHdc);
    ret = RaHdcCmdQpDestroy(qpHdc);
    if (ret) {
        hccp_err("[destroy][ra_hdc_qp]ra_hdc_cmd_qp_destroy failed ret(%d) phyId(%u)", ret, qpHdc->phyId);
    }

    free(qpHdc);
    qpHdc = NULL;
    return ret;
}

int RaHdcQpDestroyWithoutCQ(struct RaQpHandle *qpHdc)
{
    int ret;

    RaHdcLiteQpDestroyWithoutCQ(qpHdc);
    ret = RaHdcCmdQpDestroyWithoutCQ(qpHdc);
    if (ret) {
        hccp_err("[destroy][ra_hdc_qp]ra_hdc_cmd_qp_destroy_without_cq failed ret(%d) phyId(%u)", ret, qpHdc->phyId);
    }

    free(qpHdc);
    qpHdc = NULL;
    return ret;
}

int RaHdcGetQpStatus(struct RaQpHandle *qpHdc, int *status)
{
    union OpQpStatusData qpStatusData = {0};
    union OpQpInfoData qpInfoData = {0};
    unsigned int interfaceVersion = 0;
    int ret;

    ret = RaHdcGetInterfaceVersion(qpHdc->phyId, RA_RS_QP_INFO, &interfaceVersion);
    if (ret != 0) {
        hccp_warn_rma("[get][ra_hdc_qp_status]get interface version not success ret(%d) phyId(%u)", ret, qpHdc->phyId);
        interfaceVersion = 0;
    }

    if (interfaceVersion >= RA_RS_OPCODE_BASE_VERSION) {
        qpInfoData.txData.qpn = qpHdc->qpn;
        qpInfoData.txData.phyId = qpHdc->phyId;
        qpInfoData.txData.rdevIndex = qpHdc->rdevIndex;
        ret = RaHdcProcessMsg(RA_RS_QP_INFO, qpHdc->phyId, (char *)&qpInfoData, sizeof(union OpQpInfoData));
        CHK_PRT_RETURN(ret,
            hccp_err("[get][ra_hdc_qp_status]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId), ret);
        *status = qpInfoData.rxData.status;
        qpHdc->typicalQpAttr.udpSport = qpInfoData.rxData.udpSport;
    } else {
        qpStatusData.txData.qpn = qpHdc->qpn;
        qpStatusData.txData.phyId = qpHdc->phyId;
        qpStatusData.txData.rdevIndex = qpHdc->rdevIndex;
        ret = RaHdcProcessMsg(RA_RS_QP_STATUS, qpHdc->phyId, (char *)&qpStatusData, sizeof(union OpQpStatusData));
        CHK_PRT_RETURN(ret,
            hccp_err("[get][ra_hdc_qp_status]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId), ret);
        *status = qpStatusData.rxData.status;
    }

    return RaHdcLiteGetConnectedInfo(qpHdc);
}

int RaHdcTypicalQpModify(struct RaQpHandle *qpHdc, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo)
{
    union OpTypicalQpModifyData qpModifyData = {0};
    unsigned int phyId = qpHdc->phyId;
    int ret;

    qpModifyData.txData.phyId = phyId;
    qpModifyData.txData.rdevIndex = qpHdc->rdevIndex;
    ret = memcpy_s(&(qpModifyData.txData.localQpInfo), sizeof(struct TypicalQp), localQpInfo, sizeof(struct TypicalQp));
    CHK_PRT_RETURN(ret != 0, hccp_err("[modify]memcpy_s local_qp_info failed, phyId[%u] ret[%d]", phyId, ret),
        -ESAFEFUNC);
    ret = memcpy_s(&(qpModifyData.txData.remoteQpInfo), sizeof(struct TypicalQp), remoteQpInfo,
        sizeof(struct TypicalQp));
    CHK_PRT_RETURN(ret != 0, hccp_err("[modify]memcpy_s remote_qp_info failed, phyId[%u] ret[%d]", phyId, ret),
        -ESAFEFUNC);

    ret = RaHdcProcessMsg(RA_RS_TYPICAL_QP_MODIFY, phyId, (char *)&qpModifyData, sizeof(union OpTypicalQpModifyData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[modify][modify_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    qpHdc->typicalQpAttr.udpSport = qpModifyData.rxData.udpSport;
    qpHdc->typicalQpAttr.pathMtu = qpModifyData.rxData.pathMtu;
    if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
        ret = RaRdmaLiteSetQpSl(qpHdc->liteQp, localQpInfo->sl);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[modify][modify_qp]ra_rdma_lite_set_qp_sl sl(%u) failed ret(%d) phyId(%u)", localQpInfo->sl, ret,
                phyId),
            ret);
    }
    return 0;
}

int RaHdcQpConnectAsync(struct RaQpHandle *qpHdc, const void *sockHandle)
{
    union OpQpConnectData qpConnectData = {0};
    int ret;

    qpConnectData.txData.qpn = qpHdc->qpn;
    qpConnectData.txData.fd = (unsigned int)((const struct SocketHdcInfo *)sockHandle)->fd;
    qpConnectData.txData.phyId = qpHdc->phyId;
    qpConnectData.txData.rdevIndex = qpHdc->rdevIndex;
    ret = RaHdcProcessMsg(RA_RS_QP_CONNECT, qpHdc->phyId, (char *)&qpConnectData, sizeof(union OpQpConnectData));
    CHK_PRT_RETURN(ret,
        hccp_err("[connect_async][ra_hdc_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId), ret);

    return 0;
}

static void RaHdcSendDataInit(union OpSendWrData *sendWrData, struct RaQpHandle *qpHdc, struct SendWr *wr)
{
    sendWrData->txData.phyId = qpHdc->phyId;
    sendWrData->txData.rdevIndex = qpHdc->rdevIndex;
    sendWrData->txData.qpn = qpHdc->qpn;
    sendWrData->txData.bufNum = wr->bufNum;
    sendWrData->txData.dstAddr = wr->dstAddr;
    sendWrData->txData.op = wr->op;
    sendWrData->txData.sendFlags = wr->sendFlag;
}

int RaHdcSendWr(struct RaQpHandle *qpHdc, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    union OpSendWrData sendWrData = {0};
    struct LiteSendWr liteWr = {0};
    int ret;

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            liteWr.wr = *wr;
            return RaHdcLiteSendWr(qpHdc, &liteWr, opRsp, HDC_LITE_DEFAULT_WR_ID);
        }
    }

    RaHdcSendDataInit(&sendWrData, qpHdc, wr);

    ret = memcpy_s(sendWrData.txData.memList, (sizeof(struct SgList) * MAX_SGE_NUM), wr->bufList,
        (sizeof(struct SgList) * wr->bufNum));
    CHK_PRT_RETURN(ret,
        hccp_err("[send][ra_hdc_wr]memcpy_s for mem_list failed, ret(%d), phyId(%u)", ret, qpHdc->phyId), -ESAFEFUNC);

    ret = RaHdcProcessMsg(RA_RS_SEND_WR, qpHdc->phyId, (char *)&sendWrData, sizeof(union OpSendWrData));
    if (ret) {
        if (ret != -ENOENT) {
            hccp_err("[send][ra_hdc_wr]ra hdc message process failed ret(%d), phyId(%u)", ret, qpHdc->phyId);
        }
        return ret;
    }

    if (qpHdc->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
        opRsp->wqeTmp = sendWrData.rxData.wrRsp.wqeTmp;
    } else if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_GDR_ASYN_QP_MODE ||
               qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        opRsp->db = sendWrData.rxData.wrRsp.db;
    }

    return ret;
}

int RaHdcSendWrV2(struct RaQpHandle *qpHdc, struct SendWrV2 *wr, struct SendWrRsp *opRsp)
{
    struct LiteSendWr liteWr = {0};

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            liteWr.wr.bufList = wr->bufList;
            liteWr.wr.bufNum = wr->bufNum;
            liteWr.wr.dstAddr = wr->dstAddr;
            liteWr.wr.op = wr->op;
            liteWr.wr.rkey = wr->rkey;
            liteWr.wr.sendFlag = wr->sendFlag;
            liteWr.aux = wr->aux;
            liteWr.ext = wr->ext;
            return RaHdcLiteTypicalSendWr(qpHdc, &liteWr, opRsp, wr->wrId);
        }
    }

    hccp_warn_rma("qpn:%u qp_mode:%d support_lite:%d not support to send_wr", qpHdc->qpn, qpHdc->qpMode,
        qpHdc->supportLite);

    return -ENOTSUPP;
}

int RaHdcSendWrVerbs(struct RaQpHandle *qpHdc, struct SendWrVerbs *wr, struct SendWrRsp *opRsp)
{
    struct LiteSendWr liteWr = {0};

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            liteWr.wr.bufList = wr->sgList;
            liteWr.wr.bufNum = wr->numSge;
            liteWr.wr.dstAddr = wr->remoteAddr;
            liteWr.wr.op = wr->opcode;
            liteWr.wr.rkey = wr->rkey;
            liteWr.wr.sendFlag = wr->sendFlags;
            liteWr.ext.immData = wr->immData;
            return RaHdcLiteTypicalSendWr(qpHdc, &liteWr, opRsp, wr->wrId);
        }
    }

    hccp_warn_rma("qpn:%u qp_mode:%d support_lite:%d not support to send_wr", qpHdc->qpn, qpHdc->qpMode,
        qpHdc->supportLite);

    return -ENOTSUPP;
}

int RaHdcRecvWrVerbs(struct RaQpHandle *qpHdc, struct RecvWrVerbs *wr)
{
    struct rdma_lite_sge list[RA_SGLIST_MAX];
    struct rdma_lite_recv_wr liteWr = {
        .sg_list = list,
        .wr_id = wr->wrId,
        .num_sge = 0,
        .next = NULL,
    };
    struct rdma_lite_recv_wr *badWr = NULL;
    int ret;

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            for (int i = 0; i < wr->numSge && i < RA_SGLIST_MAX; i++) {
                list[i].addr = (uintptr_t)wr->sgList[i].addr;
                list[i].length = wr->sgList[i].len;
                list[i].lkey = wr->sgList[i].lkey;
                liteWr.num_sge++;
            }

            ret = RaRdmaLitePostRecv(qpHdc->liteQp, &liteWr, &badWr);
            if (ret) {
                if (ret == -ENOMEM) {
                    hccp_warn_rma("[recv][ra_hdc_wr]ra hdc post recv unsuccessful, ret(%d) phyId(%u)", ret,
                        qpHdc->phyId);
                    return 0;
                } else {
                    hccp_err("[recv][ra_hdc_wr]ra hdc post recv failed ret(%d) phyId(%u)", ret, qpHdc->phyId);
                }
                return ret;
            }

            qpHdc->recvWrNum++;
            return 0;
        }
    }

    hccp_warn_rma("qpn:%u qp_mode:%d support_lite:%d not support to recv_wr_verbs", qpHdc->qpn, qpHdc->qpMode,
        qpHdc->supportLite);

    return -ENOTSUPP;
}

int RaHdcTypicalSendWr(struct RaQpHandle *qpHdc, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    struct LiteSendWr liteWr = {0};

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            liteWr.wr = *wr;
            return RaHdcLiteTypicalSendWr(qpHdc, &liteWr, opRsp, HDC_LITE_DEFAULT_WR_ID);
        }
    }

    hccp_warn_rma("qpn:%u qp_mode:%d support_lite:%d not support to send_wr", qpHdc->qpn, qpHdc->qpMode,
        qpHdc->supportLite);

    return -ENOTSUPP;
}

int RaHdcMrDereg(struct RaQpHandle *qpHdc, struct MrInfoT *info)
{
    union OpMrDeregData mrDeregData = {0};
    mrDeregData.txData.rdevIndex = qpHdc->rdevIndex;
    mrDeregData.txData.phyId = qpHdc->phyId;
    mrDeregData.txData.qpn = qpHdc->qpn;
    mrDeregData.txData.addr = info->addr;
    int ret;

    ret = RaHdcProcessMsg(RA_RS_MR_DEREG, qpHdc->phyId, (char *)&mrDeregData, sizeof(union OpMrDeregData));
    CHK_PRT_RETURN(ret,
        hccp_err("[dereg][ra_hdc_mr]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId), ret);

    return 0;
}

int RaHdcMrReg(struct RaQpHandle *qpHdc, struct MrInfoT *info)
{
    union OpMrRegData mrRegData = {0};
    int ret;

    mrRegData.txData.phyId = qpHdc->phyId;
    mrRegData.txData.rdevIndex = qpHdc->rdevIndex;
    mrRegData.txData.qpn = qpHdc->qpn;
    mrRegData.txData.mrRegAttr.addr = info->addr;
    mrRegData.txData.mrRegAttr.len = info->size;
    mrRegData.txData.mrRegAttr.access = info->access;

    ret = RaHdcProcessMsg(RA_RS_MR_REG, qpHdc->phyId, (char *)&mrRegData, sizeof(union OpMrRegData));
    CHK_PRT_RETURN(ret, hccp_err("[reg][ra_hdc_mr]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
        ret);

    info->lkey = mrRegData.rxData.lkey;
    info->rkey = mrRegData.rxData.rkey;

    return 0;
}

int RaHdcTypicalMrReg(struct RaRdmaHandle *rdmaHandle, struct MrInfoT *info, void **mrHandle)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    union OpTypicalMrRegData mrRegData = {0};
    unsigned int opcode = RA_RS_TYPICAL_MR_REG_V1;
    struct RaMrHandle *mrHdc = NULL;
    unsigned int interfaceVersion = 0;
    int ret;

    mrHdc = (struct RaMrHandle *)calloc(1, sizeof(struct RaMrHandle));
    CHK_PRT_RETURN(mrHdc == NULL, hccp_err("[reg][ra_hdc_typical_mr]mr_hdc calloc failed phyId(%u)", phyId), -ENOMEM);

    mrRegData.txData.phyId = phyId;
    mrRegData.txData.rdevIndex = rdmaHandle->rdevIndex;
    mrRegData.txData.mrRegAttr.addr = info->addr;
    mrRegData.txData.mrRegAttr.len = info->size;
    mrRegData.txData.mrRegAttr.access = info->access;

    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_TYPICAL_MR_REG, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_OPCODE_BASE_VERSION) {
        opcode = RA_RS_TYPICAL_MR_REG;
    }
    ret = RaHdcProcessMsg(opcode, phyId, (char *)&mrRegData, sizeof(union OpTypicalMrRegData));
    if (ret) {
        hccp_err("[reg][ra_hdc_typical_mr]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        free(mrHdc);
        return ret;
    }

    info->lkey = mrRegData.rxData.lkey;
    info->rkey = mrRegData.rxData.rkey;
    if (opcode == RA_RS_TYPICAL_MR_REG_V1) {
        mrHdc->addr = (unsigned long long)(uintptr_t)info->addr;
    } else {
        mrHdc->addr = mrRegData.rxData.addr;
    }
    *mrHandle = mrHdc;
    return 0;
}

int RaHdcRemapMr(struct RaRdmaHandle *rdmaHandle, struct MemRemapInfo info[], unsigned int num)
{
    union OpRemapMrData opData = {0};
    int ret;

    ret = memcpy_s(opData.txData.memList, REMAP_MR_MAX_NUM * sizeof(struct MemRemapInfo), info,
        num * sizeof(struct MemRemapInfo));
    CHK_PRT_RETURN(ret != 0, hccp_err("[remap][ra_hdc_mr]memcpy_s mem_list failed, ret:%d", ret), -ESAFEFUNC);
    opData.txData.memNum = num;
    opData.txData.rdevIndex = rdmaHandle->rdevIndex;
    opData.txData.phyId = rdmaHandle->rdevInfo.phyId;

    ret = RaHdcProcessMsg(RA_RS_REMAP_MR, opData.txData.phyId, (char *)&opData, sizeof(union OpRemapMrData));
    CHK_PRT_RETURN(ret,
        hccp_err("[remap][ra_hdc_mr]ra hdc message process failed ret(%d) phyId(%u)", ret, rdmaHandle->rdevInfo.phyId),
        ret);

    return 0;
}

int RaHdcTypicalMrDereg(struct RaRdmaHandle *rdmaHandle, void *mrHandle)
{
    union OpTypicalMrDeregData mrDeregData = {0};
    int ret;

    mrDeregData.txData.phyId = rdmaHandle->rdevInfo.phyId;
    mrDeregData.txData.rdevIndex = rdmaHandle->rdevIndex;
    mrDeregData.txData.addr = ((struct RaMrHandle *)mrHandle)->addr;

    ret = RaHdcProcessMsg(RA_RS_TYPICAL_MR_DEREG, rdmaHandle->rdevInfo.phyId, (char *)&mrDeregData,
        sizeof(union OpTypicalMrDeregData));
    CHK_PRT_RETURN(ret,
        hccp_err("[dereg][ra_hdc_typical_mr]ra hdc message process failed ret(%d) phyId(%u)", ret,
            rdmaHandle->rdevInfo.phyId),
        ret);

    free(mrHandle);
    mrHandle = NULL;
    return 0;
}

STATIC void RaHdcSendWrlistInit(union OpSendWrlistData *sendWrlist, struct RaQpHandle *qpHdc, unsigned int completeCnt,
    struct WrlistSendCompleteNum wrlistNum)
{
    sendWrlist->txData.phyId = qpHdc->phyId;
    sendWrlist->txData.rdevIndex = qpHdc->rdevIndex;
    sendWrlist->txData.qpn = qpHdc->qpn;
    sendWrlist->txData.sendNum = (wrlistNum.sendNum - completeCnt) >= MAX_WR_NUM ? MAX_WR_NUM
                                                                                 : wrlistNum.sendNum - completeCnt;
}

STATIC void RaHdcSendWrlistExtInit(union OpSendWrlistDataExt *sendWrlist, struct RaQpHandle *qpHdc,
    unsigned int completeCnt, struct WrlistSendCompleteNum wrlistNum)
{
    sendWrlist->txData.phyId = qpHdc->phyId;
    sendWrlist->txData.rdevIndex = qpHdc->rdevIndex;
    sendWrlist->txData.qpn = qpHdc->qpn;
    sendWrlist->txData.sendNum = (wrlistNum.sendNum - completeCnt) >= MAX_WR_NUM ? MAX_WR_NUM
                                                                                 : wrlistNum.sendNum - completeCnt;
}

STATIC int RaHdcSendWrlistV1(struct RaQpHandle *qpHdc, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret = 0;
    unsigned int i, j;
    unsigned int completeCnt = 0;
    unsigned int currentSendNum = 0;
    union OpSendWrlistData *sendWrlist = NULL;

    sendWrlist = calloc(1, sizeof(union OpSendWrlistData));
    CHK_PRT_RETURN(sendWrlist == NULL, hccp_err("[send][ra_hdc_wrlist]send_wrlist calloc failed"), -ENOMEM);
    while (completeCnt < wrlistNum.sendNum) {
        RaHdcSendWrlistInit(sendWrlist, qpHdc, completeCnt, wrlistNum);
        ret = memcpy_s(sendWrlist->txData.wrlist, (sizeof(struct SendWrlistData) * MAX_WR_NUM_V1), &wr[completeCnt],
            (sizeof(struct SendWrlistData) * sendWrlist->txData.sendNum));
        if (ret) {
            hccp_err("[send][ra_hdc_wrlist]memcpy_s for wrlist failed, ret(%d).", ret);
            ret = -ESAFEFUNC;
            goto err_send_wrlist;
        }
        currentSendNum = sendWrlist->txData.sendNum;
        ret = RaHdcProcessMsg(RA_RS_SEND_WRLIST, qpHdc->phyId, (char *)sendWrlist, sizeof(union OpSendWrlistData));

        if (sendWrlist->rxData.completeNum > currentSendNum) {
            hccp_err("[send][ra_hdc_wrlist]complete_num[%u] is larger than send_num[%u], ret(%d).",
                sendWrlist->rxData.completeNum, currentSendNum, ret);
            ret = -EINVAL;
            goto err_send_wrlist;
        }

        for (i = 0; i < sendWrlist->rxData.completeNum; i++) {
            j = i + completeCnt;
            if (qpHdc->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
                opRsp[j].wqeTmp = sendWrlist->rxData.wrRsp[i].wqeTmp;
            } else if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_GDR_ASYN_QP_MODE ||
                       qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
                opRsp[j].db = sendWrlist->rxData.wrRsp[i].db;
            }
        }
        completeCnt = completeCnt + sendWrlist->rxData.completeNum;
        if (ret) {
            if (ret != -ENOENT) {
                hccp_err("[send][ra_hdc_wrlist]ra hdc message process failed ret(%d), phyId(%u)", ret, qpHdc->phyId);
            }
            goto err_send_wrlist;
        }
    }

err_send_wrlist:
    free(sendWrlist);
    sendWrlist = NULL;
    *(wrlistNum.completeNum) = completeCnt;
    return ret;
}

STATIC int RaHdcSendWrlistExtV1(struct RaQpHandle *qpHdc, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret = 0;
    unsigned int i, j;
    unsigned int completeCnt = 0;
    unsigned int currentSendNum = 0;
    union OpSendWrlistDataExt *sendWrlist = NULL;

    sendWrlist = calloc(1, sizeof(union OpSendWrlistDataExt));
    CHK_PRT_RETURN(sendWrlist == NULL, hccp_err("[send][ra_hdc_wrlist_ext]send_wrlist calloc failed"), -ENOMEM);
    while (completeCnt < wrlistNum.sendNum) {
        RaHdcSendWrlistExtInit(sendWrlist, qpHdc, completeCnt, wrlistNum);
        ret = memcpy_s(sendWrlist->txData.wrlist, (sizeof(struct SendWrlistDataExt) * MAX_WR_NUM_V1), &wr[completeCnt],
            (sizeof(struct SendWrlistDataExt) * sendWrlist->txData.sendNum));
        if (ret) {
            hccp_err("[send][ra_hdc_wrlist_ext]memcpy_s for wrlist failed, ret(%d).", ret);
            ret = -ESAFEFUNC;
            goto err_send_wrlist;
        }
        currentSendNum = sendWrlist->txData.sendNum;
        ret = RaHdcProcessMsg(RA_RS_SEND_WRLIST_EXT, qpHdc->phyId, (char *)sendWrlist,
            sizeof(union OpSendWrlistDataExt));

        if (sendWrlist->rxData.completeNum > currentSendNum) {
            hccp_err("[send][ra_hdc_wrlist_ext]complete_num[%u] is larger than send_num[%u], ret(%d).",
                sendWrlist->rxData.completeNum, currentSendNum, ret);
            ret = -EINVAL;
            goto err_send_wrlist;
        }

        for (i = 0; i < sendWrlist->rxData.completeNum; i++) {
            j = i + completeCnt;
            if (qpHdc->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
                opRsp[j].wqeTmp = sendWrlist->rxData.wrRsp[i].wqeTmp;
            } else if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_GDR_ASYN_QP_MODE ||
                       qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
                opRsp[j].db = sendWrlist->rxData.wrRsp[i].db;
            }
        }
        completeCnt = completeCnt + sendWrlist->rxData.completeNum;
        if (ret) {
            if (ret != -ENOENT) {
                hccp_err("[send][ra_hdc_wrlist_ext]ra hdc message process failed ret(%d), phyId(%u)", ret,
                    qpHdc->phyId);
            }
            goto err_send_wrlist;
        }
    }

err_send_wrlist:
    free(sendWrlist);
    sendWrlist = NULL;
    *(wrlistNum.completeNum) = completeCnt;
    return ret;
}

STATIC void RaHdcSendWrlistInitV2(union OpSendWrlistDataV2 *sendWrlist, struct RaQpHandle *qpHdc,
    unsigned int completeCnt, struct WrlistSendCompleteNum wrlistNum)
{
    sendWrlist->txData.phyId = qpHdc->phyId;
    sendWrlist->txData.rdevIndex = qpHdc->rdevIndex;
    sendWrlist->txData.qpn = qpHdc->qpn;
    sendWrlist->txData.sendNum = (wrlistNum.sendNum - completeCnt) >= MAX_WR_NUM ? MAX_WR_NUM
                                                                                 : wrlistNum.sendNum - completeCnt;
}

STATIC void RaHdcSendWrlistExtInitV2(union OpSendWrlistDataExtV2 *sendWrlist, struct RaQpHandle *qpHdc,
    unsigned int completeCnt, struct WrlistSendCompleteNum wrlistNum)
{
    sendWrlist->txData.phyId = qpHdc->phyId;
    sendWrlist->txData.rdevIndex = qpHdc->rdevIndex;
    sendWrlist->txData.qpn = qpHdc->qpn;
    sendWrlist->txData.sendNum = (wrlistNum.sendNum - completeCnt) >= MAX_WR_NUM ? MAX_WR_NUM
                                                                                 : wrlistNum.sendNum - completeCnt;
}

STATIC int RaHdcSendWrlistV2(struct RaQpHandle *qpHdc, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret = 0;
    unsigned int i, j;
    unsigned int completeCnt = 0;
    unsigned int currentSendNum = 0;
    union OpSendWrlistDataV2 *sendWrlist = NULL;

    sendWrlist = calloc(1, sizeof(union OpSendWrlistDataV2));
    CHK_PRT_RETURN(sendWrlist == NULL, hccp_err("[send][ra_hdc_wrlist_v2]send_wrlist calloc failed"), -ENOMEM);
    while (completeCnt < wrlistNum.sendNum) {
        RaHdcSendWrlistInitV2(sendWrlist, qpHdc, completeCnt, wrlistNum);
        ret = memcpy_s(sendWrlist->txData.wrlist, (sizeof(struct SendWrlistData) * MAX_WR_NUM), &wr[completeCnt],
            (sizeof(struct SendWrlistData) * sendWrlist->txData.sendNum));
        if (ret) {
            hccp_err("[send][ra_hdc_wrlist_v2]memcpy_s for wrlist failed, ret(%d).", ret);
            ret = -ESAFEFUNC;
            goto err_send_wrlist;
        }
        currentSendNum = sendWrlist->txData.sendNum;
        ret = RaHdcProcessMsg(RA_RS_SEND_WRLIST_V2, qpHdc->phyId, (char *)sendWrlist, sizeof(union OpSendWrlistDataV2));

        if (sendWrlist->rxData.completeNum > currentSendNum) {
            hccp_err("[send][ra_hdc_wrlist_v2]complete_num[%u] is larger than send_num[%u], ret(%d).",
                sendWrlist->rxData.completeNum, currentSendNum, ret);
            ret = -EINVAL;
            goto err_send_wrlist;
        }

        for (i = 0; i < sendWrlist->rxData.completeNum; i++) {
            j = i + completeCnt;
            if (qpHdc->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
                opRsp[j].wqeTmp = sendWrlist->rxData.wrRsp[i].wqeTmp;
            } else if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_GDR_ASYN_QP_MODE ||
                       qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
                opRsp[j].db = sendWrlist->rxData.wrRsp[i].db;
            }
        }
        completeCnt = completeCnt + sendWrlist->rxData.completeNum;
        if (ret) {
            if (ret != -ENOENT) {
                hccp_err("[send][ra_hdc_wrlist_v2]ra hdc message process failed ret(%d), phyId(%u)", ret, qpHdc->phyId);
            }
            goto err_send_wrlist;
        }
    }

err_send_wrlist:
    free(sendWrlist);
    sendWrlist = NULL;
    *(wrlistNum.completeNum) = completeCnt;
    return ret;
}

STATIC int RaHdcSendWrlistExtV2(struct RaQpHandle *qpHdc, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret = 0;
    unsigned int i, j;
    unsigned int completeCnt = 0;
    unsigned int currentSendNum = 0;
    union OpSendWrlistDataExtV2 *sendWrlist = NULL;

    sendWrlist = calloc(1, sizeof(union OpSendWrlistDataExtV2));
    CHK_PRT_RETURN(sendWrlist == NULL, hccp_err("[send][ra_hdc_wrlist_ext_v2]send_wrlist calloc failed"), -ENOMEM);
    while (completeCnt < wrlistNum.sendNum) {
        RaHdcSendWrlistExtInitV2(sendWrlist, qpHdc, completeCnt, wrlistNum);
        ret = memcpy_s(sendWrlist->txData.wrlist, (sizeof(struct SendWrlistDataExt) * MAX_WR_NUM), &wr[completeCnt],
            (sizeof(struct SendWrlistDataExt) * sendWrlist->txData.sendNum));
        if (ret) {
            hccp_err("[send][ra_hdc_wrlist_ext_v2]memcpy_s for wrlist failed, ret(%d).", ret);
            ret = -ESAFEFUNC;
            goto err_send_wrlist;
        }
        currentSendNum = sendWrlist->txData.sendNum;
        ret = RaHdcProcessMsg(RA_RS_SEND_WRLIST_EXT_V2, qpHdc->phyId, (char *)sendWrlist,
            sizeof(union OpSendWrlistDataExtV2));

        if (sendWrlist->rxData.completeNum > currentSendNum) {
            hccp_err("[send][ra_hdc_wrlist_ext_v2]complete_num[%u] is larger than send_num[%u], ret(%d).",
                sendWrlist->rxData.completeNum, currentSendNum, ret);
            ret = -EINVAL;
            goto err_send_wrlist;
        }

        for (i = 0; i < sendWrlist->rxData.completeNum; i++) {
            j = i + completeCnt;
            if (qpHdc->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
                opRsp[j].wqeTmp = sendWrlist->rxData.wrRsp[i].wqeTmp;
            } else if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_GDR_ASYN_QP_MODE ||
                       qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
                opRsp[j].db = sendWrlist->rxData.wrRsp[i].db;
            }
        }
        completeCnt = completeCnt + sendWrlist->rxData.completeNum;
        if (ret) {
            if (ret != -ENOENT) {
                hccp_err("[send][ra_hdc_wrlist_ext_v2]ra hdc message process failed ret(%d), phyId(%u)", ret,
                    qpHdc->phyId);
            }
            goto err_send_wrlist;
        }
    }

err_send_wrlist:
    free(sendWrlist);
    sendWrlist = NULL;
    *(wrlistNum.completeNum) = completeCnt;
    return ret;
}

int RaHdcSendWrlist(struct RaQpHandle *qpHdc, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret;
    unsigned int interfaceVersion = 0;

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            return RaHdcLiteSendWrlist(qpHdc, wr, opRsp, wrlistNum);
        }
    }

    ret = RaHdcGetInterfaceVersion(qpHdc->phyId, RA_RS_SEND_WRLIST_V2, &interfaceVersion);
    if (ret != 0 || interfaceVersion != RA_RS_SEND_WRLIST_V2_VERSION) {
        return RaHdcSendWrlistV1(qpHdc, wr, opRsp, wrlistNum);
    }

    return RaHdcSendWrlistV2(qpHdc, wr, opRsp, wrlistNum);
}

int RaHdcSendWrlistExt(struct RaQpHandle *qpHdc, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret;
    unsigned int interfaceVersion = 0;

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            return RaHdcLiteSendWrlistExt(qpHdc, wr, opRsp, wrlistNum);
        }
    }

    ret = RaHdcGetInterfaceVersion(qpHdc->phyId, RA_RS_SEND_WRLIST_EXT_V2, &interfaceVersion);
    if (ret != 0 || interfaceVersion != RA_RS_SEND_WRLIST_EXT_V2_VERSION) {
        return RaHdcSendWrlistExtV1(qpHdc, wr, opRsp, wrlistNum);
    }

    return RaHdcSendWrlistExtV2(qpHdc, wr, opRsp, wrlistNum);
}

STATIC void RaHdcSendWrlistNormalInit(union OpSendNormalWrlistData *sendWrlist, struct RaQpHandle *qpHdc,
    unsigned int completeCnt, struct WrlistSendCompleteNum wrlistNum)
{
    (void)memset_s(sendWrlist, sizeof(union OpSendNormalWrlistData), 0, sizeof(union OpSendNormalWrlistData));
    sendWrlist->txData.phyId = qpHdc->phyId;
    sendWrlist->txData.rdevIndex = qpHdc->rdevIndex;
    sendWrlist->txData.qpn = qpHdc->qpn;
    sendWrlist->txData.sendNum = ((wrlistNum.sendNum - completeCnt) >= MAX_WR_NUM) ? MAX_WR_NUM
                                                                                   : wrlistNum.sendNum - completeCnt;
}

STATIC int RaHdcSendWrlistNormal(struct RaQpHandle *qpHdc, struct WrInfo wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    union OpSendNormalWrlistData *sendWrlist = NULL;
    unsigned int currentSendNum = 0;
    unsigned int completeCnt = 0;
    unsigned int i;
    int ret = 0;

    sendWrlist = calloc(1, sizeof(union OpSendNormalWrlistData));
    CHK_PRT_RETURN(sendWrlist == NULL, hccp_err("[send][send_wrlist]send_wrlist calloc failed"), -ENOMEM);

    while (completeCnt < wrlistNum.sendNum) {
        RaHdcSendWrlistNormalInit(sendWrlist, qpHdc, completeCnt, wrlistNum);
        ret = memcpy_s(sendWrlist->txData.wrlist, (sizeof(struct WrInfo) * MAX_WR_NUM), &wr[completeCnt],
            (sizeof(struct WrInfo) * sendWrlist->txData.sendNum));
        if (ret != 0) {
            hccp_err("[send][send_wrlist]memcpy_s for wrlist failed, ret(%d)", ret);
            ret = -ESAFEFUNC;
            goto err_send_wrlist;
        }
        currentSendNum = sendWrlist->txData.sendNum;
        ret = RaHdcProcessMsg(RA_RS_SEND_NORMAL_WRLIST, qpHdc->phyId, (char *)sendWrlist,
            sizeof(union OpSendNormalWrlistData));

        if (sendWrlist->rxData.completeNum > currentSendNum) {
            hccp_err("[send][send_wrlist]complete_num[%u] is larger than send_num[%u], ret(%d)",
                sendWrlist->rxData.completeNum, currentSendNum, ret);
            ret = -EINVAL;
            goto err_send_wrlist;
        }

        for (i = 0; i < sendWrlist->rxData.completeNum; i++) {
            if (qpHdc->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
                opRsp[completeCnt + i].wqeTmp = sendWrlist->rxData.wrRsp[i].wqeTmp;
            } else if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_GDR_ASYN_QP_MODE ||
                       qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
                opRsp[completeCnt + i].db = sendWrlist->rxData.wrRsp[i].db;
            }
        }
        completeCnt += sendWrlist->rxData.completeNum;
        // send wrlist success, continue to send
        if (ret == 0) {
            continue;
        }
        if (ret != -ENOENT && ret != -ENOMEM) {
            hccp_err("[send][send_wrlist]ra hdc message process failed ret(%d), phyId(%u)", ret, qpHdc->phyId);
        }
        goto err_send_wrlist;
    }

err_send_wrlist:
    free(sendWrlist);
    sendWrlist = NULL;
    *(wrlistNum.completeNum) = completeCnt;
    return ret;
}

int RaHdcSendNormalWrlist(struct RaQpHandle *qpHdc, struct WrInfo wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            return RaHdcLiteSendNormalWrlist(qpHdc, wr, opRsp, wrlistNum);
        }
    }

    return RaHdcSendWrlistNormal(qpHdc, wr, opRsp, wrlistNum);
}

int RaHdcGetNotifyBaseAddr(struct RaRdmaHandle *rdmaHandle, unsigned long long *va, unsigned long long *size)
{
    int ret;
    union OpGetNotifyBaData getNotifyBaData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;

    getNotifyBaData.txData.phyId = phyId;
    getNotifyBaData.txData.rdevIndex = rdmaHandle->rdevIndex;

    ret = RaHdcProcessMsg(RA_RS_GET_NOTIFY_BA, phyId, (char *)&getNotifyBaData, sizeof(union OpGetNotifyBaData));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_notify_base_addr]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    *va = getNotifyBaData.rxData.va;
    *size = getNotifyBaData.rxData.size;
    return 0;
}

int RaHdcGetNotifyMrInfo(struct RaRdmaHandle *rdmaHandle, struct MrInfoT *info)
{
    union OpGetNotifyBaData getNotifyBaData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    unsigned int interfaceVersion = 0;
    int ret;

    // check opcode version, reuse RA_RS_GET_NOTIFY_BA
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_GET_NOTIFY_BA, &interfaceVersion);
    if (ret != 0 || interfaceVersion == RA_RS_GET_NOTIFY_BA_VERSION) {
        hccp_err("[get][ra_hdc_notify_mr_info]interface_version(%u) not support, ret(%d)", interfaceVersion, ret);
        return -ENOTSUPP;
    }

    getNotifyBaData.txData.phyId = phyId;
    getNotifyBaData.txData.rdevIndex = rdmaHandle->rdevIndex;

    ret = RaHdcProcessMsg(RA_RS_GET_NOTIFY_BA, phyId, (char *)&getNotifyBaData, sizeof(union OpGetNotifyBaData));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_notify_mr_info]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    info->addr = (void *)(uintptr_t)getNotifyBaData.rxData.va;
    info->size = getNotifyBaData.rxData.size;
    info->access = getNotifyBaData.rxData.access;
    info->lkey = getNotifyBaData.rxData.lkey;
    return 0;
}

int RaHdcRecvWrlist(struct RaQpHandle *qpHdc, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum)
{
    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            return RaHdcLiteRecvWrlist(qpHdc, wr, recvNum, completeNum);
        }
    }

    hccp_warn_rma("qpn:%u qp_mode:%d support_lite:%d not support to recv_wrlist", qpHdc->qpn, qpHdc->qpMode,
        qpHdc->supportLite);

    return -ENOTSUPP;
}

int RaHdcPollCq(struct RaQpHandle *qpHdc, bool isSendCq, unsigned int numEntries, void *wc)
{
    struct rdma_lite_wc_v2 *liteWc = (struct rdma_lite_wc_v2 *)wc;

    if (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT) {
        if (qpHdc->supportLite != LITE_NOT_SUPPORT) {
            return RaHdcLitePollCq(qpHdc, isSendCq, numEntries, liteWc);
        }
    }

    hccp_warn_rma("qpn:%u qp_mode:%d support_lite:%d not support to poll_cq", qpHdc->qpn, qpHdc->qpMode,
        qpHdc->supportLite);

    return -ENOTSUPP;
}

int RaHdcPollTypicalCq(struct RaTypicalCqHandle *cqHdc, unsigned int numEntries, void *wc)
{
    struct rdma_lite_wc_v2 *liteWc = (struct rdma_lite_wc_v2 *)wc;
    struct rdma_lite_cq *liteCq = RaHdcLiteFindTypicalCq(cqHdc->rdmaHandle, cqHdc->cqn);
    if (liteCq == NULL) {
        hccp_warn_rma("cqn:%u liteCq not found in table", cqHdc->cqn);
        return -ENOTSUPP;
    }
    int ret = RaRdmaLitePollCqV2(liteCq, (int)numEntries, liteWc);
    if (ret < 0) {
        hccp_err("ra_rdma_lite_poll_cq_v2 failed, ret %d", ret);
        return ret;
    }
    return ret;
}

int RaHdcNotifyCfgSet(unsigned int phyId, unsigned long long va, unsigned long long size)
{
    union OpNotifyCfgSetData setNotifyBaData = {0};
    int ret;

    setNotifyBaData.txData.phyId = phyId;
    setNotifyBaData.txData.va = va;
    setNotifyBaData.txData.size = size;

    ret = RaHdcProcessMsg(RA_RS_NOTIFY_CFG_SET, phyId, (char *)&setNotifyBaData, sizeof(union OpNotifyCfgSetData));
    CHK_PRT_RETURN(ret,
        hccp_err("[set][ra_hdc_notify_cfg]ra hdc message process failed ret(%d), phyId(%u)", ret, phyId), ret);

    return 0;
}

int RaHdcNotifyCfgGet(unsigned int phyId, unsigned long long *va, unsigned long long *size)
{
    union OpNotifyCfgGetData getNotifyBaData = {0};
    int ret;

    getNotifyBaData.txData.phyId = phyId;

    ret = RaHdcProcessMsg(RA_RS_NOTIFY_CFG_GET, phyId, (char *)&getNotifyBaData, sizeof(union OpNotifyCfgGetData));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_notify_cfg]ra hdc message process failed ret(%d), phyId(%u)", ret, phyId), ret);
    *va = getNotifyBaData.rxData.va;
    *size = getNotifyBaData.rxData.size;
    return 0;
}

STATIC int RaHdcRdevInitWithBackup(struct RaRdmaHandle *rdmaHandle, unsigned int *rdevIndex)
{
    union OpRdevInitWithBackupData rdevInitData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    unsigned int interfaceVersion = 0;
    int ret;

    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_RDEV_INIT_WITH_BACKUP, &interfaceVersion);
    // check opcode version, not support to init rdev with backup info
    if (ret != 0 || interfaceVersion < RA_RS_OPCODE_BASE_VERSION) {
        hccp_warn_init("[init][ra_hdc_rdev]get opcode[%d] not support, ret[%d] != 0 or interfaceVersion[%u] is 0",
            RA_RS_RDEV_INIT_WITH_BACKUP, ret, interfaceVersion);
        return -ENOTSUPP;
    }

    (void)memcpy_s(&(rdevInitData.txData.rdevInfo), sizeof(struct rdev), &rdmaHandle->rdevInfo, sizeof(struct rdev));
    (void)memcpy_s(&(rdevInitData.txData.backupRdevInfo), sizeof(struct rdev), &rdmaHandle->backupInfo.rdevInfo,
        sizeof(struct rdev));
    ret = RaHdcProcessMsg(RA_RS_RDEV_INIT_WITH_BACKUP, phyId, (char *)&rdevInitData,
        sizeof(union OpRdevInitWithBackupData));
    if (ret) {
        hccp_err("[init][ra_hdc_rdev]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        return ret;
    }

    *rdevIndex = rdevInitData.rxData.rdevIndex;
    return 0;
}

int RaHdcRdevInit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType, struct rdev rdevInfo,
    unsigned int *rdevIndex)
{
    union OpRdevInitData rdevInitData = {0};
    unsigned long long *notifyVa = NULL;
    int ret, drvRet;

    ret = RaHdcNotifyBaseAddrInit(notifyType, rdevInfo.phyId, &notifyVa);
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_rdev]ra_hdc_notify_base_addr_init failed, ret(%d), phyId(%u)", ret, rdevInfo.phyId),
        ret);

    // need to init backup rdev: reg notify & normal mr, prepare for aicpu unfold
    if (rdmaHandle->backupInfo.backupFlag) {
        ret = RaHdcRdevInitWithBackup(rdmaHandle, rdevIndex);
        if (ret) {
            goto free_mem;
        }
    } else {
        (void)memcpy_s(&(rdevInitData.txData.rdevInfo), sizeof(struct rdev), &rdevInfo, sizeof(struct rdev));
        ret = RaHdcProcessMsg(RA_RS_RDEV_INIT, rdevInfo.phyId, (char *)&rdevInitData, sizeof(union OpRdevInitData));
        if (ret) {
            hccp_err("[init][ra_hdc_rdev]ra hdc message process failed ret(%d) phyId(%u)", ret, rdevInfo.phyId);
            goto free_mem;
        }
        *rdevIndex = rdevInitData.rxData.rdevIndex;
    }

    ret = RaHdcLiteInit(rdmaHandle, rdevInfo.phyId, *rdevIndex);
    if (ret) {
        hccp_err("[init][ra_hdc_rdev]ra_hdc_lite_init failed ret(%d) phyId(%u)", ret, rdevInfo.phyId);
        goto free_mem;
    }

    return 0;

free_mem:
    drvRet = DlHalMemFree((void *)notifyVa);
    if (drvRet) {
        hccp_err("[init][ra_hdc_rdev]halMemFree failed! drv_ret(%d)", drvRet);
    }
    return ret;
}

int RaHdcRdevGetPortStatus(struct RaRdmaHandle *rdmaHandle, enum PortStatus *status)
{
    union OpRdevGetPortStatusData statusData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret;

    statusData.txData.rdevIndex = rdmaHandle->rdevIndex;
    statusData.txData.phyId = phyId;

    ret = RaHdcProcessMsg(RA_RS_RDEV_GET_PORT_STATUS, phyId, (char *)&statusData,
        sizeof(union OpRdevGetPortStatusData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_hdc_port_status]ra hdc message process failed, ret(%d) phyId(%u)", ret, phyId), ret);

    *status = statusData.rxData.status;
    return 0;
}

int RaHdcRdevRestoreDeinit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType)
{
    // lite thread is an inner thread, make sure it will exit
    RaHdcLiteDeinit(rdmaHandle);

    CHK_PRT_RETURN(notifyType != NOTIFY, hccp_err("[deinit][ra_hdc_rdev]notify_type[%u] error", notifyType), -EINVAL);

    return 0;
}

int RaHdcRdevDeinit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType)
{
    union OpRdevDeinitData rdevDeinitData = {0};
    unsigned long long va, size;
    int ret;

    // lite thread is an inner thread, make sure it will exit
    RaHdcLiteDeinit(rdmaHandle);

    CHK_PRT_RETURN(notifyType != NOTIFY, hccp_err("[deinit][ra_hdc_rdev]notify_type[%u] error", notifyType), -EINVAL);

    rdevDeinitData.txData.rdevIndex = rdmaHandle->rdevIndex;
    rdevDeinitData.txData.phyId = rdmaHandle->rdevInfo.phyId;

    ret = RaHdcProcessMsg(RA_RS_RDEV_DEINIT, rdmaHandle->rdevInfo.phyId, (char *)&rdevDeinitData,
        sizeof(union OpRdevDeinitData));
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_hdc_rdev]ra_hdc_notify_cfg_get failed, ret(%d)", ret), ret);

    ret = RaHdcNotifyCfgGet(rdmaHandle->rdevInfo.phyId, &va, &size);
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_hdc_rdev]ra_hdc_notify_cfg_get failed, ret(%d)", ret), ret);

    ret = DlHalMemFree((void *)(uintptr_t)va);
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_hdc_rdev]halMemFree failed, ret(%d)", ret), ret);

    return 0;
}

int RaHdcSetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int tempDepth, unsigned int *qpNum)
{
    union OpSetTsqpDepthData setTsqpDepthData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret;

    setTsqpDepthData.txData.phyId = rdmaHandle->rdevInfo.phyId;
    setTsqpDepthData.txData.rdevIndex = rdmaHandle->rdevIndex;
    setTsqpDepthData.txData.tempDepth = tempDepth;

    ret = RaHdcProcessMsg(RA_RS_SET_TSQP_DEPTH, phyId, (char *)&setTsqpDepthData, sizeof(union OpSetTsqpDepthData));
    CHK_PRT_RETURN(ret,
        hccp_err("[set][ra_hdc_tsqp_depth]ra hdc message process failed ret(%d), opcode(%d), "
                 "phyId(%u)",
            ret, RA_RS_SET_TSQP_DEPTH, phyId),
        ret);

    *qpNum = setTsqpDepthData.rxData.qpNum;
    return 0;
}

int RaHdcGetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int *tempDepth, unsigned int *qpNum)
{
    union OpGetTsqpDepthData getTsqpDepthData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret;

    getTsqpDepthData.txData.phyId = rdmaHandle->rdevInfo.phyId;
    getTsqpDepthData.txData.rdevIndex = rdmaHandle->rdevIndex;

    ret = RaHdcProcessMsg(RA_RS_GET_TSQP_DEPTH, phyId, (char *)&getTsqpDepthData, sizeof(union OpGetTsqpDepthData));
    CHK_PRT_RETURN(ret,
        hccp_err("[get][ra_hdc_tsqp_depth]ra hdc message process failed ret(%d), opcode(%d), "
                 "phyId(%u)",
            ret, RA_RS_GET_TSQP_DEPTH, phyId),
        ret);

    *tempDepth = getTsqpDepthData.rxData.tempDepth;
    *qpNum = getTsqpDepthData.rxData.qpNum;

    return 0;
}

int RaHdcSetQpAttrQos(struct RaQpHandle *qpHdc, struct QosAttr *attr)
{
    union OpSetQpAttrQosData qpAttrQosData = {0};
    int ret;

    qpAttrQosData.txData.phyId = qpHdc->phyId;
    qpAttrQosData.txData.rdevIndex = qpHdc->rdevIndex;
    qpAttrQosData.txData.qpn = qpHdc->qpn;
    qpAttrQosData.txData.qosAttr.tc = attr->tc;
    qpAttrQosData.txData.qosAttr.sl = attr->sl;

    ret = RaHdcProcessMsg(RA_RS_SET_QP_ATTR_QOS, qpHdc->phyId, (char *)&qpAttrQosData,
        sizeof(union OpSetQpAttrQosData));
    CHK_PRT_RETURN(ret,
        hccp_err("[set][ra_hdc_qp_attr_qos]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId), ret);

    return 0;
}

int RaHdcSetQpAttrTimeout(struct RaQpHandle *qpHdc, unsigned int *timeout)
{
    union OpSetQpAttrTimeoutData qpAttrTimeoutData = {0};
    int ret;

    qpAttrTimeoutData.txData.phyId = qpHdc->phyId;
    qpAttrTimeoutData.txData.rdevIndex = qpHdc->rdevIndex;
    qpAttrTimeoutData.txData.qpn = qpHdc->qpn;
    qpAttrTimeoutData.txData.timeout = *timeout;

    ret = RaHdcProcessMsg(RA_RS_SET_QP_ATTR_TIMEOUT, qpHdc->phyId, (char *)&qpAttrTimeoutData,
        sizeof(union OpSetQpAttrTimeoutData));
    CHK_PRT_RETURN(ret,
        hccp_err("[set][ra_hdc_qp_attr_timeout]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
        ret);

    return 0;
}

int RaHdcSetQpAttrRetryCnt(struct RaQpHandle *qpHdc, unsigned int *retryCnt)
{
    union OpSetQpAttrRetryCntData qpAttrRetryCntData = {0};
    int ret;

    qpAttrRetryCntData.txData.phyId = qpHdc->phyId;
    qpAttrRetryCntData.txData.rdevIndex = qpHdc->rdevIndex;
    qpAttrRetryCntData.txData.qpn = qpHdc->qpn;
    qpAttrRetryCntData.txData.retryCnt = *retryCnt;

    ret = RaHdcProcessMsg(RA_RS_SET_QP_ATTR_RETRY_CNT, qpHdc->phyId, (char *)&qpAttrRetryCntData,
        sizeof(union OpSetQpAttrRetryCntData));
    CHK_PRT_RETURN(ret,
        hccp_err("[set][ra_hdc_qp_attr_retry_cnt]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
        ret);

    return 0;
}

STATIC int RaHdcGetCqeErrInfoNum(struct RaRdmaHandle *rdmaHandle, unsigned int *num)
{
    union OpGetCqeErrInfoNumData cqeErrInfoNumData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    unsigned int interfaceVersion = 0;
    int ret;

    *num = 0;
    // check opcode version, not support to get cqe err info
    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_GET_CQE_ERR_INFO_NUM, &interfaceVersion);
    if (ret != 0 || interfaceVersion == 0) {
        hccp_warn_rma("[get][cqe_err_info_list]get opcode[%d] not support, ret[%d] != 0 or interfaceVersion[%u] is 0",
            RA_RS_GET_CQE_ERR_INFO_NUM, ret, interfaceVersion);
        return 0;
    }

    cqeErrInfoNumData.txData.phyId = phyId;
    cqeErrInfoNumData.txData.rdevIndex = rdmaHandle->rdevIndex;
    ret = RaHdcProcessMsg(RA_RS_GET_CQE_ERR_INFO_NUM, phyId, (char *)&cqeErrInfoNumData,
        sizeof(union OpGetCqeErrInfoNumData));
    CHK_PRT_RETURN(ret, hccp_err("ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    *num = cqeErrInfoNumData.rxData.num;
    return 0;
}

int RaHdcGetCqeErrInfoList(struct RaRdmaHandle *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num)
{
    union OpGetCqeErrInfoListData cqeErrInfoListData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    unsigned int liteCqeErrNum = *num;
    unsigned int cqeErrInfoNum = 0;
    unsigned int hdcCqeErrNum = 0;
    int ret = 0;

    ret = RaHdcLiteGetCqeErrInfoList(rdmaHandle, infoList, &liteCqeErrNum);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][cqe_err_info_list]get lite err info list failed, ret(%d) phyId(%u)", ret, phyId), ret);

    hdcCqeErrNum = *num - liteCqeErrNum;
    *num = liteCqeErrNum;
    // lite cqe err info full up info_list
    if (hdcCqeErrNum == 0) {
        return 0;
    }

    // get cqe err info num failed or not support to get cqe err info, skip get cqe err info list
    ret = RaHdcGetCqeErrInfoNum(rdmaHandle, &cqeErrInfoNum);
    if (ret != 0 || cqeErrInfoNum == 0) {
        return ret;
    }

    cqeErrInfoListData.txData.phyId = phyId;
    cqeErrInfoListData.txData.rdevIndex = rdmaHandle->rdevIndex;
    cqeErrInfoListData.txData.num = hdcCqeErrNum;
    ret = RaHdcProcessMsg(RA_RS_GET_CQE_ERR_INFO_LIST, phyId, (char *)&cqeErrInfoListData,
        sizeof(union OpGetCqeErrInfoListData));
    CHK_PRT_RETURN(ret, hccp_err("ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    if (cqeErrInfoListData.rxData.num > hdcCqeErrNum) {
        hccp_err("[get][cqe_err_info_list]rx_data.num(%u) is invalid, num(%u), phyId(%u)",
            cqeErrInfoListData.rxData.num, hdcCqeErrNum, phyId);
        return -EINVAL;
    }
    ret = memcpy_s(&infoList[liteCqeErrNum], sizeof(struct CqeErrInfo) * hdcCqeErrNum,
        &cqeErrInfoListData.rxData.infoList, sizeof(struct CqeErrInfo) * cqeErrInfoListData.rxData.num);
    if (ret) {
        hccp_err("[get][cqe_err_info_list]memcpy_s info_list failed, ret(%d) phyId(%u) num(%u) rxData.num(%u)", ret,
            phyId, hdcCqeErrNum, cqeErrInfoListData.rxData.num);
        return ret;
    }

    *num = liteCqeErrNum + cqeErrInfoListData.rxData.num;
    return 0;
}

STATIC int RaHdcLiteCleanCq(struct RaQpHandle *qpHandle, bool isSendCq, unsigned int numEntries)
{
    void *wc = NULL;
    int ret;

    if (numEntries == 0) {
        return 0;
    }

    wc = calloc(numEntries, sizeof(struct rdma_lite_wc_v2));
    if (wc == NULL) {
        hccp_err("calloc failed, phyId:%u, qpn:%u", qpHandle->phyId, qpHandle->qpn);
        return -ENOMEM;
    }

    ret = RaHdcLitePollCq(qpHandle, isSendCq, numEntries, wc);
    free(wc);
    if (ret < 0) {
        hccp_err("ra_hdc_lite_poll_cq failed, ret:%d, phyId:%u, qpn:%u", ret, qpHandle->phyId, qpHandle->qpn);
        return ret;
    }

    return 0;
}

STATIC int RaHdcLiteCleanQp(struct RaQpHandle *qpHandle)
{
    unsigned int interfaceVersion = 0;
    int ret;

    // check opcode version, not support to clean qp
    ret = RaHdcGetInterfaceVersion(qpHandle->phyId, RA_RS_QP_BATCH_MODIFY, &interfaceVersion);
    if (ret != 0 || interfaceVersion <= RA_RS_OPCODE_BASE_VERSION) {
        hccp_warn_rma("RA_RS_QP_BATCH_MODIFY interface_version:%u <= %u, not support to clean qp, phyId:%u, qpn:%u",
            interfaceVersion, RA_RS_OPCODE_BASE_VERSION, qpHandle->phyId, qpHandle->qpn);
        return 0;
    }

    // lite qp clean
    ret = RaRdmaLiteCleanQp(qpHandle->liteQp);
    if (ret != 0) {
        hccp_err("ra_rdma_lite_clean_qp failed, ret:%d, phyId:%u, qpn:%u", ret, qpHandle->phyId, qpHandle->qpn);
        return ret;
    }

    return 0;
}

STATIC int RaHdcLiteCleanQueue(struct RaQpHandle *qpHandle, int expectStatus)
{
    int ret;

    // not pause status, no need to clean
    if (expectStatus != RA_QP_STATUS_PAUSE) {
        return 0;
    }

    // not lite or not op mode, no need to clean
    if ((qpHandle->supportLite == 0) ||
        (qpHandle->qpMode != RA_RS_OP_QP_MODE && qpHandle->qpMode != RA_RS_OP_QP_MODE_EXT)) {
        return 0;
    }

    // poll cq to clean lite send cq
    if (qpHandle->sendWrNum > qpHandle->pollCqeNum) {
        ret = RaHdcLiteCleanCq(qpHandle, true, qpHandle->sendWrNum - qpHandle->pollCqeNum);
        if (ret != 0) {
            hccp_err("ra_hdc_lite_clean_cq send_cq failed, ret:%d, phyId:%u, qpn:%u", ret, qpHandle->phyId,
                qpHandle->qpn);
            return ret;
        }
    }

    // poll cq to clean lite recv cq
    if (qpHandle->recvWrNum > qpHandle->pollRecvCqeNum) {
        ret = RaHdcLiteCleanCq(qpHandle, false, qpHandle->recvWrNum - qpHandle->pollRecvCqeNum);
        if (ret < 0) {
            hccp_err("ra_hdc_lite_clean_cq recv_cq failed, ret:%d, phyId:%u, qpn:%u", ret, qpHandle->phyId,
                qpHandle->qpn);
            return ret;
        }
    }

    // lite qp clean
    ret = RaHdcLiteCleanQp(qpHandle);
    if (ret != 0) {
        hccp_err("ra_hdc_lite_clean_qp failed, ret:%d, phyId:%u, qpn:%u", ret, qpHandle->phyId, qpHandle->qpn);
        return ret;
    }

    return 0;
}

int RaHdcQpBatchModify(struct RaRdmaHandle *rdmaHandle, void *qpHdc[], unsigned int num, int expectStatus)
{
    union OpQpBatchModifyData *qpBatchModifyData = NULL;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct RaQpHandle *qpHandle = NULL;
    unsigned int currentSendCnt = 0;
    unsigned int completeCnt = 0;
    int opQpnCnt = 0;
    unsigned int i;
    int ret = 0;

    qpBatchModifyData = calloc(1, sizeof(union OpQpBatchModifyData));
    CHK_PRT_RETURN(qpBatchModifyData == NULL, hccp_err("[send][ra_hdc_qp_batch_modify]qp_batch_modify calloc failed"),
        -ENOMEM);
    while (completeCnt < num) {
        qpBatchModifyData->txData.phyId = phyId;
        qpBatchModifyData->txData.rdevIndex = rdmaHandle->rdevIndex;
        qpBatchModifyData->txData.status = expectStatus;

        currentSendCnt = (num - completeCnt) < RA_MAX_BATCH_QP_MODIFY_NUM ? (num - completeCnt)
                                                                          : RA_MAX_BATCH_QP_MODIFY_NUM;
        opQpnCnt = 0;
        for (i = completeCnt; i < completeCnt + currentSendCnt; i++) {
            qpHandle = (struct RaQpHandle *)qpHdc[i];
            qpBatchModifyData->txData.qpn[opQpnCnt] = (int)qpHandle->qpn;
            opQpnCnt++;

            // avoid poll invalid cqe after modify to RESET state, make sure lite cq ci & qp pointer are valid
            ret = RaHdcLiteCleanQueue(qpHandle, expectStatus);
            if (ret != 0) {
                hccp_err("[modify][qp_batch_modify]ra_hdc_lite_clean_queue failed ret(%d) phyId(%u) qpn(%u)", ret,
                    phyId, qpHandle->qpn);
                goto err_qp_batch_modify;
            }
        }
        qpBatchModifyData->txData.qpnNum = opQpnCnt;

        ret = RaHdcProcessMsg(RA_RS_QP_BATCH_MODIFY, phyId, (char *)qpBatchModifyData,
            sizeof(union OpQpBatchModifyData));
        if (ret) {
            hccp_err("[modify][qp_batch_modify]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
            goto err_qp_batch_modify;
        }
        completeCnt += (unsigned int)opQpnCnt;
    }

err_qp_batch_modify:
    free(qpBatchModifyData);
    qpBatchModifyData = NULL;

    return ret;
}

int RaHdcRdmaSetOps(struct RaRdmaHandle *rdmaHandle, struct RaRdmaOps *rdmaOps)
{
    CHK_PRT_RETURN(rdmaHandle == NULL, hccp_err("ra_hdc_rdma_set_ops rdma_handle is NULL"), -EINVAL);

    rdmaHandle->rdmaOps = rdmaOps;
    return 0;
}

int RaHdcRdmaSaveSnapshot(struct RaRdmaHandle *rdmaHandle, enum SaveSnapshotAction action)
{
    int ret = 0;

    if (rdmaHandle == NULL || rdmaHandle->supportLite == LITE_NOT_SUPPORT || rdmaHandle->disabledLiteThread) {
        return 0;
    }

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->rdevMutex);
    if (action == SAVE_SNAPSHOT_ACTION_PRE_PROCESSING && rdmaHandle->threadStatus == LITE_THREAD_STATUS_RUNNING) {
        rdmaHandle->threadStatus = LITE_THREAD_STATUS_SUSPEND;
    } else if (action == SAVE_SNAPSHOT_ACTION_POST_PROCESSING &&
               rdmaHandle->threadStatus == LITE_THREAD_STATUS_SUSPEND) {
        rdmaHandle->threadStatus = LITE_THREAD_STATUS_RUNNING;
    } else {
        hccp_err("duplicate or incorrect order calls are not allowed, threadStatus[%d] action[%d]",
            rdmaHandle->threadStatus, action);
        ret = -EPERM;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);

    return ret;
}

int RaHdcRdmaRestoreSnapshot(struct RaRdmaHandle *rdmaHandle, struct RaRdmaOps *rdmaOps)
{
    int ret = 0;

    if (rdmaHandle == NULL) {
        return 0;
    }
    ret = RaHdcRdmaSetOps(rdmaHandle, rdmaOps);
    CHK_PRT_RETURN(ret != 0, hccp_err("ra_hdc_rdma_set_ops failed, ret[%d]", ret), ret);

    if (rdmaHandle->supportLite == LITE_NOT_SUPPORT || rdmaHandle->disabledLiteThread) {
        return 0;
    }

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->rdevMutex);
    if (rdmaHandle->threadStatus != LITE_THREAD_STATUS_SUSPEND) {
        hccp_err("incorrect order calls are not allowed, threadStatus[%d]", rdmaHandle->threadStatus);
        ret = -EPERM;
        goto unlock_mutex;
    }
    ret = RaRdmaLiteRestoreSnapshot(rdmaHandle->liteCtx);
    if (ret != 0) {
        hccp_err("ra_rdma_lite_restore_snapshot failed, ret[%d]", ret);
    }

unlock_mutex:
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);
    return ret;
}
