/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include "ra_client_host.h"

HCCP_ATTRI_VISI_DEF int RaQpCreateWithCQWithAttrs(void *rdevHandle, struct QpExtAttrs *extAttrs, unsigned int sendCqn,
    unsigned int recvCqn, void **qpHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raQpCreateWithCQWithAttrs == NULL,
        hccp_err("[create][ra_qp_with_cq_attrs]rdev_handle is NULL or func is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[create][ra_qp_with_cq_attrs]qp_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(extAttrs == NULL, hccp_err("[create][ra_qp_with_cq_attrs]ext_attrs is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(extAttrs->version != QP_CREATE_WITH_ATTR_VERSION,
        hccp_err("[create][ra_qp_with_cq_attrs]attr version[%d] mismatch, expect [%d]", extAttrs->version,
            QP_CREATE_WITH_ATTR_VERSION),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(extAttrs->qpMode < 0 || extAttrs->qpMode >= RA_RS_ERR_QP_MODE,
        hccp_err("[create][ra_qp_with_cq_attrs]QP mode[%d] must be greater than or equal to 0 and less than %d",
            extAttrs->qpMode, RA_RS_ERR_QP_MODE),
        ConverReturnCode(RDMA_OP, -EINVAL));
    // no need and disallow to set data_plane_flag, set it to default value 0
    extAttrs->dataPlaneFlag.value = 0;

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_qp_with_cq_attrs]phyId(%u) must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u] qp_mode[%d] sendCqn[%u] recvCqn[%u] qpAttr.cap{%u,%u,%u,%u,%u}"
                  " qp_type[%u] sqSigAll[%d], cnt[%u]",
        phyId, extAttrs->qpMode, sendCqn, recvCqn, extAttrs->qpAttr.cap.max_send_wr, extAttrs->qpAttr.cap.max_recv_wr,
        extAttrs->qpAttr.cap.max_send_sge, extAttrs->qpAttr.cap.max_recv_sge, extAttrs->qpAttr.cap.max_inline_data,
        extAttrs->qpAttr.qp_type, extAttrs->qpAttr.sq_sig_all, rdmaHandleTmp->qpCnt);

    rdmaHandleTmp->qpCnt++;
    ret = rdmaHandleTmp->rdmaOps->raQpCreateWithCQWithAttrs(rdmaHandleTmp, extAttrs, sendCqn, recvCqn, qpHandle);
    CHK_PRT_RETURN(ret != 0 || *qpHandle == NULL,
        hccp_err("[create][ra_qp_with_cq_attrs]create qp failed, ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaTypicalCqCreate(void *rdevHandle, unsigned int cqDepth, unsigned int *cqn, void **cqHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raTypicalCqCreate == NULL,
        hccp_err("[create][ra_typical_cq]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(cqHandle == NULL || cqn == NULL, hccp_err("[create][ra_typical_cq]cq_handle or cqn is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_typical_cq]phyId(%u) exceeds max(%u)", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("RaTypicalCqCreate: phyId[%u], cqDepth[%u]", phyId, cqDepth);

    ret = rdmaHandleTmp->rdmaOps->raTypicalCqCreate(rdmaHandleTmp, cqDepth, cqn, cqHandle);
    CHK_PRT_RETURN(ret != 0 || *cqHandle == NULL,
        hccp_err("[create][ra_typical_cq]create cq failed ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaTypicalCqDestroy(void *rdevHandle, unsigned int cqn, void *cqHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raTypicalCqDestroy == NULL,
        hccp_err("[destroy][ra_typical_cq]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(cqHandle == NULL, hccp_err("[destroy][ra_typical_cq]cq_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[destroy][ra_typical_cq]phyId(%u) exceeds max(%u)", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("RaTypicalCqDestroy: phyId[%u], cqn[%u]", phyId, cqn);

    ret = rdmaHandleTmp->rdmaOps->raTypicalCqDestroy(rdmaHandleTmp, cqn, cqHandle);
    CHK_PRT_RETURN(ret != 0, hccp_err("[destroy][ra_typical_cq]destroy cq failed ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaQpDestroyWithoutCQ(void *qpHandle)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[destroy][ra_qp]qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raQpDestroyWithoutCQ == NULL,
        hccp_err(
            "[destroy][ra_qp]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_qp_destroy_without_cq is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: qpn[%u], phyId[%u], rdevIndex[%u] qpMode[%d] flag[%d]", raQpHandle->qpn,
        raQpHandle->phyId, raQpHandle->rdevIndex, raQpHandle->qpMode, raQpHandle->flag);

    ret = raQpHandle->rdmaOps->raQpDestroyWithoutCQ(raQpHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSendWrVerbs(void *qpHandle, struct SendWrVerbs *wr, struct SendWrRsp *opRsp)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || wr->sgList == NULL || opRsp == NULL,
        hccp_err("[send][ra_wr]qp_handle or wr or sg_list or op_rsp is NULL, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSendWrVerbs == NULL,
        hccp_err("[send][ra_wr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_send_wr_verbs is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raSendWrVerbs(raQpHandle, wr, opRsp);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaRecvWrVerbs(void *qpHandle, struct RecvWrVerbs *wr)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || wr->sgList == NULL,
        hccp_err("[recv][ra_wr]qp_handle or wr or sg_list is NULL, para error!"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raRecvWrVerbs == NULL,
        hccp_err("[recv][ra_wr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_recv_wr_verbs is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raRecvWrVerbs(raQpHandle, wr);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaPollTypicalCq(void *cqHandle, unsigned int numEntries, void *wc)
{
    struct RaTypicalCqHandle *cqHdc = (struct RaTypicalCqHandle *)cqHandle;
    if (cqHandle == NULL || wc == NULL) {
        hccp_err("[ra_poll]cq_handle is NULL or wc is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }
    if (cqHdc->rdmaOps == NULL || cqHdc->rdmaOps->raPollTypicalCq == NULL) {
        hccp_err("[ra_poll]rdma_ops is NULL or ra_poll_typical_cq is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }
    int ret = cqHdc->rdmaOps->raPollTypicalCq(cqHdc, numEntries, wc);
    if (ret < 0) {
        return ConverReturnCode(RDMA_OP, ret);
    }
    return ret;
}