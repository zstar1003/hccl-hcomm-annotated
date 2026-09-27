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
#include "user_log.h"
#include "config_log.h"
#include "hccp_ctx.h"
#include "hccp_async_ctx.h"
#include "ra_hdc_async_ctx.h"
#include "ra_ctx.h"

HCCP_ATTRI_VISI_DEF int RaGetEidByIpAsync(void *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num,
    void **reqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || ip == NULL || eid == NULL || num == NULL || reqHandle == NULL,
        hccp_err("[get][eid_by_ip]ctx_handle or ip or eid or num or req_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM,
        hccp_err("[get][eid_by_ip]num(%u) must be greater than 0"
                 " and less than or equal to %d",
            *num, HCCP_EID_IP_QUERY_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;

    hccp_run_info("Input parameters: phy_id:%u, devIndex:0x%x num:%u", ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex,
        *num);
    ret = RaHdcGetEidByIpAsync(ctxHandleTmp, ip, eid, num, reqHandle);
    if (ret != 0) {
        hccp_err("[get][eid_by_ip]ra_hdc_get_eid_by_ip_async failed, ret(%d) phyId(%u), devIndex(0x%x)", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex);
    }

    return ret;
}

HCCP_ATTRI_VISI_DEF int RaGetIpByEidAsync(void *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num,
    void **reqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || eid == NULL || ip == NULL || num == NULL || reqHandle == NULL,
        hccp_err("[get][IpByEid]ctxHandle or eid or ip or num or reqHandle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM,
        hccp_err("[get][IpByEid]num(%u) must be greater than 0"
                 " and less than or equal to %d",
            *num, HCCP_EID_IP_QUERY_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;

    hccp_run_info("Input parameters: phy_id:%u, devIndex:0x%x num:%u", ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex,
        *num);
    ret = RaHdcGetIpByEidAsync(ctxHandleTmp, eid, ip, num, reqHandle);
    if (ret != 0) {
        hccp_err("[get][IpByEid]RaHdcGetIpByEidAsync failed, ret(%d) phyId(%u), devIndex(0x%x)", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex);
    }

    return ret;
}

HCCP_ATTRI_VISI_DEF int RaCtxLmemRegisterAsync(void *ctxHandle, struct MrRegInfoT *lmemInfo, void **lmemHandle,
    void **reqHandle)
{
    struct RaLmemHandle *lmemHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ctxHandle == NULL || lmemInfo == NULL || lmemHandle == NULL || reqHandle == NULL,
        hccp_err("[init][ra_lmem]ctx_handle or lmem_info or lmem_handle or req_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    lmemHandleTmp = calloc(1, sizeof(struct RaLmemHandle));
    CHK_PRT_RETURN(lmemHandleTmp == NULL,
        hccp_err("[init][ra_lmem]calloc lmem_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = RaHdcCtxLmemRegisterAsync(ctxHandleTmp, lmemInfo, lmemHandleTmp, reqHandle);
    if (ret != 0) {
        hccp_err("[init][ra_lmem]register_async failed, ret(%d) phyId(%u), devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
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

HCCP_ATTRI_VISI_DEF int RaCtxLmemUnregisterAsync(void *ctxHandle, void *lmemHandle, void **reqHandle)
{
    struct RaLmemHandle *lmemHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || lmemHandle == NULL || reqHandle == NULL,
        hccp_err("[deinit][ra_lmem]ctx_handle or lmem_handle or req_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    lmemHandleTmp = (struct RaLmemHandle *)lmemHandle;
    ret = RaHdcCtxLmemUnregisterAsync(ctxHandleTmp, lmemHandleTmp, reqHandle);
    if (ret != 0) {
        hccp_err("[deinit][ra_lmem]unregister_async failed, ret(%d) phyId(%u), devIndex(%u)", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex);
    }

    free(lmemHandleTmp);
    lmemHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpCreateAsync(void *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info,
    void **qpHandle, void **reqHandle)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || attr == NULL || info == NULL || qpHandle == NULL || reqHandle == NULL,
        hccp_err("[init][ra_qp]ctx_handle or attr or info or qp_handle or req_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    qpHandleTmp = calloc(1, sizeof(struct RaCtxQpHandle));
    CHK_PRT_RETURN(qpHandleTmp == NULL,
        hccp_err("[init][ra_qp]calloc qp_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = RaHdcCtxQpCreateAsync(ctxHandleTmp, attr, info, qpHandleTmp, reqHandle);
    if (ret != 0) {
        hccp_err("[init][ra_qp]create_async failed, ret(%d) phyId(%u), devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
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

HCCP_ATTRI_VISI_DEF int RaCtxQpDestroyAsync(void *qpHandle, void **reqHandle)
{
    struct RaCtxQpHandle *qpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || reqHandle == NULL, hccp_err("[deinit][ra_qp]qp_handle or req_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    qpHandleTmp = (struct RaCtxQpHandle *)qpHandle;
    ret = RaHdcCtxQpDestroyAsync(qpHandleTmp, reqHandle);
    if (ret != 0) {
        hccp_err("[deinit][ra_qp]destroy_async failed, ret(%d) phyId(%u), devIndex(%u) qp_id(%u)", ret,
            qpHandleTmp->phyId, qpHandleTmp->devIndex, qpHandleTmp->id);
    }

    free(qpHandleTmp);
    qpHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

STATIC void RaFreeQpHandleBatch(void *qpHandle[], unsigned int num)
{
    unsigned int i;

    for (i = 0; i < num; ++i) {
        if (qpHandle[i] != NULL) {
            free(qpHandle[i]);
            qpHandle[i] = NULL;
        }
    }
}

HCCP_ATTRI_VISI_DEF int RaCtxQpDestroyBatchAsync(void *ctxHandle, void *qpHandle[], unsigned int *num, void **reqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || qpHandle == NULL || reqHandle == NULL || num == NULL,
        hccp_err("[destroy_batch][ra_qp]ctx_handle or qp_handle or req_handle or num is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(*num == 0 || *num > HCCP_MAX_QP_DESTROY_BATCH_NUM,
        hccp_err("[destroy_batch][ra_qp]num(%u) is out of range(0, %u]", *num, HCCP_MAX_QP_DESTROY_BATCH_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaHdcCtxQpDestroyBatchAsync(ctxHandleTmp, qpHandle, num, reqHandle);
    if (ret != 0) {
        hccp_err("[destroy_batch][ra_qp]qp_destroy_batch_async failed, ret[%d] phyId[%u] num[%u] devIndex[%u]", ret,
            ctxHandleTmp->attr.phyId, *num, ctxHandleTmp->devIndex);
    }

    RaFreeQpHandleBatch(qpHandle, *num);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCtxQpImportAsync(void *ctxHandle, struct QpImportInfoT *info, void **remQpHandle,
    void **reqHandle)
{
    struct RaCtxRemQpHandle *remQpHandleTmp = NULL;
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || info == NULL || remQpHandle == NULL || reqHandle == NULL,
        hccp_err("[init][ra_qp]ctx_handle or info or rem_qp_handle or req_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    remQpHandleTmp = calloc(1, sizeof(struct RaCtxRemQpHandle));
    CHK_PRT_RETURN(remQpHandleTmp == NULL,
        hccp_err("[init][ra_qp]calloc rem_qp_handle_tmp failed, errno(%d) phyId(%u) devIndex(%u)", errno,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, -ENOMEM));

    ret = RaHdcCtxQpImportAsync(ctxHandleTmp, info, remQpHandleTmp, reqHandle);
    if (ret != 0) {
        hccp_err("[init][ra_qp]import_async failed, ret(%d) phyId(%u), devIndex(%u)", ret, ctxHandleTmp->attr.phyId,
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

HCCP_ATTRI_VISI_DEF int RaCtxQpUnimportAsync(void *remQpHandle, void **reqHandle)
{
    struct RaCtxRemQpHandle *remQpHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(remQpHandle == NULL || reqHandle == NULL,
        hccp_err("[deinit][ra_qp]rem_qp_handle or req_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    remQpHandleTmp = (struct RaCtxRemQpHandle *)remQpHandle;
    ret = RaHdcCtxQpUnimportAsync(remQpHandleTmp, reqHandle);
    if (ret != 0) {
        hccp_err("[deinit][ra_qp]unimport_async failed, ret(%d) phyId(%u) devIndex(%u) qp_id(%u)", ret,
            remQpHandleTmp->phyId, remQpHandleTmp->devIndex, remQpHandleTmp->id);
    }

    free(remQpHandleTmp);
    remQpHandleTmp = NULL;
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetTpInfoListAsync(void *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num, void **reqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || cfg == NULL || reqHandle == NULL,
        hccp_err("[get][ra_tp_info]ctx_handle or cfg or req_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(infoList == NULL || num == NULL, hccp_err("[get][ra_tp_info]info_list or num is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(*num == 0 || *num > HCCP_MAX_TPID_INFO_NUM,
        hccp_err("[get][ra_tp_info]*num(%u) is out of range[0, %d]", *num, HCCP_MAX_TPID_INFO_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaHdcGetTpInfoListAsync(ctxHandleTmp, cfg, infoList, num, reqHandle);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_tp_info]get_tp_info_list_async failed, ret[%d] phyId[%u], devIndex"
                 "[%u]",
            ret, ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr,
    void **reqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || attrBitmap == NULL || reqHandle == NULL || attr == NULL,
        hccp_err("[get][ra_tp_attr]ctx_handle or attr_bitmap or req_handle or attr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaHdcGetTpAttrAsync(ctxHandleTmp, tpHandle, attrBitmap, attr, reqHandle);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_tp_attr]get_tp_attr_async failed, ret[%d] phyId[%u] devIndex[0x%x]", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr,
    void **reqHandle)
{
    struct RaCtxHandle *ctxHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(ctxHandle == NULL || attr == NULL || reqHandle == NULL,
        hccp_err("[set][ra_tp_attr]ctx_handle or attr or req_handle or attr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ctxHandleTmp = (struct RaCtxHandle *)ctxHandle;
    ret = RaHdcSetTpAttrAsync(ctxHandleTmp, tpHandle, attrBitmap, attr, reqHandle);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[set][ra_tp_attr]set_tp_attr_async failed, ret[%d] phyId[%u] devIndex[0x%x]", ret,
            ctxHandleTmp->attr.phyId, ctxHandleTmp->devIndex),
        ConverReturnCode(RDMA_OP, ret));

    return ConverReturnCode(RDMA_OP, ret);
}
