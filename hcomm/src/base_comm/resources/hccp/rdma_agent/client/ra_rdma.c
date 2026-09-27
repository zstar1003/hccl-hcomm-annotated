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
#include <errno.h>
#include "hccp.h"
#include "ra.h"
#include "ra_rs_comm.h"
#include "ra_client_host.h"
#include "ra_rdma.h"

HCCP_ATTRI_VISI_DEF int RaSendNormalWrlist(void *qpHandle, struct WrInfo wr[], struct SendWrRsp opRsp[],
    unsigned int sendNum, unsigned int *completeNum)
{
    struct WrlistSendCompleteNum wrlistNum = {0};
    struct RaQpHandle *raQpHandle = NULL;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || opRsp == NULL || sendNum == 0 || completeNum == NULL,
        hccp_err("[send][ra_wrlist]qp_handle or wr or op_rsp or complete_num is NULL or send_num is zero, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    raQpHandle = (struct RaQpHandle *)qpHandle;
    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSendNormalWrlist == NULL,
        hccp_err("[send][ra_wrlist]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_send_normal_wrlist is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    for (i = 0; i < sendNum; i++) {
        CHK_PRT_RETURN(wr[i].memList.len > MAX_SG_LIST_LEN_MAX,
            hccp_err("[send][ra_wrlist]wr[%u] mem_list.len(%u) > %d", i, wr[i].memList.len, MAX_SG_LIST_LEN_MAX),
            ConverReturnCode(RDMA_OP, -EINVAL));
    }

    wrlistNum.sendNum = sendNum;
    wrlistNum.completeNum = completeNum;
    ret = raQpHandle->rdmaOps->raSendNormalWrlist(raQpHandle, wr, opRsp, wrlistNum);
    return ConverReturnCode(RDMA_OP, ret);
}

void RaRdevSaveNotifyMr(struct RaRdmaHandle *rdmaHandle, int ret, uint64_t va, uint64_t size)
{
    // ret != 0 means unsuccessful, no need to save notify mr info
    if (ret != 0) {
        return;
    }

    rdmaHandle->notifyVa = va;
    rdmaHandle->notifySize = size;
}

STATIC void RaRdevCheckNotifyMr(struct RaRdmaHandle *rdmaHandle, uint64_t va, uint64_t size)
{
    if ((rdmaHandle->notifyVa <= (va + size)) && (va <= (rdmaHandle->notifyVa + rdmaHandle->notifySize))) {
        hccp_run_warn("[check][ra_mr]phyId:%u notify{va:0x%llx size:0x%llx} overlap input{va:0x%llx size:0x%llx}",
            rdmaHandle->rdevInfo.phyId, rdmaHandle->notifyVa, rdmaHandle->notifySize, va, size);
    }
}

HCCP_ATTRI_VISI_DEF int RaRegisterMr(const void *rdmaHandle, struct MrInfoT *info, void **mrHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(rdmaHandle == NULL || info == NULL || mrHandle == NULL,
        hccp_err("[init][ra_mr]rdma_handle or info or mr_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    rdmaHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    CHK_PRT_RETURN(rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raRegisterMr == NULL,
        hccp_err("[init][ra_mr]rdma_ops or rdma_ops->ra_register_mr is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    RaRdevCheckNotifyMr(rdmaHandleTmp, (uint64_t)(uintptr_t)info->addr, info->size);
    ret = rdmaHandleTmp->rdmaOps->raRegisterMr(rdmaHandleTmp, info, mrHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaRemapMr(const void *rdmaHandle, struct MemRemapInfo info[], unsigned int num)
{
    struct RaRdmaHandle *rdmaHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(rdmaHandle == NULL || info == NULL || num == 0 || num > REMAP_MR_MAX_NUM,
        hccp_err("[remap][ra_mr]rdma_handle or info is NULL or num:%u is out of range (0:%u]", num, REMAP_MR_MAX_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    rdmaHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    CHK_PRT_RETURN(rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raRemapMr == NULL,
        hccp_err("[remap][ra_mr]rdma_ops or rdma_ops->ra_remap_mr is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ret = rdmaHandleTmp->rdmaOps->raRemapMr(rdmaHandleTmp, info, num);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaDeregisterMr(const void *rdmaHandle, void *mrHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(rdmaHandle == NULL || mrHandle == NULL, hccp_err("[deinit][ra_mr]rdma_handle or mr_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    rdmaHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    CHK_PRT_RETURN(rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raDeregisterMr == NULL,
        hccp_err("[deinit][ra_mr]rdma_ops or rdma_ops->ra_deregister_mr is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    ret = rdmaHandleTmp->rdmaOps->raDeregisterMr(rdmaHandleTmp, mrHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetLbMax(void *rdevHandle, int *lbMax)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    int ret = 0;

    CHK_PRT_RETURN(rdevHandle == NULL || lbMax == NULL, hccp_err("[get][lbMax]rdevHandle or lbMax is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(rdevHandleTmp->rdmaOps == NULL || rdevHandleTmp->rdmaOps->raGetLbMax == NULL,
        hccp_err("[get][lbMax]rdmaOps is NULL or rdmaOps->raGetLbMax is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = rdevHandleTmp->rdmaOps->raGetLbMax(rdevHandleTmp, lbMax);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSetQpLbValue(void *qpHandle, int lbValue)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret = 0;

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[set][lbValue]qpHandle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSetQpLbValue == NULL,
        hccp_err("[set][lbValue]rdmaOps is NULL or rdmaOps->raSetQpLbValue is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raSetQpLbValue(raQpHandle, lbValue);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetQpLbValue(void *qpHandle, int *lbValue)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret = 0;

    CHK_PRT_RETURN(qpHandle == NULL || lbValue == NULL, hccp_err("[get][lbValue]qpHandle or lbValue is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raGetQpLbValue == NULL,
        hccp_err("[get][lbValue]rdmaOps is NULL or rdmaOps->raGetQpLbValue is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raGetQpLbValue(raQpHandle, lbValue);
    return ConverReturnCode(RDMA_OP, ret);
}