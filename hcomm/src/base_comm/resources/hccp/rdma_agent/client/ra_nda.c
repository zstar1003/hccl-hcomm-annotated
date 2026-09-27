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
#include "hccp_nda.h"
#include "ra_peer_nda.h"
#include "ra_rs_comm.h"

HCCP_ATTRI_VISI_DEF int RaNdaGetDirectFlag(void *rdmaHandle, int *directFlag)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    int ret = 0;

    CHK_PRT_RETURN(rdmaHandle == NULL || directFlag == NULL,
        hccp_err("[get][directFlag]rdmaHandle or directFlag is NULL, invalid"), ConverReturnCode(RDMA_OP, -EINVAL));

    ret = RaPeerNdaGetDirectFlag(rdevHandleTmp, directFlag);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaNdaCqCreate(void *rdmaHandle, struct NdaCqInitAttr *attr, struct NdaCqInfo *info,
    void **cqHandle)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    int ret = 0;

    CHK_PRT_RETURN(rdmaHandle == NULL || attr == NULL || info == NULL || cqHandle == NULL,
        hccp_err("[create][RaNdaCq]rdmaHandle or attr or info or cqHandle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = RaPeerNdaCqCreate(rdevHandleTmp, attr, info, cqHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaNdaCqDestroy(void *rdmaHandle, void *cqHandle)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    int ret = 0;

    CHK_PRT_RETURN(rdmaHandle == NULL || cqHandle == NULL, hccp_err("[destroy][RaNdaCq]rdmaHandle or cqHandle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = RaPeerNdaCqDestroy(rdevHandleTmp, cqHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaNdaQpCreate(void *rdmaHandle, struct NdaQpInitAttr *attr, struct NdaQpInfo *info,
    void **qpHandle)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    int ret = 0;

    CHK_PRT_RETURN(rdmaHandle == NULL || attr == NULL || info == NULL || qpHandle == NULL,
        hccp_err("[create][RaNdaQp]rdmaHandle or attr or info or qpHandle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = RaPeerNdaQpCreate(rdevHandleTmp, attr, info, qpHandle);
    return ConverReturnCode(RDMA_OP, ret);
}