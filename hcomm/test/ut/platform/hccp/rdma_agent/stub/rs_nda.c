/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_nda.h"
#include "ra_rs_comm.h"
#include "rs_nda.h"

int RsNdaGetDirectFlag(unsigned int phyId, unsigned int rdevIndex, int* directFlag)
{
    *directFlag = DIRECT_FLAG_PCIE;
    return 0;
}

int RsNdaCqCreate(
    unsigned int phyId, unsigned int rdevIndex, struct NdaCqInitAttr* attr, struct NdaCqInfo* info, void** ibvCqExt)
{
    return 0;
}

int RsNdaCqDestroy(unsigned int phyId, unsigned int rdevIndex, void* ibvCqExt) { return 0; }

int RsNdaQpCreate(
    unsigned int phyId, unsigned int rdevIndex, struct NdaQpInitAttr* attr, struct NdaQpInfo* info,
    struct RsQpResp* qpResp)
{
    return 0;
}

int RsNdaQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn) { return 0; }

int RsGetQpHyperFeature(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct HyperFeature* hyperFeature)
{
    return 0;
}
