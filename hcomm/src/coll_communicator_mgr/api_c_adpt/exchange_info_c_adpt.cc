/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_res_expt.h"
#include "hccl_comm_pub.h"
#include "coll_comm_config_consistency.h"

using namespace hccl;

HcclResult HcclCommAddExchangeInfo(HcclComm comm, const void* data, uint32_t length)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(data);
    CHK_PRT_RET(length == 0, HCCL_ERROR("[HcclCommAddExchangeInfo] length is 0."), HCCL_E_PARA);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    return collCommConfigConsistency.AddExchangeInfo(data, length);
}

HcclResult
HcclCommGetExchangeInfo(HcclComm comm, uint32_t remoteRank, uint32_t length, void* data, uint32_t* actualLength)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(data);
    CHK_PTR_NULL(actualLength);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    return collCommConfigConsistency.GetExchangeInfo(remoteRank, length, data, actualLength);
}

HcclResult HcclCommResetExchangeInfo(HcclComm comm)
{
    CHK_PTR_NULL(comm);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    return collCommConfigConsistency.ResetExchangeInfo();
}
