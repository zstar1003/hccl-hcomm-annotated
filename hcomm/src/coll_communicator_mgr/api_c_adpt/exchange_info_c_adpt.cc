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

// [中文导读] 域级一致性信息的登记入口：将上层提供的描述交给本Rank的CollCommConfigConsistency管理。
// [中文导读] length是描述字节数，不是张量元素数；描述在后续建链时交换，由上层读取后比较算子参数。
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

// [中文导读] 按remoteRank取已交换的描述；非空描述要求length精确等于记录长度，actualLength返回记录长度。
// [中文导读] 成功读取后会消费该Peer的记录；长度不符时即使已写actualLength，也不表示数据已复制成功。
// [中文导读] HCCL用这些元信息比较各Rank参数；该查询不等同于HcommRead类用户数据原语。
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

// [中文导读] 重置一致性信息管理器的交换信息，不是销毁通信域、Thread或Channel。
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
