/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_c_adpt.h"
#include "coll_comm_mgr.h"

using namespace hccl;

/**
 * @note 职责：集合通信的通信域管理的C接口的C到C++适配
 */
HcclResult HcclCommGetStatus(const char* commId, HcclCommStatus* status)
{
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(status);
    HcclComm comm = nullptr;
    // 公共流程, 需要同时获取单算子和图模式的通信域
    CHK_RET(HcomGetCommHandleByGroup(commId, &comm));
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("%s hcclComm is nullptr, commId[%s]", __func__, commId), HCCL_E_PTR);
    return static_cast<hccl::hcclComm*>(comm)->GetCommStatus(*status);
}
