/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins.h"
#include "ccu_ctx_mgr.h"
#include "orion_adapter_rts.h"

namespace Hccl {

void CcuInstruction::Translate(std::vector<std::vector<CcuTaskParam>>& taskParam) const
{
    s32 deviceLogicId = HrtGetDevice();
    std::unique_ptr<CcuTaskArg> ccuTaskArg = GetTaskArg();
    if (ccuTaskArg == nullptr) {
        HCCL_ERROR("[CcuInstruction][Translate]ccuTaskArg is null");
        return;
    }

    HcclResult res = CcuCtxMgr::GetTaskParam(deviceLogicId, *ccuTaskArg, GetExecId(), taskParam);
    if (res != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuInstruction][Translate]GetTaskParam failed, res[%d]", res);
        return;
    }

    return;
}

} // namespace Hccl
