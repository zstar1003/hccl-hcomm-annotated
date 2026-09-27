/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_comm_destroy_func.h"
#include "kfc.h"
#include "communicator_impl_lite.h"
#include "communicator_impl_lite_manager.h"

namespace Hccl {

AicpuCommDestroyFunc& AicpuCommDestroyFunc::GetInstance()
{
    static AicpuCommDestroyFunc func;
    return func;
}

void AicpuCommDestroyFunc::Call()
{
    std::vector<CommunicatorImplLite*> commLites = CommunicatorImplLiteMgr::GetInstance().GetAll();
    for (auto& comm : commLites) {
        if (!comm->IsCommReady()) {
            continue;
        }

        KfcCommand cmd = comm->BackGroundGetCmd();
        if (cmd != KfcCommand::DESTROY_AICPU_COMM) {
            continue;
        }
        HCCL_INFO("Received KfcCommand[DESTROY_AICPU_COMM]");
        comm->BackGroundSetStatus(KfcStatus::DESTROY_AICPU_COMM_DONE);
        auto idIndex = comm->GetCommIdIndex();
        CommunicatorImplLiteMgr::GetInstance().DestroyComm(idIndex);
        HCCL_INFO("Send KfcStatus[DESTROY_AICPU_COMM_DONE]");
    }
}

} // namespace Hccl
