/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opbase_stream_manager.h"
#include "log.h"
#include "exception_util.h"
#include "communicator_impl.h"
#include "invalid_params_exception.h"

namespace Hccl {

OpbaseStreamManager::~OpbaseStreamManager() { DECTOR_TRY_CATCH("OpbaseStreamManager", Clear()); }

void OpbaseStreamManager::ReplaceMaster(std::unique_ptr<Stream> stream)
{
    master = std::move(stream);
    master->SetStmMode(HrtStreamGetMode(master->GetPtr()));
}

void OpbaseStreamManager::RegisterMaster(std::unique_ptr<Stream> stream)
{
    HCCL_INFO("[OpbaseStreamManager::%s] start.", __func__);

    if (master == nullptr) {
        ReplaceMaster(std::move(stream));
        HCCL_INFO("[OpbaseStreamManager::%s] end, master is nullptr.", __func__);
        return;
    }
    if (*master == *stream) {
        HCCL_INFO("[OpbaseStreamManager::%s] end, master is same as stream.", __func__);
        return;
    }
    ReplaceMaster(std::move(stream));

    HCCL_INFO("[OpbaseStreamManager::%s] end.", __func__);
}

void OpbaseStreamManager::Clear() { slaves.clear(); }

Stream* OpbaseStreamManager::GetOrCreateSlave()
{
    HCCL_INFO("[OpbaseStreamManager::%s] start.", __func__);

    // 如果从流不够用，就申请一个新的从流
    u32 slavesSize = slaves.size();
    HCCL_INFO("[OpbaseStreamManager::%s] slavesSize[%u] slaveIndex[%u]", __func__, slavesSize, slaveIndex);
    if (slaveIndex >= slavesSize) {
        slaves.emplace_back(std::make_unique<Stream>(comm->GetOpAiCpuTSFeatureFlag(), false)); // 算子粒度
        if (master != nullptr && !comm->GetOpAiCpuTSFeatureFlag()) {                           // 算子粒度
            slaves[slaveIndex]->SetStmMode(master->GetMode());
        }
    }

    HCCL_INFO("[OpbaseStreamManager::%s] end, get index[%u] slave stream", __func__, slaveIndex);
    return slaves[slaveIndex++].get();
}

Stream* OpbaseStreamManager::GetSlave(u32 index) const
{
    if (index >= slaves.size()) {
        THROW<InvalidParamsException>(StringFormat("[OpbaseStreamManager::%s] index[%u] is invalid.", __func__, index));
    }
    return slaves[index].get();
}

} // namespace Hccl
