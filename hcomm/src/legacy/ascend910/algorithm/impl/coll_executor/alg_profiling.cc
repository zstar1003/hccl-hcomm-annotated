/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_profiling.h"
#include "adapter_rts_common.h"

namespace hccl {

AlgWrap& AlgWrap::GetInstance()
{
    static AlgWrap algWrap;
    return algWrap;
}

HcclResult
AlgWrap::RegisterAlgCallBack(const std::string& comm, void* userPtr, TaskCallBack callback, s32 deviceLogicID)
{
    CHK_PRT_RET(
        initialized_ == false, HCCL_WARNING("[alg_profiling][RegisterAlgCallBack] AlgWrap has not initialized"),
        HCCL_SUCCESS);

    if (deviceLogicID < 0 || static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_ERROR("[alg_profiling][RegisteralgCallBack] deviceLogicID %d is invalid", deviceLogicID);
        return HCCL_E_PARA;
    }
    std::lock_guard<std::mutex> lock(aivCallBackMutex_);
    aivCallBackMap_[comm][deviceLogicID] = callback;
    aivCallBackUserPtrMap_[comm][deviceLogicID] = userPtr;
    return HCCL_SUCCESS;
}

void AlgWrap::UnregisterAlgCallBack(const std::string& comm)
{
    if (!initialized_) {
        HCCL_WARNING("[alg_profiling][UnRegisterAlgCallBack] AlgWrap has not initialized yet");
        return;
    }

    std::lock_guard<std::mutex> lock(aivCallBackMutex_);
    aivCallBackMap_.erase(comm);
    aivCallBackUserPtrMap_.erase(comm);
}

HcclResult AlgWrap::TaskAivProfiler(const std::string& comm, struct TaskParaGeneral& taskParaGeneral)
{
    CHK_PRT_RET(
        initialized_ == false, HCCL_WARNING("[alg_profiling][RegisterAlgCallBack] AlgWrap has not initialized"),
        HCCL_SUCCESS);

    s32 deviceLogicID = INVALID_INT;
    CHK_RET(hrtGetDevice(&deviceLogicID));
    if (deviceLogicID < 0 || static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_ERROR("[alg_profiling][TaskAivProfiler] deviceLogicID %d is invalid", deviceLogicID);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(aivCallBackMutex_);
    if (aivCallBackMap_.find(comm) == aivCallBackMap_.end()
        || aivCallBackUserPtrMap_.find(comm) == aivCallBackUserPtrMap_.end()) {
        HCCL_ERROR("[alg_profiling][TaskAivProfiler] comm %s is invalid", comm.c_str());
        return HCCL_E_PARA;
    }

    auto* aivCallBack = aivCallBackMap_[comm][deviceLogicID];
    auto* aivCallBackUserPtr = aivCallBackUserPtrMap_[comm][deviceLogicID];
    if (aivCallBack == nullptr || aivCallBackUserPtr == nullptr) {
        HCCL_ERROR("[alg_profiling][TaskAivProfiler] aivCallBack or aivCallBackUserPtr is invalid");
        return HCCL_E_PTR;
    }

    // 回调
    (aivCallBack)(aivCallBackUserPtr, static_cast<void*>(&taskParaGeneral), sizeof(struct TaskParaGeneral));
    return HCCL_SUCCESS;
}

} // namespace hccl
