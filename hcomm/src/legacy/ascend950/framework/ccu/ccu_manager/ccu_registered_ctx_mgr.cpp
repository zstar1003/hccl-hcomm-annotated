/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_registered_ctx_mgr.h"
#include "ins_exe_que.h"
#include "internal_exception.h"

namespace Hccl {

// 输入ccuIns和resPack，输出执行Id，判断对应指令是否注册过
bool RegisteredCcuCtxMgr::HasRegistered(const CcuCtxSignature& ctxSignature, const uintptr_t& resPackId, u64& execId)
{
    // 查询是否注册过
    if (registeredIds.find(ctxSignature) != registeredIds.end()
        && registeredIds[ctxSignature].find(resPackId) != registeredIds[ctxSignature].end()) {
        execId = registeredIds[ctxSignature][resPackId];
        return true;
    }
    return false;
}

u64 RegisteredCcuCtxMgr::Register(
    std::unique_ptr<CcuCtxGroup> ccuCtxGroupPtr, const CcuCtxSignature& ctxSignature, const uintptr_t& resPackId,
    bool isFuncBlock)
{
    CHECK_NULLPTR(ccuCtxGroupPtr, "[RegisteredCcuCtxMgr::Register] ccuCtxGroupPtr is nullptr!");
    // 注册
    InsExeQue::ExtInsExeEntityId execId = 0;
    InsExeQue::ExtInsExeEntity entity;
    entity.isFuncBlock = isFuncBlock; //  是否需要将ctxGroup翻译为FuncBlock
    entity.ctxGroup = std::move(*ccuCtxGroupPtr);
    HcclResult res = InsExeQue::RegisterExtendInstruction(devLogicId, entity, execId);
    if (res != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>(
            StringFormat("[RegisteredCcuCtxMgr::%s] errNo[0x%016llx] Register fail.", __func__, HCCL_ERROR_CODE(res)));
    }

    // 保存execId
    registeredIds[ctxSignature][resPackId] = static_cast<u64>(execId);

    HCCL_INFO(
        "[RegisteredCcuCtxMgr::%s] register ctxSignature[%s] resPackId[%u] end, execId[%llu].", __func__,
        ctxSignature.Describe().c_str(), resPackId, execId);
    return execId;
}

RegisteredCcuCtxMgr::~RegisteredCcuCtxMgr()
{
    HCCL_INFO("[RegisteredCcuCtxMgr::%s] start.", __func__);

    for (auto& registeredId : registeredIds) {
        for (auto& execIdInfo : registeredId.second) {
            InsExeQue::ExtInsExeEntityId entityId = static_cast<InsExeQue::ExtInsExeEntityId>(execIdInfo.second);
            InsExeQue::DeregisterExtendInstruction(devLogicId, entityId);
            HCCL_INFO("[RegisteredCcuCtxMgr:%s]Destroy execId[%u]", __func__, entityId);
        }
    }

    registeredIds.clear();
    HCCL_INFO("[RegisteredCcuCtxMgr::%s] end.", __func__);
}

} // namespace Hccl
