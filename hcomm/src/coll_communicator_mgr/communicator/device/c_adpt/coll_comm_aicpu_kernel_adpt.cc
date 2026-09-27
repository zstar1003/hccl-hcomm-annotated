/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu_kernel_adpt.h"
#include "coll_comm_aicpu_mgr.h"
#include "log.h"

namespace {
// Acquire → 操作 → Release 的公共骨架：获取 CommEngineResMgr
inline HcclResult
AcquireCommEngineRes(const char* groupName, CollCommAicpu*& outComm, CommEngineResAicpuMgr*& outResMgr)
{
    outComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(groupName);
    CHK_PRT_RET(outComm == nullptr, HCCL_ERROR("%s aicpuComm is null, group[%s]", __func__, groupName), HCCL_E_PTR);

    outResMgr = outComm->GetCommEngineResMgr();
    if (outResMgr == nullptr) {
        HCCL_ERROR("[%s] commEngineResMgr is null, group[%s]", __func__, groupName);
        CollCommAicpuMgr::GetInstance().ReleaseComm(groupName);
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult CollCommAicpuKernelAdptInitThreads(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    std::string group = param->hcomId;
    HCCL_INFO("[%s]group[%s]", __func__, group.c_str());

    CollCommAicpu* aicpuComm = nullptr;
    CommEngineResAicpuMgr* commEngineResMgr = nullptr;
    CHK_RET(AcquireCommEngineRes(group.c_str(), aicpuComm, commEngineResMgr));

    HcclResult ret = commEngineResMgr->InitThreads(param);
    CHK_PRT_CONT(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] Failed to init threads group[%s]", __func__, HCCL_ERROR_CODE(ret), group.c_str()));
    CollCommAicpuMgr::GetInstance().ReleaseComm(group);
    return ret;
}

HcclResult CollCommAicpuKernelAdptInitChannel(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(commParam);
    std::string group = commParam->hcomId;
    HCCL_INFO("[%s]group[%s]", __func__, group.c_str());

    CollCommAicpu* aicpuComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(group);
    CHK_PRT_RET(
        aicpuComm == nullptr, HCCL_ERROR("%s aicpuComm is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);

    ChannelAicpuMgr* channelMgr = aicpuComm->GetChannelMgr();
    if (channelMgr == nullptr) {
        HCCL_ERROR("[%s] channelMgr is null, group[%s]", __func__, group.c_str());
        CollCommAicpuMgr::GetInstance().ReleaseComm(group);
        return HCCL_E_PTR;
    }
    HcclResult ret = channelMgr->AllocChannelResource(commParam);
    CHK_PRT_CONT(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] Failed to init channels group[%s]", __func__, HCCL_ERROR_CODE(ret), group.c_str()));
    CollCommAicpuMgr::GetInstance().ReleaseComm(group);
    return ret;
}

HcclResult CollCommAicpuKernelAdptUpdateChannel(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(commParam);
    std::string group = commParam->hcomId;
    HCCL_INFO("[%s]group[%s]", __func__, group.c_str());

    CollCommAicpu* aicpuComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(group);
    CHK_PRT_RET(
        aicpuComm == nullptr, HCCL_ERROR("%s aicpuComm is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);

    // 通过 CollCommAicpu::Resume 统一处理：通道恢复 + commStatus/isErrorReported/nsRecovery 状态重置
    HcclResult ret = aicpuComm->Resume(commParam);
    CHK_PRT_CONT(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] Failed to update channels group[%s]", __func__, HCCL_ERROR_CODE(ret), group.c_str()));
    CollCommAicpuMgr::GetInstance().ReleaseComm(group);
    return ret;
}

HcclResult CollCommAicpuKernelAdptInitNotify(NotifyMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    std::string group = param->hcomId;
    HCCL_INFO("[%s]group[%s]", __func__, group.c_str());

    CollCommAicpu* aicpuComm = nullptr;
    CommEngineResAicpuMgr* commEngineResMgr = nullptr;
    CHK_RET(AcquireCommEngineRes(group.c_str(), aicpuComm, commEngineResMgr));
    HcclResult ret = HCCL_E_INTERNAL;
    const char* opName = nullptr;
    if (param->freeFlag) {
        ret = commEngineResMgr->NotifyFree(param);
        opName = "free";
    } else {
        ret = commEngineResMgr->NotifyAlloc(param);
        opName = "alloc";
    }
    CHK_PRT_CONT(
        ret != HCCL_SUCCESS, HCCL_ERROR(
                                 "[%s]errNo[0x%016llx] Failed to %s notify group[%s]", __func__, HCCL_ERROR_CODE(ret),
                                 opName, group.c_str()));
    HCCL_INFO(
        "[%s] comm identifier[%s], notify op[%u] end, num[%u]", __func__, group.c_str(), param->freeFlag,
        param->notifyNum);
    CollCommAicpuMgr::GetInstance().ReleaseComm(group);
    return ret;
}
