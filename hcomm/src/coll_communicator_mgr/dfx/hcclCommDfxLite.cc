/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommDfxLite.h"
#include "hccl_common.h"
#include "dfx_profiling_handler_lite.h"
#include "res_pub.h"
#include "dfx_circular_queue.h"
#include "prof_sal_lite.h"
#include "unified_platform/pub_inc/config_plf_log_v2.h"

namespace hccl {
// HcclCommDfxLite构造函数实现
HcclCommDfxLite::HcclCommDfxLite() {}

HcclCommDfxLite::~HcclCommDfxLite()
{
    delete profilingImpl_;
    profilingImpl_ = nullptr;
    if (queueInitialized_ && opInfoQueue_ != nullptr) {
        delete static_cast<Hccl::DfxOpInfoCircularQueue*>(opInfoQueue_);
        opInfoQueue_ = nullptr;
    }
}

// HcclCommDfxLite初始化流程 - 修改为返回HcclResult类型
HcclResult HcclCommDfxLite::Init(u32 deviceId, const std::string& commTag, u32 rankSize, u32 localRank)
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[HcclCommDfxLite][Init] Init begin deviceId[%u], commTag[%s]", deviceId, commTag.c_str());
    deviceId_ = deviceId;
    commTag_ = commTag;
    rankSize_ = rankSize;
    localRank_ = localRank;

    EXCEPTION_CATCH(profilingImpl_ = new HcclCommProfilingLite(deviceId_), return HCCL_E_PTR);
    CHK_RET(profilingImpl_->Init());
    opInfoQueue_ = new Hccl::DfxOpInfoCircularQueue();
    queueInitialized_ = true;

    groupNameHash_ = Hccl::DfxProfilingHandlerLite::GetInstance().GetProfHashId(commTag_.c_str(), commTag_.length());
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

// HcclCommDfxLite接口实现 - 修改为返回HcclResult类型
HcclResult HcclCommDfxLite::SetCurrDfxOpInfo(const Hccl::DfxDfxOpInfo* newDfxOpInfo)
{
    CHK_PTR_NULL(newDfxOpInfo);
    auto* queue = static_cast<Hccl::DfxOpInfoCircularQueue*>(opInfoQueue_);
    CHK_PTR_NULL(queue);
    auto* slot = static_cast<Hccl::DfxDfxOpInfo*>(queue->NextSlot());
    CHK_PTR_NULL(slot);
    *slot = *newDfxOpInfo;
    auto it = Hccl::CMD_OP_TYPE_INFO_MAP.find(static_cast<HcclCMDType>(slot->opType));
    if (it == Hccl::CMD_OP_TYPE_INFO_MAP.end()) {
        HCCL_WARNING("[%s] opType[%u] not supported.", __func__, slot->opType);
    } else {
        slot->opType = static_cast<u8>(it->second.first);
    }
    slot->dataType = static_cast<u8>(Hccl::HcclDataTypeToDataType(static_cast<HcclDataType>(slot->dataType)));
    slot->hcclCommDfxLite = this;
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

void HcclCommDfxLite::ReportAllTasks(const std::vector<hccl::Thread*>& threads)
{
    profilingImpl_->ReportAllTasks(threads, GetDfxCommContext());
}

HcclResult HcclCommDfxLite::ReportStreamTask(Hccl::TaskInfoCircularQueue* taskQueue)
{
    profilingImpl_->ReportStreamTask(taskQueue, GetDfxCommContext());
    return HCCL_SUCCESS;
}

HcclResult HcclCommDfxLite::UpdateProfStat()
{
    profilingImpl_->UpdateProfStat();
    return HCCL_SUCCESS;
}

void HcclCommDfxLite::AddChannelRemoteRankId(u64 handle, u32 remoteRankId)
{
    channelRemoteRankIdLite_[handle] = remoteRankId;
    PLF_CONFIG_INFO(
        Hccl::PLF_TASK, "[%s] commTag[%s], handle[0x%llx], remoteRankId[%u]", __func__, commTag_.c_str(), handle,
        remoteRankId);
}

u32 HcclCommDfxLite::GetChannelRemoteRankId(u64 handle) const
{
    if (handle == DFX_INVALID_U64) {
        return INVALID_UINT;
    }
    auto it = channelRemoteRankIdLite_.find(handle);
    if (UNLIKELY(it == channelRemoteRankIdLite_.end())) {
        HCCL_ERROR("[%s]handle[%llu] not found, commTag[%s]", __func__, handle, commTag_.c_str());
        return INVALID_UINT;
    }
    return it->second;
}

const void* HcclCommDfxLite::GetLatestDfxOpInfo() const
{
    if (opInfoQueue_ == nullptr) {
        return nullptr;
    }
    auto* queue = static_cast<Hccl::DfxOpInfoCircularQueue*>(opInfoQueue_);
    if (queue->IsEmpty()) {
        return nullptr;
    }
    u16 end = queue->GetEnd();
    u16 latest = (end == 0) ? static_cast<u16>(queue->GetCapacity() - 1) : static_cast<u16>(end - 1);
    return queue->GetSlot(latest);
}

Hccl::DfxCommContext HcclCommDfxLite::GetDfxCommContext() const
{
    return {&channelRemoteRankIdLite_, groupNameHash_, localRank_, rankSize_};
}
} // namespace hccl
