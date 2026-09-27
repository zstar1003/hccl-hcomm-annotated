/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <hccl/hccl_types.h>
#include "hccd_impl_pml.h"
#include "hccd_comm.h"
namespace hccl {
HccdComm::HccdComm(std::string identifier) : impl_(nullptr), identifier_(identifier) {}

HccdComm::~HccdComm() { impl_ = nullptr; }

HcclResult HccdComm::init(HcclCommParams& params, const RankTable_t& rankTable)
{
    HCCL_INFO("HccdComm init workmode [%d]", params.commWorkMode);

    CHK_RET(InitImpl());

    /* 强行将最后一个字符置0, 确保其可以做字符串操作 */
    params.id.internal[HCCL_ROOT_INFO_BYTES - 1] = '\0';

    /* 入参判断 */
    if (params.rank >= params.totalRanks) {
        HCCL_ERROR(
            "[HcclComm][Init]errNo[0x%016llx] rank[%u] out of range[0, %u]", HCCL_ERROR_CODE(HCCL_E_PARA), params.rank,
            params.totalRanks - 1);
        return HCCL_E_PARA;
    }
    params.identifier = identifier_;
    CHK_RET(impl_->AtomicInitSet());                 /* 初始化竞争, 只允许被初始化一次 */
    HcclResult ret = impl_->Init(params, rankTable); /* 初始化实例, 失败则重新开放初始化竞争 */
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] hccl initialize failed", HCCL_ERROR_CODE(ret));
        impl_->AtomicInitClear();
        return ret;
    }

    HCCL_RUN_INFO(
        "hccdCommInitInfo:commId[%s], rank[%u], totalRanks[%u], serverId[%s], deviceType[%d],"
        "logicDevId[%d], identifier[%s]",
        params.id.internal, params.rank, params.totalRanks, params.serverId.c_str(), params.deviceType,
        params.logicDevId, params.identifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult HccdComm::RegisterMemory(void* buffer, uint64_t size) { return impl_->RegisterMemory(buffer, size); }

HcclResult HccdComm::UnregisterMemory(void* buffer) { return impl_->UnregisterMemory(buffer); }

HcclResult HccdComm::Isend(
    void* buffer, s32 count, HcclDataType dataType, u32 peerRank, s32 tag, HcclRequest& request, u32 userRequire) const
{
    /* 入参检查 */
    CHK_RET(impl_->CheckCount(count));
    CHK_RET(impl_->CheckDataType(dataType, false));
    return impl_->Isend(buffer, count, dataType, peerRank, tag, request, userRequire);
}

HcclResult HccdComm::Improbe(u32 peerRank, s32 tag, s32& flag, HcclMessage& msgHandle, HcclStatus& status) const
{
    return impl_->Improbe(peerRank, tag, flag, msgHandle, status);
}

HcclResult HccdComm::Imrecv(void* buffer, s32 count, HcclDataType dataType, HcclMessage msg, HcclRequest& request) const
{
    /* 入参检查 */
    CHK_RET(impl_->CheckCount(count));
    CHK_RET(impl_->CheckDataType(dataType, false));
    return impl_->Imrecv(buffer, count, dataType, msg, request);
}

HcclResult HccdComm::HcclTest(HcclRequest hcclRequest, s32& flag, HcclStatus& compState) const
{
    return impl_->HcclTest(hcclRequest, flag, compState);
}

HcclResult HccdComm::GetUserRank(u32& userRank)
{
    userRank = impl_->GetUserRank();
    return HCCL_SUCCESS;
}

HcclResult HccdComm::GetRankSize(u32& rankSize)
{
    rankSize = impl_->GetRankSize();
    return HCCL_SUCCESS;
}

const std::string& HccdComm::GetIdentifier() { return identifier_; }

HcclResult HccdComm::InitImpl()
{
    impl_.reset(new (std::nothrow) HccdImplPml());
    CHK_SMART_PTR_NULL(impl_);
    return HCCL_SUCCESS;
}

HcclResult HccdComm::GetUniqueId(HcclRootInfo* uniqueId)
{
    CHK_PTR_NULL(uniqueId);

    std::string uniqueIdGot = HccdImplPml::GetUniqueId();
    s32 ret = snprintf_s(
        uniqueId->internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s", "hccl-", uniqueIdGot.c_str());
    CHK_PRT_RET(
        (ret == -1),
        HCCL_ERROR("[Get][UniqueId]errNo[0x%016llx] get unique id failed,uniqueId[%p]", HCCL_ERROR_CODE(ret), uniqueId),
        HCCL_E_MEMORY);

    return HCCL_SUCCESS;
}
} // namespace hccl
