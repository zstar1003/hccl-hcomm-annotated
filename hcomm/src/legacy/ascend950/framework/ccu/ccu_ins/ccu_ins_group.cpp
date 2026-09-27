/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_group.h"

namespace Hccl {

void CcuInsGroup::SetExecId(u64 id)
{
    for (auto& ins : ccuInstructions) {
        ins->SetExecId(id);
    }

    execId = id;
}

u64 CcuInsGroup::GetExecId() const { return execId; }

void CcuInsGroup::Append(std::unique_ptr<CcuInstruction> ins) { ccuInstructions.emplace_back(std::move(ins)); }

const std::vector<std::unique_ptr<CcuInstruction>>& CcuInsGroup::GetCcuInstructions() const { return ccuInstructions; }

CcuCtxSignature CcuInsGroup::GetCtxSignature() const
{
    HCCL_INFO("[CcuInsGroup::GetCtxSignature] start");
    CcuCtxSignature ctxSignature;
    for (auto& ins : ccuInstructions) {
        ctxSignature.Append(ins->GetCtxArg()->GetCtxSignature());
        HCCL_INFO(
            "[CcuInsGroup::GetCtxSignature] ins->GetCtxSignature()[%s], ctxSignature[%s]",
            ins->GetCtxSignature().Describe().c_str(), ctxSignature.Describe().c_str());
    }
    HCCL_INFO("[CcuInsGroup::GetCtxSignature] end, ctxSignature[%s]", ctxSignature.Describe().c_str());
    return ctxSignature;
}

std::string CcuInsGroup::Describe() const
{
    return StringFormat("CcuInsGroup[ccuInstructions_size=%zu, execId=%llu]", ccuInstructions.size(), execId);
}

CcuInstType CcuInsGroup::GetInstType() const { return CcuInstType::CCU_INS_GROUP; }

std::unique_ptr<CcuTaskArg> CcuInsGroup::GetTaskArg() const
{
    if (ccuInstructions.size() == 0) {
        THROW<InternalException>("[CcuInsGroup] size is 0");
    }
    return ccuInstructions[0]->GetTaskArg();
}

std::unique_ptr<CcuCtxArg> CcuInsGroup::GetCtxArg() const
{
    THROW<NotSupportException>("[CcuInsGroup] not support GetCtxArg");
    return nullptr;
}

std::vector<LinkData> CcuInsGroup::GetLinks() const
{
    vector<LinkData> links;
    for (const auto& ins : GetCcuInstructions()) {
        vector<LinkData> tmpLinks = ins->GetLinks();
        links.insert(links.end(), tmpLinks.begin(), tmpLinks.end());
    }
    HCCL_INFO("[CcuInsGroup][GetInsLinks] Get all ins ccuLinks of ccuInsGroup, links size[%zu]", links.size());
    return links;
}

RankGroup CcuInsGroup::GetRankGroup() const
{
    THROW<NotSupportException>("[CcuInsGroup]  not support GetRankGroup");
    return RankGroup();
}

} // namespace Hccl
