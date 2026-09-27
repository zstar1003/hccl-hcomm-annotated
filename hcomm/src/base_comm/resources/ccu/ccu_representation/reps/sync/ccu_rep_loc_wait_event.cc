/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_loc_wait_event.h"
#include <climits>
#include "ccu_rep_v1.h"
#include "string_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"
namespace hcomm {
namespace CcuRep {

    CcuRepLocWaitEvent::CcuRepLocWaitEvent(
        CcuInsGeneratorBase* insGenPtr, const CompletedEvent& event, uint32_t mask, bool isProfiling)
        : insGenPtr(insGenPtr),
          event_(event),
          mask_(mask),
          isProfiling_(isProfiling)
    {
        type = CcuRepType::LOC_WAIT_EVENT;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    void CcuRepLocWaitEvent::SetDependencyInfo(
        const std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>& depInfo)
    {
        depInfo_ = depInfo;
    }

    std::vector<std::shared_ptr<CcuRepBase>> CcuRepLocWaitEvent::GetDependencyInfo(uint32_t bit)
    {
        // 查找给定 bit 是否存在于 depInfo_ 中
        auto it = depInfo_.find(bit);
        // 如果找到 bit，返回与之关联的 vector
        if (it != depInfo_.end()) {
            return it->second;
        }
        // 如果未找到 bit，返回一个空的 vector
        return std::vector<std::shared_ptr<CcuRepBase>>();
    }

    bool CcuRepLocWaitEvent::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepLocWaitEventTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepLocWaitEvent][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepLocWaitEvent translate failed");

        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLocWaitEvent::Translate]uint16 integer overflow occurs, "
                "instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            Hccl::CcuApiException, "integer overflow");
        instrId += instrCount;
        return translated;
    }

    std::string CcuRepLocWaitEvent::Describe()
    {
        return Hccl::StringFormat("CcuRepLocWaitEvent=id[%u], mask[%04x]", event_.Id(), mask_);
    }

}; // namespace CcuRep
}; // namespace hcomm
