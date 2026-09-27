/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOC_WAIT_EVENT_H
#define HCOMM_CCU_REPRESENTATION_LOC_WAIT_EVENT_H

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLocWaitEvent : public CcuRepBase {
    public:
        // mask 由调用方显式传入（与 CompletedEvent 解耦）。
        CcuRepLocWaitEvent(
            CcuInsGeneratorBase* insGenPtr, const CompletedEvent& event, uint32_t mask, bool isProfiling = true);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint32_t GetMask() const { return mask_; };
        uint32_t GetEventId() { return event_.Id(); };

        CompletedEvent GetEvent() { return event_; }

        bool GetIsProfiling() const { return isProfiling_; }
        void SetDependencyInfo(const std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>& depInfo);
        std::vector<std::shared_ptr<CcuRepBase>> GetDependencyInfo(uint32_t bit);

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        CompletedEvent event_{};
        uint32_t mask_{1};
        bool isProfiling_{true};
        std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>> depInfo_;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOC_WAIT_EVENT_H
