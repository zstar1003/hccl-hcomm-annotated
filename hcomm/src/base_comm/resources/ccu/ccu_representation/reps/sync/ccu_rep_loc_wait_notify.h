/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOC_WAIT_NOTIFY_H
#define HCOMM_CCU_REPRESENTATION_LOC_WAIT_NOTIFY_H
#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLocWaitNotify : public CcuRepBase {
    public:
        CcuRepLocWaitNotify(
            CcuInsGeneratorBase* insGenPtr, const LocalNotify& notify, const uint32_t mask, bool isProfiling = true);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint32_t GetMask() const { return mask_; };
        uint16_t GetNotifyId() { return notify_.Id(); };

        LocalNotify GetNotify() { return notify_; }

        bool GetIsProfiling() const { return isProfiling_; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        LocalNotify notify_{};
        uint32_t mask_{0};
        bool isProfiling_{true};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOC_WAIT_NOTIFY_H
