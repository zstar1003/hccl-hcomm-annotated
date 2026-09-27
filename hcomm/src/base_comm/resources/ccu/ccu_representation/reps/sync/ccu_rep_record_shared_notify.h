/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_RECORD_SHARED_NOTIFY_H
#define HCOMM_CCU_REPRESENTATION_RECORD_SHARED_NOTIFY_H

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepRecordSharedNotify : public CcuRepBase {
    public:
        explicit CcuRepRecordSharedNotify(CcuInsGeneratorBase* insGenPtr, const LocalNotify& notify, uint16_t mask);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t GetNotifyId() { return notify_.Id(); }
        LocalNotify GetNotify() { return notify_; }

        uint16_t GetMask() const { return mask_; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        LocalNotify notify_{};
        uint16_t mask_{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_RECORD_SHARED_NOTIFY_H
