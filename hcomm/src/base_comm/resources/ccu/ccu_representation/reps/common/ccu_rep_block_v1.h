/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_BLOCK_H
#define CCU_REPRESENTATION_BLOCK_H

#include <vector>
#include <memory>

#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepBlock : public CcuRepBase {
    public:
        explicit CcuRepBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label = "");
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;
        const std::string& GetLabel() const;
        std::vector<std::shared_ptr<CcuRepBase>>& GetReps();
        void Append(std::shared_ptr<CcuRepBase> rep);
        std::shared_ptr<CcuRepBase> GetRepByInstrId(uint16_t instrId);

    protected:
        CcuInsGeneratorBase* insGeneratorPtr_{nullptr};

    private:
        std::string label;
        std::vector<std::shared_ptr<CcuRepBase>> repVec;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_BLOCK_H
