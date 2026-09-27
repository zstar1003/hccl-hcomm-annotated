/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_FUNC_CALL_H
#define HCCL_CCU_REPRESENTATION_FUNC_CALL_H

#include "ccu_rep_base.h"
#include "ccu_rep_funcblock.h"
#include "ccu_rep_reference_manager.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepFuncCall : public CcuRepBase {
    public:
        explicit CcuRepFuncCall(const std::string& label);
        explicit CcuRepFuncCall(const Variable& funcAddrVar);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;
        const std::string& GetLabel() const;

        void Reference(std::shared_ptr<CcuRepFuncBlock> refRep);
        void SetFuncManager(CcuRepReferenceManager* funcManager);

        void SetInArg(const Variable& var);
        void SetOutArg(const Variable& var);
        void SetInArg(const std::vector<Variable>& varList);
        void SetOutArg(const std::vector<Variable>& varList);

        int32_t GetCallLayer();

    private:
        CcuRepReferenceManager* funcManager{nullptr};

        std::string label;
        std::shared_ptr<CcuRepFuncBlock> funcBlock{nullptr};
        Variable funcAddrVar;

        std::vector<CcuRepArg> inArgs;
        std::vector<CcuRepArg> outArgs;
        uint32_t inArgCount{0};
        uint32_t outArgCount{0};

        CcuInstr* instr{nullptr};
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_FUNC_CALL_H
