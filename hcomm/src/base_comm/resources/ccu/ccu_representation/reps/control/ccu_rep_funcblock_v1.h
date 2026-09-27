/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_FUNC_BLOCK_H
#define CCU_REPRESENTATION_FUNC_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_arg_v1.h"
#include "ccu_rep_reference_manager_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepFuncBlock : public CcuRepBlock {
    public:
        explicit CcuRepFuncBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;

        void SetFuncManager(CcuRepReferenceManager* funcManager);

        void DefineInArg(const Variable& var);
        void DefineOutArg(const Variable& var);
        void DefineInArg(const std::vector<Variable>& varList);
        void DefineOutArg(const std::vector<Variable>& varList);

        void SetCallLayer(uint16_t callLayer);
        uint16_t GetCallLayer() const;
        std::vector<Variable> GetInArgVars() const;

        CcuRepReferenceManager* GetFuncManager() { return funcManager; }

        std::vector<CcuRepArg>& GetInArgs() { return inArgs; }

        std::vector<CcuRepArg>& GetOutArgs() { return outArgs; }

    private:
        CcuRepReferenceManager* funcManager{nullptr};

        std::vector<CcuRepArg> inArgs;
        std::vector<CcuRepArg> outArgs;
        uint64_t inArgCount{0};
        uint64_t outArgCount{0};

        uint16_t callLayer{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_FUNC_BLOCK_H
