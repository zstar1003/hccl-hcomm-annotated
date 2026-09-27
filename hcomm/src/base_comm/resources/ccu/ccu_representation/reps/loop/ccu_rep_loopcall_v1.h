/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOP_CALL_H
#define HCOMM_CCU_REPRESENTATION_LOOP_CALL_H

#include "ccu_rep_base_v1.h"
#include "ccu_rep_loopblock_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLoopCall : public CcuRepBase {
    public:
        explicit CcuRepLoopCall(CcuInsGeneratorBase* insGeneratorPtr, const std::string& label);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;
        const std::string& GetLabel() const;

        void Reference(std::shared_ptr<CcuRepLoopBlock> refRep);

        void SetInArg(const Variable& var);
        void SetInArg(const std::vector<Variable>& varList);
        void SetInArg(const Memory& mem);
        void SetInArg(const std::vector<Memory>& memList);

        void SetInArg(const LocalAddr& addr);
        void SetInArg(const RemoteAddr& addr);
        void SetInArg(const std::vector<LocalAddr>& addrList);
        void SetInArg(const std::vector<RemoteAddr>& addrList);

        std::shared_ptr<CcuRepLoopBlock> GetLoopBlock() { return loopBlock; }
        std::vector<CcuRepArg>& GetInArgs() { return inArgs; }
        uint32_t GetInArgCount() const { return inArgCount; }

    private:
        CcuInsGeneratorBase* insGeneratorPtr_;
        std::string label;
        std::shared_ptr<CcuRepLoopBlock> loopBlock{nullptr};

        std::vector<CcuRepArg> inArgs;
        uint32_t inArgCount{0};
        uint32_t inArgInstrCount{0}; // 处理LoopCall的入参需要的指令数

        CcuInstr* instr{nullptr};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOOP_CALL_H
