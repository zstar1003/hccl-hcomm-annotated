/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOP_BLOCK_H
#define HCOMM_CCU_REPRESENTATION_LOOP_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_arg_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLoopBlock : public CcuRepBlock {
    public:
        explicit CcuRepLoopBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label);
        std::string Describe() override;

        void DefineArg(Variable var);
        void DefineArg(Memory mem);
        void DefineArg(LocalAddr addr);
        void DefineArg(RemoteAddr addr);

        void DefineArg(const std::vector<Variable> varList);
        void DefineArg(const std::vector<Memory> memList);
        void DefineArg(const std::vector<LocalAddr> addrList);
        void DefineArg(const std::vector<RemoteAddr> addrList);

        CcuRepArg& GetArg(uint16_t index);

    private:
        std::vector<CcuRepArg> args;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOOP_BLOCK_H
