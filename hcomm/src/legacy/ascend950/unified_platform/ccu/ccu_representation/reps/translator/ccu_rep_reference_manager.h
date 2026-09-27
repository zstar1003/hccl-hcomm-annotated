/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REP_REFERENCE_MANAGER_H
#define HCCL_CCU_REP_REFERENCE_MANAGER_H

#include <unordered_map>
#include <vector>
#include <memory>

#include "ccu_device_manager.h"
#include "ccu_rep_block.h"
#include "ccu_context_resource.h"

namespace Hccl {
namespace CcuRep {

    constexpr uint16_t FUNC_IN_MAX = 32;
    constexpr uint16_t FUNC_OUT_MAX = 1;
    constexpr uint16_t FUNC_NEST_MAX = 1;
    constexpr uint16_t FUNC_CALL_LAYER_INVALID = 0xFFFF;

    class CcuRepReferenceManager {
    public:
        explicit CcuRepReferenceManager(uint8_t deiId);
        static CcuResReq GetResReq(uint8_t reqDieId);
        void GetRes(CcuRepResource& res);
        std::shared_ptr<CcuRepBlock> GetRefBlock(const std::string& label);
        void SetRefBlock(const std::string& label, std::shared_ptr<CcuRepBlock> refBlock);
        uint16_t GetFuncAddr(const std::string& label);
        const Variable& GetFuncCall();
        const Variable& GetFuncRet(uint16_t callLayer);
        const std::vector<Variable>& GetFuncIn();
        const std::vector<Variable>& GetFuncOut();
        void Dump() const;
        void ClearRepReference();

    private:
        bool CheckValid(const std::string& label);
        bool CheckUnique(const std::string& label);

    private:
        uint8_t dieId{0};
        std::unordered_map<std::string, std::shared_ptr<CcuRepBlock>> referenceMap;
        std::vector<Variable> funcCallVar;
        std::vector<Variable> funcInVar;
        std::vector<Variable> funcOutVar;
    };

}; // namespace CcuRep
}; // namespace Hccl

#endif // HCCL_CCU_REP_REFERENCE_MANAGER_H
