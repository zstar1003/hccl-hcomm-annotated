/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_BUFREDUCE_H
#define HCOMM_CCU_REPRESENTATION_BUFREDUCE_H

#include <vector>

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepBufReduce : public CcuRepBase {
    public:
        CcuRepBufReduce(
            CcuInsGeneratorBase* insGenPtr, const std::vector<CcuBuf>& mem, uint16_t count, uint16_t dataType,
            uint16_t outputDataType, uint16_t opType, CompletedEvent sem, const CcuRep::Variable& len,
            uint16_t mask = 1);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        const std::vector<CcuBuf>& GetMem() { return mem; }
        uint16_t GetCount() const { return count; }
        uint16_t GetDataType() const { return dataType; }
        uint16_t GetOutputDataType() const { return outputDataType; }
        uint16_t GetOpType() const { return opType; }
        uint16_t GetXnLengthId() { return xnIdLength_.Id(); }
        uint16_t GetMask() const { return mask; }
        uint16_t GetSemId() { return sem.Id(); }

        CompletedEvent GetSem() { return sem; }
        CcuRep::Variable GetXnIdLength() { return xnIdLength_; }

    private:
        CcuInsGeneratorBase* insGenPtr;
        std::vector<CcuBuf> mem;
        uint16_t count;
        uint16_t dataType;
        uint16_t outputDataType;
        uint16_t opType;
        CompletedEvent sem;
        CcuRep::Variable xnIdLength_;

        uint16_t mask{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_BUFREDUCE_H
