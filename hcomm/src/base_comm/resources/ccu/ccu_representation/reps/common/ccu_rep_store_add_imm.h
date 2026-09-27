/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_STORE_ADD_IMM_H
#define HCOMM_CCU_REPRESENTATION_STORE_ADD_IMM_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepStoreAddImm : public CcuRepBase {
    public:
        CcuRepStoreAddImm(
            CcuInsGeneratorBase* insGenPtr, Variable& dst, uint16_t dstNum, Variable& dstOffset, uint16_t immAddValue,
            Variable& src);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        uint16_t GetDstId() { return dst.Id(); }
        uint16_t GetDstOffsetId() { return dstOffset.Id(); }
        uint16_t GetSrcId() { return src.Id(); }
        uint16_t GetImmAddValue() { return immAddValue; }

        const Variable& GetDst() const { return dst; }
        const Variable& GetDstOffset() const { return dstOffset; }
        const Variable& GetSrc() const { return src; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        Variable dst;
        Variable dstOffset;
        Variable src;
        uint16_t immAddValue{0};
        uint16_t dstNum{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_STORE_ADD_IMM_H
