/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_BUFLOCWRITE_H
#define HCOMM_CCU_REPRESENTATION_BUFLOCWRITE_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepBufLocWrite : public CcuRepBase {
    public:
        CcuRepBufLocWrite(
            CcuInsGeneratorBase* insGenPtr, CcuBuf src, LocalAddr dst, Variable len, CompletedEvent sem, uint32_t mask);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t GetSrcAddrId() { return src.Id(); }
        uint16_t GetDstTokenId() { return dst.token.Id(); }
        uint16_t GetDstAddrId() { return dst.addr.Id(); }
        uint16_t GetLenId() { return len.Id(); }
        uint16_t GetSemId() { return sem.Id(); }

        Variable GetLen() { return len; }
        CcuBuf GetSrc() { return src; }
        LocalAddr GetDst() { return dst; }
        CompletedEvent GetSem() { return sem; }
        uint16_t GetMask() const { return mask; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        CcuBuf src;
        LocalAddr dst;
        Variable len;

        CompletedEvent sem;
        uint32_t mask{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_BUFLOCWRITE_H
