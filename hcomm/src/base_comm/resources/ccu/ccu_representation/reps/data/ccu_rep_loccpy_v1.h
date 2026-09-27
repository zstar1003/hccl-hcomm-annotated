/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOCCPY_H
#define HCOMM_CCU_REPRESENTATION_LOCCPY_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLocCpy : public CcuRepBase {
    public:
        CcuRepLocCpy(
            CcuInsGeneratorBase* insGenPtr, LocalAddr dst, LocalAddr src, Variable len, CompletedEvent sem,
            uint16_t mask);
        CcuRepLocCpy(
            CcuInsGeneratorBase* insGenPtr, LocalAddr dst, LocalAddr src, Variable len, uint16_t dataType,
            uint16_t opType, CompletedEvent sem, uint16_t mask);

        // A6场景预埋
        CcuRepLocCpy(
            CcuInsGeneratorBase* insGenPtr, LocalAddr dst, LocalAddr src, Variable len, const std::vector<CcuBuf>& bufs,
            CompletedEvent sem, uint16_t mask);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t GetSrcAddrId() { return src.addr.Id(); }
        uint16_t GetSrcTokenId() { return src.token.Id(); }
        uint16_t GetDstAddrId() { return dst.addr.Id(); }
        uint16_t GetDstTokenId() { return dst.token.Id(); }
        uint16_t GetLenId() { return len.Id(); }
        uint16_t GetSemId() { return sem.Id(); }
        uint16_t GetDataType() const { return dataType; }
        uint16_t GetOpType() const { return opType; }
        const std::vector<CcuBuf>& GetBufs() { return bufs; }

        LocalAddr GetDst() { return dst; }
        LocalAddr GetSrc() { return src; }
        Variable GetLen() { return len; }
        CompletedEvent GetSem() { return sem; }
        uint16_t GetMask() const { return mask; }
        uint16_t GetReduceFlag() const { return reduceFlag; }
        bool GetUseCcuBuffer() const { return useCcuBuffer; }

        uint16_t GetFirstBufId();
        uint16_t GetUsedBufNum();

    private:
        void ValidateInsGeneratorForLocCpy() const;

        CcuInsGeneratorBase* insGenPtr{nullptr};
        LocalAddr dst;
        LocalAddr src;
        Variable len;

        // 用于A6场景locmem2locmem搬运
        std::vector<CcuBuf> bufs;

        CompletedEvent sem;
        uint16_t mask{0};

        uint16_t dataType{0};
        uint16_t opType{0};
        uint16_t reduceFlag{0};

        bool useCcuBuffer = false;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOCCPY_H
